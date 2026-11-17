"""Models.

RehabNet  : kinematic-tree-aware spatial Transformer (per frame, 25 joints) + temporal mixer over frames
            (temporal='mamba' : selective state-space blocks with input-dependent dt/B/C, Mamba-style, pure PyTorch;
             temporal='s4d'   : diagonal linear SSM via FFT convolution)
LSTMNet   : bidirectional GRU baseline (same heads) for the baseline ladder
Student   : small 1D-CNN distilled from the teacher, exported to ONNX for the headset browser
Heads     : exercise class, form score + log-variance (heteroscedastic uncertainty); SSL heads for pretraining.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# WebXR hand bones: parent of each of the 25 joints (wrist = -1)
PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23]
FINGERS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8, 9], "middle": [10, 11, 12, 13, 14], "ring": [15, 16, 17, 18, 19], "little": [20, 21, 22, 23, 24]}


def tree_bias():
    """[25,25] attention bias: 0 for self, decreasing with graph distance along the bone tree."""
    d = torch.full((25, 25), 99.0)
    for i in range(25):
        d[i, i] = 0
        if PARENT[i] >= 0:
            d[i, PARENT[i]] = d[PARENT[i], i] = 1
    for k in range(25):
        d = torch.minimum(d, d[:, k:k + 1] + d[k:k + 1, :])
    return -0.5 * d.clamp(max=8)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DiagSSM(nn.Module):
    """Diagonal linear state-space layer applied as a long convolution via FFT (S4D-style, real poles)."""
    def __init__(self, d, n=16):
        super().__init__()
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.A = nn.Parameter(-torch.arange(1, n + 1).float().repeat(d, 1) * 0.5)
        self.C = nn.Parameter(torch.randn(d, n) * n ** -0.5)
        self.D = nn.Parameter(torch.ones(d))
        self.glu = nn.Linear(d, 2 * d)

    def forward(self, x):  # [B,T,d]
        B, T, d = x.shape
        dt = self.log_dt.exp()[:, None, None]
        t = torch.arange(T, device=x.device)[None, None, :]
        K = (self.C[:, :, None] * torch.exp(self.A[:, :, None] * dt * t)).sum(1) * dt[:, 0]
        n = 2 * T
        y = torch.fft.irfft(torch.fft.rfft(x.transpose(1, 2), n) * torch.fft.rfft(K, n)[None], n)[..., :T]
        y = y.transpose(1, 2) + x * self.D
        return F.glu(self.glu(F.gelu(y)), dim=-1)


class MambaBlock(nn.Module):
    """Selective state-space block (Mamba): dt, B, C depend on the input; recurrence h_t = exp(dt*A) h_{t-1} + dt*B*x_t."""
    def __init__(self, d, n=16, expand=2, dconv=4):
        super().__init__()
        self.n, self.di = n, expand * d
        self.inp = nn.Linear(d, 2 * self.di)
        self.conv = nn.Conv1d(self.di, self.di, dconv, padding=dconv - 1, groups=self.di)
        self.x_proj = nn.Linear(self.di, 1 + 2 * n)
        self.dt_proj = nn.Linear(1, self.di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(self.di, 1))
        self.D = nn.Parameter(torch.ones(self.di))
        self.out = nn.Linear(self.di, d)

    def forward(self, x):  # [B,T,d]
        B, T, _ = x.shape
        n = self.n
        xi, z = self.inp(x).chunk(2, -1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :T].transpose(1, 2))
        dbc = self.x_proj(xi)
        dt = F.softplus(self.dt_proj(dbc[..., :1]))                      # [B,T,di]
        Bm, Cm = dbc[..., 1:1 + n], dbc[..., 1 + n:]                       # [B,T,n]
        A = -torch.exp(self.A_log)                                         # [di,n]
        dA = torch.exp(dt.unsqueeze(-1) * A)                               # [B,T,di,n]
        dBx = (dt * xi).unsqueeze(-1) * Bm.unsqueeze(2)                    # [B,T,di,n]
        h = x.new_zeros(B, self.di, n)
        ys = []
        for t in range(T):  # sequential selective scan (T = 300)
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + xi * self.D
        return self.out(y * F.silu(z))


class Block(nn.Module):
    def __init__(self, d, heads, temporal):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ssm = MambaBlock(d) if temporal == "mamba" else DiagSSM(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.ssm(self.n1(x))
        h = self.n2(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n3(x))


class Heads(nn.Module):
    def __init__(self, d, n_ex=3):
        super().__init__()
        self.h_ex = nn.Linear(d, n_ex)
        self.h_score = nn.Linear(d, 2)  # mean (score/100), log-variance
        self.h_dom = nn.Linear(d, 2)    # domain classifier (synthetic vs real) for adversarial adaptation

    def task(self, z, dom_lambda=0.0):
        s = self.h_score(z)
        out = {"ex": self.h_ex(z), "score": s[:, 0], "logvar": s[:, 1].clamp(-7, 1), "z": z}
        if dom_lambda > 0:
            out["dom"] = self.h_dom(GradReverse.apply(z, dom_lambda))
        return out


class RehabNet(nn.Module):
    def __init__(self, d=96, heads=4, depth=3, n_ex=3, temporal="mamba", stride=2):
        super().__init__()
        self.d, self.stride = d, stride  # stride 2: the encoder runs at 15 Hz (2x faster; 30 Hz input is subsampled inside)
        self.embed = nn.Linear(3, d)
        self.jemb = nn.Parameter(torch.randn(25, d) * 0.02)
        self.register_buffer("tbias", tree_bias())
        self.spatial = nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True, norm_first=True, dropout=0.0)
        self.pool = nn.Linear(25 * d, d)
        self.blocks = nn.ModuleList([Block(d, heads, temporal) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.mask_tok = nn.Parameter(torch.zeros(3))
        self.recon = nn.Linear(d, 75)                                   # masked-joint reconstruction head
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 64))  # contrastive projection head
        self.heads = Heads(d, n_ex)

    def encode(self, x, mask=None):
        """x [B,T,75] (30 Hz); mask [B,T/stride,25] bool (True = hidden) -> frame embeddings [B,T/stride,d]."""
        x = x[:, ::self.stride]
        B, T, _ = x.shape
        j = x.reshape(B, T, 25, 3)
        if mask is not None:
            j = torch.where(mask[..., None], self.mask_tok.view(1, 1, 1, 3).expand_as(j), j)
        h = (self.embed(j) + self.jemb).reshape(B * T, 25, self.d)
        h = self.spatial(h, src_mask=self.tbias.to(h.dtype))
        h = self.pool(h.reshape(B * T, -1)).reshape(B, T, self.d)
        for b in self.blocks:
            h = b(h)
        return self.norm(h)

    def forward(self, x, mask=None, mode="task", dom_lambda=0.0):
        h = self.encode(x, mask)
        if mode == "ssl":
            return self.recon(h)
        if mode == "contrast":
            return F.normalize(self.proj(h.mean(1)), dim=-1)
        return self.heads.task(h.mean(1), dom_lambda)


class LSTMNet(nn.Module):
    """Bidirectional GRU baseline with the same heads."""
    def __init__(self, d=96, n_ex=3):
        super().__init__()
        self.rnn = nn.GRU(75, d, num_layers=2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d, d)
        self.heads = Heads(d, n_ex)

    def forward(self, x, mask=None, mode="task", dom_lambda=0.0):
        h, _ = self.rnn(x[:, ::2])  # 15 Hz is plenty for a GRU baseline
        return self.heads.task(F.gelu(self.proj(h.mean(1))), dom_lambda)


def make_model(arch="mamba"):
    if arch == "lstm":
        return LSTMNet()
    return RehabNet(temporal=arch)  # 'mamba' | 's4d'


class Student(nn.Module):
    """Small 1D-CNN for on-headset inference (distilled from the teacher)."""
    def __init__(self, c=48, n_ex=3):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv1d(i, o, 5, s, 2), nn.BatchNorm1d(o), nn.GELU())
        self.net = nn.Sequential(blk(75, c, 2), blk(c, c, 2), blk(c, 2 * c, 2), blk(2 * c, 2 * c, 1))
        self.h_ex = nn.Linear(2 * c, n_ex)
        self.h_score = nn.Linear(2 * c, 2)

    def forward(self, x):  # [B,T,75]
        z = self.net(x.transpose(1, 2)).mean(-1)
        s = self.h_score(z)
        return self.h_ex(z), s[:, 0], s[:, 1]
