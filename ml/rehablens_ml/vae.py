"""Motion-manifold VAE.
 (1) one-class scoring : trained on GOOD-form reps only; reconstruction error = distance from the good-motion manifold
                         -> a second, label-free quality score (isotonic-mapped to 0-100 on training data).
 (2) generative augment: conditional VAE trained on all reps with condition = (exercise one-hot, score/100); sampling new
                         latents with a target score yields extra labelled sequences.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 300


class MotionVAE(nn.Module):
    def __init__(self, cond=3, zdim=32, c=64):
        super().__init__()
        self.cond, self.zdim, self.c = cond, zdim, c
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, 2, 2), nn.GELU())
        self.enc = nn.Sequential(blk(75 + cond, c), blk(c, c), blk(c, 2 * c), blk(2 * c, 2 * c))  # 300 -> 19
        self.mu = nn.Linear(2 * c * 19, zdim)
        self.lv = nn.Linear(2 * c * 19, zdim)
        self.fc = nn.Linear(zdim + cond, 2 * c * 19)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(2 * c, 2 * c, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(2 * c, c, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(c, c, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(c, c, 4, 2, 1), nn.GELU(),
        )
        self.out = nn.Conv1d(c, 75, 5, 1, 2)

    def encode(self, x, cond):
        ct = cond[:, :, None].expand(-1, -1, x.shape[1])
        h = self.enc(torch.cat([x.transpose(1, 2), ct], 1)).flatten(1)
        return self.mu(h), self.lv(h)

    def decode(self, z, cond):
        h = self.fc(torch.cat([z, cond], 1)).view(-1, 2 * self.c, 19)
        h = self.dec(h)
        h = F.interpolate(h, size=T, mode="linear", align_corners=False)
        return self.out(h).transpose(1, 2)

    def forward(self, x, cond):
        mu, lv = self.encode(x, cond)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.decode(z, cond), mu, lv


def vae_loss(model, x, cond, beta):
    rec, mu, lv = model(x, cond)
    r = F.mse_loss(rec, x)
    kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
    return r + beta * kl, r.item(), kl.item()


def fit_vae(X, cond, epochs, dev, bs=32, beta=0.02, lr=1e-3, log=print):
    m = MotionVAE(cond.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr)
    Xt, Ct = torch.tensor(X), torch.tensor(cond, dtype=torch.float32)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            loss, r, k = vae_loss(m, Xt[b].to(dev), Ct[b].to(dev), beta * min(1.0, (ep + 1) / max(1, epochs // 3)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += r
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"  vae ep{ep + 1}/{epochs} recon {tot / (len(Xt) / bs):.4f}")
    return m


@torch.no_grad()
def recon_error(m, X, cond, dev, bs=64):
    m.eval()
    errs = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs]).to(dev)
        c = torch.tensor(cond[i:i + bs], dtype=torch.float32).to(dev)
        mu, _ = m.encode(x, c)
        errs.append(((m.decode(mu, c) - x) ** 2).mean((1, 2)).cpu().numpy())
    return np.concatenate(errs)


@torch.no_grad()
def generate(m, cond, dev, temp=0.9):
    m.eval()
    c = torch.tensor(cond, dtype=torch.float32).to(dev)
    z = torch.randn(len(c), m.zdim, device=dev) * temp
    return m.decode(z, c).cpu().numpy()
