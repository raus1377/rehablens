"""Torch augmentations for self-supervised views and training. x: [B,T,75] (wrist-centred, size-normalised)."""
import math
import torch


def time_warp(x, lo=0.8, hi=1.25):
    """Random playback-speed change with periodic wrap (exercises are periodic)."""
    B, T, C = x.shape
    s = lo + (hi - lo) * torch.rand(B, 1, device=x.device)
    pos = (torch.arange(T, device=x.device)[None] * s) % (T - 1)
    i0 = pos.floor().long()
    w = (pos - i0)[..., None]
    i1 = (i0 + 1).clamp(max=T - 1)
    g = lambda i: torch.gather(x, 1, i[..., None].expand(B, T, C))
    return g(i0) * (1 - w) + g(i1) * w


def rotate(x, max_deg=12.0):
    B, T, _ = x.shape
    a = (torch.rand(B, 3, device=x.device) * 2 - 1) * math.radians(max_deg)
    cx, cy, cz, sx, sy, sz = a[:, 0].cos(), a[:, 1].cos(), a[:, 2].cos(), a[:, 0].sin(), a[:, 1].sin(), a[:, 2].sin()
    R = torch.stack([
        torch.stack([cy * cz, -cy * sz, sy], -1),
        torch.stack([sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy], -1),
        torch.stack([-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy], -1)], 1)  # [B,3,3]
    return (x.reshape(B, T, 25, 3) @ R.transpose(1, 2)[:, None]).reshape(B, T, 75)


def mirror(x, p=0.5):
    B, T, _ = x.shape
    flip = (torch.rand(B, 1, 1, device=x.device) < p).float()          # [B,1,1]
    j = x.reshape(B, T, 25, 3).clone()
    j[..., 0] = j[..., 0] * (1 - 2 * flip)                              # mirror the x axis
    return j.reshape(B, T, 75)


def view(x, strength=1.0):
    """One stochastic augmented view (used for contrastive learning)."""
    x = time_warp(x)
    x = rotate(x, 12.0 * strength)
    x = mirror(x)
    x = x * (1 + 0.05 * strength * torch.randn(x.shape[0], 1, 1, device=x.device))
    return x + 0.01 * strength * torch.randn_like(x)


def light(x):
    """Mild augmentation for supervised training (does not change score-relevant motion statistics)."""
    x = x * (1 + 0.05 * torch.randn(x.shape[0], 1, 1, device=x.device)) + 0.01 * torch.randn_like(x)
    return rotate(x, 6.0)


def nt_xent(z1, z2, tau=0.2):
    """SimCLR NT-Xent loss between two batches of normalised embeddings."""
    B = z1.shape[0]
    z = torch.cat([z1, z2])
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return torch.nn.functional.cross_entropy(sim, tgt)
