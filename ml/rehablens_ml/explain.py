"""Integrated-Gradients joint attributions + a deletion-based faithfulness test; split-conformal prediction intervals;
few-shot per-user personalization with a low-rank adapter."""
import numpy as np
import torch
import torch.nn as nn

from .model import FINGERS


# ---------------------------------------------------------------- Integrated Gradients
def integrated_gradients(model, x, dev, steps=12, target="score"):
    """x [B,T,75] -> attribution [B,T,75] for the model's score output. Baseline = each sequence's own mean pose
    (a frozen hand), so attributions measure how MOTION of each joint contributes to the score."""
    model.eval()
    x = x.to(dev)
    base = x.mean(1, keepdim=True).expand_as(x)
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xi = (base + (k / steps) * (x - base)).clone().requires_grad_(True)
        out = model(xi)[target]
        out.sum().backward()
        total += xi.grad
    return ((x - base) * total / steps).detach()


def joint_importance(attr):
    """attr [B,T,75] -> [B,25] mean |attribution| per joint (summed over time and xyz), normalised to sum 1."""
    a = attr.abs().reshape(attr.shape[0], attr.shape[1], 25, 3).sum((1, 3))
    return (a / (a.sum(1, keepdim=True) + 1e-12)).cpu().numpy()


def finger_importance(ji):
    """ji [B,25] -> {finger: mean importance share}."""
    return {f: float(ji[:, idx].sum(1).mean()) for f, idx in FINGERS.items()}


@torch.no_grad()
def deletion_test(model, x, imp, dev, k=6):
    """Faithfulness: freeze the k most-important joints (replace by their time-mean) vs k random joints.
    A faithful explanation changes the score more when the *important* joints are frozen."""
    model.eval()
    x = x.to(dev)
    base = model(x)["score"]
    j = x.reshape(x.shape[0], x.shape[1], 25, 3)
    mean = j.mean(1, keepdim=True)
    imp = torch.tensor(imp, device=dev)
    top = imp.topk(k, dim=1).indices
    rnd = torch.stack([torch.randperm(25, device=dev)[:k] for _ in range(len(x))])
    res = {}
    for name, idx in (("top", top), ("random", rnd)):
        m = torch.zeros(len(x), 1, 25, 1, device=dev)
        m.scatter_(2, idx[:, None, :, None], 1.0)
        xd = (j * (1 - m) + mean * m).reshape_as(x)
        res[name] = float((model(xd)["score"] - base).abs().mean() * 100)
    return res  # mean |score change| in points


# ---------------------------------------------------------------- split conformal
def conformal_eval(y, pred, sigma, groups, alpha=0.1, seed=0):
    """Out-of-fold predictions are valid calibration data (each came from a model that never saw that subject).
    Calibrate on half of the SUBJECTS, test coverage on the other half. Nonconformity = |y - pred| / sigma."""
    rng = np.random.default_rng(seed)
    subj = np.unique(groups)
    rng.shuffle(subj)
    cal_s = set(subj[: len(subj) // 2])
    cal = np.array([g in cal_s for g in groups])
    s = np.abs(y - pred) / np.maximum(sigma, 1e-3)
    n = cal.sum()
    q = np.quantile(s[cal], min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    test = ~cal
    cover = float(np.mean(s[test] <= q))
    width = float(np.mean(2 * q * sigma[test]))
    naive = float(np.mean(np.abs(y[test] - pred[test]) <= 1.645 * sigma[test]))  # uncalibrated Gaussian 90% interval
    qall = float(np.quantile(s, min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s))))
    return dict(alpha=alpha, qhat=float(q), qhat_all=qall, coverage=cover, naive_gauss_coverage=naive, mean_width_points=width)


# ---------------------------------------------------------------- few-shot personalization
def personalize_eval(z, y, pred0, groups, head, ks=(0, 3, 6, 12), reps=3, rank=4, steps=60, seed=0):
    """For each unseen subject: K labelled windows (e.g. rated once by the clinician) -> fit a low-rank adapter on the
    frozen embedding z (score = head(z + (z U) V)) and measure MAE on that subject's remaining windows.
    z [N,d] torch (frozen features), y [N] true score/100 (np), pred0 [N] base predictions/100."""
    rng = np.random.default_rng(seed)
    z = z.detach()
    out = {k: [] for k in ks}
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        for k in ks:
            maes = []
            for _ in range(reps if k > 0 else 1):
                perm = rng.permutation(idx)
                sup, qry = perm[:k], perm[k:]
                if len(qry) < 5:
                    continue
                if k == 0:
                    pred = pred0[qry]
                else:
                    U = nn.Parameter(torch.zeros(z.shape[1], rank, device=z.device))
                    V = nn.Parameter(torch.randn(rank, z.shape[1], device=z.device) * 0.05)
                    b = nn.Parameter(torch.zeros(1, device=z.device))
                    opt = torch.optim.Adam([U, V, b], 1e-2, weight_decay=1e-3)
                    zs, ys = z[sup], torch.tensor(y[sup], dtype=torch.float32, device=z.device)
                    for _ in range(steps):
                        pr = head(zs + (zs @ U) @ V)[:, 0] + b
                        loss = ((pr - ys) ** 2).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        zq = z[qry]
                        pred = (head(zq + (zq @ U) @ V)[:, 0] + b).cpu().numpy()
                maes.append(np.abs(pred - y[qry]).mean() * 100)
            if maes:
                out[k].append(float(np.mean(maes)))
    return {int(k): float(np.mean(v)) for k, v in out.items() if v}
