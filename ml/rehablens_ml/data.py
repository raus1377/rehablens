"""Load real Quest recordings (JSON from the WebXR recorder) into model-ready arrays."""
import glob, json, os
import numpy as np

FORM_SCORE = {"good": 88.0, "limited_rom": 50.0, "tremor": 58.0, "slow": 62.0, "irregular": 55.0}  # WEAK labels
EX = ["fist", "finger_tap", "wrist_rot"]


def load_recordings(folder, fs=30, T=300):
    X, y, ex, groups, forms = [], [], [], [], []
    for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(fp))
        m = d["meta"]
        fr = np.asarray(d["frames"], dtype=np.float32)  # [n, 2+150]
        t, mask = fr[:, 0], fr[:, 1].astype(int)
        side = 2 + 75 if (mask & 2).mean() >= (mask & 1).mean() else 2  # use the better-tracked hand
        pos = fr[:, side:side + 75]
        ok = (mask & (2 if side > 2 else 1)) > 0
        if ok.sum() < 30:
            continue
        grid = np.arange(0, t[-1], 1 / fs)
        seq = np.stack([np.interp(grid, t[ok], pos[ok, k]) for k in range(75)], axis=1)
        seq = seq[:T] if len(seq) >= T else np.pad(seq, ((0, T - len(seq)), (0, 0)), mode="edge")
        X.append(seq); y.append(FORM_SCORE.get(m["form"], 70.0)); ex.append(EX.index(m["exercise"])); groups.append(m["subject"]); forms.append(m["form"])
    return np.stack(X), np.array(y, np.float32), np.array(ex), np.array(groups), np.array(forms)


def normalise(x):
    """x [..., T, 75] -> wrist-centred, hand-size-normalised [..., T, 75].
    Hand size = mean wrist->middle-proximal (knuckle, joint 11) distance; identical rule in src/scoring.js."""
    s = x.shape
    j = x.reshape(*s[:-1], 25, 3)
    j = j - j[..., :1, :]
    scale = np.linalg.norm(j[..., 11:12, :], axis=-1, keepdims=True).mean(axis=-3, keepdims=True) + 1e-6
    return (j / scale).reshape(s)
