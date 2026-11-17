"""Kinematic synthetic hand-exercise generator (pipeline development + pretraining data).

NOT clinical data. It exists so the whole pipeline (features -> baselines -> deep model ->
distillation -> ONNX) can be built and tested before real Quest recordings exist, and to
give the self-supervised model more motion to pretrain on. Real recordings should be
mixed in / used for evaluation (see data.py).

Joint order = WebXR hand: wrist, thumb(4), index(5), middle(5), ring(5), little(5) = 25.
Output per sequence: [T, 25, 3] metres, wrist at origin.
"""
import numpy as np

EXERCISES = ["fist", "finger_tap", "wrist_rot"]
FORMS = ["good", "limited_rom", "tremor", "slow", "irregular"]
FS = 30  # Hz
T = 300  # 10 s

# (n_segments, base offset x (m), segment lengths (m)) per digit; thumb has 4 joints, others 5
_DIGITS = [
    ("thumb", 4, -0.03, [0.04, 0.035, 0.03, 0.025]),
    ("index", 5, -0.025, [0.06, 0.04, 0.025, 0.02, 0.018]),
    ("middle", 5, -0.005, [0.062, 0.045, 0.028, 0.022, 0.018]),
    ("ring", 5, 0.015, [0.058, 0.042, 0.026, 0.02, 0.017]),
    ("little", 5, 0.033, [0.05, 0.03, 0.02, 0.016, 0.014]),
]


def _fk(flex, size):
    """flex: [T, 5 digits, 3 flexion angles (rad)] -> joints [T, 25, 3]."""
    t = flex.shape[0]
    out = np.zeros((t, 25, 3))
    k = 1
    for d, (_, nj, bx, seg) in enumerate(_DIGITS):
        pos = np.tile(np.array([bx, 0.0, 0.0]), (t, 1)) * size
        ang = np.zeros(t)
        # joint 0 of each digit = metacarpal base (fixed to palm)
        pos = pos + np.array([0.0, 0.0, -0.01])
        out[:, k, :] = pos
        k += 1
        for s in range(nj - 1):
            if s >= 1:
                ang = ang + flex[:, d, min(s - 1, 2)]
            L = seg[s] * size
            pos = pos + np.stack([np.zeros(t), -np.sin(ang) * L, -np.cos(ang) * L], axis=1)
            out[:, k, :] = pos
            k += 1
    return out


def _roll(joints, ang):
    c, s = np.cos(ang), np.sin(ang)
    x, y = joints[..., 0].copy(), joints[..., 1].copy()
    joints[..., 0] = c[:, None] * x - s[:, None] * y
    joints[..., 1] = s[:, None] * x + c[:, None] * y
    return joints


def make_sequence(rng, exercise, form, size=1.0, base_hz=0.9):
    """Return (joints[T,25,3], score 0-100, params dict)."""
    t = np.arange(T) / FS
    amp, tremor, jitter, speed = 1.0, 0.0, 0.05, 1.0
    if form == "limited_rom":
        amp = rng.uniform(0.35, 0.65)
    elif form == "tremor":
        tremor = rng.uniform(0.4, 1.0)
        amp = rng.uniform(0.7, 1.0)
    elif form == "slow":
        speed = rng.uniform(0.35, 0.6)
        amp = rng.uniform(0.75, 1.0)
    elif form == "irregular":
        jitter = rng.uniform(0.25, 0.5)
        amp = rng.uniform(0.7, 1.0)
    else:
        amp = rng.uniform(0.85, 1.0)
        jitter = rng.uniform(0.02, 0.1)
        tremor = rng.uniform(0.0, 0.1)
    f = base_hz * speed
    # phase with rhythm irregularity
    phase = 2 * np.pi * f * t + np.cumsum(rng.normal(0, jitter, T)) * 0.5
    tremor_sig = tremor * 0.06 * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t + rng.uniform(0, 6.28))
    base = 0.5 - 0.5 * np.cos(phase)  # 0..1 open->closed
    flex = np.zeros((T, 5, 3))
    roll = np.zeros(T)
    max_flex = np.array([0.9, 1.0, 0.8])  # per-joint max flexion (rad)
    if exercise == "fist":
        for d in range(5):
            flex[:, d, :] = (amp * base[:, None] + tremor_sig[:, None]) * max_flex
    elif exercise == "finger_tap":
        for d in range(1, 5):  # fingers tap thumb in sequence
            ph = phase - (d - 1) * np.pi / 2
            b = np.clip(np.sin(ph), 0, 1) ** 2
            flex[:, d, :] = (amp * b[:, None] + tremor_sig[:, None]) * max_flex
        flex[:, 0, :] = (amp * 0.4 * base[:, None]) * max_flex
    else:  # wrist_rot
        roll = amp * 0.9 * np.sin(phase) + tremor_sig * 2
    flex += rng.normal(0, 0.01, flex.shape)
    j = _fk(flex, size)
    j = _roll(j, roll)
    j += rng.normal(0, 0.0015, j.shape)  # sensor noise
    # --- quality score (rubric): ROM 40%, tremor 25%, rhythm 20%, speed 15% ---
    q = (
        0.40 * np.clip((amp - 0.3) / 0.7, 0, 1)
        + 0.25 * (1 - np.clip(tremor, 0, 1))
        + 0.20 * (1 - np.clip((jitter - 0.02) / 0.5, 0, 1))
        + 0.15 * np.clip((speed - 0.3) / 0.7, 0, 1)
    )
    score = float(np.clip(100 * q + rng.normal(0, 3), 0, 100))  # +-3 rater noise
    return j.astype(np.float32), score, dict(amp=amp, tremor=tremor, jitter=jitter, speed=speed)


def make_dataset(n_subjects=24, reps=4, seed=0):
    """Returns X[N,T,75], y score[N], meta list (subject, exercise, form)."""
    rng = np.random.default_rng(seed)
    X, y, meta = [], [], []
    for s in range(n_subjects):
        size = rng.uniform(0.85, 1.2)  # hand size
        hz = rng.uniform(0.7, 1.1)  # personal tempo
        for ex in EXERCISES:
            for form in FORMS:
                for _ in range(reps if form == "good" else max(1, reps // 2)):
                    j, sc, _ = make_sequence(rng, ex, form, size, hz)
                    X.append(j.reshape(T, -1))
                    y.append(sc)
                    meta.append((f"s{s:02d}", ex, form))
    return np.stack(X), np.array(y, dtype=np.float32), meta
