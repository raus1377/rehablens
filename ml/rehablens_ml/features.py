"""Clinically-meaningful signal features from a hand-joint sequence [T,25,3] (or [T,75])."""
import numpy as np
from scipy.signal import find_peaks, welch

TIPS = [4, 9, 14, 19, 24]  # thumb, index, middle, ring, little fingertips (WebXR order)


def _prep(x, fs):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), 25, 3)
    x = x - x[:, :1, :]  # wrist-centred
    return x


def sparc(sig, fs, fc=10.0, pad=4):
    """Spectral arc length smoothness (Balasubramanian 2012); more negative = less smooth."""
    n = int(2 ** (np.ceil(np.log2(len(sig))) + pad))
    f = np.arange(n) * fs / n
    m = np.abs(np.fft.fft(sig, n))
    m = m / (m.max() + 1e-12)
    sel = f <= fc
    mf, ff = m[sel], f[sel]
    return -float(np.sum(np.sqrt(np.diff(ff / (ff[-1] + 1e-12)) ** 2 + np.diff(mf) ** 2)))


def extract(x, fs=30):
    x = _prep(x, fs)
    tips = x[:, TIPS, :]
    d = np.linalg.norm(tips, axis=2)  # fingertip->wrist distance [T,5]
    sig = d.mean(axis=1)  # openness proxy
    sig = sig - sig.mean()
    # rotation proxy for wrist roll: angle of index->little axis in the palm plane
    ax = x[:, 24, :2] - x[:, 9, :2]
    roll = np.unwrap(np.arctan2(ax[:, 1], ax[:, 0]))
    roll = roll - roll.mean()
    main = sig if sig.std() * 3 > roll.std() * 0.5 else roll  # pick the dominant motion channel
    # ROM
    rom_f = float(np.mean(np.percentile(d, 95, axis=0) - np.percentile(d, 5, axis=0)))
    rom_r = float(np.percentile(roll, 95) - np.percentile(roll, 5))
    # spectrum
    f, p = welch(main, fs=fs, nperseg=min(len(main), 128))
    p = p / (p.sum() + 1e-12)
    dom = float(f[np.argmax(p)])
    tremor_band = float(p[(f >= 4) & (f <= 12)].sum())
    # kinematics
    v = np.gradient(tips, 1 / fs, axis=0)
    a = np.gradient(v, 1 / fs, axis=0)
    j = np.gradient(a, 1 / fs, axis=0)
    speed = np.linalg.norm(v, axis=2).mean()
    jerk = float(np.sqrt((np.linalg.norm(j, axis=2) ** 2).mean()) / (speed + 1e-9) ** 1.0)
    # rhythm
    pk, _ = find_peaks(main, distance=max(2, int(fs * 0.25)), prominence=main.std() * 0.3)
    if len(pk) >= 3:
        iv = np.diff(pk) / fs
        cv, rate = float(iv.std() / (iv.mean() + 1e-9)), float(len(pk) / (len(main) / fs))
    else:
        cv, rate = 1.0, 0.0
    return np.array(
        [rom_f, rom_r, dom, tremor_band, speed, np.log1p(jerk), sparc(main, fs), cv, rate, sig.std(), roll.std()],
        dtype=np.float64,
    )


FEATURE_NAMES = ["rom_finger", "rom_roll", "dom_freq", "tremor_band", "mean_speed", "log_jerk", "sparc", "rhythm_cv", "rep_rate", "sig_std", "roll_std"]
