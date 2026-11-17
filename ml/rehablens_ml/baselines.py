"""Baseline ladder with leave-subjects-out CV. Reports MAE / Pearson / Spearman for the form score
and accuracy for exercise recognition. Run: python -m rehablens_ml.baselines [--real DIR]"""
import argparse, json, time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from . import features, synth


def build(n_subjects=24, reps=4, seed=0):
    X, y, meta = synth.make_dataset(n_subjects, reps, seed)
    t0 = time.time()
    F = np.stack([features.extract(x) for x in X])
    print(f"features: {F.shape} in {time.time()-t0:.1f}s")
    groups = np.array([m[0] for m in meta])
    ex = np.array([synth.EXERCISES.index(m[1]) for m in meta])
    return X, F, y, ex, groups


def loso_eval(F, y, ex, groups, n_splits=6):
    gkf = GroupKFold(n_splits=n_splits)
    P = {k: np.zeros_like(y) for k in ["mean", "ridge", "xgb"]}
    acc = []
    for tr, te in gkf.split(F, y, groups):
        P["mean"][te] = y[tr].mean()
        P["ridge"][te] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(F[tr], y[tr]).predict(F[te])
        P["xgb"][te] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, verbosity=0).fit(F[tr], y[tr]).predict(F[te])
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, verbosity=0).fit(F[tr], ex[tr])
        acc.append(float((clf.predict(F[te]) == ex[te]).mean()))
    rows = {}
    for k, p in P.items():
        rows[k] = dict(
            MAE=float(np.abs(p - y).mean()),
            pearson=float(pearsonr(p, y)[0]) if p.std() > 0 else 0.0,
            spearman=float(spearmanr(p, y)[0]) if p.std() > 0 else 0.0,
        )
    rows["exercise_acc_xgb"] = float(np.mean(acc))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=24)
    a = ap.parse_args()
    X, F, y, ex, g = build(a.subjects)
    r = loso_eval(F, y, ex, g)
    print(json.dumps(r, indent=2))
    json.dump(r, open("results/baselines_synth.json", "w"), indent=2)
