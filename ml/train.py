"""Hand Rehab Coach - full training / evaluation pipeline. Colab-ready, resumable (everything lands in --ckpt).

  python train.py all --ckpt DRIVE_DIR [--real ../recordings] [--preset standard|quick|smoke]

or step by step (the Colab notebook does this):
  baselines | pretrain | cv | final | vae | conformal | explain | personalize | distill | export | report

Every advanced method has its own evaluation row (subject-wise cross-validation: a subject is never in both train and test):
  baselines (mean / ridge / XGBoost on FFT-tremor-smoothness features) | BiGRU | Mamba teacher from scratch |
  + masked-joint SSL | + masked-joint + contrastive SSL | + generative (cVAE) augmentation | + domain-adversarial (needs real data) |
  one-class VAE scoring | conformal intervals | Integrated-Gradients explanations (+deletion test) |
  few-shot personalization | distilled int8 student (size, latency, accuracy kept).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

from rehablens_ml import augment as A, data as rdata, explain as E, features, synth, vae as V
from rehablens_ml.model import Student, make_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
EX = synth.EXERCISES
PRESETS = {
    "standard": dict(subjects=48, reps=3, epochs=14, pre_epochs=25, bs=32, folds=3, folds_total=4, vae_epochs=40, distill_epochs=40, gen_aug=600),
    "quick": dict(subjects=24, reps=2, epochs=8, pre_epochs=10, bs=32, folds=2, folds_total=4, vae_epochs=20, distill_epochs=25, gen_aug=300),
    "smoke": dict(subjects=8, reps=1, epochs=1, pre_epochs=1, bs=16, folds=1, folds_total=4, vae_epochs=2, distill_epochs=2, gen_aug=16),
}


def log(*a):
    print(*a, flush=True)


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=float)


def P(a, name):
    return os.path.join(a.ckpt, name)


# ------------------------------------------------------------------ data
def get_data(a):
    X, y, meta = synth.make_dataset(a.subjects, a.reps, seed=0)
    ex = np.array([EX.index(m[1]) for m in meta]); g = np.array([m[0] for m in meta]); fm = np.array([m[2] for m in meta])
    dom = np.zeros(len(X), dtype=np.int64)
    if a.real and os.path.isdir(a.real) and any(f.endswith(".json") for f in os.listdir(a.real)):
        Xr, yr, er, gr, fr = rdata.load_recordings(a.real)
        log(f"+ {len(Xr)} real recordings (weak form-based labels)")
        X, y, ex, g, fm = (np.concatenate([X, Xr]), np.concatenate([y, yr]), np.concatenate([ex, er]), np.concatenate([g, gr]), np.concatenate([fm, fr]))
        dom = np.concatenate([dom, np.ones(len(Xr), dtype=np.int64)])
    return rdata.normalise(X).astype(np.float32), y.astype(np.float32), ex, g, fm, dom


def make_folds(g, a):
    n = min(a.folds_total, len(np.unique(g)))
    return list(GroupKFold(n_splits=n).split(np.zeros(len(g)), groups=g))[: a.folds]


def metrics(y, p, ex=None, exp=None):
    r = dict(MAE=float(np.abs(p - y).mean()), RMSE=float(np.sqrt(((p - y) ** 2).mean())),
             pearson=float(pearsonr(p, y)[0]) if p.std() > 1e-9 else 0.0, spearman=float(spearmanr(p, y)[0]) if p.std() > 1e-9 else 0.0)
    if ex is not None:
        r["ex_acc"] = float((exp == ex).mean())
    return r


# ------------------------------------------------------------------ training core
def task_loss(out, ex, y, dom=None, dom_w=0.0):
    nll = 0.5 * (out["logvar"] + (out["score"] - y / 100) ** 2 * torch.exp(-out["logvar"]))
    d = y[:, None] - y[None, :]
    s = out["score"][:, None] - out["score"][None, :]
    sel = d.abs() > 5.0
    rank = F.softplus(-torch.sign(d) * s * 8)[sel].mean() if sel.any() else 0.0  # pairwise ranking loss
    loss = F.cross_entropy(out["ex"], ex) + 5 * nll.mean() + rank
    if dom is not None and "dom" in out and dom_w > 0:
        loss = loss + dom_w * F.cross_entropy(out["dom"], dom)
    return loss


def fit(model, X, y, ex, dom, a, epochs, dann=False, lr=None):
    lr = lr or a.lr
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    Xt, yt, et, dt = torch.tensor(X), torch.tensor(y), torch.tensor(ex), torch.tensor(dom)
    use_dann = dann and dom.max() > 0 and dom.min() == 0
    steps_total, step = epochs * math.ceil(len(Xt) / a.bs), 0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=max(2, steps_total), pct_start=0.15)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), a.bs):
            b = perm[i:i + a.bs]
            if len(b) < 2:
                continue
            lam = (2 / (1 + math.exp(-10 * step / max(1, steps_total))) - 1) * 0.5 if use_dann else 0.0
            out = model(A.light(Xt[b].to(dev)), dom_lambda=lam)
            loss = task_loss(out, et[b].to(dev), yt[b].to(dev), dt[b].to(dev), 0.2 if use_dann else 0.0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            tot += loss.item()
        log(f"    ep{ep + 1}/{epochs} loss {tot / math.ceil(len(Xt) / a.bs):.4f}")
    return model


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    outs = {"score": [], "sigma": [], "ex": [], "z": []}
    for i in range(0, len(X), bs):
        o = model(torch.tensor(X[i:i + bs]).to(dev))
        outs["score"].append(o["score"].cpu().numpy() * 100)
        outs["sigma"].append(torch.exp(0.5 * o["logvar"]).cpu().numpy() * 100)
        outs["ex"].append(o["ex"].argmax(1).cpu().numpy())
        outs["z"].append(o["z"].cpu())
    return {k: (torch.cat(v) if k == "z" else np.concatenate(v)) for k, v in outs.items()}


def load_init(model, a, init):
    if init == "scratch":
        return
    path = P(a, f"pretrain_{a.arch}_{init}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run: python train.py pretrain --ssl {init}")
    model.load_state_dict(torch.load(path, map_location=dev)["m"])


# ------------------------------------------------------------------ steps
def step_baselines(a, data):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier, XGBRegressor
    X, y, ex, g, fm, dom = data
    t0 = time.time()
    Fm = np.stack([features.extract(x) for x in X])
    log(f"features {Fm.shape} in {time.time() - t0:.0f}s (ROM, tremor band, SPARC smoothness, jerk, rhythm, ...)")
    res = {"mean": [], "ridge": [], "xgb": []}
    for tr, te in make_folds(g, a):
        preds = {"mean": np.full(len(te), y[tr].mean()),
                 "ridge": make_pipeline(StandardScaler(), Ridge(1.0)).fit(Fm[tr], y[tr]).predict(Fm[te]),
                 "xgb": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, verbosity=0).fit(Fm[tr], y[tr]).predict(Fm[te])}
        expred = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, verbosity=0).fit(Fm[tr], ex[tr]).predict(Fm[te])
        for k, p in preds.items():
            res[k].append(metrics(y[te], p, ex[te], expred if k == "xgb" else None) if k == "xgb" else metrics(y[te], p))
    for k, v in res.items():
        jdump(v, P(a, f"cv_{k}.json"))
        log(f"  {k:6s} MAE {np.mean([r['MAE'] for r in v]):.2f}  pearson {np.mean([r['pearson'] for r in v]):.3f}")


def step_pretrain(a, data):
    X = data[0]
    path = P(a, f"pretrain_{a.arch}_{a.ssl}.pt")
    m = make_model(a.arch).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.01)
    ep0 = 0
    if os.path.exists(path) and not a.force:
        c = torch.load(path, map_location=dev); m.load_state_dict(c["m"]); opt.load_state_dict(c["opt"]); ep0 = c["ep"]
        log(f"resumed {path} at epoch {ep0}")
    Xt = torch.tensor(X)
    for ep in range(ep0, a.pre_epochs):
        m.train(); perm = torch.randperm(len(Xt)); tr_, tc_, n = 0.0, 0.0, 0
        for i in range(0, len(Xt), a.bs):
            x = A.light(Xt[perm[i:i + a.bs]].to(dev))
            if len(x) < 4:
                continue
            loss = 0.0
            if a.ssl in ("recon", "both"):
                xs = x[:, ::m.stride]
                mask = torch.rand(xs.shape[0], xs.shape[1], 25, device=dev) < 0.3
                pred = m(x, mask, mode="ssl")
                lrec = ((pred - xs) ** 2).reshape(*xs.shape[:2], 25, 3)[mask].mean()
                loss = loss + lrec; tr_ += lrec.item()
            if a.ssl in ("contrastive", "both"):
                lc = A.nt_xent(m(A.view(x), mode="contrast"), m(A.view(x), mode="contrast"))
                loss = loss + a.lam * lc; tc_ += lc.item()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); n += 1
        log(f"pretrain[{a.ssl}] ep{ep + 1}/{a.pre_epochs} recon {tr_ / max(n, 1):.4f} contrastive {tc_ / max(n, 1):.4f}")
        torch.save(dict(m=m.state_dict(), opt=opt.state_dict(), ep=ep + 1), path + ".tmp"); os.replace(path + ".tmp", path)


def gen_extra(a, Xtr, ytr, etr, n):
    """Train a conditional VAE on the training subjects only, then sample n new labelled sequences."""
    cond = np.concatenate([np.eye(3)[etr], (ytr / 100)[:, None]], 1)
    m = V.fit_vae(Xtr, cond, a.vae_epochs, dev, log=log)
    rng = np.random.default_rng(0)
    e = rng.integers(0, 3, n); tgt = rng.uniform(0.3, 0.95, n)
    Xg = V.generate(m, np.concatenate([np.eye(3)[e], tgt[:, None]], 1), dev)
    return Xg.astype(np.float32), (tgt * 100).astype(np.float32), e


def step_cv(a, data):
    X, y, ex, g, fm, dom = data
    tag = a.tag
    log(f"== CV [{tag}] arch={a.arch} init={a.init} gen_aug={a.gen_aug if a.gen else 0} dann={a.dann}")
    res, oof = [], {k: np.full(len(X), np.nan) for k in ("score", "sigma", "ex")}
    for k, (tr, te) in enumerate(make_folds(g, a)):
        model = make_model(a.arch).to(dev)
        load_init(model, a, a.init)
        Xtr, ytr, etr, dtr = X[tr], y[tr], ex[tr], dom[tr]
        if a.gen and a.gen_aug > 0:
            Xg, yg, eg = gen_extra(a, Xtr, ytr, etr, a.gen_aug)
            Xtr, ytr, etr, dtr = np.concatenate([Xtr, Xg]), np.concatenate([ytr, yg]), np.concatenate([etr, eg]), np.concatenate([dtr, np.zeros(len(yg), dtype=np.int64)])
        fit(model, Xtr, ytr, etr, dtr, a, a.epochs, dann=a.dann)
        pr = predict(model, X[te])
        m_ = metrics(y[te], pr["score"], ex[te], pr["ex"]); res.append(m_)
        log(f"  fold{k}: MAE {m_['MAE']:.2f} pearson {m_['pearson']:.3f} ex-acc {m_['ex_acc']:.3f}")
        for key, arr in (("score", pr["score"]), ("sigma", pr["sigma"]), ("ex", pr["ex"])):
            oof[key][te] = arr
        if k == 0:
            torch.save(model.state_dict(), P(a, f"fold0_{tag}.pt"))
            jdump(dict(train=tr.tolist(), test=te.tolist(), arch=a.arch), P(a, f"fold0_{tag}.json"))
    jdump(res, P(a, f"cv_{tag}.json"))
    np.savez(P(a, f"oof_{tag}.npz"), y=y, groups=g, **oof)


def step_final(a, data):
    X, y, ex, g, fm, dom = data
    model = make_model(a.arch).to(dev)
    load_init(model, a, a.init)
    fit(model, X, y, ex, dom, a, a.epochs, dann=a.dann)
    torch.save(model.state_dict(), P(a, f"teacher_{a.tag}.pt"))
    log(f"teacher_{a.tag}.pt saved")


def load_fold0(a, tag):
    info = json.load(open(P(a, f"fold0_{tag}.json")))
    m = make_model(info["arch"]).to(dev)
    m.load_state_dict(torch.load(P(a, f"fold0_{tag}.pt"), map_location=dev))
    return m.eval(), np.array(info["train"]), np.array(info["test"])


def step_vae(a, data):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    X, y, ex, g, fm, dom = data
    folds = make_folds(g, a)
    out = []
    for k, (tr, te) in enumerate(folds):
        good = tr[fm[tr] == "good"]
        cond = np.eye(3)[ex]
        m = V.fit_vae(X[good], cond[good], a.vae_epochs, dev, log=log)
        e_tr, e_te = V.recon_error(m, X[tr], cond[tr], dev), V.recon_error(m, X[te], cond[te], dev)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(e_tr, y[tr])
        p = iso.predict(e_te)
        r = dict(spearman=float(spearmanr(-e_te, y[te])[0]), auroc_good_vs_bad=float(roc_auc_score(fm[te] == "good", -e_te)), MAE=float(np.abs(p - y[te]).mean()))
        out.append(r); log(f"  one-class VAE fold{k}: {r}")
        if k == 0:
            torch.save(m.state_dict(), P(a, "vae_oneclass.pt"))
    jdump(out, P(a, "vae_oneclass.json"))


def step_conformal(a, data):
    z = np.load(P(a, f"oof_{a.tag}.npz"), allow_pickle=True)
    ok = ~np.isnan(z["score"])
    res = {}
    for alpha in (0.05, 0.1, 0.2):
        res[str(alpha)] = E.conformal_eval(z["y"][ok], z["score"][ok], z["sigma"][ok], z["groups"][ok], alpha=alpha)
        log(f"  alpha={alpha}: target coverage {1 - alpha:.2f}  conformal {res[str(alpha)]['coverage']:.3f}  uncalibrated-Gaussian {res[str(alpha)]['naive_gauss_coverage']:.3f}  width {res[str(alpha)]['mean_width_points']:.1f} pts")
    jdump(res, P(a, f"conformal_{a.tag}.json"))


def step_explain(a, data):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, y, ex, g, fm, dom = data
    model, tr, te = load_fold0(a, a.tag)
    rng = np.random.default_rng(0)
    imp_ex, fing, dele = {}, {}, {"top": [], "random": []}
    allimp = np.zeros((3, 25))
    for e in range(3):
        idx = te[ex[te] == e]
        idx = rng.choice(idx, min(len(idx), a.n_explain), replace=False)
        ji = []
        for i in range(0, len(idx), 8):
            xb = torch.tensor(X[idx[i:i + 8]])
            ji.append(E.joint_importance(E.integrated_gradients(model, xb, dev)))
            d = E.deletion_test(model, xb, ji[-1], dev)
            for kk in dele: dele[kk].append(d[kk])
        ji = np.concatenate(ji)
        allimp[e] = ji.mean(0); fing[EX[e]] = E.finger_importance(ji)
        log(f"  {EX[e]:10s} finger importance: " + ", ".join(f"{f} {v:.2f}" for f, v in fing[EX[e]].items()))
    res = dict(finger_importance=fing, deletion_mean_abs_score_change_points={k: float(np.mean(v)) for k, v in dele.items()}, joint_importance=allimp.tolist())
    log(f"  deletion test (|score change| when freezing 6 joints): top-attributed {res['deletion_mean_abs_score_change_points']['top']:.2f} vs random {res['deletion_mean_abs_score_change_points']['random']:.2f} points")
    jdump(res, P(a, f"explain_{a.tag}.json"))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(allimp, aspect="auto", cmap="magma"); ax.set_yticks(range(3)); ax.set_yticklabels(EX); ax.set_xlabel("joint index (0 wrist, 1-4 thumb, 5-9 index, 10-14 middle, 15-19 ring, 20-24 little)")
    plt.colorbar(im, label="share of attribution"); plt.title("Integrated Gradients: which joints drive the form score"); plt.tight_layout()
    os.makedirs(P(a, "report"), exist_ok=True); plt.savefig(P(a, "report/joint_importance.png"), dpi=140); plt.close()


def step_personalize(a, data):
    X, y, ex, g, fm, dom = data
    model, tr, te = load_fold0(a, a.tag)
    pr = predict(model, X[te])
    head = lambda zz: model.heads.h_score(zz.to(dev))[:, :1]
    res = E.personalize_eval(pr["z"].to(dev), y[te] / 100, pr["score"] / 100, g[te], head)
    log("  few-shot personalization (K rated windows from an unseen user -> MAE in points):", res)
    jdump(res, P(a, f"personalize_{a.tag}.json"))


def step_distill(a, data):
    X, y, ex, g, fm, dom = data
    mode = a.frm
    if mode == "fold0":
        teacher, tr, te = load_fold0(a, a.tag)
    else:
        teacher = make_model(a.arch).to(dev); teacher.load_state_dict(torch.load(P(a, f"teacher_{a.tag}.pt"), map_location=dev)); teacher.eval()
        tr, te = np.arange(len(X)), None
    s = Student().to(dev)
    opt = torch.optim.AdamW(s.parameters(), 2e-3)
    Xt = torch.tensor(X[tr])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.distill_epochs)
    for ep in range(a.distill_epochs):
        s.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(Xt), a.bs):
            x = A.light(Xt[perm[i:i + a.bs]].to(dev))
            if len(x) < 2: continue
            with torch.no_grad():
                o = teacher(x)
            e, sc, lv = s(x)
            loss = F.mse_loss(sc, o["score"]) * 20 + F.kl_div(F.log_softmax(e, 1), F.softmax(o["ex"], 1), reduction="batchmean") + F.mse_loss(lv, o["logvar"]) * 0.5
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if (ep + 1) % max(1, a.distill_epochs // 5) == 0: log(f"  distill ep{ep + 1}/{a.distill_epochs} loss {tot / math.ceil(len(Xt) / a.bs):.4f}")
    torch.save(dict(m=s.state_dict()), P(a, f"student_{mode}.pt"))
    if mode == "final":
        torch.save(dict(m=s.state_dict()), P(a, "student.pt"))
    if te is not None:  # honest evaluation on unseen subjects
        s.eval()
        with torch.no_grad():
            Xe = torch.tensor(X[te]).to(dev)
            e, sc, lv = s(Xe)
        pt = predict(teacher, X[te])
        r = dict(student=metrics(y[te], sc.cpu().numpy() * 100, ex[te], e.argmax(1).cpu().numpy()), teacher=metrics(y[te], pt["score"], ex[te], pt["ex"]),
                 student_params=sum(p.numel() for p in s.parameters()), teacher_params=sum(p.numel() for p in teacher.parameters()))
        log(f"  held-out subjects: teacher MAE {r['teacher']['MAE']:.2f} -> student MAE {r['student']['MAE']:.2f} ({r['teacher_params']:,} -> {r['student_params']:,} params)")
        jdump(r, P(a, "student_eval.json"))


def step_export(a, data=None):
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    s = Student(); s.load_state_dict(torch.load(P(a, "student.pt"), map_location="cpu")["m"]); s.eval()
    x = torch.randn(1, 300, 75); p = P(a, "student.onnx")
    torch.onnx.export(s, x, p, input_names=["seq"], output_names=["ex", "score", "logvar"], dynamic_axes={"seq": {0: "b"}}, opset_version=17, dynamo=False)
    q = P(a, "student.int8.onnx"); quantize_dynamic(p, q, weight_type=QuantType.QInt8)
    card = {}
    for f in (p, q):
        sess = ort.InferenceSession(f); o = sess.run(None, {"seq": x.numpy()}); r = s(x)
        t0 = time.time(); [sess.run(None, {"seq": x.numpy()}) for _ in range(30)]
        name = os.path.basename(f)
        card[name] = dict(kb=os.path.getsize(f) // 1024, max_abs_score_diff_vs_pytorch=float(abs(o[1] - r[1].detach().numpy()).max()), ms_per_inference_cpu=round((time.time() - t0) / 30 * 1000, 2))
        log(" ", name, card[name])
    cp = P(a, f"conformal_{a.tag}.json")
    if os.path.exists(cp):
        c = json.load(open(cp))["0.1"]
        jdump(dict(alpha=0.1, qhat=c["qhat_all"], note="interval = score +/- qhat * sigma (split-conformal, subject-wise out-of-fold calibration)"), P(a, "calibration.json"))
        log("  calibration.json written (qhat = %.3f)" % c["qhat_all"])
    jdump(card, P(a, "model_card.json"))
    log("Copy student.int8.onnx and calibration.json to public/models/ in the app.")


def step_report(a, data=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(P(a, "report"), exist_ok=True)
    rows = [("mean", "Predict the mean"), ("ridge", "Ridge (signal features)"), ("xgb", "XGBoost (signal features)"), ("lstm", "BiGRU"),
            ("scratch", "Mamba teacher, no pretraining"), ("recon", "+ masked-joint pretraining"), ("both", "+ masked-joint + contrastive pretraining"),
            ("both_gen", "+ cVAE generative augmentation"), ("both_dann", "+ domain-adversarial (real data)")]
    lines = ["# Results (subject-wise cross-validation: held-out subjects only)\n", "| model | score MAE (pts, lower better) | Pearson | exercise acc |", "|---|---|---|---|"]
    names, maes = [], []
    for key, name in rows:
        pth = P(a, f"cv_{key}.json")
        if not os.path.exists(pth): continue
        v = json.load(open(pth))
        mae, pe = [r["MAE"] for r in v], [r["pearson"] for r in v]
        acc = [r["ex_acc"] for r in v if "ex_acc" in r]
        lines.append(f"| {name} | {np.mean(mae):.2f} +- {np.std(mae):.2f} | {np.mean(pe):.3f} | {np.mean(acc):.3f} |" if acc else f"| {name} | {np.mean(mae):.2f} +- {np.std(mae):.2f} | {np.mean(pe):.3f} | - |")
        names.append(name); maes.append(np.mean(mae))
    def add(title, path, fmt):
        if os.path.exists(path):
            lines.append(f"\n## {title}\n" + fmt(json.load(open(path))))
    add("One-class VAE (trained on good form only; no bad-form labels)", P(a, "vae_oneclass.json"), lambda v: "\n".join(f"- fold {i}: AUROC good-vs-bad {r['auroc_good_vs_bad']:.3f}, Spearman {r['spearman']:.3f}, MAE {r['MAE']:.2f}" for i, r in enumerate(v)))
    add("Conformal prediction intervals", P(a, "conformal_both.json"), lambda v: "\n".join(f"- target {1 - float(k):.0%}: conformal coverage {r['coverage']:.1%} (uncalibrated Gaussian {r['naive_gauss_coverage']:.1%}), mean width {r['mean_width_points']:.1f} pts" for k, r in v.items()))
    add("Few-shot personalization (MAE in points vs number of rated windows)", P(a, "personalize_both.json"), lambda v: ", ".join(f"K={k}: {x:.2f}" for k, x in v.items()))
    add("Explainability (Integrated Gradients)", P(a, "explain_both.json"), lambda v: f"- deletion test: freezing top-attributed joints changes the score by {v['deletion_mean_abs_score_change_points']['top']:.2f} pts vs {v['deletion_mean_abs_score_change_points']['random']:.2f} pts for random joints\n" + "\n".join(f"- {e}: " + ", ".join(f"{f} {s:.0%}" for f, s in d.items()) for e, d in v["finger_importance"].items()))
    add("Distilled on-headset student", P(a, "student_eval.json"), lambda v: f"- teacher {v['teacher']['MAE']:.2f} MAE ({v['teacher_params']:,} params) -> student {v['student']['MAE']:.2f} MAE ({v['student_params']:,} params)")
    add("Deployment (ONNX)", P(a, "model_card.json"), lambda v: "\n".join(f"- {k}: {d['kb']} KB, {d['ms_per_inference_cpu']} ms per 10-s window (CPU), max score diff vs PyTorch {d['max_abs_score_diff_vs_pytorch']:.1e}" for k, d in v.items()))
    lines.append("\n_Labels come from the synthetic rubric unless real recordings were supplied; real-data rows must be reported separately._")
    open(P(a, "report/summary.md"), "w").write("\n".join(lines)); log("\n".join(lines))
    if names:
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1)); ax.barh(names[::-1], maes[::-1], color="#38bdf8"); ax.set_xlabel("score MAE (points, lower is better)"); plt.title("Baseline ladder + ablations"); plt.tight_layout(); plt.savefig(P(a, "report/ladder.png"), dpi=140); plt.close()
    pz = P(a, "personalize_both.json")
    if os.path.exists(pz):
        v = json.load(open(pz)); fig, ax = plt.subplots(figsize=(5, 3.2)); ax.plot([int(k) for k in v], list(v.values()), "o-"); ax.set_xlabel("rated windows from the new user (K)"); ax.set_ylabel("MAE (points)"); plt.title("Few-shot personalization"); plt.tight_layout(); plt.savefig(P(a, "report/personalization.png"), dpi=140); plt.close()


def step_all(a, data):
    def done(name): return os.path.exists(P(a, name)) and not a.force
    if not done("cv_xgb.json"): step_baselines(a, data)
    for ssl in ("recon", "both"):
        a.ssl = ssl
        step_pretrain(a, data)
    configs = [("lstm", dict(arch="lstm", init="scratch")), ("scratch", dict(arch="mamba", init="scratch")), ("recon", dict(arch="mamba", init="recon")),
               ("both", dict(arch="mamba", init="both")), ("both_gen", dict(arch="mamba", init="both", gen=True))]
    if data[5].max() > 0: configs.append(("both_dann", dict(arch="mamba", init="both", dann=True)))
    for tag, kw in configs:
        if done(f"cv_{tag}.json"): log(f"skip cv_{tag} (done)"); continue
        a.tag, a.arch, a.init, a.gen, a.dann = tag, kw.get("arch", "mamba"), kw.get("init", "both"), kw.get("gen", False), kw.get("dann", False)
        step_cv(a, data)
    a.tag, a.arch, a.init, a.gen, a.dann = "both", "mamba", "both", False, False
    if not done("teacher_both.pt"): step_final(a, data)
    if not done("vae_oneclass.json"): step_vae(a, data)
    step_conformal(a, data); step_explain(a, data); step_personalize(a, data)
    a.frm = "fold0"; step_distill(a, data)
    a.frm = "final"; step_distill(a, data)
    step_export(a); step_report(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "baselines", "pretrain", "cv", "final", "vae", "conformal", "explain", "personalize", "distill", "export", "report"])
    ap.add_argument("--ckpt", default="ckpt"); ap.add_argument("--real", default="../recordings"); ap.add_argument("--preset", default="standard", choices=list(PRESETS))
    ap.add_argument("--arch", default="mamba", choices=["mamba", "s4d", "lstm"]); ap.add_argument("--ssl", default="both", choices=["recon", "contrastive", "both"])
    ap.add_argument("--init", default="both", choices=["scratch", "recon", "contrastive", "both"]); ap.add_argument("--tag", default="both")
    ap.add_argument("--gen", action="store_true", help="cVAE generative augmentation"); ap.add_argument("--dann", action="store_true", help="domain-adversarial (needs real data)")
    ap.add_argument("--from", dest="frm", default="fold0", choices=["fold0", "final"]); ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--n_explain", type=int, default=24); ap.add_argument("--force", action="store_true")
    for k in ("subjects", "reps", "epochs", "pre_epochs", "bs", "folds", "folds_total", "vae_epochs", "distill_epochs", "gen_aug"):
        ap.add_argument(f"--{k}", type=int, default=None)
    a = ap.parse_args()
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k) is None: setattr(a, k, v)
    os.makedirs(a.ckpt, exist_ok=True)
    log(f"device {dev} | preset {a.preset} | cmd {a.cmd}")
    if a.cmd == "export": return step_export(a)
    if a.cmd == "report": return step_report(a)
    data = get_data(a)
    log(f"data: {len(data[0])} windows, {len(np.unique(data[3]))} subjects")
    dict(all=step_all, baselines=step_baselines, pretrain=step_pretrain, cv=step_cv, final=step_final, vae=step_vae, conformal=step_conformal,
         explain=step_explain, personalize=step_personalize, distill=step_distill)[a.cmd](a, data)


if __name__ == "__main__":
    main()
