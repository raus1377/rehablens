# RehabLens

**Hands-only hand rehab for Meta Quest, in the browser (WebXR).** No controllers, no install.
Built for the Meta VR Start Developer Competition 2026 (Productivity).

**Live demo:** https://raus1377.github.io/rehablens/  
- **On a Meta Quest (Quest Browser):** tap Enter VR. This is the real app, driven by your tracked hands.
- **On a laptop/desktop (no headset):** the page plays a self-running **simulated-hand demo** of the same app, labelled "DESKTOP DEMO - simulated hand". Real hand tracking needs a Quest (or the Immersive Web Emulator).

## What it does
Pinch and hold to start. A ghost hand demonstrates each exercise (fist open/close, finger tapping, wrist rotation); you copy it for 12 s while RehabLens scores your form live from the 25 tracked joints per hand (range of motion, 4-12 Hz tremor, rhythm), counts reps, names the finger that under-moves, and shows a 90% confidence interval (about +/-13 points). A summary screen follows.

## Session history
After each real session the summary shows your last overall score and the change (stored locally in the browser).

## Honest status
- Scoring models were trained and evaluated on **synthetic** kinematic data (subject-wise CV). Not clinically validated.
- Demo video was recorded in the Meta Quest 3 emulator (IWER); not yet tested on a physical Quest.
- The deep models did **not** beat hand-crafted features (14-20 vs 3.3 MAE), so the shipped on-device scorer is the calibrated classical one (MAE ~6.4, 90% interval +/-12.8 on held-out synthetic subjects). The full ablation is in `results/`.

## Results (synthetic data, subject-wise CV)
| Model | MAE (points) |
|---|---|
| Mean baseline | 9.20 |
| Ridge (signal features) | 4.61 |
| XGBoost (signal features) | 3.26 |
| Mamba + Transformer, best deep variant | 14.45 - 20.15 |

Split-conformal intervals: 89.7% coverage at a 90% target. Few-shot personalization: 17.7 -> 9.8 MAE with 12 rated windows. Distilled int8 ONNX student: 112 KB, ~1.4 ms. Figures and raw numbers: `results/`.

## Run locally
    npm install
    npm run dev:runtime     # https://localhost:8081 (accept the local certificate; use the Immersive Web Emulator in Chrome, or open the network URL on a Quest)
URL flags: `?demo=1` force the simulated-hand demo, `?demo=0` disable it, `?mode=record&subject=s01&exercise=finger_tap&form=good` record real hand data.

## Stack
IWSDK / WebXR, Three.js, Vite. ML in `ml/` (PyTorch, scikit-learn, XGBoost); Colab notebook `rehablens_colab.ipynb`.

## Deploy
GitHub Pages via `.github/workflows/pages.yml` (Settings -> Pages -> Source: GitHub Actions).
