# Results (subject-wise cross-validation: held-out subjects only)

| model | score MAE (pts, lower better) | Pearson | exercise acc |
|---|---|---|---|
| Predict the mean | 9.20 +- 0.21 | nan | - |
| Ridge (signal features) | 4.61 +- 0.17 | 0.843 | - |
| XGBoost (signal features) | 3.26 +- 0.11 | 0.920 | 0.999 |
| BiGRU | 8.69 +- 0.28 | 0.396 | 0.627 |
| Mamba teacher, no pretraining | 14.45 +- 8.39 | 0.502 | 0.858 |
| + masked-joint pretraining | 14.84 +- 0.94 | 0.426 | 0.749 |
| + masked-joint + contrastive pretraining | 18.09 +- 0.37 | 0.449 | 0.343 |
| + cVAE generative augmentation | 20.15 +- 0.38 | 0.249 | 0.747 |

## One-class VAE (trained on good form only; no bad-form labels)
- fold 0: AUROC good-vs-bad 0.416, Spearman -0.321, MAE 9.17
- fold 1: AUROC good-vs-bad 0.374, Spearman -0.413, MAE 8.97
- fold 2: AUROC good-vs-bad 0.374, Spearman -0.299, MAE 9.48

## Conformal prediction intervals
- target 95%: conformal coverage 95.5% (uncalibrated Gaussian 28.6%), mean width 65.4 pts
- target 90%: conformal coverage 89.7% (uncalibrated Gaussian 28.6%), mean width 61.5 pts
- target 80%: conformal coverage 81.0% (uncalibrated Gaussian 28.6%), mean width 56.6 pts

## Few-shot personalization (MAE in points vs number of rated windows)
K=0: 17.66, K=3: 11.69, K=6: 10.48, K=12: 9.75

## Explainability (Integrated Gradients)
- deletion test: freezing top-attributed joints changes the score by 1.02 pts vs 0.42 pts for random joints
- fist: thumb 10%, index 14%, middle 61%, ring 10%, little 6%
- finger_tap: thumb 5%, index 11%, middle 68%, ring 8%, little 8%
- wrist_rot: thumb 26%, index 20%, middle 4%, ring 8%, little 42%

## Distilled on-headset student
- teacher 17.66 MAE (868,088 params) -> student 17.91 MAE (99,989 params)

## Deployment (ONNX)
- student.onnx: 394 KB, 0.33 ms per 10-s window (CPU), max score diff vs PyTorch 1.2e-07
- student.int8.onnx: 112 KB, 1.37 ms per 10-s window (CPU), max score diff vs PyTorch 3.3e-03

_Labels come from the synthetic rubric unless real recordings were supplied; real-data rows must be reported separately._