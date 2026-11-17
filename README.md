# Hand Rehab Coach (working title)
Hands-only WebXR hand-rehab coach with deep-learning form scoring — Meta VR Start Developer Competition 2026.

## Run
    npm install
    npm run dev            # https://localhost:8081  (use the Immersive Web Emulator in Chrome, or open the network URL on a Quest)
The IWSDK dev tool may try to download its own Chromium; if that fails the server still runs — open the URL in normal Chrome.

## Modes
- **Coach (default):** pinch & hold -> 3 guided exercises (ghost-hand demo + 12 s live scoring each) -> summary. No controllers.
  Uses `public/models/student.onnx` if present (badge shows "AI model active"), otherwise the calibrated signal-analysis fallback.
- **Recorder:** `?mode=record&subject=s01&exercise=finger_tap&form=good&dur=10`
  (exercise: fist | finger_tap | wrist_rot; form: good | limited_rom | tremor | slow | irregular).
  Pinch & hold -> countdown -> records 10 s of 25-joint hand data -> saved to `./recordings/*.json`.

## ML pipeline (ml/) — run on Colab (GPU), checkpoints to Drive, auto-resume
    pip install -r ml/requirements.txt
    python -m rehablens_ml.baselines                                   # CPU: mean / ridge / XGBoost ladder, subject-wise CV
    python train.py pretrain --ckpt DRIVE --epochs 30                  # masked-joint self-supervised pretraining
    python train.py finetune --ckpt DRIVE --real ../recordings         # teacher (Transformer + state-space), 5-fold subject-wise
    python train.py finetune --ckpt DRIVE --real ../recordings --scratch   # ablation: no pretraining
    python train.py distill  --ckpt DRIVE                              # teacher -> small student CNN
    python train.py export   --ckpt DRIVE                              # student.onnx (fp32, ~0.4 MB) + parity/latency check
Then copy `student.onnx` to `public/models/` and redeploy. Synthetic data is only for pipeline development and
pretraining — report real-recording results separately.

## Deploy
Push to GitHub, enable Pages -> Source: GitHub Actions (workflow included). Submission needs the live URL.

## Full ML stack (ml/train.py, Colab notebook `rehablens_colab.ipynb`)
Baseline ladder (mean/Ridge/XGBoost/BiGRU) -> Mamba teacher -> masked-joint + contrastive SSL -> cVAE augmentation -> domain-adversarial (real data) -> one-class VAE -> conformal intervals -> Integrated Gradients + deletion test -> few-shot personalization -> distilled int8 ONNX student.
`cd ml && pip install -r requirements.txt && python train.py all --preset quick --ckpt ckpt`. Copy `student.int8.onnx` + `calibration.json` to `public/models/`.
All numbers are on synthetic data unless you add real recordings (`?mode=record`).
