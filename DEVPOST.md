# <APP NAME> - hands-only hand-rehab coach for Meta Quest (WebXR)

## Inspiration
Hand-rehab exercises are boring, hard to do correctly alone, and progress is rarely measured. Quest hand tracking already reads 25 joints per hand - enough to coach form, not just count reps.

## What it does
Put on the headset, pinch and hold to start - no controllers, no menus. A ghost hand demonstrates each exercise (fist, finger tapping, wrist rotation); you copy it for 12 s while the coach scores form live (range of motion, tremor, rhythm), counts reps, marks which finger under-moves, and says what to fix ("Bend your ring finger more"). The score always comes with a 90% interval (about +/-13 points) so it says how sure it is.

## How we built it
IWSDK / WebXR, runs in the Quest browser - no install. Scoring runs on-device from the 25-joint stream (wrist-centred, hand-size-normalised). Models were trained and evaluated with subject-wise cross-validation on a synthetic kinematic exercise generator:
- classical signal features (ROM, 4-12 Hz tremor, SPARC smoothness, jerk, rhythm): Ridge 4.6, XGBoost 3.3 MAE (points)
- deep: spatial Transformer with bone-tree attention + Mamba state-space temporal blocks, masked-joint and contrastive pretraining, cVAE augmentation, domain-adversarial head, one-class VAE
- split-conformal intervals: 89.7% coverage at a 90% target (naive Gaussian: 28.6%)
- Integrated Gradients explanations: freezing the top-attributed joints moves the score 1.02 pts vs 0.42 for random joints
- few-shot low-rank personalization: 17.7 -> 9.8 MAE with 12 rated windows
- distillation to a 112 KB int8 ONNX model, ~1 ms per 10 s window

## Challenges / honest findings
The deep models did NOT beat hand-crafted features on this data (14-20 MAE vs 3.3), so the shipped on-device scorer is the calibrated classical one (MAE ~6.4, 90% interval +/-12.8 on held-out synthetic subjects). We report the whole ablation ladder, including what failed. All numbers are on synthetic data; not clinically validated.

## What's next
Real recordings from patients with clinician ratings (record mode is built in: ?mode=record), domain adaptation to real hands, learned rep segmentation.

# Demo video script (<3 min)
0:00 Problem in one sentence + headset shot. 0:15 Pinch-hold start, no controllers. 0:30 Ghost hand demo, copy it; show live score, +/-13 interval, rep counter. 1:10 Deliberately move badly -> cue changes ("Open and close wider", finger cue). 1:40 Summary screen. 1:55 Cut to results slide: ladder chart, conformal coverage, personalization curve. 2:30 Honest limits + what's next. Record on Quest with the browser capture, or the emulator if no headset.
