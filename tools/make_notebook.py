import json, sys
cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(True)})

md("""
# Hand Rehab Coach — full ML training & evaluation notebook
Runtime → Change runtime type → **T4 GPU**, then **Run all** (about 40–70 min on `standard`; use `quick` for a 15-min pass).

What this notebook trains and measures (all with **subject-wise cross-validation** — a subject is never in train and test):

| # | Method | Why |
|---|---|---|
| 1 | Mean / Ridge / XGBoost on FFT-tremor, SPARC smoothness, jerk, rhythm features | honest classical baseline |
| 2 | BiGRU | recurrent baseline |
| 3 | Spatial Transformer (bone-tree attention) + **Mamba selective state-space** temporal blocks | the teacher |
| 4 | Masked-joint **self-supervised** pretraining, + **contrastive (NT-Xent)** pretraining | ablation ladder |
| 5 | **Conditional VAE** generative augmentation | ablation |
| 6 | **Domain-adversarial** training (only if you upload real recordings) | sim-to-real |
| 7 | **One-class VAE** scoring trained on good form only | label-free quality score |
| 8 | Heteroscedastic head + **split-conformal** intervals | calibrated uncertainty |
| 9 | **Integrated Gradients** + deletion faithfulness test | which finger/joint caused the cue |
| 10 | **Few-shot low-rank personalization** | adapt to a new patient |
| 11 | **Distillation** to a 1D-CNN student → **ONNX int8** | runs inside the Quest browser |

⚠️ Labels come from a synthetic kinematic rubric unless you add real recordings. Report synthetic numbers as synthetic.
""")
code("""
import os, sys, subprocess, zipfile, glob
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    CKPT = '/content/drive/MyDrive/handrehab_ckpt'   # checkpoints survive disconnects; re-running resumes
else:
    CKPT = './ckpt'
os.makedirs(CKPT, exist_ok=True)
import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available() and torch.cuda.get_device_name(0))
""")
md("## 0. Get the code\nUpload `handrehab.zip` (the project zip) with the file panel on the left, or run the upload cell.")
code("""
if not os.path.exists('ml/train.py'):
    zs = glob.glob('*.zip')
    if not zs and IN_COLAB:
        from google.colab import files; up = files.upload(); zs = list(up)
    zipfile.ZipFile(zs[0]).extractall('.')
    if not os.path.exists('ml/train.py'):   # zip may contain a top folder
        top = [d for d in os.listdir('.') if os.path.exists(f'{d}/ml/train.py')][0]; os.chdir(top)
os.chdir('ml'); print(os.getcwd())
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements.txt'])
""")
code("""
PRESET = 'standard'      # 'quick' | 'standard'
REAL_DIR = '../recordings'   # optional: JSON takes recorded in the app with ?mode=record (enables the domain-adversarial row)
def run(cmd):
    cmd = f'{sys.executable} -u train.py {cmd} --ckpt {CKPT} --preset {PRESET} --real {REAL_DIR}'
    print('$', cmd); subprocess.run(cmd, shell=True, check=True)
""")
md("## 1. Baselines — classical signal features\nROM, tremor-band (4–12 Hz) power, SPARC spectral smoothness, normalised jerk, rhythm CV … fed to Ridge and XGBoost.")
code("run('baselines')")
md("## 2. Self-supervised pretraining (no labels)\nMask 30 % of joint×frame tokens and reconstruct them; optionally add a SimCLR objective over augmented views (time-warp, rotation, mirror, noise).")
code("run('pretrain --ssl recon')")
code("run('pretrain --ssl both')")
md("## 3. Cross-validation ablation ladder\nEach row = same folds, same data. Only the method changes.")
code("run('cv --tag lstm    --arch lstm   --init scratch')")
code("run('cv --tag scratch --arch mamba  --init scratch')")
code("run('cv --tag recon   --arch mamba  --init recon')")
code("run('cv --tag both    --arch mamba  --init both')")
md("### 3b. + generative augmentation (conditional VAE trained inside each training fold only)")
code("run('cv --tag both_gen --arch mamba --init both --gen')")
md("### 3c. + domain-adversarial (only meaningful if `recordings/` contains real takes)")
code("""
if os.path.isdir(REAL_DIR) and glob.glob(REAL_DIR+'/*.json'): run('cv --tag both_dann --arch mamba --init both --dann')
else: print('No real recordings found - skipping (this is expected on a first run).')
""")
md("## 4. Final teacher, one-class VAE, uncertainty, explanations, personalization")
code("run('final --tag both --arch mamba --init both')")
code("run('vae')")
code("run('conformal --tag both')")
code("run('explain --tag both')")
code("run('personalize --tag both')")
md("## 5. Distil to the headset model and export ONNX (int8)")
code("run('distill --tag both --from fold0')   # honest held-out comparison teacher vs student")
code("run('distill --tag both --from final')   # model you ship")
code("run('export --tag both')")
md("## 6. Results")
code("""
run('report')
from IPython.display import Image, Markdown, display
display(Markdown(open(f'{CKPT}/report/summary.md').read()))
for f in ('ladder','joint_importance','personalization'):
    p=f'{CKPT}/report/{f}.png'
    if os.path.exists(p): display(Image(p))
""")
md("## 7. Download the files the app needs\nCopy `student.int8.onnx` and `calibration.json` into the app's `public/models/`, then redeploy.")
code("""
out = 'handrehab_results.zip'
with zipfile.ZipFile(out,'w') as z:
    for f in ['student.int8.onnx','student.onnx','calibration.json','model_card.json','student_eval.json']:
        if os.path.exists(f'{CKPT}/{f}'): z.write(f'{CKPT}/{f}', f)
    for f in glob.glob(f'{CKPT}/*.json')+glob.glob(f'{CKPT}/report/*'): z.write(f, 'metrics/'+os.path.basename(f))
if IN_COLAB:
    from google.colab import files; files.download(out)
print('wrote', out)
""")
md("""
### Optional: run everything in one cell
```python
run('all')   # resumable; skips finished stages
```
""")
nb={"cells":cells,"metadata":{"accelerator":"GPU","colab":{"provenance":[]},"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb,open(sys.argv[1],"w"),indent=1)
