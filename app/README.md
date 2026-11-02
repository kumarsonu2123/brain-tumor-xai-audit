---
title: Brain Tumor XAI Audit
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Can You Trust the Heatmap? — live demo

Upload a T1-weighted brain MRI slice; the model predicts the tumor class and
returns **audited** explanations (Grad-CAM + Integrated Gradients), MC-dropout
uncertainty (predictive entropy, mutual information), and an automated
**trust flag** that warns when a confident prediction rests on an untrustworthy
explanation.

Full pipeline and methodology: https://github.com/kumarsonu2123/brain-tumor-xai-audit

**To run this Space you must add two files** (from the Kaggle run):
- `best_model.pt` — the trained checkpoint (Notebook 01 output)
- `trust_thresholds.json` — optional (Notebook 06 output); sensible defaults are used if absent

_Research demo, not a medical device. Do not use for diagnosis._
