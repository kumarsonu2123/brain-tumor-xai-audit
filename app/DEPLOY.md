# Deploying the live demo to Hugging Face Spaces

The Space is this `app/` folder + the repo's `src/` package + two files from the
Kaggle run (`best_model.pt`, optional `trust_thresholds.json`).

## Steps
1. Create a free account at https://huggingface.co and a **Write** token
   (Settings -> Access Tokens).
2. Create a new **Space** (SDK: Gradio). Clone it locally.
3. Into the Space repo copy: `app/app.py` and `app/requirements.txt` and
   `app/README.md` (as the Space root `README.md`) and `app/.gitattributes`,
   the repo's `src/` folder, and your `best_model.pt` (+ `trust_thresholds.json`).
4. `git lfs install && git add . && git commit -m "demo" && git push`.
   The Space builds and goes live automatically.

`best_model.pt` is tracked via Git LFS (see `.gitattributes`). If `trust_thresholds.json`
is absent, the app uses sensible default thresholds.
