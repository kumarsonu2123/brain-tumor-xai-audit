import os, sys, json
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get("CKPT_PATH", "best_model.pt")
TH_PATH = os.environ.get("THRESHOLDS_PATH", "trust_thresholds.json")

# Falls back to sensible defaults if trust_thresholds.json (from Notebook 06) is absent.
DEFAULT_TH = {
    "n_forward_passes": 30, "ig_steps": 50, "target_layer": "layer4",
    "gradcam_area_threshold": 0.5, "min_confidence": 0.70,
    "max_entropy": 0.60, "max_gradcam_area": 0.60,
}
DISCLAIMER = "Research demo, not a medical device. Do not use for diagnosis."

def _load_thresholds():
    th = dict(DEFAULT_TH)
    if os.path.exists(TH_PATH):
        try: th.update(json.load(open(TH_PATH)))
        except Exception as e: print("warning: thresholds unreadable:", e)
    return th

explainer, load_error = None, None
if os.path.exists(CKPT):
    try:
        from src.demo.inference import TumorExplainer, DISCLAIMER as D
        DISCLAIMER = D
        explainer = TumorExplainer(CKPT, _load_thresholds())
    except Exception as e:
        load_error = f"{type(e).__name__}: {e}"
else:
    load_error = (f"Checkpoint '{CKPT}' not found. Upload your trained best_model.pt "
                  "to this Space (and optionally trust_thresholds.json).")

def predict(image):
    if explainer is None:
        return None, None, {}, f"**Model not loaded.**\n\n{load_error}", ""
    if image is None:
        return None, None, {}, "Upload an MRI slice to begin.", ""
    out = explainer(image)
    probs = out["mc_probs"]
    metrics = (
        f"**Prediction:** {out['predicted_class']}  \n"
        f"**MC confidence:** {out['mc_confidence']:.3f}  \n"
        f"**Predictive entropy:** {out['predictive_entropy']:.3f}  \n"
        f"**Mutual information:** {out['mutual_information']:.3f}  \n"
        f"**Grad-CAM area:** {out['gradcam_area']:.1%}"
    )
    reasons = "\n".join(f"- {r}" for r in out["flag_reasons"]) or "- none"
    trust = f"### {out['trust_flag']}\n{reasons}"
    return out["gradcam_overlay"], out["ig_overlay"], probs, metrics, trust

with gr.Blocks(title="Brain Tumor XAI Audit") as demo:
    gr.Markdown("# Can You Trust the Heatmap?\n"
                "Brain tumor MRI classifier with **audited** explanations "
                "(Grad-CAM + Integrated Gradients), MC-dropout uncertainty, and an "
                "automated **trust flag**. Upload a T1-weighted MRI slice.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="MRI slice", image_mode="L")
            btn = gr.Button("Analyze", variant="primary")
            probs_out = gr.Label(num_top_classes=3, label="MC-averaged class probabilities")
            metrics_out = gr.Markdown()
            trust_out = gr.Markdown()
        with gr.Column():
            gc_out = gr.Image(label="Grad-CAM (where the model looked)")
            ig_out = gr.Image(label="Integrated Gradients")
    btn.click(predict, inputs=inp, outputs=[gc_out, ig_out, probs_out, metrics_out, trust_out])
    gr.Markdown(f"<sub>{DISCLAIMER}  \nCode: https://github.com/kumarsonu2123/brain-tumor-xai-audit</sub>")

if __name__ == "__main__":
    demo.launch()
