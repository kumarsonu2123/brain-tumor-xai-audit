from src.explain import gradcam, integrated_gradients

# keys match cfg["explain"]["methods"]
EXPLAINERS = {
    "gradcam": gradcam.explain,
    "integrated_gradients": integrated_gradients.explain,
}