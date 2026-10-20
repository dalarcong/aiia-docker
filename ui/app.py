"""Gradio front-end. Sends the image to the inference API and shows the result."""
import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", "http://api:8000")


def classify(image_path: str | None, top_k: int):
    if image_path is None:
        raise gr.Error("Please upload an image first.")

    try:
        with open(image_path, "rb") as f:
            response = httpx.post(
                f"{API_URL}/predict",
                params={"top_k": int(top_k)},
                files={"file": f},
                timeout=30,
            )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise gr.Error(f"Inference API error: {e}") from None

    result = response.json()
    labels = {p["label"]: p["score"] for p in result["predictions"]}
    info = f"Model: **{result['model']}** - Device: **{result['device']}** - Inference: **{result['inference_ms']} ms**"

    return labels, info


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="filepath", label="Image"),
        gr.Slider(1, 10, value=5, step=1, label="Top-k"),
    ],
    outputs=[gr.Label(label="Predictions"), gr.Markdown()],
    title="ResNet-18 image classifier",
    description="Upload an image. The prediction is computed by the FastAPI inference API.",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    