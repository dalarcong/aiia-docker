"""Hugging Face Space entry point. FastAPI inference API and Gradio UI
in a single process."""
import gradio as gr

from app import app
from ui import demo

# Routes declared on the FastAPI app (/health, /predict) take precedence over the mount.
app = gr.mount_gradio_app(app, demo, path="/")
