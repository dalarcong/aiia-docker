---
title: AIIA Pet Breed Classifier
emoji: 🐶
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ResNet-18 pet breed classifier

> Not deployed: Hugging Face now requires a PRO subscription to host Docker Spaces. The image is built and tested locally.

ResNet-18 pretrained on ImageNed and fine-tuned on Oxford-IIIT Pet (37 cat and dog breeds, 89.4 % validation accuracy).
Single container serving a FastAPI inference API (`/predict`, `/health`, `/docs`) and a Gradio UI (`/`).
