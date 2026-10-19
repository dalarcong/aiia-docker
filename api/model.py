"""Model loading and inference, shared by the API, the CLI and the training script."""
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "/models/resnet18_pets.pt"))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier:
    """ResNet-18 image classifier.

    Uses the ImageNet-pretrained weights by default. If a fine-tuned checkpoint
    exists at CHECKPOINT_PATH, the classification head is replaced and loaded from it.
    """

    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        weights = ResNet18_Weights.DEFAULT
        self.transform = weights.transforms()
        self.model = resnet18(weights=weights)
        self.labels = weights.meta["categories"]
        self.source = "imagenet"

        if CHECKPOINT_PATH.exists():
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(ckpt["labels"]))
            self.model.load_state_dict(ckpt["state_dict"])
            self.labels = ckpt["labels"]
            self.source = "fine-tuned"

        self.model.eval().to(self.device)


    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0]
        scores, indices = probs.topk(min(top_k, len(self.labels)))
        return [
            {"label": self.labels[i], "score": round(s, 4)}
            for s, i in zip(scores.tolist(), indices.tolist())
        ]