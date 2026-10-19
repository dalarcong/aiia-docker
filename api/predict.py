"""Command-line inference: python predict.py image.jpg [--top-k 5]"""
import argparse
import json

from PIL import Image

from model import Classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify an image with ResNet-18")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to show")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    clf = Classifier()
    result = {
        "device": str(clf.device),
        "model": clf.source,
        "predictions": clf.predict(Image.open(args.image), args.top_k)
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()