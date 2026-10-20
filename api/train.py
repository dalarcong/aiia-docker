"""
Fine-tune ResNet-18 on Oxford-IIIT Pet (37 cat and dog breeds).

Saves the best checkpoint (by validation accuracy) to CHECKPOINT_PATH,
where the API picks it up.
"""
import argparse
import time
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet18_Weights, resnet18

from model import CHECKPOINT_PATH, get_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on Oxford-IIIT Pet")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the new head (backbone uses lr/10)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classification head")
    parser.add_argument("--subset", type=int, default=0, help="Use only N images per split (0 = all) for quick tests")
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False, help="float16 mixed precision on GPU (off by default. GTX 16 GPUs produce NaNs in float16)")

    return parser.parse_args(argv)


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader, list[str]]:
    eval_tf = ResNet18_Weights.DEFAULT.transforms() # same preprocessing as inference
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(eval_tf.mean, eval_tf.std),
    ])

    train_set = OxfordIIITPet(args.data_dir, split="trainval", transform=train_tf, download=True)
    val_set = OxfordIIITPet(args.data_dir, split="test", transform=eval_tf, download=True)
    labels = train_set.classes

    if args.subset:
        gen = torch.Generator().manual_seed(0)
        train_set = Subset(train_set, torch.randperm(len(train_set), generator=gen)[: args.subset].tolist())
        val_set = Subset(val_set, torch.randperm(len(val_set), generator=gen)[: args.subset].tolist())

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
        labels
    )


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes) # new head, always trainable

    return model


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=False) -> tuple[float, float]:
    """One pass over the loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)

    return total_loss / seen, correct / seen


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_loader, val_loader, labels = build_loaders(args, device)
    print(f"Train images: {len(train_loader.dataset)} - Val images: {len(val_loader.dataset)} - Classes: {len(labels)}")

    model = build_model(len(labels), args.freeze_backbone).to(device)
    param_groups = [{"params": model.fc.parameters(), "lr": args.lr}]

    if not args.freeze_backbone:
        backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
        param_groups.append({"params": backbone, "lr": args.lr / 10})

    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss()

    use_amp = args.amp and device.type == "cuda"
    print(f"Mixed precision (float16): {'on' if use_amp else 'off'}")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
        if not math.isfinite(train_loss):
            raise SystemExit("Training diverged (loss is NaN/inf). Try --no-amp or a lower --lr.")
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, amp=use_amp)
        print(
            f"Epoch {epoch}/{args.epochs} - train loss {train_loss:.3f} acc {train_acc:.3f} - "
            f"val loss {val_loss:.3f} acc {val_acc:.3f} - {time.perf_counter() - start:.1f} s"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "labels": labels, "val_acc": val_acc}, CHECKPOINT_PATH)
            print(f"  Saved best checkpoint to {CHECKPOINT_PATH}")

    print(f"Best val accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()
