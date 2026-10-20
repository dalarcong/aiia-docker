#!/usr/bin/env bash
# Assemble the Hugging Face Space in build/space/ from the project sources.
# Usage: hf_space/stage.sh path/to/resnet18_pets.pt
set -euo pipefail

CKPT=$(realpath "${1:?usage: hf_space/stage.sh path/to/resnet18_pets.pt}")
cd "$(dirname "$0")/.."
OUT=build/space

rm -rf "$OUT" && mkdir -p "$OUT"
cp hf_space/{Dockerfile,README.md,requirements.txt,space.py} "$OUT"/
cp api/model.py api/app.py "$OUT"/
cp ui/app.py "$OUT"/ui.py
cp "$CKPT" "$OUT"/resnet18_pets.pt
echo "Space staged in $OUT"
