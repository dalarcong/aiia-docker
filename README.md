# AIIA - Deployment of Deep Learning Architectures in Docker Containers

Image classification with ResNet-18 (ImageNet-pretrained, fine-tuned on Oxford-IIIT Pet, 37 breeds), served by a FastAPI inference container and a separate Gradio UI container, orchestrated with Docker Compose.

## Structure

```
api/                            Inference API (FastAPI), CLI inference and training script
ui/                             Gradio UI, calls the API over HTTP
docker-compose.yaml             api + ui (+ train)
docker-compose.gpu.yaml         NVIDIA GPU override (CUDA PyTorch + GPU reservation)
hf_space/                       Single-container image for Hugging Face Spaces (tested locally)
```

## Run (CPU)

```
cp .env.example .env
docker compose up -d --build --wait
```

UI at http://localhost:7860. The API is only reachable from the internal network.

## Inference from the terminal

```bash
docker compose run --rm --no-deps -v "$PWD/samples:/samples:ro" api python predict.py /samples/cat.jpg
```

## Fine-tuning

```bash
docker compose --profile train run --rm train python train.py --epochs 5
docker compose restart api   # loads the new checkpoint
```

## GPU (Nvidia Container Toolkit required)

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml --profile train build
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml --profile train run --rm train python train.py --epochs 5 --batch-size 64
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --wait
```

Or set `COMPOSE_FILE=docker-compose.yaml:docker-compose.gpu.yaml` in `.env`.
