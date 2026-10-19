# Base image
FROM python:3.12-slim

# Image metadata
LABEL org.opencontainers.image.title="aiia-api" \
      org.opencontainers.image.description="ResNet-18 image classification inference API" \
      org.opencontainers.image.authors="David Alarcón"

ARG DEBIAN_FRONTEND=noninteractive

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TORCH_HOME=/opt/torch

# download curl for the HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models "$TORCH_HOME" \
    && chown app:app /models "$TORCH_HOME"

WORKDIR /app


# Pytorch build: cpu or a CUDA variant.
# Changing it does not invalidate the layers above.
ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1

# 1. Download PyTorch (heaviest layer)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# 2. Download app dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

USER app

# 3. Bake the pretrained weights into the image
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"


# 4. Copy source code (it changes most, so an edit only rebuilds this layer)
COPY --chown=app:app *.py ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

    
