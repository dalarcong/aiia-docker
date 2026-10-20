FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    GRADIO_ANALYTICS_ENABLED=False \
    TORCH_HOME=/opt/torch \
    CHECKPOINT_PATH=/app/resnet18_pets.pt \
    API_URL=http://localhost:7860

RUN useradd --create-home --uid 1000 app \ 
    && mkdir -p "$TORCH_HOME" \
    && chown app:app "$TORCH_HOME"

WORKDIR /app

RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER app
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"

COPY --chown=app:app resnet18_pets.pt .
COPY --chown=app:app *.py ./

EXPOSE 7860
CMD ["uvicorn", "space:app", "--host", "0.0.0.0", "--port", "7860"]
