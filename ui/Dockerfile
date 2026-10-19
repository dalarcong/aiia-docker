# Base image
FROM python:3.12-slim

LABEL org.opencontainers.image.title="aiia-ui" \
      org.opencontainers.image.description="Gradio front-end for the ResNet-18 inference API"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    GRADIO_ANALYTICS_ENABLED=False

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Download app dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

USER app
# Copy source code
COPY --chown=app:app app.py .

EXPOSE 7860
# Healthcheck with Python's standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=4)" || exit 1

CMD ["python", "app.py"]