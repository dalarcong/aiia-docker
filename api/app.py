"""FastAPI inference server. Exposes the Classifier over HTTP."""
import io
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from model import Classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at start-up instead of on every request.
    app.state.classifier = Classifier()
    yield


app = FastAPI(title="AIIA ResNet-18 inference API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    clf = app.state.classifier
    return {"status": "ok", "device": str(clf.device), "model": clf.source}


@app.post("/predict")
def predict(
    file: Annotated[UploadFile, File(description="Image to classify")], 
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> dict:
    try:
        image = Image.open(io.BytesIO(file.file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from None

    clf = app.state.classifier
    start = time.perf_counter()
    predictions = clf.predict(image, top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "device": str(clf.device),
        "model": clf.source,
        "inference_ms": round(elapsed_ms, 1),
        "predictions": predictions
    }