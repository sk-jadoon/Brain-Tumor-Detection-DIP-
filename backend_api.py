import base64
import os
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


app = FastAPI(title="Brain Tumor Detection Backend")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_grayscale(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image. Upload a valid MRI image.")
    return img


def _preprocess(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced


def _segment(enhanced: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep largest connected component to reduce false positives.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_idx] = 255
    return cleaned


def _encode_image_base64(img: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode output image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _save_upload(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Only .png, .jpg, .jpeg files are supported.")

    file_id = str(uuid.uuid4())
    target = os.path.join(OUTPUT_DIR, f"{file_id}{ext}")
    data = upload.file.read()
    with open(target, "wb") as f:
        f.write(data)
    return target


class DetectRequest(BaseModel):
    image_path: str
    pixel_spacing_mm: Optional[float] = None


class PreprocessRequest(BaseModel):
    image_path: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/select-input-image")
def select_input_image(file: UploadFile = File(...)) -> dict:
    try:
        image_path = _save_upload(file)
        return {"message": "Image uploaded successfully.", "image_path": image_path}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/pre-processing")
def pre_processing(payload: PreprocessRequest) -> dict:
    try:
        img = _load_grayscale(payload.image_path)
        enhanced = _preprocess(img)
        out_path = os.path.join(OUTPUT_DIR, f"enhanced_{uuid.uuid4()}.png")
        cv2.imwrite(out_path, enhanced)

        return {
            "message": "Pre-processing completed.",
            "enhanced_image_path": out_path,
            "enhanced_image_base64": _encode_image_base64(enhanced),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/detection")
def detection(payload: DetectRequest) -> dict:
    start = time.perf_counter()
    try:
        img = _load_grayscale(payload.image_path)
        enhanced = _preprocess(img)
        mask = _segment(enhanced)
        tumor_area_pixels = int(np.sum(mask == 255))

        # If pixel spacing is given, convert to mm^2; otherwise area in pixels.
        tumor_area_mm2 = None
        if payload.pixel_spacing_mm is not None and payload.pixel_spacing_mm > 0:
            tumor_area_mm2 = float(tumor_area_pixels * (payload.pixel_spacing_mm ** 2))

        execution_time_sec = round(time.perf_counter() - start, 4)

        mask_path = os.path.join(OUTPUT_DIR, f"mask_{uuid.uuid4()}.png")
        cv2.imwrite(mask_path, mask)

        return {
            "detection_result": "Tumor detected" if tumor_area_pixels > 0 else "No tumor detected",
            "tumor_area_pixels": tumor_area_pixels,
            "tumor_area_mm2": tumor_area_mm2,
            "execution_time_sec": execution_time_sec,
            "segmentation_mask_path": mask_path,
            "segmentation_mask_base64": _encode_image_base64(mask),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset() -> dict:
    removed = 0
    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
    return {"message": "Reset completed.", "files_removed": removed}

