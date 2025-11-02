import os
import time
import uuid
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference import load_model, Preprocessor, predict_image, gradcam_on_image, encode_image_b64


WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", os.path.join("../outputs", "best_model.pth"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 256))
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", 150))
FRAME_SAMPLES = int(os.environ.get("FRAME_SAMPLES", 8))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
LOG_CSV = os.environ.get("PRED_LOG", "../outputs/predictions_log.csv")
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", 60))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(WEIGHTS_PATH, device)
pre = Preprocessor(image_size=IMAGE_SIZE)

app = FastAPI(title="Deepfake Detector API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiter
_rl_bucket = {}


def rate_limit_ok(client_id: str) -> bool:
    now = time.time()
    window = int(now // 60)
    key = (client_id, window)
    count = _rl_bucket.get(key, 0)
    if count >= RATE_LIMIT_PER_MIN:
        return False
    _rl_bucket[key] = count + 1
    return True


def ensure_dirs() -> None:
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def log_prediction(kind: str, prob: float, label: int) -> None:
    ensure_dirs()
    header_needed = not os.path.isfile(LOG_CSV)
    with open(LOG_CSV, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,kind,prob,label\n")
        f.write(f"{int(time.time())},{kind},{prob:.6f},{label}\n")


@app.get("/api/health")
def health():
    return {
        "status": "ok", 
        "device": str(device),
        "model_loaded": model is not None,
        "threshold": THRESHOLD
    }


@app.post("/api/predict-image")
async def predict_image_endpoint(request: Request, file: UploadFile = File(...), heatmap: Optional[bool] = False):
    client = request.client.host if request.client else "unknown"
    if not rate_limit_ok(client):
        raise HTTPException(429, detail="Rate limit exceeded. Try again later.")
    
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(415, detail="Unsupported media type. Please upload a JPEG or PNG image.")
    
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, detail="File too large")
    
    npimg = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, detail="Invalid image file")

    result = predict_image(model, pre, img_bgr, device)
    prob = result["prob"]
    result["label"] = int(prob >= THRESHOLD)

    resp = {
        "prob": prob, 
        "label": result["label"], 
        "threshold": THRESHOLD, 
        "no_face_detected": result.get("no_face_detected", False),
        "confidence": "high" if prob > 0.8 or prob < 0.2 else "medium" if prob > 0.6 or prob < 0.4 else "low"
    }

    if heatmap:
        overlay, prob_cam = gradcam_on_image(model, pre, img_bgr, device)
        if overlay is not None:
            resp["heatmap_b64_jpg"] = encode_image_b64(overlay)
            resp["prob_cam"] = prob_cam
    
    log_prediction("image", resp["prob"], resp["label"])
    return JSONResponse(resp)


@app.post("/api/predict-video")
async def predict_video_endpoint(request: Request, file: UploadFile = File(...), heatmap: Optional[bool] = False):
    client = request.client.host if request.client else "unknown"
    if not rate_limit_ok(client):
        raise HTTPException(429, detail="Rate limit exceeded. Try again later.")
    
    if file.content_type not in {"video/mp4", "video/avi", "video/x-matroska", "application/octet-stream"}:
        raise HTTPException(415, detail="Unsupported media type. Please upload an MP4/AVI/MKV video.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, detail="File too large")
    
    ensure_dirs()
    tmp_path = os.path.join(UPLOAD_DIR, f"_tmp_{uuid.uuid4().hex}_{file.filename}")
    with open(tmp_path, "wb") as f:
        f.write(data)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        raise HTTPException(400, detail="Invalid video file")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); os.remove(tmp_path)
        raise HTTPException(400, detail="Empty video")

    # Sample frames evenly
    idxs = np.linspace(0, total - 1, num=max(1, FRAME_SAMPLES), dtype=int)
    probs: List[float] = []
    frames_for_heatmap: List[Tuple[float, np.ndarray]] = []
    
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        result = predict_image(model, pre, frame, device)
        p = float(result["prob"])
        probs.append(p)
        if heatmap:
            frames_for_heatmap.append((p, frame))
    
    cap.release()
    os.remove(tmp_path)

    if not probs:
        raise HTTPException(400, detail="Could not read frames from the video")
    
    prob = float(np.mean(probs))
    label = int(prob >= THRESHOLD)
    
    resp = {
        "prob": prob, 
        "label": label, 
        "threshold": THRESHOLD, 
        "frames": len(probs),
        "confidence": "high" if prob > 0.8 or prob < 0.2 else "medium" if prob > 0.6 or prob < 0.4 else "low"
    }
    
    if heatmap and frames_for_heatmap:
        top_p, top_frame = max(frames_for_heatmap, key=lambda t: t[0])
        overlay, prob_cam = gradcam_on_image(model, pre, top_frame, device)
        if overlay is not None:
            resp["heatmap_b64_jpg"] = encode_image_b64(overlay)
            resp["prob_cam"] = prob_cam
            resp["top_frame_prob"] = top_p
    
    log_prediction("video", resp["prob"], resp["label"])
    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

