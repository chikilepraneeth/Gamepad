# app.py
import os
import io
import time
from typing import List, Optional

import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

# ============================================================
# CONFIG
# ============================================================

# Where the model will be stored on disk inside the container / server
MODEL_PATH = "best_tuned_hasy_symbols.pt"

# Where to download it from if it's not present.
# You can override this via environment variable on Render.
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/praneeth143/symbolrecognizer/resolve/main/best_tuned_hasy_symbols.pt"  # <-- change this!
)

# Same canvas size as training preprocessing (we resize to 32×32)
TARGET_SIZE = 32

# ============================================================
# MODEL LOADING
# ============================================================

def ensure_model_file():
    """
    Download the model from MODEL_URL if it doesn't exist locally.
    """
    if os.path.exists(MODEL_PATH):
        print(f"[API] Using existing model file: {MODEL_PATH}")
        return

    if not MODEL_URL or MODEL_URL.startswith("https://YOUR-STORAGE-URL"):
        raise RuntimeError(
            "Model file not found and MODEL_URL is not set correctly.\n"
            "Set the env var MODEL_URL or hard-code a direct download link."
        )

    print(f"[API] Model file not found. Downloading from: {MODEL_URL}")
    resp = requests.get(MODEL_URL, stream=True)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download model from {MODEL_URL}: {e}")

    tmp_path = MODEL_PATH + ".tmp"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, MODEL_PATH)
    print(f"[API] Downloaded model to {MODEL_PATH}")


def build_model_gray(num_classes: int, arch: str):
    """
    Rebuilds the same architecture you used in your training/Tkinter script:
    - resnet18 or resnet34
    - first conv changed to 1-channel (grayscale)
    - fc changed to num_classes
    """
    if arch == "resnet34":
        m = models.resnet34(weights=None)
    else:
        m = models.resnet18(weights=None)

    # 1-channel input instead of 3
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_model(ckpt_path: str, device: torch.device):
    """
    Loads checkpoint with keys:
      - class_names_latex: list of LaTeX labels
      - state_dict: model weights
      - (optional) arch: 'resnet18' or 'resnet34'
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names_latex"]
    num_classes = len(class_names)

    arch = ckpt.get("arch", None)
    model = None
    tried = []

    if arch is not None:
        tried.append(arch)
        model = build_model_gray(num_classes, arch).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Fallback: try a couple of arches
        for a in ["resnet34", "resnet18"]:
            try:
                tried.append(a)
                tmp = build_model_gray(num_classes, a).to(device)
                tmp.load_state_dict(ckpt["state_dict"])
                model, arch = tmp, a
                break
            except Exception:
                continue

    if model is None:
        raise RuntimeError(f"Could not rebuild model. Tried arches: {tried}")

    model.eval()
    return model, class_names, arch


# ============================================================
# PREPROCESSING (same logic as your Tkinter script)
# ============================================================

to_tensor_32 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=Image.BILINEAR),
    transforms.ToTensor(),  # -> [0,1], shape [1,H,W]
])


@torch.no_grad()
def predict_pil(model, class_names, device, pil_img, topk=3, threshold=None):
    """
    pil_img: PIL.Image (L or RGB), white bg, black strokes.
    Returns: (pred_label, conf_pct, topk_list, pred_idx)
      topk_list: list of (class_idx, latex_label, confidence_pct)
    """
    img = pil_img.convert("L")

    # auto-trim whitespace, then pad to square before resizing (keeps aspect)
    bbox = ImageOps.invert(img).getbbox()
    if bbox:
        img = img.crop(bbox)

    w, h = img.size
    s = max(w, h)
    sq = Image.new("L", (s, s), color=255)  # white bg
    sq.paste(img, ((s - w) // 2, (s - h) // 2))
    img32 = to_tensor_32(sq).unsqueeze(0).to(device)  # [1,1,32,32]

    logits = model(img32)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    conf, idx = torch.max(probs, dim=0)

    pred_idx = int(idx)
    k = min(topk, probs.numel())
    topk_conf, topk_idx = torch.topk(probs, k)

    topk_list = [
        (int(i), class_names[int(i)], float(p) * 100.0)
        for p, i in zip(topk_conf.tolist(), topk_idx.tolist())
    ]

    pred = class_names[pred_idx]
    conf_pct = float(conf) * 100.0
    if threshold is not None and float(conf) < threshold:
        pred = "<unknown>"

    return pred, conf_pct, topk_list, pred_idx


def matrix_to_pil(matrix: List[List[float]]) -> Image.Image:
    """
    Converts a 2D Python list (e.g. 32x32 or 64x64) with values 0..255 (white→black)
    into a grayscale PIL image, matching the semantics of your draw pad.
    """
    arr = np.array(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2D")

    # Clip to [0,255] and convert to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # In draw pad, 255=white background, 0=black stroke -> good
    img = Image.fromarray(arr, mode="L")
    return img


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="HASY Symbol API", version="1.0.0")

# Allow calls from anywhere (pad.html, Unity dev, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model + labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensure_model_file()
model, class_names, arch = load_model(MODEL_PATH, device)
print(f"[API] Model loaded: arch={arch}, classes={len(class_names)}, device={device}")


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class MatrixInput(BaseModel):
    matrix: List[List[float]]
    room: Optional[str] = None
    t: Optional[int] = None  # optional timestamp from client (Firebase)


class PredictionResponse(BaseModel):
    pred_label: str
    class_index: int
    confidence_pct: float
    topk: List[dict]
    input_h: int
    input_w: int
    room: Optional[str] = None
    t: Optional[int] = None
    server_time: float


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "HASY Symbol API",
        "arch": arch,
        "num_classes": len(class_names),
        "device": str(device),
    }


@app.post("/predict_matrix", response_model=PredictionResponse)
def predict_matrix(payload: MatrixInput):
    start_time = time.time()

    if not payload.matrix or not isinstance(payload.matrix, list):
        raise HTTPException(status_code=400, detail="matrix must be a non-empty 2D list")

    # Convert matrix → PIL
    try:
        pil_img = matrix_to_pil(payload.matrix)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid matrix: {e}")

    h, w = pil_img.size[1], pil_img.size[0]

    try:
        pred, conf_pct, topk_list, pred_idx = predict_pil(
            model, class_names, device, pil_img, topk=3, threshold=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    elapsed = time.time() - start_time

    # Prepare top-k for JSON
    topk_dicts = [
        {"class_index": idx, "label": label, "confidence_pct": conf}
        for idx, label, conf in topk_list
    ]

    return PredictionResponse(
        pred_label=pred,
        class_index=pred_idx,
        confidence_pct=conf_pct,
        topk=topk_dicts,
        input_h=h,
        input_w=w,
        room=payload.room,
        t=payload.t,
        server_time=elapsed,
    )


# ============================================================
# LOCAL DEV ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
