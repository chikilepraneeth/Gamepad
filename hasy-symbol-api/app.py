import time
from typing import List, Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_tuned_hasy_symbols.pt"  # in same folder as app.py

# =========================
# SCHEMAS
# =========================
class PredictRequest(BaseModel):
    matrix: List[List[float]]

class TopKItem(BaseModel):
    idx: int
    label: str
    conf: float

class PredictResponse(BaseModel):
    label: str
    class_idx: int
    confidence: float
    top3: List[TopKItem]
    server_time: float

# =========================
# MODEL LOADING
# =========================
def build_model_gray(num_classes: int, arch: str):
    if arch == "resnet34":
        m = models.resnet34(weights=None)
    else:
        m = models.resnet18(weights=None)

    # 1-channel input instead of 3
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_model(ckpt_path: str, device: torch.device):
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

# Preprocess (same as your Tk app)
to_tensor_32 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32), interpolation=Image.BILINEAR),
    transforms.ToTensor(),  # [0,1]
])

@torch.no_grad()
def predict_pil(model, class_names, device, pil_img, topk=3, threshold=None):
    img = pil_img.convert("L")
    bbox = ImageOps.invert(img).getbbox()
    if bbox:
        img = img.crop(bbox)

    w, h = img.size
    s = max(w, h)
    sq = Image.new("L", (s, s), color=255)
    sq.paste(img, ((s - w)//2, (s - h)//2))
    img32 = to_tensor_32(sq).unsqueeze(0).to(device)

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
    arr = np.array(matrix, dtype=np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    return img

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="HASY Symbol API")

# CORS: allow pad.html from anywhere (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names, arch = load_model(MODEL_PATH, device)
print(f"[API] Model loaded: arch={arch}, classes={len(class_names)}, device={device}")

@app.get("/")
def root():
    return {"status": "ok", "arch": arch, "num_classes": len(class_names)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.matrix or not isinstance(req.matrix, list):
        raise HTTPException(status_code=400, detail="matrix must be a non-empty 2D list")

    try:
        pil_img = matrix_to_pil(req.matrix)
        pred, conf, topk_raw, pred_idx = predict_pil(
            model, class_names, device, pil_img, topk=3, threshold=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    top3 = [
        TopKItem(idx=i, label=lbl, conf=c)
        for (i, lbl, c) in topk_raw
    ]
    return PredictResponse(
        label=str(pred),
        class_idx=int(pred_idx),
        confidence=float(conf),
        top3=top3,
        server_time=time.time()
    )
