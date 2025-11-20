import os
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================
MODEL_URL = (
    "https://huggingface.co/praneeth143/symbolrecognizer/resolve/main/"
    "best_tuned_hasy_symbols_simplified.onnx"
)
MODEL_LOCAL_PATH = os.getenv(
    "MODEL_LOCAL_PATH",
    "best_tuned_hasy_symbols_simplified.onnx"
)

EXPECTED_SIZE = 32  # your model is 32x32

# Labels taken from your class_names_latex
CLASS_NAMES = [
    r"\pi",
    r"\alpha",
    r"\beta",
    r"\sum",
    r"\delta",
    r"\triangle",
    r"\theta",
    r"\epsilon",
    r"\lambda",
    r"\mu",
    r"\diameter",
    r"\sharp",
    r"\%",
    r"\triangleright",
    r"\diamond",
    r"\pm",
    r"\div",
    r"\uplus",
    r"\star",
    r"\fint",
    r"\approx",
    r"\sim",
    r"\pitchfork",
    r"\lightning",
    r"\notin",
    r"\infty",
    r"\heartsuit",
    r"\triangledown",
    r"\ohm",
]

# ============================================================
# DOWNLOAD MODEL IF NEEDED
# ============================================================
def ensure_model_downloaded():
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f"[app] Using existing ONNX model at: {MODEL_LOCAL_PATH}")
        return

    print(f"[app] ONNX model not found, downloading from:\n{MODEL_URL}")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()

    total = 0
    with open(MODEL_LOCAL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    print(f"[app] Downloaded ONNX model ({total/1024/1024:.2f} MB) to {MODEL_LOCAL_PATH}")


ensure_model_downloaded()

# ============================================================
# LOAD ONNX MODEL
# ============================================================
if not os.path.exists(MODEL_LOCAL_PATH):
    raise RuntimeError(f"ONNX model not found at {MODEL_LOCAL_PATH}")

session = ort.InferenceSession(
    MODEL_LOCAL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"[app] Loaded model. Input: {input_name}, Output: {output_name}")

# We assume input shape is (1,1,32,32) NCHW as exported from PyTorch.


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="HASY Symbol ONNX API",
    description="Predict LaTeX symbol from 32x32 grayscale matrix.",
    version="1.0.0",
)


class MatrixPayload(BaseModel):
    matrix: List[List[float]]  # 2D array 32x32, 0..255
    size: Optional[int] = None  # optional, for debugging


@app.get("/")
def root():
    return {
        "status": "ok",
        "info": "HASY Symbol ONNX inference API",
        "expected_size": EXPECTED_SIZE,
        "num_classes": len(CLASS_NAMES),
    }


@app.post("/predict_matrix")
def predict_matrix(payload: MatrixPayload):
    mat = payload.matrix

    # ---- Basic checks ----
    if not mat or not isinstance(mat, list) or not isinstance(mat[0], list):
        raise HTTPException(status_code=400, detail="matrix must be a 2D list")

    h = len(mat)
    w = len(mat[0])

    if h != EXPECTED_SIZE or w != EXPECTED_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_SIZE}x{EXPECTED_SIZE}, got {h}x{w}",
        )

    # ---- Convert to numpy ----
    arr = np.array(mat, dtype=np.float32)  # shape (H, W)

    # Normalize: 0..255 → 0..1, and invert so black strokes become 1.0
    arr = np.clip(arr, 0.0, 255.0) / 255.0
    arr = 1.0 - arr

    # Add N,C dims → (1, 1, H, W) = (1, 1, 32, 32)
    arr = arr[np.newaxis, np.newaxis, :, :]  # NCHW

    # ---- Run ONNX inference ----
    ort_inputs = {input_name: arr}
    ort_outs = session.run([output_name], ort_inputs)
    logits = ort_outs[0]  # (1, num_classes)

    # ---- Softmax ----
    logits = logits.astype(np.float32)
    logits = logits[0]  # (num_classes,)
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    probs = exps / np.sum(exps)

    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])

    if 0 <= best_idx < len(CLASS_NAMES):
        label = CLASS_NAMES[best_idx]
    else:
        label = f"class_{best_idx}"

    # Top-k (3)
    k = min(3, probs.shape[0])
    topk_idx = np.argsort(probs)[::-1][:k]
    topk = []
    for i in topk_idx:
        lab = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{int(i)}"
        topk.append(
            {
                "index": int(i),
                "label": lab,
                "prob": float(probs[i]),
            }
        )

    return {
        "index": best_idx,
        "label": label,
        "prob": best_prob,
        "topk": topk,
    }


# Optional: run locally with `python app.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
