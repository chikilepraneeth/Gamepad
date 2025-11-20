from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

# -----------------------------
# ONNX model setup
# -----------------------------
ONNX_PATH = "best_tuned_hasy_symbols_simplified.onnx"  # in the same folder as app.py

# class names (same order as training)
CLASS_NAMES = [
    r"\pi", r"\alpha", r"\beta", r"\sum", r"\delta", r"\triangle", r"\theta",
    r"\epsilon", r"\lambda", r"\mu", r"\diameter", r"\sharp", r"\%", r"\triangleright",
    r"\diamond", r"\pm", r"\div", r"\uplus", r"\star", r"\fint", r"\approx",
    r"\sim", r"\pitchfork", r"\lightning", r"\notin", r"\infty", r"\heartsuit",
    r"\triangledown", r"\ohm"
]

# Load ONNX
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI()

# 🔥 IMPORTANT: enable CORS so your pad.html origin can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or set to ["https://gamepad-1-e9w6.onrender.com"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request / Response models
# -----------------------------
class MatrixRequest(BaseModel):
    size: int
    matrix: list[list[int]]

class TopKEntry(BaseModel):
    index: int
    label: str
    prob: float

class PredictResponse(BaseModel):
    index: int
    label: str
    prob: float
    topk: list[TopKEntry]

# -----------------------------
# Utility
# -----------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "info": "HASY Symbol ONNX inference API",
        "expected_size": 32,
        "num_classes": len(CLASS_NAMES),
    }

@app.post("/predict_matrix", response_model=PredictResponse)
def predict_matrix(req: MatrixRequest):
    # 1) validate size
    h = len(req.matrix)
    w = len(req.matrix[0]) if h > 0 else 0
    if h != req.size or w != req.size or req.size != 32:
        return {
            "index": -1,
            "label": "<bad_size>",
            "prob": 0.0,
            "topk": [],
        }

    # 2) to numpy (1,1,32,32)
    arr = np.array(req.matrix, dtype=np.float32)  # [32,32], 0..255
    arr = arr / 255.0          # 0..1
    arr = 1.0 - arr            # invert (black strokes = 1.0)
    arr = arr[None, None, :, :]  # [1,1,32,32]

    # 3) run ONNX
    logits = sess.run([output_name], {input_name: arr})[0]  # [1,29]
    probs = softmax(logits)
    probs1 = probs[0]

    best_idx = int(np.argmax(probs1))
    best_p   = float(probs1[best_idx])
    label    = CLASS_NAMES[best_idx] if 0 <= best_idx < len(CLASS_NAMES) else f"class_{best_idx}"

    # top-3
    k = min(3, probs1.shape[0])
    topk_idx = np.argsort(-probs1)[:k]
    topk = []
    for i in topk_idx:
        idx = int(i)
        topk.append(
            TopKEntry(
                index=idx,
                label=CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"class_{idx}",
                prob=float(probs1[idx]),
            )
        )

    return PredictResponse(
        index=best_idx,
        label=label,
        prob=best_p,
        topk=topk,
    )
