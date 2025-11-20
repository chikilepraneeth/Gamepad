import os
import time
import requests
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ONNX_PATH = "best_tuned_hasy_symbols_simplified.onnx"

ONNX_URL = (
    "https://huggingface.co/praneeth143/symbolrecognizer/resolve/main/"
    "best_tuned_hasy_symbols_simplified.onnx"
)

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

# --------------------------------------------------
# DOWNLOAD MODEL IF MISSING
# --------------------------------------------------
def ensure_onnx_present(path: str, url: str) -> None:
    if os.path.exists(path):
        print(f"[app] ONNX file already present at: {path}")
        return

    print(f"[app] ONNX file not found. Downloading from:\n  {url}")
    t0 = time.time()
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"[app] Downloaded ONNX model to {path} in {time.time() - t0:.1f}s")


ensure_onnx_present(ONNX_PATH, ONNX_URL)

# --------------------------------------------------
# LOAD ONNX MODEL (CPU)
# --------------------------------------------------
print("[app] Loading ONNXRuntime session...")
sess = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"],
)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print("[app] ONNXRuntime session ready.")

# --------------------------------------------------
# FASTAPI + CORS
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # later you can restrict to your pad origin:
    # allow_origins=["https://gamepad-1-e9w6.onrender.com"],
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],   # POST, GET, OPTIONS, etc.
    allow_headers=["*"],   # Content-Type, etc.
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class MatrixRequest(BaseModel):
    size: int
    matrix: list[list[int]]  # 2D list of 0..255


class TopKEntry(BaseModel):
    index: int
    label: str
    prob: float


class PredictResponse(BaseModel):
    index: int
    label: str
    prob: float
    topk: list[TopKEntry]


# --------------------------------------------------
# UTILS
# --------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "info": "HASY Symbol ONNX inference API",
        "expected_size": 32,
        "num_classes": len(CLASS_NAMES),
    }


# Explicit OPTIONS handler to make preflight super happy
@app.options("/predict_matrix")
def options_predict_matrix():
    # CORS middleware will add the headers
    return Response(status_code=200)


@app.post("/predict_matrix", response_model=PredictResponse)
def predict_matrix(req: MatrixRequest):
    print("[app] /predict_matrix called")

    # 1) Validate shape
    h = len(req.matrix)
    w = len(req.matrix[0]) if h > 0 else 0
    if h != req.size or w != req.size or req.size != 32:
        print(f"[app] Bad size: got {h}x{w}, expected 32x32")
        return PredictResponse(
            index=-1,
            label=f"<bad_size:{h}x{w}>",
            prob=0.0,
            topk=[],
        )

    # 2) Convert to numpy [1,1,32,32], normalize & invert
    arr = np.array(req.matrix, dtype=np.float32)  # [32,32], 0..255
    arr = arr / 255.0
    arr = 1.0 - arr  # black strokes -> 1.0
    arr = arr[None, None, :, :]  # [1,1,32,32]

    # 3) Run ONNX
    logits = sess.run([output_name], {input_name: arr})[0]  # [1,num_classes]
    probs = softmax(logits)
    probs1 = probs[0]

    best_idx = int(np.argmax(probs1))
    best_prob = float(probs1[best_idx])

    if 0 <= best_idx < len(CLASS_NAMES):
        label = CLASS_NAMES[best_idx]
    else:
        label = f"class_{best_idx}"

    # top-3
    k = min(3, probs1.shape[0])
    top_indices = np.argsort(-probs1)[:k]

    topk: list[TopKEntry] = []
    for i in top_indices:
        idx = int(i)
        lbl = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"class_{idx}"
        topk.append(
            TopKEntry(
                index=idx,
                label=lbl,
                prob=float(probs1[idx]),
            )
        )

    print(f"[app] Predicted index={best_idx}, label={label}, prob={best_prob:.3f}")
    return PredictResponse(
        index=best_idx,
        label=label,
        prob=best_prob,
        topk=topk,
    )
