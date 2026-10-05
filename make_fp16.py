"""Convert inswapper_128.onnx to FP16 for faster CUDA inference, then validate the result.

keep_io_types=True leaves graph inputs/outputs float32 so InsightFace's existing feeds work.
The model builds instance normalization out of ReduceMean/Sub/Sqrt/Div primitives; that variance
math overflows in half precision and produces a black frame, so those ops stay in float32 while
the heavy Conv/Gemm layers run in FP16.
"""

import pathlib
import warnings

import numpy as np
import onnx
from onnx import numpy_helper
from onnxconverter_common import float16

warnings.filterwarnings("ignore")

SRC = pathlib.Path("inswapper_128.onnx")
DST = pathlib.Path("inswapper_128_fp16.onnx")
KEEP_FP32 = ["ReduceMean", "Sub", "Sqrt", "Div", "Pow", "Add", "Mul"]
MAX_MEAN_DIFF = 2.0  # on a 0-255 scale


def emap_of(model):
    """InsightFace reads the identity mapping matrix as graph.initializer[-1]."""
    return numpy_helper.to_array(model.graph.initializer[-1])


print(f"loading {SRC} ({SRC.stat().st_size / 1e6:.0f} MB)")
model = onnx.load(str(SRC))
emap32 = emap_of(model)

print(f"converting to fp16, keeping {KEEP_FP32} in fp32...")
model16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
    disable_shape_infer=True,
    op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + KEEP_FP32,
)

if emap_of(model16).shape != emap32.shape:
    raise SystemExit("emap moved during conversion - insightface would read garbage")

onnx.save(model16, str(DST))
print(f"saved {DST} ({DST.stat().st_size / 1e6:.0f} MB)")

# ---- validate against fp32 on a real face ----
import os
import sys
import time

site = pathlib.Path(sys.prefix) / "Lib" / "site-packages"
for d in (site / "nvidia").glob("*/bin"):
    if d.is_dir():
        os.add_dll_directory(str(d))

import cv2
import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis

try:
    ort.preload_dlls(cuda=True, cudnn=True)
except Exception:
    pass

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread("source.jpg")
face = max(app.get(img), key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

results = {}
for label, path in [("fp32", SRC), ("fp16", DST)]:
    sw = insightface.model_zoo.get_model(str(path), providers=providers)
    for _ in range(5):
        sw.get(img, face, face, paste_back=False)
    n = 30
    start = time.perf_counter()
    for _ in range(n):
        sw.get(img, face, face, paste_back=False)
    ms = (time.perf_counter() - start) / n * 1000
    patch = sw.get(img, face, face, paste_back=False)[0].astype(np.float32)
    results[label] = (ms, patch)
    print(f"{label:5} model-only {ms:6.1f} ms")
    del sw

fp32_patch, fp16_patch = results["fp32"][1], results["fp16"][1]
mean_diff = float(np.abs(fp32_patch - fp16_patch).mean())
print(f"speedup {results['fp32'][0] / results['fp16'][0]:.2f}x")
print(f"mean abs diff vs fp32: {mean_diff:.3f}/255 (max {np.abs(fp32_patch - fp16_patch).max():.1f})")

if mean_diff > MAX_MEAN_DIFF:
    DST.unlink(missing_ok=True)
    raise SystemExit(f"FP16 output drifted too far (>{MAX_MEAN_DIFF}/255) - deleted {DST}")
print("FP16 model validated.")
