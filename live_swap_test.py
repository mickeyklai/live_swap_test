"""Live webcam face swapping with InsightFace + inswapper_128 on ONNX Runtime GPU.

Usage:
    python live_swap_test.py --source source.jpg
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
from pathlib import Path

import cv2
import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis

SWAPPER_NAME = "inswapper_128.onnx"
SWAPPER_FP16_NAME = "inswapper_128_fp16.onnx"
WINDOW = "Live Face Swap"
TRT_CACHE = Path(__file__).resolve().parent / "trt_cache"

TRT_PROVIDER = (
    "TensorrtExecutionProvider",
    {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(TRT_CACHE),
    },
)


def build_providers(use_trt, cpu_only):
    """On Blackwell (sm_120) the stock onnxruntime-gpu wheel ships only PTX, and JIT compilation
    aborts the process for some kernels - intermittently, so it cannot be caught. --cpu is the
    reliable fallback until a Blackwell-native wheel is in place."""
    if cpu_only:
        return ["CPUExecutionProvider"]
    if use_trt:
        return [TRT_PROVIDER, "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def preload_nvidia_dlls():
    """Register CUDA/cuDNN/TensorRT wheel DLL dirs so the GPU providers can resolve them on Windows."""
    site = Path(sys.prefix) / "Lib" / "site-packages"
    dirs = []
    nvidia_root = site / "nvidia"
    if nvidia_root.is_dir():
        dirs += list(nvidia_root.glob("*/bin")) + list(nvidia_root.glob("*/bin/x86_64"))
    dirs += [site / "tensorrt_libs", site / "tensorrt"]
    for d in dirs:
        if d.is_dir():
            os.add_dll_directory(str(d))
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Live webcam face swap (InsightFace + ONNX Runtime GPU)")
    parser.add_argument("--source", default="source.jpg", help="source face image")
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    parser.add_argument("--width", type=int, default=640, help="requested capture width")
    parser.add_argument("--height", type=int, default=480, help="requested capture height")
    parser.add_argument("--swapper", default=None, help="explicit path to the inswapper onnx file")
    parser.add_argument(
        "--trt",
        action="store_true",
        help="prioritize TensorrtExecutionProvider (compiles engines on first run)",
    )
    parser.add_argument("--cpu", action="store_true", help="force CPUExecutionProvider only")
    parser.add_argument(
        "--fp32", action="store_true", help="use the fp32 swapper even if the fp16 one exists"
    )
    return parser.parse_args()


def print_session_providers(label, session):
    print(f"  {label:<24} -> {session.get_providers()}")


def announce_compile_stage(use_trt, cpu_only):
    print("-" * 70)
    if cpu_only:
        print("provider: CPUExecutionProvider only (--cpu)")
    elif not use_trt:
        print("provider priority: CUDAExecutionProvider -> CPUExecutionProvider")
        print("(pass --trt to put TensorrtExecutionProvider first)")
    elif TRT_CACHE.is_dir() and any(TRT_CACHE.iterdir()):
        print(f"TensorRT engine cache found in {TRT_CACHE} - loading prebuilt engines.")
    else:
        print(f"No TensorRT engine cache in {TRT_CACHE}.")
        print("First run COMPILES engines for your GPU: expect 30-60s per model, NOT a hang.")
        print("Subsequent runs reuse the cache and start in a couple of seconds.")
    print("-" * 70)


def verify_gpu(source_app, webcam_app, swapper):
    print("=" * 70)
    print(f"onnxruntime {ort.__version__}  |  ort.get_device() = {ort.get_device()}")
    print(f"available providers: {ort.get_available_providers()}")
    print("-" * 70)
    print("confirmed active providers per model:")
    active = set()
    for name, model in source_app.models.items():
        print_session_providers(f"source/{name}", model.session)
        active.update(model.session.get_providers())
    for name, model in webcam_app.models.items():
        print_session_providers(f"webcam/{name}", model.session)
        active.update(model.session.get_providers())
    print_session_providers("inswapper", swapper.session)
    active.update(swapper.session.get_providers())
    print("=" * 70)

    if "TensorrtExecutionProvider" in active:
        print("GPU OK: running on TensorRT.")
    elif "CUDAExecutionProvider" in active:
        print("GPU OK: running on CUDA.")
    elif active == {"CPUExecutionProvider"}:
        print("running on CPU - expect single-digit FPS.")


def resolve_swapper_path(explicit, prefer_fp16=True):
    candidates = []
    if explicit:
        candidates.append(explicit)
    here = osp.dirname(osp.abspath(__file__))
    models = osp.expanduser(osp.join("~", ".insightface", "models"))
    names = [SWAPPER_FP16_NAME, SWAPPER_NAME] if prefer_fp16 else [SWAPPER_NAME]
    for name in names:
        candidates.append(osp.join(here, name))
        candidates.append(osp.join(models, name))
    for path in candidates:
        if osp.isfile(path):
            return path
    return None


def load_swapper(explicit, providers, prefer_fp16=True):
    local = resolve_swapper_path(explicit, prefer_fp16)
    if local is not None:
        print(f"      loading swapper from {osp.basename(local)}")
        return insightface.model_zoo.get_model(local, providers=providers)

    print(f"{SWAPPER_NAME} not found locally, attempting download...")
    try:
        return insightface.model_zoo.get_model(
            SWAPPER_NAME, download=True, download_zip=True, providers=providers
        )
    except Exception as exc:
        raise SystemExit(
            f"could not obtain {SWAPPER_NAME} ({exc}).\n"
            f"Download it manually and place it next to this script or in ~/.insightface/models/."
        )


def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def load_source_face(app, path):
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"could not read source image: {path}")
    faces = app.get(image)
    if not faces:
        raise SystemExit(f"no face detected in source image: {path}")
    face = largest_face(faces)
    print(f"source face locked in: bbox={face.bbox.astype(int).tolist()}, {len(faces)} face(s) found")
    return face


def open_camera(index, width, height):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(
        f"camera {index} open at "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    return cap


def main():
    args = parse_args()
    preload_nvidia_dlls()
    providers = build_providers(args.trt, args.cpu)
    if args.trt:
        TRT_CACHE.mkdir(exist_ok=True)
    announce_compile_stage(args.trt, args.cpu)

    # Full pipeline (detection + recognition) is only needed to embed the source face.
    print("[1/3] loading buffalo_l detection + recognition (source embedding)...", flush=True)
    started = time.perf_counter()
    source_app = FaceAnalysis(name="buffalo_l", providers=providers)
    source_app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"      ready in {time.perf_counter() - started:.1f}s", flush=True)

    # Detection-only pipeline for the live stream: no embeddings, so no wasted GPU time.
    print("[2/3] loading buffalo_l detection only (webcam, det_size=320)...", flush=True)
    started = time.perf_counter()
    webcam_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
    webcam_app.prepare(ctx_id=0, det_size=(320, 320))
    print(f"      ready in {time.perf_counter() - started:.1f}s", flush=True)

    print("[3/3] loading inswapper_128...", flush=True)
    started = time.perf_counter()
    swapper = load_swapper(args.swapper, providers, prefer_fp16=not args.fp32)
    print(f"      ready in {time.perf_counter() - started:.1f}s", flush=True)

    verify_gpu(source_app, webcam_app, swapper)

    source_face = load_source_face(source_app, args.source)

    cap = open_camera(args.camera, args.width, args.height)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("running - press 'q' or ESC to quit")

    fps = 0.0
    prev = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed, stopping", file=sys.stderr)
                break

            faces = webcam_app.get(frame)
            if faces:
                frame = swapper.get(frame, largest_face(faces), source_face, paste_back=True)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0.0 else fps * 0.9 + instant * 0.1

            cv2.putText(
                frame,
                f"FPS: {fps:5.1f}  faces: {len(faces)}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"stopped at {fps:.1f} FPS")


if __name__ == "__main__":
    main()
