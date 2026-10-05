"""Slim CUDA face-swap engine for the desktop app.

Restores threaded capture, landmark EMA, soft ROI paste, and non-blocking
pyvirtualcam send — without the old camEffects effect stack.
"""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import insightface
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from PIL import Image

ROOT = Path(__file__).resolve().parent
SWAPPER_NAME = "inswapper_128.onnx"
SWAPPER_FP16_NAME = "inswapper_128_fp16.onnx"
TRT_CACHE = ROOT / "trt_cache"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TRT_PROVIDER = (
    "TensorrtExecutionProvider",
    {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(TRT_CACHE),
    },
)


def build_providers(use_trt: bool = False, cpu_only: bool = False):
    if cpu_only:
        return ["CPUExecutionProvider"]
    if use_trt:
        return [TRT_PROVIDER, "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def preload_nvidia_dlls():
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


def resolve_swapper_path(explicit: Optional[str] = None, prefer_fp16: bool = True) -> Optional[str]:
    candidates = []
    if explicit:
        candidates.append(explicit)
    models = osp.expanduser(osp.join("~", ".insightface", "models"))
    names = [SWAPPER_FP16_NAME, SWAPPER_NAME] if prefer_fp16 else [SWAPPER_NAME]
    for name in names:
        candidates.append(str(ROOT / name))
        candidates.append(osp.join(models, name))
    for path in candidates:
        if osp.isfile(path):
            return path
    return None


def load_swapper(providers, prefer_fp16: bool = True, explicit: Optional[str] = None):
    local = resolve_swapper_path(explicit, prefer_fp16)
    if local is not None:
        print(f"[engine] loading swapper from {osp.basename(local)}", flush=True)
        return insightface.model_zoo.get_model(local, providers=providers)
    print(f"[engine] {SWAPPER_NAME} not found locally, attempting download...", flush=True)
    return insightface.model_zoo.get_model(
        SWAPPER_NAME, download=True, download_zip=True, providers=providers
    )


def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def open_camera(index: int, width: int, height: int):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraStream:
    """Background grabber that keeps only the newest camera frame."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, name="CameraStream", daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2.0)
        self.cap.release()


class LandmarkSmoother:
    """EMA on the 5 alignment landmarks (and bbox) to reduce glasses flicker."""

    def __init__(self, alpha: float = 0.7):
        self.alpha = float(alpha)
        self.kps = None
        self.bbox = None

    def reset(self):
        self.kps = None
        self.bbox = None

    def update(self, face):
        kps = face.kps.astype(np.float32)
        bbox = face.bbox.astype(np.float32)
        if self.kps is None:
            self.kps = kps.copy()
            self.bbox = bbox.copy()
        else:
            a = self.alpha
            self.kps = a * self.kps + (1.0 - a) * kps
            self.bbox = a * self.bbox + (1.0 - a) * bbox
        return Face(bbox=self.bbox.copy(), kps=self.kps.copy(), det_score=getattr(face, "det_score", 1.0))


def shift_face_into_roi(face, x0, y0):
    bbox = face.bbox.astype(np.float32).copy()
    bbox[0] -= x0
    bbox[1] -= y0
    bbox[2] -= x0
    bbox[3] -= y0
    kps = None
    if getattr(face, "kps", None) is not None:
        kps = face.kps.astype(np.float32).copy()
        kps[:, 0] -= x0
        kps[:, 1] -= y0
    return Face(bbox=bbox, kps=kps, det_score=getattr(face, "det_score", 1.0))


def soft_paste_back(target_img, bgr_fake, M):
    """Feathered elliptical paste without InsightFace's black-border warp seams.

    Stock paste_back uses borderValue=0 and a hard |fake-diff| mask, which paints thin
    black strokes when landmarks jitter during head motion. We warp with BORDER_REPLICATE
    and a soft face ellipse only.
    """
    im = cv2.invertAffineTransform(M)
    th, tw = target_img.shape[:2]
    size = bgr_fake.shape[0]

    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(
        mask,
        (size // 2, int(size * 0.48)),
        (int(size * 0.44), int(size * 0.56)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    # Soften the outer rim so motion cannot leave hard black edges.
    k = max((size // 6) | 1, 11)
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    # Zero a thin outer ring so warped black/empty texels never composite in.
    ring = np.zeros_like(mask)
    cv2.ellipse(
        ring,
        (size // 2, int(size * 0.48)),
        (int(size * 0.48), int(size * 0.60)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    mask *= ring
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    warped = cv2.warpAffine(
        bgr_fake,
        im,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.warpAffine(
        mask,
        im,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_mask = np.clip(warped_mask, 0.0, 1.0).astype(np.float32)[..., None]

    target_f = target_img.astype(np.float32)
    warped_f = warped.astype(np.float32)
    out = warped_mask * warped_f + (1.0 - warped_mask) * target_f
    return np.clip(out, 0, 255).astype(np.uint8)


def swap_face(frame, target_face, source_face, swapper):
    bgr_fake, M = swapper.get(frame, target_face, source_face, paste_back=False)
    return soft_paste_back(frame, bgr_fake, M)


def _roi_feather_mask(h: int, w: int, border: int) -> np.ndarray:
    """1 inside, fades to 0 at the crop border so ROI paste never leaves a hard seam."""
    border = max(1, min(border, h // 3, w // 3))
    mask = np.ones((h, w), np.float32)
    if border > 0:
        ramp = np.linspace(0.0, 1.0, border, dtype=np.float32)
        mask[:border, :] *= ramp[:, None]
        mask[-border:, :] *= ramp[::-1, None]
        mask[:, :border] *= ramp[None, :]
        mask[:, -border:] *= ramp[None, ::-1]
    return mask[..., None]


def swap_on_roi(frame, target_face, source_face, swapper, margin: float = 0.45):
    x1, y1, x2, y2 = target_face.bbox.astype(int)
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    pad_x = int(fw * margin)
    pad_y = int(fh * margin)
    h, w = frame.shape[:2]
    x0 = max(0, x1 - pad_x)
    y0 = max(0, y1 - pad_y)
    x1c = min(w, x2 + pad_x)
    y1c = min(h, y2 + pad_y)
    if x1c - x0 < 16 or y1c - y0 < 16:
        return swap_face(frame, target_face, source_face, swapper)

    crop = np.ascontiguousarray(frame[y0:y1c, x0:x1c])
    swapped = swap_face(crop, shift_face_into_roi(target_face, x0, y0), source_face, swapper)
    roi_h, roi_w = swapped.shape[:2]
    feather = _roi_feather_mask(roi_h, roi_w, border=max(10, min(roi_h, roi_w) // 12))
    base = frame[y0:y1c, x0:x1c].astype(np.float32)
    blended = feather * swapped.astype(np.float32) + (1.0 - feather) * base
    frame[y0:y1c, x0:x1c] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame


def obs_app_is_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq obs64.exe"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return "obs64.exe" in (result.stdout or "").lower()
    except Exception:
        return False


def create_virtual_camera(width: int, height: int, fps: int = 30):
    """Create pyvirtualcam device (OBS Virtual Camera driver). Never raises."""
    try:
        import pyvirtualcam
    except ImportError:
        return None, "pyvirtualcam is not installed"

    if obs_app_is_running():
        warn = "OBS Studio is running and may hold the virtual camera"
    else:
        warn = None

    try:
        cam = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            print_fps=False,
        )
        msg = f"{cam.device} ({width}x{height})"
        if warn:
            msg = f"{msg} — warning: {warn}"
        return cam, msg
    except Exception as exc:
        detail = str(exc)
        if obs_app_is_running():
            detail += " (quit OBS Studio and retry)"
        return None, detail


@dataclass
class Portrait:
    path: Path
    name: str
    thumb_rgb: np.ndarray  # HxWx3 RGB uint8 for UI
    face: object  # InsightFace Face with embedding frozen at load time
    face_id: int = 0


class SwapEngine:
    """Background swap pipeline with precomputed source embeddings."""

    def __init__(
        self,
        camera: int = 0,
        width: int = 640,
        height: int = 480,
        det_every: int = 3,
        roi_margin: float = 0.45,
        prefer_fp16: bool = True,
        use_trt: bool = False,
        cpu_only: bool = False,
    ):
        self.camera = camera
        self.width = width
        self.height = height
        self.det_every = max(1, det_every)
        self.roi_margin = roi_margin
        self.prefer_fp16 = prefer_fp16
        self.use_trt = use_trt
        self.cpu_only = cpu_only

        self.providers = None
        self.embed_app = None
        self.detect_app = None
        self.swapper = None
        self.gpu_status = "not loaded"

        self._face_lock = threading.Lock()
        self._active_face = None
        self._active_name = ""
        self._active_id = -1
        self._face_generation = 0
        self._frame_lock = threading.Lock()
        self._preview = None
        self._fps = 0.0
        self._status = ""

        self._vcam_lock = threading.Lock()
        self._vcam = None
        self._vcam_wanted = False
        self._vcam_status = "Virtual camera off"

        self._stream = None
        self._worker = None
        self._stop = threading.Event()
        self._running = False
        self._next_face_id = 1

    def load(self):
        preload_nvidia_dlls()
        self.providers = build_providers(self.use_trt, self.cpu_only)
        if self.use_trt:
            TRT_CACHE.mkdir(exist_ok=True)

        print("[engine] loading buffalo_l (embeddings)...", flush=True)
        self.embed_app = FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.embed_app.prepare(ctx_id=0, det_size=(640, 640))

        print("[engine] loading buffalo_l detection only...", flush=True)
        self.detect_app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection"], providers=self.providers
        )
        self.detect_app.prepare(ctx_id=0, det_size=(320, 320))

        print("[engine] loading inswapper...", flush=True)
        self.swapper = load_swapper(self.providers, prefer_fp16=self.prefer_fp16)

        active = set()
        for model in list(self.embed_app.models.values()) + list(self.detect_app.models.values()):
            active.update(model.session.get_providers())
        active.update(self.swapper.session.get_providers())
        if "CUDAExecutionProvider" in active:
            self.gpu_status = "CUDA"
        elif "TensorrtExecutionProvider" in active:
            self.gpu_status = "TensorRT"
        else:
            self.gpu_status = "CPU"
        print(f"[engine] providers active -> {self.gpu_status}", flush=True)
        return self.gpu_status

    def load_portraits(self, faces_dir) -> list[Portrait]:
        faces_dir = Path(faces_dir)
        faces_dir.mkdir(parents=True, exist_ok=True)
        portraits: list[Portrait] = []
        paths = sorted(
            p for p in faces_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"[engine] skip unreadable {path.name}", flush=True)
                continue
            faces = self.embed_app.get(image)
            if not faces:
                print(f"[engine] skip no-face {path.name}", flush=True)
                continue
            src = largest_face(faces)
            # Freeze a private Face copy so later InsightFace mutations cannot clobber the
            # precomputed identity, and so each portrait has a distinct object identity.
            emb = np.asarray(src.embedding, dtype=np.float32).copy()
            frozen = Face(
                bbox=np.asarray(src.bbox, dtype=np.float32).copy(),
                kps=np.asarray(src.kps, dtype=np.float32).copy() if src.kps is not None else None,
                det_score=float(getattr(src, "det_score", 1.0)),
            )
            frozen.embedding = emb
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            thumb = np.array(Image.fromarray(rgb).resize((120, 120), Image.Resampling.LANCZOS))
            face_id = self._next_face_id
            self._next_face_id += 1
            portraits.append(
                Portrait(path=path, name=path.stem, thumb_rgb=thumb, face=frozen, face_id=face_id)
            )
            print(
                f"[engine] embedded {path.name} id={face_id} emb_norm={float(np.linalg.norm(emb)):.3f}",
                flush=True,
            )
        return portraits

    def set_active_face(self, face, name: str = "", face_id: int = -1):
        """Thread-safe switch. Call from UI when a gallery portrait is clicked."""
        with self._face_lock:
            self._active_face = face
            self._active_name = name or self._active_name
            self._active_id = face_id
            self._face_generation += 1
            gen = self._face_generation
        print(f"[engine] active face -> {name or '?'} id={face_id} gen={gen}", flush=True)
        self._status = f"switched to {name}" if name else "face updated"

    def set_active_portrait(self, portrait: Portrait):
        self.set_active_face(portrait.face, name=portrait.name, face_id=portrait.face_id)

    def get_active_face(self):
        with self._face_lock:
            return self._active_face, self._active_name, self._active_id, self._face_generation
    def start(self):
        if self._running:
            return
        cap = open_camera(self.camera, self.width, self.height)
        self._stream = CameraStream(cap)
        for _ in range(50):
            ok, _ = self._stream.read()
            if ok:
                break
            time.sleep(0.02)
        self._stop.clear()
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, name="SwapWorker", daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        self.set_virtual_cam(False)

    def get_preview_bgr(self):
        with self._frame_lock:
            frame = None if self._preview is None else self._preview.copy()
            fps = self._fps
        return frame, fps, self._status, self._vcam_status

    @property
    def virtual_cam_enabled(self) -> bool:
        with self._vcam_lock:
            return self._vcam is not None

    def set_virtual_cam(self, enabled: bool) -> str:
        self._vcam_wanted = bool(enabled)
        with self._vcam_lock:
            if not enabled:
                if self._vcam is not None:
                    try:
                        self._vcam.close()
                    except Exception:
                        pass
                    self._vcam = None
                self._vcam_status = "Virtual camera off"
                return self._vcam_status

            if self._vcam is not None:
                return self._vcam_status

            cam, msg = create_virtual_camera(self.width, self.height, fps=30)
            if cam is None:
                self._vcam = None
                self._vcam_wanted = False
                self._vcam_status = f"Virtual camera failed: {msg}"
                return self._vcam_status

            self._vcam = cam
            self._vcam_status = f"Virtual camera on: {msg}"
            return self._vcam_status

    def _send_vcam(self, frame_bgr):
        with self._vcam_lock:
            vcam = self._vcam
        if vcam is None:
            return
        try:
            # CRITICAL: do not call sleep_until_next_frame() — it blocks the worker.
            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
            vcam.send(frame_bgr)
        except Exception as exc:
            self._vcam_status = f"Virtual camera send error: {exc}"

    def _worker_loop(self):
        smoother = LandmarkSmoother(alpha=0.8)
        cached = None
        frame_i = 0
        fps = 0.0
        prev = time.perf_counter()

        while not self._stop.is_set():
            ok, frame = self._stream.read() if self._stream else (False, None)
            if not ok or frame is None:
                time.sleep(0.001)
                continue

            # Selfie-style mirror for preview + Zoom (same as camEffects).
            frame = cv2.flip(frame, 1)

            if cached is None or frame_i % self.det_every == 0:
                faces = self.detect_app.get(frame)
                if faces:
                    cached = smoother.update(largest_face(faces))
                else:
                    cached = None
                    smoother.reset()

            source, source_name, source_id, _gen = self.get_active_face()
            if cached is not None and source is not None:
                try:
                    if getattr(source, "embedding", None) is None and getattr(source, "normed_embedding", None) is None:
                        raise RuntimeError("active face has no embedding")
                    frame = swap_on_roi(
                        frame, cached, source, self.swapper, margin=self.roi_margin
                    )
                except Exception as exc:
                    self._status = f"swap error: {exc}"

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0.0 else fps * 0.9 + instant * 0.1

            display = frame.copy()
            label = f"FPS: {fps:5.1f}  {self.gpu_status}"
            if source_name:
                label += f"  [{source_name}]"
            cv2.putText(
                display,
                label,
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            with self._frame_lock:
                self._preview = display
                self._fps = fps

            if self._vcam_wanted:
                self._send_vcam(frame)

            frame_i += 1

        self._status = "stopped"
