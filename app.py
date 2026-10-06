"""Desktop face-swap app: CustomTkinter gallery + live preview + virtual camera.

Usage:
    python app.py
    run_app.bat
"""

from __future__ import annotations

import threading
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from engine import SwapEngine

ROOT = Path(__file__).resolve().parent
FACES_DIR = ROOT / "faces"
PREVIEW_W = 640
PREVIEW_H = 480


class FaceSwapApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Live Face Swap")
        self.geometry("1100x640")
        self.minsize(900, 560)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.engine = SwapEngine(width=PREVIEW_W, height=PREVIEW_H, det_every=1)
        self.portraits = []
        self._photo = None
        self._selected_btn = None
        self._selected_card = None
        self._selected_name = ""
        self._closing = False

        self._build_loading_ui()
        self.after(50, self._bootstrap)

    def _build_loading_ui(self):
        self.loading = ctk.CTkLabel(
            self,
            text="Loading CUDA models…\nThis may take a few seconds.",
            font=ctk.CTkFont(size=18),
        )
        self.loading.pack(expand=True)

    def _build_main_ui(self):
        self.loading.destroy()

        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=12, pady=12)
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # ---- Left: gallery ----
        left = ctk.CTkFrame(root, width=220)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.grid_propagate(False)

        ctk.CTkLabel(left, text="Faces", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=12, pady=(12, 6)
        )
        ctk.CTkLabel(
            left,
            text=f"From ./{FACES_DIR.name}",
            text_color=("gray40", "gray70"),
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.gallery = ctk.CTkScrollableFrame(left, width=190)
        self.gallery.pack(fill="both", expand=True, padx=8, pady=(0, 12))

        # ---- Right: preview + controls ----
        right = ctk.CTkFrame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(right, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        top.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(top, text="Starting…", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="w")

        self.vcam_btn = ctk.CTkButton(
            top,
            text="Enable Virtual Camera",
            width=200,
            command=self._toggle_vcam,
        )
        self.vcam_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.preview_label = ctk.CTkLabel(right, text="", width=PREVIEW_W, height=PREVIEW_H)
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)

        self.footer = ctk.CTkLabel(
            right,
            text="",
            anchor="w",
            text_color=("gray30", "gray65"),
            font=ctk.CTkFont(size=12),
        )
        self.footer.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        self._populate_gallery()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._tick_preview)

    def _bootstrap(self):
        err = {"msg": None}

        def work():
            try:
                self.engine.load()
                self.portraits = self.engine.load_portraits(FACES_DIR)
                if not self.portraits:
                    raise RuntimeError(
                        f"No faces with a detectable face in {FACES_DIR}. "
                        "Drop jpg/png portraits there and restart."
                    )
            except Exception as exc:
                err["msg"] = str(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def poll():
            if thread.is_alive():
                self.after(100, poll)
                return
            if err["msg"]:
                self.loading.configure(text=f"Failed to start:\n{err['msg']}")
                return
            self._build_main_ui()
            # Prefer source.jpg if present, else first portrait.
            preferred = next((p for p in self.portraits if p.path.name.lower() == "source.jpg"), None)
            self._select_portrait(preferred or self.portraits[0])
            self.engine.start()
            self.status_label.configure(text=f"GPU: {self.engine.gpu_status}  |  ready")

        poll()

    def _populate_gallery(self):
        for child in self.gallery.winfo_children():
            child.destroy()
        self._thumb_refs = []

        if not self.portraits:
            ctk.CTkLabel(
                self.gallery,
                text="No faces found.\nAdd jpg/png files to ./faces\nand restart.",
                justify="left",
            ).pack(padx=8, pady=12)
            return

        if len(self.portraits) < 2:
            ctk.CTkLabel(
                self.gallery,
                text="Tip: add more photos to\n./faces to switch identities.",
                text_color=("gray40", "gray65"),
                font=ctk.CTkFont(size=11),
                justify="left",
            ).pack(anchor="w", padx=8, pady=(0, 8))

        for portrait in self.portraits:
            img = Image.fromarray(portrait.thumb_rgb)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(100, 100))
            self._thumb_refs.append(photo)

            card = ctk.CTkFrame(self.gallery, fg_color=("gray80", "gray25"), corner_radius=8)
            card.pack(padx=6, pady=6, fill="x")

            btn = ctk.CTkButton(
                card,
                text=portrait.name,
                image=photo,
                compound="top",
                width=160,
                height=140,
                fg_color="transparent",
                hover_color=("gray70", "gray35"),
                command=lambda p=portrait: self._select_portrait(p),
            )
            btn.pack(padx=4, pady=4)
            # Extra binds — some CustomTkinter builds miss clicks on the image area.
            for widget in (card, btn):
                widget.bind("<Button-1>", lambda _e, p=portrait: self._select_portrait(p))

            portrait._btn = btn
            portrait._card = card

    def _select_portrait(self, portrait):
        self.engine.set_active_portrait(portrait)
        self._selected_name = portrait.name
        if self._selected_card is not None:
            try:
                self._selected_card.configure(fg_color=("gray80", "gray25"), border_width=0)
            except Exception:
                pass

        btn = getattr(portrait, "_btn", None)
        card = getattr(portrait, "_card", None)
        if card is not None:
            card.configure(border_width=2, border_color="#6cb6ff", fg_color=("#2a5a8c", "#1f538d"))
            self._selected_card = card
        self._selected_btn = btn
        self.footer.configure(text=f"Active face: {portrait.name}")
        if hasattr(self, "status_label"):
            self.status_label.configure(text=f"Switched to {portrait.name}")

    def _toggle_vcam(self):
        enable = not self.engine.virtual_cam_enabled
        msg = self.engine.set_virtual_cam(enable)
        if self.engine.virtual_cam_enabled:
            self.vcam_btn.configure(text="Disable Virtual Camera", fg_color="#a33", hover_color="#822")
        else:
            self.vcam_btn.configure(
                text="Enable Virtual Camera",
                fg_color=("#3a7ebf", "#1f538d"),
                hover_color=("#325ea8", "#14375e"),
            )
        self.status_label.configure(text=msg)

    def _tick_preview(self):
        if self._closing:
            return
        frame, fps, status, vcam_status = self.engine.get_preview_bgr()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(PREVIEW_W / w, PREVIEW_H / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
            y0 = (PREVIEW_H - nh) // 2
            x0 = (PREVIEW_W - nw) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
            image = Image.fromarray(canvas)
            self._photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self._photo, text="")

            bits = [f"FPS {fps:.1f}", f"GPU {self.engine.gpu_status}"]
            if status:
                bits.append(status)
            self.status_label.configure(text="  |  ".join(bits))
            self.footer.configure(text=f"Active face: {self._selected_name}   ·   {vcam_status}")

        self.after(33, self._tick_preview)

    def _on_close(self):
        self._closing = True
        try:
            self.engine.stop()
        except Exception:
            pass
        self.destroy()


def main():
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    app = FaceSwapApp()
    app.mainloop()


if __name__ == "__main__":
    main()
