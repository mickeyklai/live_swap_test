"""Desktop face-swap app: premium CustomTkinter UI + live preview + virtual camera.

Usage:
    python app.py
    run_app.bat

UI-only redesign — swap logic lives in engine.py and is unchanged.
"""

from __future__ import annotations

import shutil
import threading
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- Design tokens (visual only) ----
BG = "#0B0F17"
PANEL = "#121826"
CARD = "#171E2B"
ELEVATED = "#1C2433"
BORDER = "#2A3344"
ACCENT = "#3B82F6"
ACCENT_SOFT = "#1E3A5F"
ACCENT_SEL = "#152238"
TEXT = "#E8EEF7"
MUTED = "#8B9BB4"
SUCCESS = "#22C55E"

SUBTITLES = {
    "face_01": "Business Portrait",
    "face_02": "Casual Look",
    "face_03": "Studio Portrait",
    "face_04": "Professional",
    "face_05": "Natural Look",
    "face_06": "Portrait",
    "face_07": "Portrait",
    "face_08": "Portrait",
    "source": "Default Source",
}


def _font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)


class FaceSwapApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Live Face Swap")
        self.geometry("1280x800")
        self.minsize(1000, 640)
        self.configure(fg_color=BG)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.engine = SwapEngine(width=PREVIEW_W, height=PREVIEW_H, det_every=1)
        self.portraits = []
        self._photo = None
        self._active_thumb = None
        self._selected_card = None
        self._selected_portrait = None
        self._selected_name = ""
        self._closing = False
        self._zoomed = False
        self._vcam_syncing = False
        self._card_widgets = {}  # name -> card frame

        self._build_loading_ui()
        self.after(50, self._bootstrap)

    # ------------------------------------------------------------------ bootstrap
    def _build_loading_ui(self):
        self.loading = ctk.CTkLabel(
            self,
            text="Loading CUDA models…\nThis may take a few seconds.",
            font=_font(16),
            text_color=MUTED,
            fg_color=BG,
        )
        self.loading.pack(expand=True)

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
                self.loading.configure(text=f"Failed to start:\n{err['msg']}", text_color="#F87171")
                return
            self._build_main_ui()
            preferred = next(
                (p for p in self.portraits if p.path.name.lower() == "source.jpg"), None
            )
            self._select_portrait(preferred or self.portraits[0])
            self.engine.start()

        poll()

    # ------------------------------------------------------------------ layout
    def _build_main_ui(self):
        self.loading.destroy()

        shell = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        shell.pack(fill="both", expand=True)
        shell.grid_rowconfigure(1, weight=1)
        shell.grid_columnconfigure(0, weight=1)

        self._build_top_nav(shell)

        body = ctk.CTkFrame(shell, fg_color=BG, corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_preview_panel(body)
        self._build_status_bar(shell)

        self._populate_gallery()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._tick_preview)

    def _build_top_nav(self, parent):
        nav = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, height=64)
        nav.grid(row=0, column=0, sticky="ew")
        nav.grid_propagate(False)
        nav.grid_columnconfigure(1, weight=1)

        brand = ctk.CTkFrame(nav, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w", padx=20, pady=10)
        ctk.CTkLabel(
            brand, text="Live Face Swap", font=_font(16, "bold"), text_color=TEXT, anchor="w"
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand,
            text="Real-time face swapping with AI",
            font=_font(11),
            text_color=MUTED,
            anchor="w",
        ).pack(anchor="w")

        self.vcam_btn = ctk.CTkButton(
            nav,
            text="Enable Virtual Camera",
            width=200,
            height=36,
            corner_radius=10,
            font=_font(13, "bold"),
            fg_color=ACCENT,
            hover_color="#2563EB",
            text_color="#FFFFFF",
            command=self._toggle_vcam,
        )
        self.vcam_btn.grid(row=0, column=2, sticky="e", padx=20)

    def _build_sidebar(self, parent):
        side = ctk.CTkFrame(
            parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER, width=280
        )
        side.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        side.grid_propagate(False)
        side.grid_rowconfigure(3, weight=1)
        side.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(side, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head, text="Face Library", font=_font(15, "bold"), text_color=TEXT, anchor="w"
        ).grid(row=0, column=0, sticky="w")
        self.face_count_lbl = ctk.CTkLabel(
            head, text="0 Faces", font=_font(11), text_color=MUTED, anchor="e"
        )
        self.face_count_lbl.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            side,
            text="Select a face to swap onto your camera",
            font=_font(11),
            text_color=MUTED,
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        ctk.CTkButton(
            side,
            text="+  Add Face",
            height=36,
            corner_radius=10,
            font=_font(13, "bold"),
            fg_color=ACCENT,
            hover_color="#2563EB",
            text_color="#FFFFFF",
            command=self._add_face,
        ).grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))

        self.gallery = ctk.CTkScrollableFrame(
            side,
            fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT_SOFT,
        )
        self.gallery.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 12))

    def _build_preview_panel(self, parent):
        main = ctk.CTkFrame(
            parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER
        )
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Header row
        hdr = ctk.CTkFrame(main, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        hdr.grid_columnconfigure(0, weight=1)

        titles = ctk.CTkFrame(hdr, fg_color="transparent")
        titles.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            titles, text="Live Preview", font=_font(15, "bold"), text_color=TEXT, anchor="w"
        ).pack(anchor="w")
        ctk.CTkLabel(
            titles,
            text="Real-time face swap using your camera and selected face",
            font=_font(11),
            text_color=MUTED,
            anchor="w",
        ).pack(anchor="w")

        tools = ctk.CTkFrame(hdr, fg_color="transparent")
        tools.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(
            tools,
            text="⛶",
            width=36,
            height=32,
            corner_radius=8,
            fg_color=ELEVATED,
            hover_color=ACCENT_SOFT,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            font=_font(14),
            command=self._toggle_fullscreen,
        ).pack(side="left", padx=(0, 6))

        # Preview frame with overlay chips
        self.preview_frame = ctk.CTkFrame(
            main, fg_color="#070A10", corner_radius=12, border_width=1, border_color=BORDER
        )
        self.preview_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(self.preview_frame, text="", fg_color="#070A10")
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        chips = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        chips.place(relx=0.0, rely=0.0, x=18, y=16)
        self.chip_fps = self._make_chip(chips, "FPS —")
        self.chip_fps.pack(side="left", padx=(0, 6))
        self.chip_gpu = self._make_chip(chips, "CUDA")
        self.chip_gpu.pack(side="left", padx=(0, 6))
        self.chip_src = self._make_chip(chips, "Source")
        self.chip_src.pack(side="left")

        # Status cards row
        cards = ctk.CTkFrame(main, fg_color="transparent")
        cards.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        for i in range(4):
            cards.grid_columnconfigure(i, weight=1, uniform="stat")

        self.card_face = self._stat_card(cards, 0, "Active Face")
        self.active_face_thumb = ctk.CTkLabel(
            self.card_face, text="", width=40, height=40, corner_radius=8, fg_color=ELEVATED
        )
        self.active_face_thumb.pack(side="left", padx=(0, 10), pady=4)
        self.active_face_name = ctk.CTkLabel(
            self.card_face, text="—", font=_font(13, "bold"), text_color=TEXT, anchor="w"
        )
        self.active_face_name.pack(side="left", fill="x", expand=True)

        self.card_perf = self._stat_card(cards, 1, "Performance")
        self.perf_fps_lbl = ctk.CTkLabel(
            self.card_perf, text="FPS —", font=_font(18, "bold"), text_color=SUCCESS, anchor="w"
        )
        self.perf_fps_lbl.pack(anchor="w")

        self.card_proc = self._stat_card(cards, 2, "Processing")
        self.proc_gpu_lbl = ctk.CTkLabel(
            self.card_proc, text="GPU —", font=_font(14, "bold"), text_color=TEXT, anchor="w"
        )
        self.proc_gpu_lbl.pack(anchor="w")
        self.proc_sub_lbl = ctk.CTkLabel(
            self.card_proc, text="Starting…", font=_font(11), text_color=MUTED, anchor="w"
        )
        self.proc_sub_lbl.pack(anchor="w")

        self.card_vcam = self._stat_card(cards, 3, "Virtual Camera")
        vrow = ctk.CTkFrame(self.card_vcam, fg_color="transparent")
        vrow.pack(fill="x")
        self.card_vcam_state = ctk.CTkLabel(
            vrow, text="Off", font=_font(14, "bold"), text_color=MUTED, anchor="w"
        )
        self.card_vcam_state.pack(side="left")
        self.vcam_switch_card = ctk.CTkSwitch(
            vrow,
            text="",
            width=42,
            command=self._on_vcam_card_switch,
            progress_color=SUCCESS,
            button_color=TEXT,
            button_hover_color=TEXT,
            fg_color=BORDER,
        )
        self.vcam_switch_card.pack(side="right")

    def _make_chip(self, parent, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent,
            text=text,
            font=_font(11, "bold"),
            text_color=TEXT,
            fg_color=("#000000", "#0E1522"),
            corner_radius=8,
            padx=10,
            pady=4,
        )

    def _stat_card(self, parent, col: int, title: str) -> ctk.CTkFrame:
        wrap = ctk.CTkFrame(
            parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER
        )
        wrap.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0 if col == 3 else 6))
        ctk.CTkLabel(wrap, text=title, font=_font(11), text_color=MUTED, anchor="w").pack(
            anchor="w", padx=12, pady=(10, 2)
        )
        body = ctk.CTkFrame(wrap, fg_color="transparent")
        body.pack(fill="x", padx=12, pady=(0, 12))
        return body

    def _build_status_bar(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, height=32)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=16)
        self.status_dot = ctk.CTkLabel(left, text="●", font=_font(10), text_color=SUCCESS)
        self.status_dot.pack(side="left")
        self.status_left = ctk.CTkLabel(
            left,
            text=" Ready   |   Active face: —   |   Source: Camera   |   Virtual camera: Off",
            font=_font(11),
            text_color=MUTED,
            anchor="w",
        )
        self.status_left.pack(side="left")

        self.status_right = ctk.CTkLabel(
            bar,
            text=f"{PREVIEW_W} × {PREVIEW_H}   ·   — FPS   ·   GPU (—)",
            font=_font(11),
            text_color=MUTED,
            anchor="e",
        )
        self.status_right.grid(row=0, column=1, sticky="e", padx=16)

    # ------------------------------------------------------------------ gallery
    def _add_face(self):
        path = filedialog.askopenfilename(
            title="Add Face",
            filetypes=[
                ("Images", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        src = Path(path)
        if src.suffix.lower() not in IMAGE_EXTS:
            messagebox.showerror("Add Face", "Please choose a jpg, png, webp, or bmp image.")
            return

        FACES_DIR.mkdir(parents=True, exist_ok=True)
        dest = FACES_DIR / src.name
        if dest.exists():
            stem, suf = src.stem, src.suffix.lower()
            n = 2
            while True:
                dest = FACES_DIR / f"{stem}_{n}{suf}"
                if not dest.exists():
                    break
                n += 1

        try:
            shutil.copy2(src, dest)
        except Exception as exc:
            messagebox.showerror("Add Face", f"Could not copy image:\n{exc}")
            return

        prev_names = {p.name for p in self.portraits}
        try:
            self.portraits = self.engine.load_portraits(FACES_DIR)
        except Exception as exc:
            messagebox.showerror("Add Face", f"Failed to load faces:\n{exc}")
            return

        self._populate_gallery()
        added = next((p for p in self.portraits if p.name not in prev_names), None)
        if added is None:
            # Same stem may already exist; prefer file we just wrote.
            added = next((p for p in self.portraits if p.path.name == dest.name), None)
        if added is None:
            messagebox.showwarning(
                "Add Face",
                "Image was saved, but no face was detected.\nTry a clearer front-facing portrait.",
            )
            return
        self._select_portrait(added)

    def _populate_gallery(self):
        for child in self.gallery.winfo_children():
            child.destroy()
        self._thumb_refs = []
        self._card_widgets = {}

        n = len(self.portraits)
        self.face_count_lbl.configure(text=f"{n} Face{'s' if n != 1 else ''}")

        if not self.portraits:
            ctk.CTkLabel(
                self.gallery,
                text="No faces found.\nAdd jpg/png files to ./faces\nand restart.",
                justify="left",
                text_color=MUTED,
                font=_font(12),
            ).pack(padx=8, pady=12)
            return

        for portrait in self.portraits:
            self._add_face_card(portrait)

    def _add_face_card(self, portrait):
        img = Image.fromarray(portrait.thumb_rgb)
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=(48, 48))
        self._thumb_refs.append(photo)

        card = ctk.CTkFrame(
            self.gallery,
            fg_color=CARD,
            corner_radius=10,
            border_width=1,
            border_color=BORDER,
            height=64,
        )
        card.pack(fill="x", padx=4, pady=4)
        card.pack_propagate(False)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=8, pady=8)

        thumb = ctk.CTkLabel(inner, text="", image=photo, width=48, height=48)
        thumb.pack(side="left", padx=(0, 10))

        text_col = ctk.CTkFrame(inner, fg_color="transparent")
        text_col.pack(side="left", fill="both", expand=True)
        name_lbl = ctk.CTkLabel(
            text_col, text=portrait.name, font=_font(13, "bold"), text_color=TEXT, anchor="w"
        )
        name_lbl.pack(anchor="w")
        sub = SUBTITLES.get(portrait.name, "Portrait")
        sub_lbl = ctk.CTkLabel(
            text_col, text=sub, font=_font(11), text_color=MUTED, anchor="w"
        )
        sub_lbl.pack(anchor="w")

        def on_click(_event=None, p=portrait):
            self._select_portrait(p)

        for w in (card, inner, thumb, text_col, name_lbl, sub_lbl):
            w.bind("<Button-1>", on_click)
            w.bind("<Enter>", lambda _e, c=card: self._hover_card(c, True))
            w.bind("<Leave>", lambda _e, c=card, p=portrait: self._hover_card(c, False, p))

        portrait._card = card
        self._card_widgets[portrait.name] = card

    def _hover_card(self, card, entering: bool, portrait=None):
        if self._selected_card is card:
            return
        if entering:
            card.configure(fg_color=ELEVATED)
        else:
            card.configure(fg_color=CARD)

    def _select_portrait(self, portrait):
        self.engine.set_active_portrait(portrait)
        self._selected_name = portrait.name
        self._selected_portrait = portrait

        if self._selected_card is not None:
            try:
                self._selected_card.configure(fg_color=CARD, border_color=BORDER, border_width=1)
            except Exception:
                pass

        card = getattr(portrait, "_card", None)
        if card is not None:
            card.configure(border_width=2, border_color=ACCENT, fg_color=ACCENT_SEL)
            self._selected_card = card

        self.active_face_name.configure(text=portrait.name)
        thumb_img = Image.fromarray(portrait.thumb_rgb)
        self._active_thumb = ctk.CTkImage(light_image=thumb_img, dark_image=thumb_img, size=(40, 40))
        self.active_face_thumb.configure(image=self._active_thumb, text="")

    # ------------------------------------------------------------------ vcam / window
    def _sync_vcam_ui(self, enabled: bool, msg: str = ""):
        state = "On" if enabled else "Off"
        color = SUCCESS if enabled else MUTED
        self.card_vcam_state.configure(text=state, text_color=color)
        if enabled:
            self.vcam_btn.configure(
                text="Virtual Camera ON",
                fg_color="#15803D",
                hover_color="#166534",
            )
        else:
            self.vcam_btn.configure(
                text="Enable Virtual Camera",
                fg_color=ACCENT,
                hover_color="#2563EB",
            )
        self._vcam_syncing = True
        try:
            if enabled and not self.vcam_switch_card.get():
                self.vcam_switch_card.select()
            elif not enabled and self.vcam_switch_card.get():
                self.vcam_switch_card.deselect()
        finally:
            self._vcam_syncing = False
        if msg:
            self.proc_sub_lbl.configure(text=msg[:48])

    def _apply_vcam(self, want: bool):
        if want == self.engine.virtual_cam_enabled:
            self._sync_vcam_ui(want)
            return
        msg = self.engine.set_virtual_cam(want)
        self._sync_vcam_ui(self.engine.virtual_cam_enabled, msg)

    def _toggle_vcam(self):
        self._apply_vcam(not self.engine.virtual_cam_enabled)

    def _on_vcam_card_switch(self):
        if self._vcam_syncing:
            return
        self._apply_vcam(bool(self.vcam_switch_card.get()))

    def _toggle_fullscreen(self):
        self._zoomed = not self._zoomed
        try:
            self.state("zoomed" if self._zoomed else "normal")
        except Exception:
            self.attributes("-fullscreen", self._zoomed)

    # ------------------------------------------------------------------ preview tick
    def _tick_preview(self):
        if self._closing:
            return
        frame, fps, status, vcam_status = self.engine.get_preview_bgr()
        if frame is not None:
            # Fit into available preview label size (fallback to fixed).
            try:
                pw = max(320, self.preview_label.winfo_width())
                ph = max(240, self.preview_label.winfo_height())
            except Exception:
                pw, ph = PREVIEW_W, PREVIEW_H

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(pw / w, ph / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            y0 = (ph - nh) // 2
            x0 = (pw - nw) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
            image = Image.fromarray(canvas)
            self._photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self._photo, text="")

            gpu = self.engine.gpu_status
            self.chip_fps.configure(text=f"FPS {fps:.1f}")
            self.chip_gpu.configure(text=gpu if gpu else "GPU")
            self.chip_src.configure(text="Source")

            fps_color = SUCCESS if fps >= 12 else "#EAB308" if fps >= 6 else "#F87171"
            self.perf_fps_lbl.configure(text=f"FPS {fps:.1f}", text_color=fps_color)
            self.proc_gpu_lbl.configure(text=f"GPU {gpu}")
            if status:
                self.proc_sub_lbl.configure(text=status[:56])
            elif gpu == "CUDA":
                self.proc_sub_lbl.configure(text="NVIDIA")

            vcam_on = self.engine.virtual_cam_enabled
            vcam_txt = "On" if vcam_on else "Off"
            self.status_left.configure(
                text=(
                    f" Ready   |   Active face: {self._selected_name or '—'}   |   "
                    f"Source: Camera   |   Virtual camera: {vcam_txt}"
                )
            )
            self.status_right.configure(
                text=f"{PREVIEW_W} × {PREVIEW_H}   ·   {fps:.1f} FPS   ·   GPU ({gpu})"
            )

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
