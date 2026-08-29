"""Tkinter GUI: six-step calibration wizard + live monitoring + control.

Threading rules (strict):
* Tkinter widgets are touched ONLY from the main thread.
* Worker/process data arrives through queues; ``after(100, ...)`` polls and
  applies updates.  No worker ever calls a Tk method.
* The preview grabber thread only produces PIL images into a queue.

Calibration steps (requirement §5):
 1. select region (transparent overlay, live rectangle, WxH + DPI info)
 2. lock region (live preview, size validation, relative coordinates)
 3. colour anchor (click, 5x5 patch, median baseline, stability score,
    mandatory alive-calibration window, optional dead-sample capture)
 4. respawn click (click position, test-click with confirmation)
 5. horizon zone (slider 15-40%, live preview overlay)
 6. start/stop (F8 emergency, logs, all metrics, state machine)
"""

from __future__ import annotations

import platform
import queue as queue_mod
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Optional

from config import BotConfig
from logging_utils import get_logger, setup_logging
from states import BotState, InvalidTransitionError, StateMachine

LOGGER = get_logger("gui")

DPI_AWARENESS_SET = False


def set_dpi_awareness() -> float:
    """Ask Windows for per-monitor DPI awareness; return the Tk scale factor."""
    global DPI_AWARENESS_SET
    if platform.system() == "Windows" and not DPI_AWARENESS_SET:
        try:
            import ctypes

            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
            DPI_AWARENESS_SET = True
        except Exception as exc:
            LOGGER.warning("DPI awareness not set: %s", exc)
    return 1.0


# --------------------------------------------------------------------- #
# Step 1: region selection overlay
# --------------------------------------------------------------------- #
class RegionSelector:
    """Fullscreen semi-transparent overlay for click-and-drag selection."""

    def __init__(self, root: tk.Tk, on_accept, on_cancel) -> None:
        self.on_accept = on_accept
        self.on_cancel = on_cancel
        self.top = tk.Toplevel(root)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-topmost", True)
        try:
            self.top.attributes("-alpha", 0.30)
        except tk.TclError as exc:
            LOGGER.debug("overlay alpha unsupported: %s", exc)
        self.canvas = tk.Canvas(self.top, cursor="crosshair", bg="grey20")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.info = self.canvas.create_text(
            20, 20, anchor="nw", fill="white", font=("Consolas", 12),
            text="Drag to select the game region | Enter=accept | Esc=cancel | R=reset",
        )
        self.rect_id: Optional[int] = None
        self.start_xy: Optional[tuple[int, int]] = None
        self.cur_xy: Optional[tuple[int, int]] = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.top.bind("<Return>", self._accept)
        self.top.bind("<Escape>", self._cancel)
        self.top.bind("r", self._reset)
        self.top.bind("R", self._reset)
        self.top.focus_force()

    # ------------------------------------------------------------------ #
    def _press(self, ev) -> None:
        self.start_xy = (ev.x_root, ev.y_root)
        self._draw(ev.x_root, ev.y_root)

    def _drag(self, ev) -> None:
        if self.start_xy:
            self._draw(ev.x_root, ev.y_root)

    def _release(self, ev) -> None:
        if self.start_xy:
            self.cur_xy = (ev.x_root, ev.y_root)
            self._update_info()

    def _draw(self, x: int, y: int) -> None:
        self.cur_xy = (x, y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        if self.start_xy:
            x0, y0 = self.start_xy
            self.rect_id = self.canvas.create_rectangle(
                min(x0, x), min(y0, y), max(x0, x), max(y0, y),
                outline="#00ff88", width=2, dash=(6, 4),
            )
        self._update_info()

    def _update_info(self) -> None:
        if not (self.start_xy and self.cur_xy):
            return
        x0, y0 = self.start_xy
        x1, y1 = self.cur_xy
        w, h = abs(x1 - x0), abs(y1 - y0)
        self.canvas.itemconfigure(
            self.info,
            text=(f"Drag to select | Enter=accept | Esc=cancel | R=reset\n"
                  f"box: {min(x0,x1)},{min(y0,y1)}  size: {w}x{h}"),
        )

    def selection(self) -> Optional[tuple[int, int, int, int]]:
        if not (self.start_xy and self.cur_xy):
            return None
        x0, y0 = self.start_xy
        x1, y1 = self.cur_xy
        left, top = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 50 or h < 50:
            return None
        return left, top, w, h

    def _accept(self, _ev=None) -> None:
        sel = self.selection()
        self.top.destroy()
        if sel is None:
            self.on_cancel("selection too small (<50x50) — try again")
        else:
            self.on_accept(*sel)

    def _cancel(self, _ev=None) -> None:
        self.top.destroy()
        self.on_cancel("region selection cancelled")

    def _reset(self, _ev=None) -> None:
        self.start_xy = None
        self.cur_xy = None
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None


# --------------------------------------------------------------------- #
# Preview grabber (own thread, PIL images into a queue)
# --------------------------------------------------------------------- #
class PreviewGrabber(threading.Thread):
    """Low-rate region preview independent of the capture worker."""

    def __init__(self, monitor: dict[str, int], fps: float = 6.0) -> None:
        super().__init__(daemon=True, name="preview")
        self.monitor = monitor
        self.interval = 1.0 / fps
        self.queue: queue_mod.Queue = queue_mod.Queue(maxsize=4)
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                while not self.stop_event.is_set():
                    try:
                        raw = sct.grab(self.monitor)
                        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                        try:
                            self.queue.put_nowait(img)
                        except queue_mod.Full:
                            try:
                                self.queue.get_nowait()
                                self.queue.put_nowait(img)
                            except queue_mod.Empty as empty_exc:
                                LOGGER.debug("preview queue raced empty: %s",
                                             empty_exc)
                    except Exception as exc:
                        LOGGER.warning("preview grab failed: %s", exc)
                    self.stop_event.wait(self.interval)
        except Exception as exc:
            LOGGER.warning("preview unavailable: %s", exc)

    def stop(self) -> None:
        self.stop_event.set()


# --------------------------------------------------------------------- #
# Main GUI
# --------------------------------------------------------------------- #
class ControlGUI:
    POLL_MS = 150

    def __init__(self, root: tk.Tk, cfg: BotConfig, cli_args: Any = None) -> None:
        self.root = root
        self.cfg = cfg
        self.cli_args = cli_args
        self.sm = StateMachine(BotState.CALIBRATING)
        self.app: Optional[Any] = None  # BotApplication once running
        self.hotkey: Optional[Any] = None
        self.preview: Optional[PreviewGrabber] = None
        self.preview_img: Optional[Any] = None  # keep ref for PhotoImage
        self.anchor_frames: list[Any] = []
        self.anchor_t0 = 0.0
        self.dead_sample_frames: list[Any] = []
        self.log_lines: list[str] = []
        self._demo_recorder: Optional[Any] = None
        self._build()
        self.root.after(self.POLL_MS, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # Widget construction
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        self.root.title("Subway Surfers Research Bot — calibration & control")
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        # -- state row
        state_row = ttk.Frame(outer)
        state_row.pack(fill=tk.X, pady=(0, 6))
        self.state_var = tk.StringVar(value=self.sm.state.value)
        ttk.Label(state_row, text="State:", font=("TkDefaultFont", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(state_row, textvariable=self.state_var, foreground="#0a5",
                  font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=6)
        self.dpi_var = tk.StringVar(value="DPI scale: unknown")
        ttk.Label(state_row, textvariable=self.dpi_var).pack(side=tk.RIGHT)

        # -- steps notebook
        self.nb = ttk.Notebook(outer)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.step1 = ttk.Frame(self.nb, padding=10)
        self.step2 = ttk.Frame(self.nb, padding=10)
        self.step3 = ttk.Frame(self.nb, padding=10)
        self.step4 = ttk.Frame(self.nb, padding=10)
        self.step5 = ttk.Frame(self.nb, padding=10)
        self.step6 = ttk.Frame(self.nb, padding=10)
        for name, tab in (("1 Select region", self.step1),
                          ("2 Lock region", self.step2),
                          ("3 Colour anchor", self.step3),
                          ("4 Respawn click", self.step4),
                          ("5 Horizon", self.step5),
                          ("6 Train", self.step6)):
            self.nb.add(tab, text=name)

        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_step4()
        self._build_step5()
        self._build_step6()

        # -- error/status strip
        self.status_var = tk.StringVar(value="Calibration required (steps 1-5).")
        ttk.Label(outer, textvariable=self.status_var, foreground="#a00",
                  wraplength=900).pack(fill=tk.X, pady=(4, 0))

    # step 1 ----------------------------------------------------------- #
    def _build_step1(self) -> None:
        f = self.step1
        ttk.Label(f, text="Step 1 — drag a rectangle around the GAME area only.\n"
                          "Do not include browser UI; the bot keys arrow keys, so "
                          "the page must not scroll.", wraplength=700).pack(anchor="w")
        self.region_var = tk.StringVar(value="no region selected")
        ttk.Label(f, textvariable=self.region_var, font=("Consolas", 11)).pack(pady=6)
        ttk.Button(f, text="Select region (fullscreen overlay)",
                   command=self.start_region_select).pack(pady=4)

    def start_region_select(self) -> None:
        self._set_state(BotState.CALIBRATING)
        RegionSelector(self.root, self._region_accepted, self._region_cancelled)

    def _region_accepted(self, left: int, top: int, w: int, h: int) -> None:
        scale = set_dpi_awareness()
        self.root.update_idletasks()
        try:
            dpi_scale = self.root.winfo_fpixels("1i") / 96.0
        except tk.TclError:
            dpi_scale = 1.0
        sw = self.root.winfo_vrootwidth()
        sh = self.root.winfovrootheight()
        r = self.cfg.region
        r.left, r.top, r.width, r.height = int(left), int(top), int(w), int(h)
        r.screen_width, r.screen_height = int(sw), int(sh)
        r.dpi_scale = float(dpi_scale)
        r.frac_left = left / max(1, sw)
        r.frac_top = top / max(1, sh)
        r.frac_width = w / max(1, sw)
        r.frac_height = h / max(1, sh)
        self.region_var.set(f"region: {left},{top} {w}x{h}  "
                            f"(fractions {r.frac_left:.3f},{r.frac_top:.3f}, "
                            f"{r.frac_width:.3f},{r.frac_height:.3f})  "
                            f"DPI scale {dpi_scale:.2f}")
        self.dpi_var.set(f"DPI scale: {dpi_scale:.2f} ({sw}x{sh} virtual)")
        self.log(f"region selected: {left},{top} {w}x{h} dpi={dpi_scale:.2f}")
        self._set_state(BotState.CALIBRATING)
        self.nb.select(self.step2)
        self.status_var.set("Region selected — continue with step 2 (lock).")

    def _region_cancelled(self, reason: str) -> None:
        self.log(f"region selection: {reason}")
        self.status_var.set(f"Region selection: {reason}")

    # step 2 ----------------------------------------------------------- #
    def _build_step2(self) -> None:
        f = self.step2
        ttk.Label(f, text="Step 2 — lock the region. A live preview starts; "
                          "validate size and stability, then re-select if needed.",
                  wraplength=700).pack(anchor="w")
        self.lock_var = tk.StringVar(value="not locked")
        ttk.Label(f, textvariable=self.lock_var).pack(pady=4)
        self.preview_label = ttk.Label(f)
        self.preview_label.pack(pady=4)
        btns = ttk.Frame(f)
        btns.pack(pady=4)
        ttk.Button(btns, text="Start preview", command=self.start_preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Lock region", command=self.lock_region).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Re-select", command=self.start_region_select).pack(side=tk.LEFT, padx=4)

    def start_preview(self) -> None:
        if not self.cfg.region.is_set():
            self.status_var.set("Select a region first (step 1).")
            return
        self._stop_preview()
        self.preview = PreviewGrabber(self.cfg.region.to_monitor())
        self.preview.start()

    def _stop_preview(self) -> None:
        if self.preview is not None:
            self.preview.stop()
            self.preview = None

    def _update_preview(self) -> None:
        if self.preview is None:
            return
        try:
            img = self.preview.queue.get_nowait()
        except queue_mod.Empty:
            return
        from PIL import ImageTk

        img = img.copy()
        img.thumbnail((420, 420))
        self.preview_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_img)
        self._latest_preview = img  # for click/anchor sampling (fractions)

    _latest_preview: Optional[Any] = None
    _preview_full_size: Optional[Any] = None

    def lock_region(self) -> None:
        r = self.cfg.region
        if not r.is_set():
            self.status_var.set("Cannot lock: no region selected.")
            return
        if r.width < 240 or r.height < 320:
            self.status_var.set(
                f"Region too small ({r.width}x{r.height}); need >=240x320 so the "
                f"game scene is usable. Re-select."
            )
            return
        if r.width * r.height > self.cfg.capture.max_region_pixels:
            self.status_var.set("Region exceeds max_region_pixels; re-select smaller.")
            return
        self._set_state(BotState.READY)
        self.lock_var.set(f"locked: {r.width}x{r.height} at {r.left},{r.top}")
        self.log("region locked")
        self.nb.select(self.step3)
        self.status_var.set("Region locked — continue with step 3 (colour anchor).")

    # step 3 ----------------------------------------------------------- #
    def _build_step3(self) -> None:
        f = self.step3
        ttk.Label(f, text="Step 3 — click a STABLE UI element that is visible while "
                          "ALIVE and changes/covers on game-over (score bar corner, "
                          "coin counter...). The 5x5 patch is baselined with a median "
                          "and a stability score.", wraplength=700).pack(anchor="w")
        self.anchor_canvas = tk.Canvas(f, width=420, height=260, bg="grey15",
                                       cursor="crosshair")
        self.anchor_canvas.pack(pady=6)
        self.anchor_canvas.bind("<ButtonPress-1>", self._anchor_clicked)
        self.anchor_var = tk.StringVar(value="anchor: not set")
        ttk.Label(f, textvariable=self.anchor_var, font=("Consolas", 10)).pack()
        row = ttk.Frame(f)
        row.pack(pady=4)
        ttk.Button(row, text="Start preview", command=self.start_preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="Calibrate ALIVE (2s)", command=self._anchor_calibrate_alive).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="Capture DEAD sample", command=self._anchor_capture_dead).pack(side=tk.LEFT, padx=3)
        ttk.Label(f, text="Tip: run the alive calibration while the game is running; "
                          "capture the dead sample later on the game-over screen.",
                  wraplength=700).pack()

    def _anchor_clicked(self, ev) -> None:
        if self._latest_preview is None:
            self.anchor_var.set("anchor: start the preview first, then click")
            return
        # scale click -> region fraction
        w, h = self._latest_preview.size
        fx, fy = ev.x / max(1, w), ev.y / max(1, h)
        if not (0.01 <= fx <= 0.99 and 0.01 <= fy <= 0.99):
            return
        self.cfg.death.anchor_fx = float(fx)
        self.cfg.death.anchor_fy = float(fy)
        self.anchor_var.set(f"anchor position: fx={fx:.3f} fy={fy:.3f} "
                            f"(abs {int(fx * self.cfg.region.width)},"
                            f"{int(fy * self.cfg.region.height)}) — now calibrate ALIVE")
        self.log(f"anchor position set: fx={fx:.3f} fy={fy:.3f}")

    def _grab_anchor_patch(self) -> Optional[Any]:
        """Grab the current 5x5 patch at the anchor from a fresh capture."""
        if not (0 <= self.cfg.death.anchor_fx <= 1):
            return None
        try:
            import mss
            import numpy as np

            ax = int(self.cfg.death.anchor_fx * self.cfg.region.width)
            ay = int(self.cfg.death.anchor_fy * self.cfg.region.height)
            with mss.mss() as sct:
                raw = sct.grab(self.cfg.region.to_monitor())
                arr = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(
                    raw.height, raw.width, 4)[:, :, :3][:, :, ::-1]
            x0, y0 = max(0, ax - 2), max(0, ay - 2)
            return arr[y0 : y0 + 5, x0 : x0 + 5].copy()
        except Exception as exc:
            self.log(f"anchor grab failed: {exc}")
            return None

    def _anchor_calibrate_alive(self) -> None:
        if self.cfg.death.anchor_fx < 0:
            self.status_var.set("Click the anchor position on the preview first.")
            return
        self.anchor_frames = []
        self.anchor_t0 = time.monotonic()
        self.status_var.set("ALIVE calibration running — keep the game visible...")
        self.root.after(100, self._anchor_alive_tick)

    def _anchor_alive_tick(self) -> None:
        elapsed = time.monotonic() - self.anchor_t0
        patch = self._grab_anchor_patch()
        if patch is not None:
            self.anchor_frames.append(patch)
        if elapsed < self.cfg.anchor_calibration_s:
            self.root.after(100, self._anchor_alive_tick)
            return
        if len(self.anchor_frames) < 8:
            self.status_var.set("Too few anchor samples; is the preview running? Retry.")
            return
        from perception import patch_stability

        std, baseline = patch_stability(self.anchor_frames)
        self.cfg.death.anchor_baseline_rgb = baseline
        self.cfg.death.anchor_baseline_std = std
        ok = std <= 6.0
        self.anchor_var.set(
            f"anchor baseline RGB={baseline} stability(std)={std:.2f} "
            f"samples={len(self.anchor_frames)} "
            f"{'[ACCEPTED]' if ok else '[UNSTABLE — pick a calmer pixel]'}"
        )
        if ok:
            self.log(f"anchor calibrated: rgb={baseline} std={std:.2f}")
            self.status_var.set("Anchor calibrated — continue with step 4.")
            self._maybe_ready()
        else:
            self.status_var.set(
                f"Anchor unstable (std={std:.2f} > 6). Choose a pixel that does not "
                f"flicker, then recalibrate."
            )

    def _anchor_capture_dead(self) -> None:
        patch = self._grab_anchor_patch()
        if patch is None:
            self.status_var.set("Capture DEAD sample failed (no region/anchor).")
            return
        import numpy as np

        med = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
        self.cfg.death.dead_sample_rgb = med
        dist = float(np.linalg.norm(
            np.array(med) - np.array(self.cfg.death.anchor_baseline_rgb)))
        self.log(f"dead sample rgb={med} distance={dist:.1f}")
        self.anchor_var.set(
            self.anchor_var.get()
            + f" | DEAD sample rgb={med} (dist {dist:.1f})"
        )

    # step 4 ----------------------------------------------------------- #
    def _build_step4(self) -> None:
        f = self.step4
        ttk.Label(f, text="Step 4 — click the restart/respawn button on the "
                          "game-over screen. Nothing is clicked automatically; the "
                          "test button asks for confirmation first.", wraplength=700
                  ).pack(anchor="w")
        self.respawn_canvas = tk.Canvas(f, width=420, height=260, bg="grey15",
                                        cursor="crosshair")
        self.respawn_canvas.pack(pady=6)
        self.respawn_canvas.bind("<ButtonPress-1>", self._respawn_clicked)
        self.respawn_var = tk.StringVar(value="respawn point: not set")
        ttk.Label(f, textvariable=self.respawn_var).pack()
        row = ttk.Frame(f)
        row.pack(pady=4)
        ttk.Button(row, text="Start preview", command=self.start_preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="Test click (asks first)", command=self._respawn_test).pack(side=tk.LEFT, padx=3)

    def _respawn_clicked(self, ev) -> None:
        if self._latest_preview is None:
            self.respawn_var.set("respawn: start the preview first, then click")
            return
        w, h = self._latest_preview.size
        self.cfg.input.respawn_fx = ev.x / max(1, w)
        self.cfg.input.respawn_fy = ev.y / max(1, h)
        self.respawn_var.set(
            f"respawn point: fx={self.cfg.input.respawn_fx:.3f} "
            f"fy={self.cfg.input.respawn_fy:.3f} (abs "
            f"{self.cfg.region.left + int(self.cfg.input.respawn_fx * self.cfg.region.width)},"
            f"{self.cfg.region.top + int(self.cfg.input.respawn_fy * self.cfg.region.height)})"
        )
        self._maybe_ready()

    def _respawn_test(self) -> None:
        if not self.cfg.input.respawn_set():
            self.status_var.set("Set the respawn point first.")
            return
        x = self.cfg.region.left + int(self.cfg.input.respawn_fx * self.cfg.region.width)
        y = self.cfg.region.top + int(self.cfg.input.respawn_fy * self.cfg.region.height)
        if not messagebox.askyesno(
            "Confirm test click",
            f"Click at absolute screen position ({x}, {y}) once?\n"
            "Move the mouse to a screen corner to abort (pyautogui failsafe).",
        ):
            return
        from input_controller import InputController

        ctl = InputController(self.cfg.input, backend="auto")
        try:
            ok = ctl.click(x, y)
            self.log(f"test click at ({x},{y}): {'sent' if ok else 'BLOCKED (focus?)'}")
        finally:
            ctl.dispose()

    # step 5 ----------------------------------------------------------- #
    def _build_step5(self) -> None:
        f = self.step5
        ttk.Label(f, text="Step 5 — horizon zone: the top slice of the region used "
                          "for fast hazard detection. Default 25%.", wraplength=700
                  ).pack(anchor="w")
        self.horizon_var = tk.StringVar(value=f"horizon: {self.cfg.perception.horizon_frac*100:.0f}%")
        self.horizon_scale = ttk.Scale(
            f, from_=15, to=40, value=self.cfg.perception.horizon_frac * 100,
            command=self._horizon_changed)
        self.horizon_scale.pack(fill=tk.X, pady=6)
        ttk.Label(f, textvariable=self.horizon_var).pack()
        ttk.Label(f, text="The preview overlay shows the split line once the "
                          "preview runs.", wraplength=700).pack()

    def _horizon_changed(self, value: str) -> None:
        pct = float(value)
        self.cfg.perception.horizon_frac = round(pct, 1) / 100.0
        self.horizon_var.set(f"horizon: {self.cfg.perception.horizon_frac*100:.1f}% "
                             f"({int(self.cfg.region.height * self.cfg.perception.horizon_frac)} px)")

    # step 6 ----------------------------------------------------------- #
    def _build_step6(self) -> None:
        f = self.step6
        ctrl = ttk.Frame(f)
        ctrl.pack(fill=tk.X)
        self.btn_start = ttk.Button(ctrl, text="▶ Start training", command=self.start_training,
                                    state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=3)
        self.btn_pause = ttk.Button(ctrl, text="⏸ Pause", command=self.pause_training,
                                    state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=3)
        self.btn_stop = ttk.Button(ctrl, text="⏹ Stop", command=self.stop_training,
                                   state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        self.btn_demo = ttk.Button(ctrl, text="● Record demo (F9 stop)",
                                   command=self.toggle_demo_recording, state=tk.DISABLED)
        self.btn_demo.pack(side=tk.LEFT, padx=3)
        self.btn_pretrain = ttk.Button(ctrl, text="BC pretrain", command=self.run_pretrain,
                                       state=tk.DISABLED)
        self.btn_pretrain.pack(side=tk.LEFT, padx=3)
        self.btn_emergency = ttk.Button(ctrl, text="⛔ EMERGENCY (F8)",
                                        command=self.emergency_stop)
        self.btn_emergency.pack(side=tk.RIGHT, padx=3)

        metrics = ttk.Labelframe(f, text="Live metrics", padding=6)
        metrics.pack(fill=tk.BOTH, expand=True, pady=6)
        self.metric_vars: dict[str, tk.StringVar] = {}
        cols = [
            ("state", "Process state"), ("survival_s", "Survival (s)"),
            ("episode_id", "Episode"), ("score", "Score (not detectable: n/a)"),
            ("fps", "Capture FPS"), ("eff_fps", "Effective FPS"),
            ("dropped", "Dropped frames"), ("infer_p95", "Inference p95 (ms)"),
            ("action_p95", "Action p95 (ms)"), ("epsilon", "Epsilon"),
            ("avg_reward", "Avg reward (episode)"), ("td_loss", "TD loss"),
            ("q_mean", "Mean Q"), ("buffer", "Replay buffer"),
            ("learner_updates", "Learner updates"), ("cpu", "CPU %"),
            ("ram", "RAM (GB)"), ("profile", "Model profile"),
        ]
        for i, (key, label) in enumerate(cols):
            r, c = divmod(i, 3)
            ttk.Label(metrics, text=label + ":").grid(row=r, column=c * 2,
                                                      sticky="e", padx=4, pady=2)
            var = tk.StringVar(value="—")
            ttk.Label(metrics, textvariable=var, font=("Consolas", 10)
                      ).grid(row=r, column=c * 2 + 1, sticky="w", padx=4, pady=2)
            self.metric_vars[key] = var

        logframe = ttk.Labelframe(f, text="Training log", padding=4)
        logframe.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(logframe, height=10, width=110, font=("Consolas", 9))
        scroll = ttk.Scrollbar(logframe, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set, state=tk.DISABLED)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------ #
    # State + readiness
    # ------------------------------------------------------------------ #
    def _set_state(self, state: BotState) -> None:
        try:
            self.sm.transition(state)
        except InvalidTransitionError as exc:
            LOGGER.warning("state transition refused: %s", exc)
            return
        self.state_var.set(state.value)
        self.metric_vars["state"].set(state.value)

    def _maybe_ready(self) -> None:
        if (
            self.cfg.region.is_set()
            and self.cfg.death.anchor_set()
            and self.cfg.input.respawn_set()
        ):
            if self.sm.can(BotState.READY):
                self._set_state(BotState.READY)
            for btn in (self.btn_start, self.btn_demo, self.btn_pretrain):
                btn.configure(state=tk.NORMAL)

    # ------------------------------------------------------------------ #
    # Training control
    # ------------------------------------------------------------------ #
    def start_training(self) -> None:
        if self.app is not None:
            self.status_var.set("Already running.")
            return
        warnings = self.cfg.validate()
        for w in warnings:
            self.log(f"config warning: {w}")
        from app import BotApplication

        try:
            self.app = BotApplication(self.cfg)
        except Exception as exc:
            self._set_state(BotState.ERROR)
            self.status_var.set(f"Failed to start: {exc}")
            self.log(f"start failed: {exc}")
            return
        self.app.start(with_learner=True)
        from safety_watchdog import EmergencyHotkey

        self.hotkey = EmergencyHotkey(self.cfg.emergency_hotkey, self.app.events)
        self._set_state(BotState.RUNNING)
        self.btn_pause.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.NORMAL)
        self.status_var.set("Training started. F8 = emergency stop.")
        self.log("training started")

    def pause_training(self) -> None:
        if self.app is None:
            return
        if self.sm.state is BotState.PAUSED:
            self.app.resume()
            self._set_state(BotState.RUNNING)
            self.log("resumed")
        else:
            self.app.pause()
            self._set_state(BotState.PAUSED)
            self.log("paused (keys released)")

    def stop_training(self) -> None:
        if self.app is None:
            return
        self._set_state(BotState.STOPPING)
        self.status_var.set("Stopping (saving model + buffer)...")
        self.root.update_idletasks()
        if self.hotkey is not None:
            self.hotkey.stop()
            self.hotkey = None
        self.app.shutdown()
        self.app = None
        self._set_state(BotState.STOPPED)
        self.status_var.set("Stopped. Model, buffer and logs saved.")
        for btn in (self.btn_pause, self.btn_stop):
            btn.configure(state=tk.DISABLED)
        self.btn_start.configure(state=tk.NORMAL)
        self.log("stopped cleanly")

    def emergency_stop(self) -> None:
        self.log("EMERGENCY STOP (GUI)")
        if self.app is not None:
            self.app.emergency_stop()
            self.stop_training()
        else:
            self._set_state(BotState.STOPPED)

    def toggle_demo_recording(self) -> None:
        if self.sm.state is BotState.RECORDING_DEMO:
            self._stop_demo_recording()
            return
        if not self.cfg.region.is_set():
            self.status_var.set("Select a region before recording demos.")
            return
        self._set_state(BotState.RECORDING_DEMO)
        from demonstration_recorder import DemoRecorder
        from ipc import SharedFrameRing

        if self.app is None:
            from app import BotApplication

            self.app = BotApplication(self.cfg)
            self.app.start(with_learner=False)
        # recorder reads the shared ring (latest frame wins)
        ring: Optional[SharedFrameRing] = self.app.ring if self.app else None

        class _RingReader:
            def __init__(self, ring) -> None:
                self.ring = ring
                self._last = -1

            def __call__(self):
                if self.ring is None:
                    return None
                fr = self.ring.read_latest()
                if fr is None or fr.frame_id <= self._last:
                    return None
                self._last = fr.frame_id
                return fr

        self._demo_recorder = DemoRecorder(self.cfg, self.cfg.paths.demos_dir,
                                           _RingReader(ring))
        self._demo_recorder.start()
        self.btn_demo.configure(text="■ Stop demo (F9)")
        self.status_var.set("RECORDING demonstration — play! F9 / button stops.")

    def _stop_demo_recording(self) -> None:
        if self._demo_recorder is None:
            return
        path = self._demo_recorder.stop(done=True)
        self._demo_recorder.dispose()
        self._demo_recorder = None
        self.btn_demo.configure(text="● Record demo (F9 stop)")
        self._set_state(BotState.READY)
        if path:
            self.status_var.set(f"Demo saved: {path}")
            self.log(f"demo saved: {path}")
        else:
            self.status_var.set("Demo recording stopped (nothing saved).")

    def run_pretrain(self) -> None:
        if self.app is None:
            self.status_var.set("Start training first (the learner owns pretraining).")
            return
        self._set_state(BotState.PRETRAINING)
        self.app.command("pretrain", demos_dir=str(self.cfg.paths.demos_dir))

    # ------------------------------------------------------------------ #
    # Polling loop (the ONLY place worker data reaches Tk)
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        try:
            self._update_preview()
            self._poll_metrics()
            self._check_workers()
        except Exception as exc:
            self.log(f"gui tick error: {exc}")
        self.root.after(self.POLL_MS, self._tick)

    def _poll_metrics(self) -> None:
        if self.app is None:
            return
        for msg in self.app.drain_metrics(128):
            kind = msg.get("type")
            if kind == "metrics" and msg.get("src") == "capture":
                self.metric_vars["fps"].set(f"{msg['data'].get('capture_fps', 0):.1f}")
            elif kind == "metrics" and msg.get("src") == "learner":
                d = msg["data"]
                self.metric_vars["td_loss"].set(f"{d.get('td_loss', 0):.4f}")
                self.metric_vars["q_mean"].set(f"{d.get('q_mean', 0):.2f}")
                self.metric_vars["buffer"].set(
                    f"{int(d.get('buffer_size', 0))} ({d.get('buffer_mb', 0):.0f} MB)")
                self.metric_vars["learner_updates"].set(str(int(d.get("learner_updates", 0))))
            elif kind == "actor_stats":
                d = msg["data"]
                self.metric_vars["eff_fps"].set(f"{d.get('fps', 0):.1f}")
                self.metric_vars["dropped"].set(str(d.get("dropped", 0)))
                inf = d.get("inference_ms", {})
                act = d.get("action_ms", {})
                self.metric_vars["infer_p95"].set(f"{inf.get('p95', 0):.1f}")
                self.metric_vars["action_p95"].set(f"{act.get('p95', 0):.1f}")
                self.metric_vars["epsilon"].set(f"{d.get('epsilon', 0):.3f}")
                self.metric_vars["survival_s"].set(f"{d.get('survival_s', 0):.1f}")
                self.metric_vars["avg_reward"].set(f"{d.get('episode_reward', 0):.2f}")
                self.metric_vars["profile"].set(str(d.get("profile", "—")))
                if d.get("held_keys"):
                    self.log(f"WARNING held keys: {d['held_keys']}")
            elif kind == "log":
                self.log(f"[{msg.get('src')}] {msg.get('msg')}")
            elif kind == "error":
                self.log(f"ERROR in {msg.get('src')}: {msg.get('error')}")
                self._show_error(msg)
            elif kind == "watchdog":
                self.log(f"WATCHDOG [{msg.get('tag')}]: {msg.get('msg')}")
                if self.sm.state is BotState.RUNNING:
                    self._set_state(BotState.PAUSED)
                self.status_var.set(f"Watchdog: {msg.get('msg')}")
            elif kind == "episode_end":
                d = msg.get("data", {})
                self.log(f"episode {d.get('episode_id')} ended ({msg.get('reason')}: "
                         f"{d.get('survival_s', 0):.1f}s, reward {d.get('total_reward', 0):.1f})")
                self.metric_vars["episode_id"].set(str(d.get("episode_id", "—")))
        from metrics import system_usage

        u = system_usage()
        self.metric_vars["cpu"].set(
            f"{u['cpu_system']:.0f}% sys / {u['cpu_process']:.0f}% bot")
        self.metric_vars["ram"].set(
            f"{u['ram_process_gb']:.2f} bot / {u['ram_system_gb']:.1f} sys")
        snap = self.app.counters.snapshot()
        self.metric_vars["episode_id"].set(str(snap["episode_id"]))
        self.metric_vars["epsilon"].set(f"{snap['epsilon']:.3f}")
        if self.sm.state is BotState.RUNNING and snap.get("dead"):
            self._set_state(BotState.DEAD)

    def _show_error(self, msg: dict) -> None:
        self.status_var.set(f"ERROR in {msg.get('src')}: {msg.get('error')}")
        if self.sm.state is not BotState.ERROR:
            try:
                self._set_state(BotState.ERROR)
            except InvalidTransitionError as exc:
                LOGGER.warning("error-state transition refused: %s", exc)

    def _check_workers(self) -> None:
        if self.app is None:
            return
        alive = self.app.workers_alive()
        if not alive["learner"] and self.sm.state is BotState.RUNNING:
            self.log("learner process died — GUI stays alive; training paused")
            self.app.pause()
            self._set_state(BotState.PAUSED)
            self.status_var.set("Learner crashed — gameplay paused. Check logs.")
        if not alive["actor"]:
            self.log("actor process died — stop and restart required")
            self._set_state(BotState.ERROR)
            self.status_var.set("Actor crashed — press Stop, then Start again.")

    # ------------------------------------------------------------------ #
    def log(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_lines.append(f"{stamp} {line}")
        self.log_lines = self.log_lines[-500:]
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert("end", f"{stamp} {line}\n")
        self.log_text.see("end")
        self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------ #
    def _on_close(self) -> None:
        try:
            if self._demo_recorder is not None:
                self._demo_recorder.stop(done=False)
                self._demo_recorder.dispose()
            self._stop_preview()
            if self.app is not None:
                self._set_state(BotState.STOPPING)
                if self.hotkey is not None:
                    self.hotkey.stop()
                self.app.shutdown()
        finally:
            self.root.destroy()


def run_gui(cli_args: Any = None) -> int:
    """GUI entry point (from app.py)."""
    setup_logging("gui", "logs")
    if cli_args is not None and getattr(cli_args, "config", ""):
        cfg = BotConfig.load(cli_args.config)
    else:
        cfg = BotConfig()
    root = tk.Tk()
    try:
        root.state("zoomed")
    except tk.TclError:
        root.geometry("1080x860")
    gui = ControlGUI(root, cfg, cli_args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_gui())
