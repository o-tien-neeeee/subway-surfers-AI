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

from version import APP_VERSION, banner
from tkinter import messagebox, ttk
from typing import Any, Optional

from config import BotConfig
from logging_utils import get_logger, setup_logging
from states import BotState, InvalidTransitionError, StateMachine

LOGGER = get_logger("gui")

DPI_AWARENESS_SET = False


def set_dpi_awareness() -> bool:
    """Ask Windows for per-monitor DPI awareness.

    DEEP-FIX: three defects fixed together.

    1. The signature promised "the Tk scale factor" but the body returned a
       hardcoded ``1.0`` on every path, including the failure path.  A caller
       that trusted it would silently treat a 150%-scaled desktop as 100%.
       It now returns whether awareness was actually established -- a value
       that is true by construction.

    2. Windows only honours ``SetProcessDpiAwareness`` *before the first
       window is created*.  It was being called from ``_region_accepted``,
       i.e. long after ``tk.Tk()`` exists, where it fails with
       ``E_ACCESSDENIED`` (0x80070005).  The bare ``except`` swallowed that,
       so on a scaled Windows desktop the process stayed DPI-unaware: Tk
       reported *virtualised* screen coordinates while mss captured *physical*
       pixels, and the recorded region fractions pointed at the wrong part of
       the screen.  ``run_gui`` now calls this before ``tk.Tk()``.

    3. The return code was never inspected.  ``SetProcessDpiAwareness``
       reports failure in its HRESULT return value, not by raising, so the
       old code marked ``DPI_AWARENESS_SET = True`` even when Windows had
       refused the request.
    """
    global DPI_AWARENESS_SET
    if platform.system() != "Windows":
        return True                      # X11/Wayland have no process DPI mode
    if DPI_AWARENESS_SET:
        return True
    try:
        import ctypes

        try:
            # S_OK == 0.  E_ACCESSDENIED (0x80070005) means a window already
            # exists, which is the "called too late" signature.
            hr = ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
            if hr == 0:
                DPI_AWARENESS_SET = True
                return True
            if hr == -2147024891:        # E_ACCESSDENIED
                LOGGER.warning(
                    "DPI awareness refused (E_ACCESSDENIED): this must be set "
                    "before the first window is created. Screen coordinates "
                    "may be virtualised, so the selected region may not match "
                    "what the capture sees."
                )
                return False
            LOGGER.warning("SetProcessDpiAwareness returned HRESULT %#x", hr)
            return False
        except Exception:
            # Older Windows without shcore: fall back to the user32 call,
            # which returns non-zero on success.
            if ctypes.windll.user32.SetProcessDPIAware():
                DPI_AWARENESS_SET = True
                return True
            LOGGER.warning("SetProcessDPIAware() returned 0 (failure)")
            return False
    except Exception as exc:
        LOGGER.warning("DPI awareness not set: %s", exc)
        return False


# --------------------------------------------------------------------- #
# Step 1: region selection overlay
# --------------------------------------------------------------------- #
HELP_AZ = """HƯỚNG DẪN SỬ DỤNG TỪ A–Z
────────────────────────────
ĐIỀU KIỆN TRƯỚC (máy Windows thật):
 • Mở Chrome vào poki.com/en/g/subway-surfers, game ĐANG chạy, KHÔNG thu nhỏ.
 • Đặt cửa sổ điều khiển này sang MÀN HÌNH/ vị trí KHÔNG che vùng game.
 • Nếu preview ĐEN: Chrome → Cài đặt → tắt "Use hardware acceleration", rồi chọn lại vùng.

BƯỚC 1 – Chọn vùng: bấm "Chọn vùng", kéo khung CHỈ quanh vùng game (không dính
   thanh trình duyệt). Enter = chọn, Esc = huỷ, R = làm lại.
BƯỚC 2 – Khoá vùng: "Bật preview" để thấy game; ổn thì "Khoá vùng".
BƯỚC 3 – Neo màu (sống/chết): "Bật preview"; bấm vào một điểm UI ổn định luôn
   hiện khi sống (góc thanh điểm). "Hiệu chuẩn SỐNG (2s)" khi game đang chạy.
   Vòng ĐỎ trên preview là vị trí neo.
BƯỚC 4 – Nút hồi sinh: "Bật preview"; bấm đúng nút "chơi lại" trên màn thua.
   Vòng XANH là điểm hồi sinh. "Test click (hỏi trước)" để thử (không cần focus).
BƯỚC 5 – Chân trời: kéo thanh trượt (mặc định 25%). Đường XANH NGANG trên preview
   là ranh giới chân trời / mặt đất — phải thấy nó.
BƯỚC 6 – Chạy: "▶ Bắt đầu train". QUAN TRỌNG: ngay sau khi bấm, hãy CLICK VÀO CỬA
   SỔ CHROME để game có focus — bot chỉ chơi khi Chrome focus (để không bấm nhầm
   khi bạn đang gõ). F8 = dừng khẩn cấp. F9 = dừng quay demo.

QUAY DEMO (dạy bot bằng cách BẠN chơi):
 • Mục đích: bot HỌC BẮT CHƯỚC bạn. Bạn chơi, bot CHỈ quay lại — nó KHÔNG
   tự bấm phím.  Mỗi khung hình 84x84 được lưu kèm hành động bạn nhấn.
 • Cách dùng: (1) đã Chọn + Khoá vùng (bước 1-2); (2) bấm "● Quay demo (F9
   dừng)"; (3) CHƠI bằng phím mũi tên (trái / phải / nhảy / lướt) — ô
   "demo frames: N" đếm số khung đang ghi; (4) bấm F9 (hoặc nút) để dừng & lưu.
 • KHI ĐANG QUAY, preview hiện lớp phủ trực quan: banner ● REC đỏ + số khung +
   thời gian, ô "HĐ:" cho HÀNH ĐỘNG đang nhận (để bạn thấy phím có ăn không),
   và ảnh vùng 84×84 thực sự đang được ghi (viền vàng).  Nếu "HĐ:" kẹt ở một
   hành động dù bạn không bấm → phím bị kẹt, đã tự xoá sau 1.5s.
 • File lưu: demos/episode_YYYYmmdd_HHMMSS.npz (ghi atomic, không ghi đè).
 • Dòng lệnh: python app.py --record-demo
 • LƯU Ý QUAN TRỌNG: cần bàn phím thật + pynput.  Nếu không có hook bàn phím,
   log báo "UNAVAILABLE — NOOP only" và episode sẽ TOÀN NOOP = VÔ DỤNG cho BC.
   Bạn phải chơi thật, nhấn phím thật.
 • Kiểm tra demo đã quay: python app.py --validate-demos demos  →  in
   "X/Y episodes valid".  Episode INVALID nếu: rỗng, khung != 84x84, hành động
   ngoài 0..4, timestamp không tăng dần, hoặc cờ done không nằm ở bước cuối.

TIỀN-HUẤN LUYỆN BC (học bắt chước TRƯỚC khi RL):
 • Mục đích: huấn luyện policy bắt chước các demo ở trên TRƯỚC, để bot không
   khởi đầu hoàn toàn ngẫu nhiên.  (BC = behaviour cloning.)
 • Điều kiện: cần >= 2 episode hợp lệ (bc.min_episodes).  Thiếu thì log báo
   "BC skipped — online learning with warm-up will be used instead" — vẫn train
   RL bình thường, chỉ là không có bước bắt chước.
 • Cách dùng (GUI): phải BẬT train trước (để learner chạy), rồi bấm nút
   "Tiền-huấn luyện (BC)".
 • Cách dùng (dòng lệnh, KHÔNG cần game): python app.py --pretrain demos
 • Nó làm gì: kiểm tra demos → train 8 epoch (batch 64, Adam lr 1e-3) → mỗi
   epoch log "BC epoch K: loss / train_acc / val_acc" → lưu checkpoint best
   (theo val_acc) và latest.
 • BC KHÔNG tự chạy khi bấm "Bắt đầu train" — bạn phải chủ động bấm nút BC
   hoặc chạy lệnh --pretrain.

GIẢI QUYẾT RẮC RỐI:
 • "test click ... BLOCKED (focus?)": bản cũ bắt Chrome focus cả khi test — đã sửa,
   nút Test giờ bỏ qua cổng focus (vẫn hỏi xác nhận).
 • "Browser focus lost — pausing": bình thường khi bạn đang ở cửa sổ này; hãy click
   vào Chrome để bot chơi tiếp.
 • Chết ngay 0.2s khi vừa chạy: vùng chọn đang bị che hoặc game chưa ở trạng thái
   sống — kiểm tra cửa sổ không che vùng, và game đang chạy trước khi Bắt đầu.
 • Preview đen: tắt hardware acceleration của Chrome (xem trên).
 • "AI không tiến bộ dù đã train lâu": gần như luôn do bot CHẾT NGAY mỗi
   episode (vùng bị che / màn thua / neo sai) nên không có tín hiệu sống sót
   để học.  Log sẽ báo "BOT ĐANG CHẾT NGAY LIÊN TIẾP".  Khắc phục: quay lại
   bước 1-3, đảm bảo game đang SỐNG và vùng không bị che, rồi train lại.  Chạy
   preflight (tự động khi bấm Start) cũng cảnh báo điều này.
 • Muốn bot học NHANH hơn: quay vài demo rồi chạy Tiền-huấn luyện BC (ở trên)
   để bot bắt chước bạn trước, thay vì mò ngẫu nhiên.
"""


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
            text="Kéo chuột khoanh vùng GAME | Enter=chọn | Esc=huỷ | R=làm lại",
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
        # DEEP-FIX: F9 was advertised everywhere ("F9 dừng") but never wired, so
        # the user could not stop a recording from the game window.  A global
        # pynput listener sets this event; the Tk loop (the only thread allowed
        # to touch widgets) performs the actual stop.
        self._demo_hotkey: Optional[Any] = None
        self._demo_stop_req = threading.Event()
        self._demo_t0 = 0.0
        self._demo_started_app = False
        self._demo_last_log_t = 0.0
        self._build()
        self.root.after(self.POLL_MS, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # Widget construction
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        # DEEP-FIX: surface the build version so a user can tell at a
        # glance whether they are running the fixed binary.
        self.root.title(f"{banner()} — calibration & control")
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        # -- state row
        state_row = ttk.Frame(outer)
        state_row.pack(fill=tk.X, pady=(0, 6))
        self.state_var = tk.StringVar(value=self.sm.state.value)
        ttk.Label(state_row, text="Trạng thái:", font=("TkDefaultFont", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(state_row, textvariable=self.state_var, foreground="#0a5",
                  font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=6)
        self.dpi_var = tk.StringVar(value="Tỉ lệ DPI: chưa rõ")
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
        self.step_help = ttk.Frame(self.nb, padding=10)
        for name, tab in (("1 Chọn vùng", self.step1),
                          ("2 Khoá vùng", self.step2),
                          ("3 Neo màu (sống/chết)", self.step3),
                          ("4 Nút hồi sinh", self.step4),
                          ("5 Vùng chân trời", self.step5),
                          ("6 Chạy & số liệu", self.step6),
                          ("❓ Hướng dẫn", self.step_help)):
            self.nb.add(tab, text=name)

        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_step4()
        self._build_step5()
        self._build_step6()
        self._build_help()

        # -- error/status strip
        self.status_var = tk.StringVar(value="Calibration required (steps 1-5).")
        ttk.Label(outer, textvariable=self.status_var, foreground="#a00",
                  wraplength=900).pack(fill=tk.X, pady=(4, 0))

    # help ------------------------------------------------------------- #
    def _build_help(self) -> None:
        f = self.step_help
        txt = tk.Text(f, wrap="word", font=("Segoe UI", 10), background="#101418",
                      foreground="#e6e9ee", insertbackground="#e6e9ee")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("end", HELP_AZ)
        txt.configure(state=tk.DISABLED)

    # step 1 ----------------------------------------------------------- #
    def _build_step1(self) -> None:
        f = self.step1
        ttk.Label(f, text="Bước 1 — kéo một hình chữ nhật CHỈ quanh vùng GAME.\n"
                          "Đừng gồm thanh trình duyệt; bot bấm phím mũi tên nên "
                          "trang không được cuộn.", wraplength=700).pack(anchor="w")
        self.region_var = tk.StringVar(value="no region selected")
        ttk.Label(f, textvariable=self.region_var, font=("Consolas", 11)).pack(pady=6)
        ttk.Button(f, text="Chọn vùng (overlay toàn màn hình)",
                   command=self.start_region_select).pack(pady=4)

    def start_region_select(self) -> None:
        self._set_state(BotState.CALIBRATING)
        RegionSelector(self.root, self._region_accepted, self._region_cancelled)

    def _region_accepted(self, left: int, top: int, w: int, h: int) -> None:
        # DEEP-FIX: `scale = set_dpi_awareness()` assigned a value nothing
        # ever read (the function always returned 1.0), which hid the fact
        # that calling it here is far too late to matter.  run_gui() now sets
        # awareness before tk.Tk(); this call is kept only as an idempotent
        # safety net for anyone entering this path directly.
        dpi_aware = set_dpi_awareness()
        self.root.update_idletasks()
        try:
            dpi_scale = self.root.winfo_fpixels("1i") / 96.0
        except tk.TclError:
            dpi_scale = 1.0
        # DEEP-FIX: make the virtualised-coordinate mismatch visible instead
        # of silently recording region fractions in the wrong space.
        if not dpi_aware and abs(dpi_scale - 1.0) > 0.01:
            self.log(f"WARNING: desktop scaling is {dpi_scale:.2f}x but the "
                     "process is DPI-unaware; the saved region may not match "
                     "the captured pixels.")
            LOGGER.warning("DPI mismatch: Tk scale %.3f, process DPI-unaware",
                           dpi_scale)
        sw = self.root.winfo_vrootwidth()
        # DEEP-FIX: this was `winfovrootheight()` -- the underscores were
        # missing, so accepting a region selection raised AttributeError
        # from the Tkinter callback and calibration could never complete.
        # The neighbouring line spelled winfo_vrootwidth() correctly, which
        # is what made it read plausibly.  Reported from a real Windows
        # run; it fails identically on every platform because no test
        # builds a Tk root (there is no display in CI).
        sh = self.root.winfo_vrootheight()
        r = self.cfg.region
        r.left, r.top, r.width, r.height = int(left), int(top), int(w), int(h)
        r.screen_width, r.screen_height = int(sw), int(sh)
        r.dpi_scale = float(dpi_scale)
        r.frac_left = left / max(1, sw)
        r.frac_top = top / max(1, sh)
        r.frac_width = w / max(1, sw)
        r.frac_height = h / max(1, sh)
        self.region_var.set(f"vùng: {left},{top} {w}x{h}  "
                            f"(fractions {r.frac_left:.3f},{r.frac_top:.3f}, "
                            f"{r.frac_width:.3f},{r.frac_height:.3f})  "
                            f"tỉ lệ DPI {dpi_scale:.2f}")
        self.dpi_var.set(f"Tỉ lệ DPI: {dpi_scale:.2f} ({sw}x{sh} ảo)")
        self.log(f"region selected: {left},{top} {w}x{h} dpi={dpi_scale:.2f}")
        self._set_state(BotState.CALIBRATING)
        self.nb.select(self.step2)
        self.status_var.set("Đã chọn vùng — sang bước 2 (khoá).")

    def _region_cancelled(self, reason: str) -> None:
        self.log(f"region selection: {reason}")
        self.status_var.set(f"Region selection: {reason}")

    # step 2 ----------------------------------------------------------- #
    def _build_step2(self) -> None:
        f = self.step2
        ttk.Label(f, text="Bước 2 — khoá vùng. Preview trực tiếp chạy; "
                          "kiểm tra kích thước/ổn định, chọn lại nếu cần.",
                  wraplength=700).pack(anchor="w")
        self.lock_var = tk.StringVar(value="not locked")
        ttk.Label(f, textvariable=self.lock_var).pack(pady=4)
        self.preview_label = ttk.Label(f)
        self.preview_label.pack(pady=4)
        btns = ttk.Frame(f)
        btns.pack(pady=4)
        ttk.Button(btns, text="Bật preview", command=self.start_preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Khoá vùng", command=self.lock_region).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Chọn lại", command=self.start_region_select).pack(side=tk.LEFT, padx=4)

    def start_preview(self) -> None:
        if not self.cfg.region.is_set():
            self.status_var.set("Chọn vùng trước (bước 1).")
            return
        self._stop_preview()
        self.preview = PreviewGrabber(self.cfg.region.to_monitor())
        self.preview.start()

    def _stop_preview(self) -> None:
        if self.preview is not None:
            self.preview.stop()
            self.preview = None

    #: action id -> (nhãn, màu) cho lớp phủ quay demo
    _ACTION_STYLE = {
        0: ("KHÔNG", (120, 120, 120)),
        1: ("← TRÁI", (52, 152, 219)),
        2: ("→ PHẢI", (155, 89, 182)),
        3: ("↑ NHẢY", (46, 204, 113)),
        4: ("↓ LƯỚT", (230, 126, 34)),
    }

    def _draw_rec_overlay(self, img, dr) -> None:
        """Draw the live data-collection HUD onto the preview while recording."""
        from PIL import Image as _Image, ImageFont as _ImageFont
        rec = self._demo_recorder
        try:
            action = int(rec.current_action())
            frames = int(rec.frame_count())
        except Exception:
            return
        elapsed = max(0.0, time.monotonic() - self._demo_t0)
        try:
            font = _ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font = _ImageFont.load_default()
        # --- REC banner (top-left) ---
        label, color = self._ACTION_STYLE.get(action, ("?", (200, 0, 0)))
        banner = f"● REC  {frames} khung  {elapsed:4.1f}s"
        dr.rectangle([0, 0, 210, 22], fill=(180, 20, 20))
        dr.text((6, 4), banner, fill=(255, 255, 255), font=font)
        # --- current action chip (below banner) ---
        dr.rectangle([0, 24, 150, 48], fill=color)
        dr.text((8, 28), f"HĐ: {label}", fill=(255, 255, 255), font=font)
        # --- the 84x84 ground zone actually being saved (bottom-right) ---
        zone = rec.last_zone()
        if zone is not None:
            try:
                import numpy as _np
                zi = _Image.fromarray(_np.asarray(zone, dtype=_np.uint8)).convert("L")
                zi = zi.resize((84, 84))
                w, h = img.size
                img.paste(zi, (w - 92, h - 108))
                dr.rectangle([w - 93, h - 109, w - 7, h - 23],
                             outline=(255, 255, 0), width=2)
                dr.text((w - 92, h - 124), "vùng 84×84 đang ghi",
                        fill=(255, 255, 0), font=font)
            except Exception as exc:
                if not getattr(self, "_rec_hud_warned", False):
                    self._rec_hud_warned = True
                    self.log(f"HUD vùng 84×84 lỗi: {type(exc).__name__}: {exc}")

    def _update_preview(self) -> None:
        if self.preview is None:
            return
        try:
            img = self.preview.queue.get_nowait()
        except queue_mod.Empty:
            return
        from PIL import ImageTk

        img = img.copy()
        img.thumbnail((420, 260))
        # DEEP-FIX: vẽ lớp overlay hiệu chuẩn lên preview để bạn NHÌN THẤY
        # những gì mình đã chỉnh: đường phân cách chân trời (xanh), điểm neo
        # (đỏ) và điểm hồi sinh (xanh lá).  Trước đây "chân trời" không hề hiện
        # nên bạn không biết mình chỉnh gì.  Toạ độ dùng phân số nên đúng tỉ lệ
        # với ảnh thumbnail.
        from PIL import ImageDraw
        dw, dh = img.size
        dr = ImageDraw.Draw(img)
        hf = float(self.cfg.perception.horizon_frac)
        dr.line([(0, int(hf * dh)), (dw, int(hf * dh))], fill=(46, 204, 113), width=2)
        if getattr(self.cfg.death, "anchor_fx", -1.0) >= 0:
            ax = self.cfg.death.anchor_fx * dw
            ay = self.cfg.death.anchor_fy * dh
            dr.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], outline=(255, 59, 48), width=2)
        if self.cfg.input.respawn_set():
            rx = self.cfg.input.respawn_fx * dw
            ry = self.cfg.input.respawn_fy * dh
            dr.ellipse([rx - 5, ry - 5, rx + 5, ry + 5], outline=(46, 204, 113), width=2)
        # DEEP-FIX: quay demo trước đây gần như vô hình — chỉ có một dòng chữ
        # "demo frames: N" trong lưới số liệu, nên người dùng không biết dữ liệu
        # đang được thu thập ra sao.  Vẽ ngay lên preview: banner REC đỏ, số
        # khung, thời gian, HÀNH ĐỘNG đang nhận (để thấy phím ăn hay kẹt), và
        # ảnh vùng 84x84 thực sự đang được ghi.
        if self.sm.state is BotState.RECORDING_DEMO and self._demo_recorder is not None:
            self._draw_rec_overlay(img, dr)
        self.preview_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_img)
        self._latest_preview = img  # for click/anchor sampling (fractions)
        # DEEP-FIX: the step-3 anchor canvas used to stay a blank grey/black
        # box -- the preview image was never drawn on it, so the user picked
        # the anchor blind ("đen thui").  Draw the same live frame there.
        self.anchor_canvas.delete("preview")
        self.anchor_canvas.delete("marker")
        self._anchor_photo = self.preview_img
        self.anchor_canvas.create_image(0, 0, anchor="nw",
                                        image=self._anchor_photo, tags="preview")
        # DEEP-FIX: bước 4 (nút hồi sinh) cũng từng là một hộp đen — ảnh preview
        # không được vẽ lên respawn_canvas nên bạn phải click mò.  Vẽ cùng frame.
        self.respawn_canvas.delete("preview")
        self.respawn_canvas.delete("marker")
        self._respawn_photo = self.preview_img
        self.respawn_canvas.create_image(0, 0, anchor="nw",
                                         image=self._respawn_photo, tags="preview")
        # DEEP-FIX: luminance readout so a broken capture is obvious.
        import numpy as _np
        lum = float(_np.asarray(img, dtype=_np.float32).mean())
        if lum < 8.0:
            self.preview_luma_var.set(
                f"độ sáng {lum:.0f}/255 — ĐEN! capture hỏng: game "
                "occluded / wrong region / hardware acceleration")
        elif lum > 247.0:
            self.preview_luma_var.set(f"độ sáng {lum:.0f}/255 — TRẮNG tinh, sai vùng?")
        else:
            self.preview_luma_var.set(f"độ sáng {lum:.0f}/255 — OK")

    _latest_preview: Optional[Any] = None
    _preview_full_size: Optional[Any] = None

    def lock_region(self) -> None:
        r = self.cfg.region
        if not r.is_set():
            self.status_var.set("Không khoá được: chưa chọn vùng.")
            return
        if r.width < 240 or r.height < 320:
            self.status_var.set(
                f"Region too small ({r.width}x{r.height}); need >=240x320 so the "
                f"game scene is usable. Re-select."
            )
            return
        if r.width * r.height > self.cfg.capture.max_region_pixels:
            self.status_var.set("Vùng quá lớn; chọn lại nhỏ hơn.")
            return
        self._set_state(BotState.READY)
        self.lock_var.set(f"đã khoá: {r.width}x{r.height} tại {r.left},{r.top}")
        self.log("region locked")
        self.nb.select(self.step3)
        self.status_var.set("Đã khoá vùng — sang bước 3 (neo màu).")

    # step 3 ----------------------------------------------------------- #
    def _build_step3(self) -> None:
        f = self.step3
        ttk.Label(f, text="Bước 3 — bấm vào một điểm UI ỔN ĐỊNH luôn hiện khi "
                          "SỐNG và đổi/mất khi thua (góc thanh điểm, "
                          "bộ đếm xu...). Miếng 5x5 lấy trung vị làm gốc "
                          "và điểm ổn định.", wraplength=700).pack(anchor="w")
        self.anchor_canvas = tk.Canvas(f, width=420, height=260, bg="grey15",
                                       cursor="crosshair")
        self.anchor_canvas.pack(pady=6)
        self.anchor_canvas.bind("<ButtonPress-1>", self._anchor_clicked)
        self.anchor_var = tk.StringVar(value="neo: chưa đặt")
        ttk.Label(f, textvariable=self.anchor_var, font=("Consolas", 10)).pack()
        # DEEP-FIX: live readout of how "readable" the capture is.
        # A black or white capture is instantly visible here instead of
        # silently producing a useless anchor.
        self.preview_luma_var = tk.StringVar(value="preview: not started")
        ttk.Label(f, textvariable=self.preview_luma_var,
                  font=("Consolas", 10), foreground="#080").pack()
        row = ttk.Frame(f)
        row.pack(pady=4)
        ttk.Button(row, text="Bật preview", command=self.start_preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="Hiệu chuẩn SỐNG (2s)", command=self._anchor_calibrate_alive).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="Lấy mẫu CHẾT", command=self._anchor_capture_dead).pack(side=tk.LEFT, padx=3)
        ttk.Label(f, text="Mẹo: chạy hiệu chuẩn SỐNG khi game đang chạy; "
                          "capture the dead sample later on the game-over screen.",
                  wraplength=700).pack()

    def _anchor_clicked(self, ev) -> None:
        if self._latest_preview is None:
            self.anchor_var.set("neo: bật preview trước, rồi bấm")
            return
        # scale click -> region fraction
        w, h = self._latest_preview.size
        fx, fy = ev.x / max(1, w), ev.y / max(1, h)
        if not (0.01 <= fx <= 0.99 and 0.01 <= fy <= 0.99):
            return
        self.cfg.death.anchor_fx = float(fx)
        self.cfg.death.anchor_fy = float(fy)
        # DEEP-FIX: show where the anchor landed instead of an invisible state.
        self.anchor_canvas.delete("marker")
        self.anchor_canvas.create_oval(ev.x - 5, ev.y - 5, ev.x + 5, ev.y + 5,
                                       outline="#ff3b30", width=2, tags="marker")
        self.anchor_var.set(f"vị trí neo: fx={fx:.3f} fy={fy:.3f} "
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
            self.status_var.set("Bấm vị trí neo trên preview trước.")
            return
        self.anchor_frames = []
        self.anchor_t0 = time.monotonic()
        self.status_var.set("Đang hiệu chuẩn SỐNG — giữ game hiện...")
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
            self.status_var.set("Quá ít mẫu neo; preview có chạy không? Thử lại.")
            return
        from perception import (patch_stability, is_degenerate_patch,
                                 mean_luma)
        import numpy as np

        std, baseline = patch_stability(self.anchor_frames)
        stacked = np.stack(self.anchor_frames)
        black = mean_luma(stacked) < 8.0
        flat = is_degenerate_patch(self.anchor_frames[0])
        # DEEP-FIX: "stable" alone is the wrong gate -- a black or flat patch
        # is the *most* stable thing there is, so a broken capture used to be
        # [ACCEPTED] with std=0.00.  Require stability AND real content.
        ok = (std <= 6.0) and not black and not flat
        if black:
            verdict = "[BLACK CAPTURE — not usable, see hint below]"
        elif flat:
            verdict = "[FLAT COLOUR — pick a textured pixel]"
        elif std > 6.0:
            verdict = "[UNSTABLE — pick a calmer pixel]"
        else:
            verdict = "[ACCEPTED]"
        self.cfg.death.anchor_baseline_rgb = baseline
        self.cfg.death.anchor_baseline_std = std
        self.anchor_var.set(
            f"anchor baseline RGB={baseline} stability(std)={std:.2f} "
            f"samples={len(self.anchor_frames)} {verdict}"
        )
        if ok:
            self.log(f"anchor calibrated: rgb={baseline} std={std:.2f}")
            self.status_var.set("Đã hiệu chuẩn neo — sang bước 4.")
            self._maybe_ready()
        else:
            self.status_var.set(
                f"Anchor not usable (std={std:.2f}, black={black}, flat={flat}). "
                "Choose a pixel that does not "
                f"flicker, then recalibrate."
            )

    def _anchor_capture_dead(self) -> None:
        patch = self._grab_anchor_patch()
        if patch is None:
            self.status_var.set("Lấy mẫu CHẾT thất bại (chưa có vùng/neo).")
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
        ttk.Label(f, text="Bước 4 — bấm vào nút chơi lại/hồi sinh trên "
                          "màn hình thua. Không gì được click tự động; "
                          "nút Test sẽ hỏi xác nhận trước.", wraplength=700
                  ).pack(anchor="w")
        self.respawn_canvas = tk.Canvas(f, width=420, height=260, bg="grey15",
                                        cursor="crosshair")
        self.respawn_canvas.pack(pady=6)
        self.respawn_canvas.bind("<ButtonPress-1>", self._respawn_clicked)
        self.respawn_var = tk.StringVar(value="điểm hồi sinh: chưa đặt")
        ttk.Label(f, textvariable=self.respawn_var).pack()
        row = ttk.Frame(f)
        row.pack(pady=4)
        ttk.Button(row, text="Bật preview", command=self.start_preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="Test click (hỏi trước)", command=self._respawn_test).pack(side=tk.LEFT, padx=3)

    def _respawn_clicked(self, ev) -> None:
        if self._latest_preview is None:
            self.respawn_var.set("hồi sinh: bật preview trước, rồi bấm")
            return
        w, h = self._latest_preview.size
        self.cfg.input.respawn_fx = ev.x / max(1, w)
        self.cfg.input.respawn_fy = ev.y / max(1, h)
        # DEEP-FIX: khoanh tròn vị trí đã chọn để bạn thấy rõ mình bấm vào đâu.
        self.respawn_canvas.delete("marker")
        self.respawn_canvas.create_oval(ev.x - 6, ev.y - 6, ev.x + 6, ev.y + 6,
                                        outline="#2ecc71", width=2, tags="marker")
        self.respawn_var.set(
            f"điểm hồi sinh: fx={self.cfg.input.respawn_fx:.3f} "
            f"fy={self.cfg.input.respawn_fy:.3f} (abs "
            f"{self.cfg.region.left + int(self.cfg.input.respawn_fx * self.cfg.region.width)},"
            f"{self.cfg.region.top + int(self.cfg.input.respawn_fy * self.cfg.region.height)})"
        )
        self._maybe_ready()

    def _respawn_test(self) -> None:
        if not self.cfg.input.respawn_set():
            self.status_var.set("Đặt điểm hồi sinh trước.")
            return
        x = self.cfg.region.left + int(self.cfg.input.respawn_fx * self.cfg.region.width)
        y = self.cfg.region.top + int(self.cfg.input.respawn_fy * self.cfg.region.height)
        if not messagebox.askyesno(
            "Xác nhận test click",
            f"Bấm một lần tại toạ độ màn hình ({x}, {y})?\n"
            "Di chuột vào góc màn hình để huỷ (failsafe pyautogui).",
        ):
            return
        from input_controller import InputController

        ctl = InputController(self.cfg.input, backend="auto")
        try:
            # DEEP-FIX: cổng "chỉ bấm khi Chrome focus" là cho LÚC CHƠI, không
            # dành cho nút Test hiệu chuẩn — bạn đang bấm trên cửa sổ GUI nên
            # Chrome không focus và cú test luôn bị "BLOCKED (focus?)".  Cú test
            # đã được bạn xác nhận bằng hộp thoại, nên bỏ cổng focus ở đây.
            ok = ctl.click(x, y, confirm_focus=False)
            self.log(f"test click at ({x},{y}): {'đã gửi' if ok else 'THẤT BẠI'}")
        finally:
            ctl.dispose()

    # step 5 ----------------------------------------------------------- #
    def _build_step5(self) -> None:
        f = self.step5
        ttk.Label(f, text="Bước 5 — vùng chân trời: dải trên của vùng dùng "
                          "để phát hiện vật cản nhanh. Mặc định 25%.", wraplength=700
                  ).pack(anchor="w")
        self.horizon_var = tk.StringVar(value=f"chân trời: {self.cfg.perception.horizon_frac*100:.0f}%")
        self.horizon_scale = ttk.Scale(
            f, from_=15, to=40, value=self.cfg.perception.horizon_frac * 100,
            command=self._horizon_changed)
        self.horizon_scale.pack(fill=tk.X, pady=6)
        ttk.Label(f, textvariable=self.horizon_var).pack()
        ttk.Label(f, text="Overlay preview sẽ hiện đường phân cách khi "
                          "preview chạy.", wraplength=700).pack()

    def _horizon_changed(self, value: str) -> None:
        pct = float(value)
        self.cfg.perception.horizon_frac = round(pct, 1) / 100.0
        self.horizon_var.set(f"chân trời: {self.cfg.perception.horizon_frac*100:.1f}% "
                             f"({int(self.cfg.region.height * self.cfg.perception.horizon_frac)} px)")

    # step 6 ----------------------------------------------------------- #
    def _build_step6(self) -> None:
        f = self.step6
        ctrl = ttk.Frame(f)
        ctrl.pack(fill=tk.X)
        self.btn_start = ttk.Button(ctrl, text="▶ Bắt đầu train", command=self.start_training,
                                    state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=3)
        self.btn_pause = ttk.Button(ctrl, text="⏸ Tạm dừng", command=self.pause_training,
                                    state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=3)
        self.btn_stop = ttk.Button(ctrl, text="⏹ Dừng", command=self.stop_training,
                                   state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        self.btn_demo = ttk.Button(ctrl, text="● Quay demo (F9 dừng)",
                                   command=self.toggle_demo_recording, state=tk.DISABLED)
        self.btn_demo.pack(side=tk.LEFT, padx=3)
        self.btn_pretrain = ttk.Button(ctrl, text="Tiền-huấn luyện (BC)", command=self.run_pretrain,
                                       state=tk.DISABLED)
        self.btn_pretrain.pack(side=tk.LEFT, padx=3)
        self.btn_emergency = ttk.Button(ctrl, text="⛔ KHẨN CẤP (F8)",
                                        command=self.emergency_stop)
        self.btn_emergency.pack(side=tk.RIGHT, padx=3)

        metrics = ttk.Labelframe(f, text="Số liệu trực tiếp", padding=6)
        metrics.pack(fill=tk.BOTH, expand=True, pady=6)
        self.metric_vars: dict[str, tk.StringVar] = {}
        cols = [
            ("state", "Trạng thái"), ("survival_s", "Sống (s)"),
            ("episode_id", "Màn"), ("score", "Điểm (n/a)"),
            ("fps", "FPS capture"), ("eff_fps", "FPS hiệu dụng"),
            ("dropped", "Frame rơi"), ("infer_p95", "Suy luận p95 (ms)"),
            ("action_p95", "Hành động p95 (ms)"), ("epsilon", "Epsilon"),
            ("avg_reward", "Thưởng TB (màn)"), ("td_loss", "Mất mát TD"),
            ("q_mean", "Q TB"), ("buffer", "Replay buffer"),
            ("learner_updates", "Bước learner"), ("cpu", "CPU %"),
            ("ram", "RAM (GB)"), ("profile", "Profile model"),
            ("best", "Tốt nhất"),
        ]
        for i, (key, label) in enumerate(cols):
            r, c = divmod(i, 3)
            ttk.Label(metrics, text=label + ":").grid(row=r, column=c * 2,
                                                      sticky="e", padx=4, pady=2)
            var = tk.StringVar(value="—")
            ttk.Label(metrics, textvariable=var, font=("Consolas", 10)
                      ).grid(row=r, column=c * 2 + 1, sticky="w", padx=4, pady=2)
            self.metric_vars[key] = var

        logframe = ttk.Labelframe(f, text="Nhật ký train", padding=4)
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
    def _preflight_capture_check(self) -> None:
        """Grab one frame and warn if the capture looks broken/dead so the user
        fixes the region/focus BEFORE a run that would instantly die."""
        from perception import mean_luma, patch_rgb_distance
        patch = self._grab_anchor_patch()
        if patch is None:
            self.log("preflight: KHÔNG bắt được frame — kiểm tra vùng chọn.")
            return
        if float(mean_luma(patch)) < 8.0:
            self.log("⚠ preflight: frame ĐEN — vùng bị che hoặc hardware "
                     "acceleration. Sửa trước khi train.")
            return
        d = self.cfg.death
        if d.anchor_baseline_rgb[0] >= 0:
            dist = float(patch_rgb_distance(patch, d.anchor_baseline_rgb))
            if dist > d.threshold:
                self.log(f"⚠ preflight: neo KHÁC xa lúc sống "
                         f"(d={dist:.0f} > {d.threshold}) — vùng đang hiện màn "
                         "THUA hoặc bị che. Vào màn game SỐNG rồi bấm lại.")
            else:
                self.log(f"preflight OK: neo khớp lúc sống (d={dist:.0f}).")

    def start_training(self) -> None:
        if self.app is not None:
            self.status_var.set("Đang chạy rồi.")
            return
        warnings = self.cfg.validate()
        for w in warnings:
            self.log(f"config warning: {w}")
        self._preflight_capture_check()
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
        self.status_var.set("Đã bắt đầu train. F8 = dừng khẩn cấp.")
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
        self.status_var.set("Đang dừng (lưu model + buffer)...")
        self.root.update_idletasks()
        if self.hotkey is not None:
            self.hotkey.stop()
            self.hotkey = None
        self.app.shutdown()
        self.app = None
        self._set_state(BotState.STOPPED)
        self.status_var.set("Đã dừng. Đã lưu model, buffer, log.")
        for btn in (self.btn_pause, self.btn_stop):
            btn.configure(state=tk.DISABLED)
        self.btn_start.configure(state=tk.NORMAL)
        self.log("stopped cleanly")

    def emergency_stop(self) -> None:
        self.log("DỪNG KHẨN CẤP (GUI)")
        if self.app is not None:
            self.app.emergency_stop()
            self.stop_training()
        else:
            self._set_state(BotState.STOPPED)

    def _arm_demo_hotkey(self) -> None:
        """Global F9 listener so recording can be stopped while the GAME has
        focus.  A Tk key binding would never fire (focus is on Chrome)."""
        if self._demo_hotkey is not None:
            return
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.f9:
                    self._demo_stop_req.set()

            self._demo_hotkey = keyboard.Listener(on_press=on_press)
            self._demo_hotkey.start()
        except Exception as exc:
            self._demo_hotkey = None
            self.log(f"F9 không khả dụng ({type(exc).__name__}: {exc}) — "
                     "dùng nút ■ Dừng demo.")

    def _disarm_demo_hotkey(self) -> None:
        h = self._demo_hotkey
        self._demo_hotkey = None
        self._demo_stop_req.clear()
        if h is not None:
            try:
                h.stop()
            except Exception as exc:
                self.log(f"dừng hotkey demo thất bại: {exc}")

    def toggle_demo_recording(self) -> None:
        if self.sm.state is BotState.RECORDING_DEMO:
            self._stop_demo_recording()
            return
        if not self.cfg.region.is_set():
            self.status_var.set("Chọn vùng trước khi quay demo.")
            return
        self._set_state(BotState.RECORDING_DEMO)
        from demonstration_recorder import DemoRecorder
        from ipc import SharedFrameRing

        self._demo_started_app = self.app is None
        if self.app is None:
            from app import BotApplication

            self.app = BotApplication(self.cfg)
            # DEEP-FIX: capture only — NO actor.  The actor presses keys, so
            # starting it made the character move by itself during recording.
            self.app.start(with_learner=False, with_actor=False)
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
        self._demo_recorder.on_episode_saved = self._on_demo_episode_saved
        self._demo_recorder.start()
        self._demo_t0 = time.monotonic()
        self._demo_last_log_t = 0.0
        self._arm_demo_hotkey()
        self.btn_demo.configure(text="■ Dừng demo (F9)")
        self.status_var.set("ĐANG quay demo — chơi đi! F9 / nút để dừng.")
        # DEEP-FIX: log rõ trạng thái để người dùng biết quá trình thu thập.
        r = self.cfg.region
        self.log("=== BẮT ĐẦU QUAY DEMO ===")
        self.log(f"vùng {r.width}x{r.height} | thư mục: {self.cfg.paths.demos_dir}"
                 f" | chân trời={self.cfg.perception.horizon_frac:.2f}")
        if self._demo_recorder.keyboard_active():
            self.log("bàn phím: ĐANG NGHE — chơi bằng mũi tên (trái/phải/nhảy/lướt).")
        else:
            self.log("⚠ bàn phím: KHÔNG KHẢ DỤNG — demo sẽ TOÀN NOOP, không dùng "
                     "cho tiền-huấn luyện được! Cần chạy trên máy có bàn phím thật.")
        if self.cfg.death.anchor_set():
            self.log("tự động tách episode khi CHẾT (mỗi mạng = 1 file) — cứ chơi "
                     "tiếp sau khi hồi sinh, không cần bấm F9.")
            self.log("mỗi episode tự CẮT BỎ ~3.5s cuối (lúc loạng choạng/di chuyển "
                     "lỗi trước khi chết) để AI không bắt chước.")
        else:
            self.log("chưa hiệu chuẩn neo chết (bước 3) → KHÔNG tự tách episode; "
                     "bấm F9 thủ công để cắt mỗi episode.")
        self.log("F9 hoặc nút ■ để dừng & lưu. Đang ghi…")

    def _on_demo_episode_saved(self, path: str) -> None:
        """Logged when a life ends and the recorder auto-starts a new episode."""
        n = len(self._demo_recorder.episode_paths) if self._demo_recorder else 0
        self.log(f"☠ chết → đã lưu episode: {path}")
        self.log(f"tổng {n} episode. Đang ghi episode MỚI — chơi tiếp đi.")

    def _stop_demo_recording(self) -> None:
        self._disarm_demo_hotkey()
        if self._demo_recorder is None:
            return
        path = self._demo_recorder.stop(done=True)
        self._demo_recorder.dispose()
        self._demo_recorder = None
        self.btn_demo.configure(text="● Quay demo (F9 dừng)")
        self._set_state(BotState.READY)
        self.log("=== DỪNG QUAY DEMO ===")
        # DEEP-FIX: demo recording started a capture-only app; shut it down so
        # start_training() (which refuses when self.app is set) can run again.
        if getattr(self, "_demo_started_app", False) and self.app is not None:
            try:
                self.app.shutdown()
            except Exception as exc:
                self.log(f"dừng capture demo lỗi: {type(exc).__name__}: {exc}")
            self.app = None
            self._demo_started_app = False
        if path:
            self.status_var.set(f"Demo saved: {path}")
            self.log(f"đã lưu: {path}")
            self.log("kiểm tra: python app.py --validate-demos "
                     f"{self.cfg.paths.demos_dir}")
        else:
            self.status_var.set("Đã dừng quay demo (không lưu).")
            self.log("không lưu (0 khung — preview có chạy không?).")

    def run_pretrain(self) -> None:
        if self.app is None:
            self.status_var.set("Bắt đầu train trước (learner giữ việc tiền-huấn luyện).")
            return
        self._pre_pretrain_state = self.sm.state
        self._set_state(BotState.PRETRAINING)
        self.app.command("pretrain", demos_dir=str(self.cfg.paths.demos_dir))
        # DEEP-FIX: log rõ để người dùng biết BC đang chạy (trước đây im lặng).
        self.log("=== BẮT ĐẦU TIỀN-HUẤN LUYỆN (BC) ===")
        self.log(f"thư mục demos: {self.cfg.paths.demos_dir}")
        self.log("learner đang kiểm tra demo rồi học bắt chước — xem log bên dưới.")
        self.status_var.set("Đang tiền-huấn luyện BC… xem log.")

    # ------------------------------------------------------------------ #
    # Polling loop (the ONLY place worker data reaches Tk)
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        try:
            self._update_preview()
            # DEEP-FIX: the demo recorder was constructed with a frame reader
            # and then never pumped, so every episode saved 0 frames.  The Tk
            # loop is the only correct place to do it: it is the main thread,
            # it already runs at a bounded rate, and it never blocks on I/O
            # longer than a few ring reads.
            self._pump_demo_recorder()
            # DEEP-FIX: F9 arrives on the pynput thread; perform the stop here
            # (Tk main thread) so widget updates stay thread-safe.
            if self._demo_stop_req.is_set():
                self._demo_stop_req.clear()
                self._stop_demo_recording()
            self._poll_metrics()
            self._check_workers()
        except Exception as exc:
            self.log(f"gui tick error: {exc}")
        self.root.after(self.POLL_MS, self._tick)

    def _pump_demo_recorder(self) -> None:
        rec = self._demo_recorder
        if rec is None or not rec.recording:
            return
        try:
            got = rec.pump(max_frames=4)
        except Exception as exc:
            self.log(f"demo pump error: {type(exc).__name__}: {exc}")
            return
        if got:
            frames = rec.frame_count()
            self.metric_vars["episode_id"].set(f"demo frames: {frames}")
            # DEEP-FIX: log tiến trình ~2s/lần để người dùng THẤY dữ liệu đang vào.
            now = time.monotonic()
            if now - getattr(self, "_demo_last_log_t", 0.0) >= 2.0:
                self._demo_last_log_t = now
                label = self._ACTION_STYLE.get(rec.current_action(), ("?", None))[0]
                self.log(f"đang ghi demo: {frames} khung, "
                         f"{now - self._demo_t0:.1f}s, hành động: {label}")

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
                best = d.get("best_rolling", -1.0)
                name = d.get("best_metric_name", "survival_s")
                self.metric_vars["best"].set(
                    "—" if best is None or best < 0 else f"{best:.1f} ({name})")
            elif kind == "best_model":
                d = msg.get("data") or msg
                self.log(f"NEW BEST MODEL ({d.get('metric')}: rolling="
                         f"{d.get('rolling', 0):.2f}, window={d.get('window')})")
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
            elif kind == "pretrain_done":
                self._on_pretrain_done(msg.get("result", {}))
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

    def _on_pretrain_done(self, res: dict) -> None:
        """Log a clear Vietnamese summary when behaviour cloning finishes."""
        status = res.get("status")
        if status == "ok":
            hist = res.get("history") or []
            last = hist[-1] if hist else {}
            self.log("=== TIỀN-HUẤN LUYỆN HOÀN TẤT ===")
            self.log(f"{len(hist)} epoch | val_acc cuối="
                     f"{float(last.get('val_acc', 0)):.3f} | "
                     f"train_acc={float(last.get('train_acc', 0)):.3f}")
            self.log("đã lưu checkpoint best + latest — bot đã biết bắt chước, "
                     "sẵn sàng train RL tiếp.")
            self.status_var.set("Tiền-huấn luyện xong. Sẵn sàng train.")
        else:
            self.log("=== TIỀN-HUẤN LUYỆN BỎ QUA ===")
            self.log(f"lý do: {res.get('reason', 'không rõ')} — cần >= "
                     f"{self.cfg.bc.min_episodes} demo hợp lệ trong "
                     f"{self.cfg.paths.demos_dir}.")
            self.log("hãy quay demo (chơi thật) rồi bấm Tiền-huấn luyện lại; "
                     "train RL vẫn chạy bình thường không cần BC.")
            self.status_var.set("BC bỏ qua (thiếu demo hợp lệ). Xem log.")
        # restore the control state so the UI is not stuck in PRETRAINING
        target = getattr(self, "_pre_pretrain_state", BotState.READY)
        try:
            self._set_state(target)
        except InvalidTransitionError:
            self._set_state(BotState.READY)

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
            self.status_var.set("Learner sập — tạm dừng chơi. Xem log.")
        if not alive["actor"] and getattr(self.app, "actor_proc", None) is not None:
            # DEEP-FIX: this branch had no state guard, so after an actor
            # crash it fired on every 150 ms tick forever (~7 identical log
            # lines per second, and the status strip was rewritten just as
            # often).  Report once by moving to ERROR, which is not RUNNING.
            if self.sm.state is BotState.RUNNING or self.sm.state is BotState.PAUSED:
                self.log("actor process died — stop and restart required")
                self._set_state(BotState.ERROR)
            self.status_var.set("Actor sập — bấm Dừng, rồi Bắt đầu lại.")

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
    # DEEP-FIX: must happen before the first window exists, or Windows
    # refuses it and every screen coordinate is virtualised.
    set_dpi_awareness()
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
