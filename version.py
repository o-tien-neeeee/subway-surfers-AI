"""Single source of truth for the application build version.

Why this exists
---------------
Before this file there was no visible build number: a user running
``python app.py`` could not tell whether they had the latest fixes, so after a
round of bug-fixing they would re-run an old checkout and conclude "still
broken".  The request that prompted it was blunt: *"mỗi lần chỉnh sửa thì tăng
version lên đi"* — bump the version on every edit.

So the rule is: **whenever behaviour changes, bump ``APP_VERSION`` here** (and
only here).  It is surfaced in:

  * the Tk window title (``gui.py``),
  * the startup log line (``app.py``),
  * the web dashboard header and ``/api/health`` (``webui.py``),
  * ``runs/headless_report.json`` so a report is self-describing.

``CONFIG_VERSION`` (config.py) is a *separate* number: it describes the config
schema, not the build, and must not be conflated with this one.
"""

#: Bump on every behavioural change.  Format: MAJOR.MINOR.PATCH
APP_VERSION = "1.15.0"

#: Human-readable notes for the current build, newest first.  Kept short on
#: purpose — the full reasoning lives in DEEP_FIX_REPORT.md.
CHANGELOG = [
    "1.13.1 — README hiển thị phiên bản hiện tại (banner) + test buộc README khớp APP_VERSION mỗi phiên.",
    "1.13.0 — audit toàn bộ máy học (Double DQN/PER/n-step/reward/observation): thuật toán vững; thêm log TIẾN TRÌNH HỌC mỗi 10 episode (survival TB ↑/→/↓ + epsilon) để thấy AI có đang học thật không.",
    "1.12.1 — khi chết, CẮT BỎ ~3.5s cuối episode (lúc loạng choạng/di chuyển lỗi) để AI không bắt chước; khung chết không bao giờ được ghi (sửa mốc cắt).",
    "1.12.0 — quay demo TỰ ĐỘNG tách episode khi chết (mỗi mạng = 1 file, done=True đúng khung chết), bỏ qua màn thua/hồi sinh để episode mới bắt đầu khi sống; cần neo chết đã hiệu chuẩn.",
    "1.11.0 — quay demo KHÔNG còn tự di chuyển: recording chỉ chạy capture (with_actor=False), actor bấm phím không bị khởi động; dừng demo thì tắt app để Start train lại được.",
    "1.10.1 — log rõ từng trạng thái của quay demo (bắt đầu/tiến trình/dừng, hook bàn phím) và tiền-huấn luyện BC (bắt đầu/kiểm tra demo/mỗi epoch/hoàn tất hoặc bỏ qua) ra khung log GUI.",
    "1.10.0 — quay demo: F9 dừng THẬT (global hotkey), lớp phủ trực quan ● REC + hành động + vùng 84×84, sửa action ma (held-key set + chặn modifier + tự xoá phím kẹt); chẩn đoán bot chết-ngay liên tiếp.",
    "1.9.1 — tab Hướng dẫn bổ sung mục QUAY DEMO và TIỀN-HUẤN LUYỆN BC (giải thích + cách dùng).",
    "1.9.0 — tab HƯỚNG DẪN A–Z; preview vẽ đường chân trời + marker sống/hồi sinh; "
    "nút Test click bỏ cổng focus (không còn BLOCKED); preflight cảnh báo vùng bị che "
    "trước khi train.",
    "1.8.0 — GUI tiếng Việt toàn bộ; preview + marker vẽ lên cả bước 4 (nút hồi sinh).",
    "1.7.0 — removed a shadowed duplicate is_black_frame (round-4 regression); runtime black-frame skip now uses the single, config-driven definition.",
    "1.6.0 — activation-memory estimate no longer double-counts composite modules; config.save is atomic (failed save never bricks config.json).",
    "1.5.0 — black-capture guard in anchor calibration + live luminance readout; "
    "build version surfaced in GUI/log/webui/report.",
    "1.4.0 — hazard expiry no longer wedges; no credit for ancient dodges; "
    "browser dashboard (webui.py).",
    "1.3.0 — fixed winfovrootheight, backend_name resolution, DPI ordering, "
    "Thread._stop shadowing; Tk-API surface guard.",
    "1.2.0 — deep fix: IPC profile switch, PER per-sample TD, drain timeout, "
    "demo recorder, learner spin, seqlock telemetry, checkpoints.",
    "1.1.0 — upstream research baseline.",
]


def banner() -> str:
    return f"Subway Surfers Research Bot v{APP_VERSION}"
