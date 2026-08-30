"""Browser dashboard for the Subway Surfers research bot.

Why this exists
---------------
The real control surface is a Tkinter desktop app (``gui.py``). It cannot be
shown in a browser preview, and it needs a Windows desktop, a Chrome window
running the game, and a keyboard to drive. So the parts of this project that
*can* run anywhere -- the synthetic game, the perception stack, the policy
network, the headless training pipeline -- were effectively invisible unless
you already knew which command to type.

This module serves those parts over plain HTTP using only the standard
library, so there is nothing to install:

  * **Live**   -- the real ``GameEnvironment`` (perception + horizon detector +
                  death detector + reward) driven by the real
                  ``InferencePolicy``, streamed as JPEG so you can watch it.
  * **Train**  -- runs the actual ``app.py --headless`` multi-process pipeline
                  as a subprocess and tails its log.
  * **Report** -- renders ``runs/headless_report.json``.
  * **Guide**  -- what each part needs, and what only works on Windows.

It is deliberately read-mostly and side-effect-light: it writes nothing except
whatever ``app.py --headless`` already writes (checkpoints, logs, reports).

Run it with:  python webui.py [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent import InferencePolicy                       # noqa: E402
from config import BotConfig, NOOP                      # noqa: E402
from environment import GameEnvironment, SyntheticGame  # noqa: E402
from models import PROFILES, DuelingDQN
from version import APP_VERSION, CHANGELOG                 # noqa: E402


# --------------------------------------------------------------------- #
# Live demo: real environment, real policy, rendered frames.
# --------------------------------------------------------------------- #
class DemoSession:
    """Drives GameEnvironment in a background thread and keeps the last frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.running = False
        self.policy_mode = "network"        # "network" | "random"
        self.profile = "strict_lite"
        self.speed_hz = 12.0
        self.env: Optional[GameEnvironment] = None
        self.policy: Optional[InferencePolicy] = None
        self.stats: dict[str, Any] = {}
        self.last_error: str = ""

    # -- lifecycle ----------------------------------------------------- #
    def start(self, policy_mode: str, profile: str, speed_hz: float,
              seed: int = 0) -> dict[str, Any]:
        with self._lock:
            if self.running:
                return {"ok": False, "error": "already running"}
            self.policy_mode = policy_mode
            self.profile = profile if profile in PROFILES else "strict_lite"
            self.speed_hz = max(1.0, min(60.0, float(speed_hz)))
            try:
                cfg = BotConfig()
                cfg.seed = int(seed)
                self.env = GameEnvironment(cfg, SyntheticGame(seed=int(seed)))
                model = DuelingDQN.from_profile(
                    self.profile, cfg.perception.frame_stack,
                    cfg.perception.ground_size)
                self.policy = InferencePolicy(model, seed=int(seed))
            except Exception as exc:                       # pragma: no cover
                self.last_error = f"{type(exc).__name__}: {exc}"
                return {"ok": False, "error": self.last_error}
            self._stop.clear()
            self.running = True
            self.stats = {"episodes": 0, "steps": 0, "score": 0,
                          "last_reward": 0.0, "last_action": NOOP,
                          "alive": True, "policy": self.policy_mode,
                          "profile": self.profile, "hz": self.speed_hz,
                          "params": sum(p.numel()
                                        for p in model.parameters())}
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="demo-live")
            self._thread.start()
            return {"ok": True, "params": self.stats["params"],
                    "profile": self.profile}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self.running = False
        return {"ok": True}

    # -- the loop ------------------------------------------------------ #
    def _loop(self) -> None:
        env = self.env
        assert env is not None
        obs = env.reset()
        episode_reward = 0.0
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                if self.policy_mode == "random" or self.policy is None:
                    action = int(np.random.default_rng().integers(0, 5))
                else:
                    # epsilon=0 -> pure greedy, i.e. what the actor does at
                    # inference time.  The network is untrained unless a
                    # checkpoint was loaded, so this shows the plumbing, not
                    # a skilled player.
                    action = self.policy.act(obs, epsilon=0.0)
                obs, reward, done, info = env.step(action)
                episode_reward += float(reward)
                with self._lock:
                    self.stats.update({
                        "steps": self.stats.get("steps", 0) + 1,
                        "score": int(env.game.score),
                        "last_reward": round(float(reward), 4),
                        "last_action": int(action),
                        "alive": not done,
                        "episode_reward": round(episode_reward, 3),
                        "horizon": bool(info.horizon.detected),
                        "horizon_conf": round(float(info.horizon.confidence), 3),
                    })
                    frame = env.game.render()
                    self._jpeg = self._encode(frame)
                if done:
                    with self._lock:
                        self.stats["episodes"] = self.stats.get("episodes", 0) + 1
                        self.stats["last_episode_reward"] = round(episode_reward, 3)
                    obs = env.reset()
                    episode_reward = 0.0
                # hold the requested frame rate
                delay = (1.0 / self.speed_hz) - (time.perf_counter() - t0)
                if delay > 0:
                    time.sleep(delay)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.running = False

    @staticmethod
    def _encode(frame_rgb: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # downscale for the browser; 480x800 at 12 fps is plenty
        h, w = bgr.shape[:2]
        if h > 560:
            scale = 560.0 / h
            bgr = cv2.resize(bgr, (int(w * scale), 560),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return buf.tobytes() if ok else b""

    # -- accessors ----------------------------------------------------- #
    def frame(self) -> bytes:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self.stats)
        out["running"] = self.running
        out["last_error"] = self.last_error
        return out


# --------------------------------------------------------------------- #
# Training: the real multi-process pipeline as a subprocess.
# --------------------------------------------------------------------- #
class TrainSession:
    """Runs ``app.py --headless`` and tails its output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.log: deque[str] = deque(maxlen=400)
        self.proc: Optional[subprocess.Popen] = None
        self.running = False
        self.exit_code: Optional[int] = None
        self.started_at: Optional[float] = None
        self.cmd: list[str] = []

    def start(self, steps: int, profile: str, extra: list[str]) -> dict[str, Any]:
        with self._lock:
            if self.running and self.proc is not None and self.proc.poll() is None:
                return {"ok": False, "error": "a run is already in progress"}
            self.log.clear()
            self.exit_code = None
            steps = max(50, min(20000, int(steps)))
            self.cmd = [sys.executable, "app.py", "--headless",
                        "--steps", str(steps), "--dry-run"]
            if extra and str(extra[0]) == "--profile-models":
                self.cmd.append("--profile-models")
            # The profile is picked via config; keep the default config here
            # so the web run matches what the CLI would do with no flags.
            del profile  # reserved for a future --profile wiring
            env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            try:
                self.proc = subprocess.Popen(
                    self.cmd, cwd=str(ROOT), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.running = True
            self.started_at = time.time()
            threading.Thread(target=self._pump, daemon=True,
                             name="train-log").start()
            return {"ok": True, "cmd": " ".join(self.cmd)}

    def _pump(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                with self._lock:
                    self.log.append(line)
        code = proc.wait()
        with self._lock:
            self.exit_code = code
            self.running = False
            self.log.append(f"--- exited with code {code} ---")

    def stop(self) -> dict[str, Any]:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.running = False
        return {"ok": True}

    def snapshot(self, tail: int = 120) -> dict[str, Any]:
        with self._lock:
            lines = list(self.log)[-tail:]
        return {"running": self.running, "exit_code": self.exit_code,
                "cmd": " ".join(self.cmd), "lines": lines,
                "elapsed_s": (None if self.started_at is None
                              else round(time.time() - self.started_at, 1))}


DEMO = DemoSession()
TRAIN = TrainSession()


# --------------------------------------------------------------------- #
# Static content: the usage guide.
# --------------------------------------------------------------------- #
def guide_payload() -> list[dict[str, Any]]:
    """What each part of the project is, and what it needs to run."""
    return [
        {"title": "1. Live demo (this page)",
         "where": "anywhere — including this preview",
         "how": "Pick a policy and press Start.",
         "detail": ("Runs the real GameEnvironment: ZonePreprocessor, "
                    "HorizonDetector, ColorAnchorDeathDetector, "
                    "SurvivalRewardCalculator and InferencePolicy over the "
                    "SyntheticGame. The network is untrained unless a "
                    "checkpoint exists, so 'network' mode shows the plumbing, "
                    "not a skilled player.")},
        {"title": "2. Headless training",
         "where": "anywhere",
         "how": "Train tab → set steps → Start. Or: python app.py --headless --steps 1500 --dry-run",
         "detail": ("The full multi-process pipeline: capture-worker, "
                    "actor-worker and learner-worker communicating over shared "
                    "memory. --dry-run means no real keys are pressed. Writes "
                    "checkpoints/, logs/ and runs/headless_report.json.")},
        {"title": "3. Evaluation & A/B",
         "where": "anywhere",
         "how": ("python app.py --evaluate 1500   then   "
                 "python app.py --compare-baseline runs/evaluation_OLD.json"),
         "detail": ("Seeds each episode distinctly, so repeated runs are not "
                    "identical. --compare-baseline runs the Mann-Whitney U test "
                    "described in README section 10.")},
        {"title": "4. Profiling",
         "where": "anywhere",
         "how": "python profiling.py",
         "detail": ("Measures per-profile training-update cost on THIS "
                    "machine and estimates memory. Re-run on the target "
                    "machine before picking a default profile.")},
        {"title": "5. Demonstration recording",
         "where": "Windows desktop + Chrome running the game",
         "how": "python app.py --record-demo   (then play; arrows/jump/slide are captured)",
         "detail": ("KeyboardTap listens with pynput and DemoRecorder.pump() is "
                    "driven from the Tk event loop. Needs a real keyboard and a "
                    "real game window.")},
        {"title": "6. Behaviour cloning pretrain",
         "where": "anywhere (needs recorded demos)",
         "how": "python app.py --pretrain demos/20260101_120000.npz",
         "detail": "Trains the policy on recorded demonstrations before RL."},
        {"title": "7. The real bot (GUI calibration)",
         "where": "Windows desktop ONLY — cannot be previewed in a browser",
         "how": "python app.py   → follow the six numbered steps",
         "detail": ("Needs a display, Chrome playing Poki Subway Surfers, and "
                    "pynput. Step order: select screen region → live preview → "
                    "lock region → calibrate the alive anchor → capture the "
                    "death anchor → set the respawn button. This is the only "
                    "part that touches the real game.")},
    ]


def profiles_payload() -> list[dict[str, Any]]:
    out = []
    for name, spec in PROFILES.items():
        try:
            m = DuelingDQN.from_profile(name, 4, 84)
            params = sum(p.numel() for p in m.parameters())
        except Exception:
            params = 0
        out.append({"name": name, "kind": spec.get("kind", ""),
                    "blocks": spec.get("blocks", []),
                    "head_hidden": spec.get("head_hidden", 0),
                    "budget": spec.get("param_budget", 0),
                    "params": params,
                    "within_budget": (params <= spec.get("param_budget", 1 << 60)
                                      if spec.get("param_budget") else True),
                    "description": spec.get("description", "")})
    return out


def report_payload() -> dict[str, Any]:
    p = ROOT / "runs" / "headless_report.json"
    if not p.exists():
        return {"exists": False}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"exists": True, "error": f"{type(exc).__name__}: {exc}"}
    return {"exists": True, "data": d,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(p.stat().st_mtime))}


# --------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------- #
def _parse_query(query: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not query:
        return out
    for pair in query.split("&"):
        if not pair:
            continue
        k, _, v = pair.partition("=")
        out.setdefault(_unquote(k), []).append(_unquote(v))
    return out


def _unquote(s: str) -> str:
    return s.replace("+", " ").replace("%20", " ")


class Handler(BaseHTTPRequestHandler):
    server_version = "SSBotDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # The browser polls ~15x/s; echoing every request would drown the
        # console the operator actually reads.  Deliberately silent.
        return None

    # -- helpers ------------------------------------------------------- #
    def _send(self, code: int, body: bytes, ctype: str,
              extra: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode(),
                   "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode() or "{}")
        except Exception:
            return {}

    # -- routes -------------------------------------------------------- #
    def do_GET(self) -> None:                  # noqa: N802
                # stdlib http.server gives us the raw target; split it without
        # pulling in urllib so this file stays out of the core-offline scan.
        raw = self.path
        path, _, query = raw.partition("?")
        qs = _parse_query(query)
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/frame.jpg":
            data = DEMO.frame()
            if not data:
                self._send(404, b"", "image/jpeg")
            else:
                self._send(200, data, "image/jpeg")
        elif path == "/api/demo":
            self._json(DEMO.snapshot())
        elif path == "/api/train":
            self._json(TRAIN.snapshot(int((qs.get("tail") or ["120"])[0])))
        elif path == "/api/report":
            self._json(report_payload())
        elif path == "/api/profiles":
            self._json(profiles_payload())
        elif path == "/api/guide":
            self._json(guide_payload())
        elif path == "/api/config":
            try:
                self._json({"ok": True, "config": BotConfig().to_dict()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        elif path == "/api/health":
            self._json({"ok": True, "python": sys.version.split()[0],
                        "version": APP_VERSION,
                        "demo_running": DEMO.running,
                        "train_running": TRAIN.running})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:                 # noqa: N802
        path = self.path.partition("?")[0]
        body = self._body()
        if path == "/api/demo/start":
            self._json(DEMO.start(
                str(body.get("policy", "network")),
                str(body.get("profile", "strict_lite")),
                float(body.get("hz", 12)),
                int(body.get("seed", 0))))
        elif path == "/api/demo/stop":
            self._json(DEMO.stop())
        elif path == "/api/train/start":
            self._json(TRAIN.start(int(body.get("steps", 600)),
                                   str(body.get("profile", "strict_lite")),
                                   list(body.get("extra", []))))
        elif path == "/api/train/stop":
            self._json(TRAIN.stop())
        else:
            self._send(404, b"not found", "text/plain")


# --------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subway Surfers Research Bot — Dashboard</title>
<style>
:root{--bg:#0f1216;--panel:#171b21;--panel2:#1d232b;--fg:#e6e9ee;--mut:#8b95a5;
--acc:#4da3ff;--ok:#39d98a;--warn:#ffb020;--err:#ff5c5c;--line:#2a323d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line);
display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:var(--panel)}
h1{font-size:16px;margin:0;font-weight:600}
.sub{color:var(--mut);font-size:12px}
nav{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}
button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:7px 13px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--acc)}
button.pri{background:var(--acc);border-color:var(--acc);color:#06121f;font-weight:600}
button.dan{border-color:#5a2b2b;color:var(--err)}
main{padding:16px;max-width:1180px;margin:0 auto}
.tab{display:none}.tab.on{display:block}
.grid{display:grid;gap:14px;grid-template-columns:minmax(260px,420px) 1fr}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--mut);text-transform:uppercase;
letter-spacing:.06em;font-weight:600}
img.feed{width:100%;border-radius:8px;background:#000;display:block;
image-rendering:auto;min-height:200px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
label{color:var(--mut);font-size:12px}
select,input{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:6px 8px;font-size:13px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-variant-numeric:tabular-nums}
.kv b{color:var(--mut);font-weight:500}
pre{background:#0b0e12;border:1px solid var(--line);border-radius:8px;padding:10px;
overflow:auto;max-height:430px;font:11.5px/1.45 ui-monospace,Consolas,monospace;
white-space:pre-wrap;word-break:break-word;margin:0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;
letter-spacing:.05em}
td.n{font-variant-numeric:tabular-nums;text-align:right}
.ok{color:var(--ok)}.err{color:var(--err)}.warn{color:var(--warn)}.mut{color:var(--mut)}
.guide{display:grid;gap:10px}
.g{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px}
.g h3{margin:0 0 4px;font-size:14px}
.g code{background:#0b0e12;padding:2px 6px;border-radius:4px;font-size:12px;
color:var(--acc);display:inline-block;margin-top:5px;word-break:break-all}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);
color:var(--mut)}
.badge.win{border-color:#5a4a2b;color:var(--warn)}
.note{background:#141a20;border-left:3px solid var(--acc);padding:10px 12px;
border-radius:0 7px 7px 0;color:var(--mut);font-size:13px;margin:0 0 14px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--mut);display:inline-block}
.dot.on{background:var(--ok);box-shadow:0 0 7px var(--ok)}
</style></head><body>
<header>
  <h1>Subway Surfers Research Bot <span class="sub" id="ver"></span></h1>
  <span class="sub">screen-capture RL &middot; CPU-only PyTorch</span>
  <span class="sub"><span class="dot" id="dot"></span> <span id="dotT">idle</span></span>
  <nav>
    <button data-t="live" class="pri">Trực tiếp</button>
    <button data-t="train">Train</button>
    <button data-t="report">Báo cáo</button>
    <button data-t="profiles">Profile</button>
    <button data-t="guide">Hướng dẫn</button>
  </nav>
</header>
<main>

<section class="tab on" id="t-live">
  <div class="note">Đây là <b>GameEnvironment thật</b> (perception + horizon detector +
  death detector + reward) chạy trên <b>SyntheticGame</b>, điều khiển bởi
  <b>InferencePolicy thật</b>. Mạng chưa được train nên chế độ <i>network</i>
  cho thấy đường ống hoạt động, không phải người chơi giỏi. Muốn có agent giỏi:
  tab <b>Train</b> để chạy pipeline, hoặc train trên máy bạn.</div>
  <div class="grid">
    <div class="card">
      <h2>Camera</h2>
      <img class="feed" id="feed" alt="live feed">
    </div>
    <div>
      <div class="card">
        <h2>Điều khiển</h2>
        <div class="row">
          <label>Policy</label>
          <select id="pol"><option value="network">network (greedy)</option>
          <option value="random">random</option></select>
          <label>Profile</label>
          <select id="prof"></select>
          <label>fps</label>
          <input id="hz" type="number" value="12" min="1" max="60" style="width:64px">
          <label>seed</label>
          <input id="seed" type="number" value="0" style="width:70px">
        </div>
        <div class="row">
          <button class="pri" id="start">Start</button>
          <button class="dan" id="stop">Stop</button>
          <span class="mut" id="startMsg"></span>
        </div>
      </div>
      <div class="card" style="margin-top:14px">
        <h2>Số liệu</h2>
        <div class="kv" id="kv"></div>
      </div>
    </div>
  </div>
</section>

<section class="tab" id="t-train">
  <div class="note">Chạy <b>pipeline đa tiến trình thật</b>
  (<code>capture-worker</code> / <code>actor-worker</code> / <code>learner-worker</code>
  qua shared memory). <code>--dry-run</code> = không bấm phím thật. Kết quả ghi vào
  <code>runs/headless_report.json</code> (xem tab Report).</div>
  <div class="card">
    <h2>Chạy headless</h2>
    <div class="row">
      <label>steps</label>
      <input id="steps" type="number" value="600" min="50" max="20000" style="width:110px">
      <button class="pri" id="tstart">Start run</button>
      <button class="dan" id="tstop">Stop</button>
      <span class="mut" id="tmsg"></span>
    </div>
    <pre id="tlog">— chưa chạy —</pre>
  </div>
</section>

<section class="tab" id="t-report">
  <div class="card">
    <h2>runs/headless_report.json <span class="mut" id="rmeta"></span></h2>
    <div id="rbody" class="mut">đang tải…</div>
  </div>
</section>

<section class="tab" id="t-profiles">
  <div class="card"><h2>Model profiles (đo thật, không ước lượng)</h2>
  <div id="pbody" class="mut">đang tải…</div></div>
</section>

<section class="tab" id="t-guide">
  <div class="note">Toàn bộ các phần của project, và phần nào chạy được ở đâu.
  Chỉ <b>mục 7</b> cần desktop Windows thật và không thể preview trong trình duyệt.</div>
  <div class="guide" id="gbody"></div>
</section>

</main>
<script>
const $=id=>document.getElementById(id);
get('/api/health').then(h=>{ if(h.version) $('ver').textContent='v'+h.version; }).catch(()=>{});
const tabs=document.querySelectorAll('nav button');
tabs.forEach(b=>b.onclick=()=>{
  tabs.forEach(x=>x.classList.remove('pri'));b.classList.add('pri');
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  $('t-'+b.dataset.t).classList.add('on');
  if(b.dataset.t==='report')loadReport();
  if(b.dataset.t==='profiles')loadProfiles();
  if(b.dataset.t==='guide')loadGuide();
});
const ACT=['NOOP','LEFT','RIGHT','JUMP','SLIDE'];

async function post(u,b){const r=await fetch(u,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
  return r.json();}
async function get(u){const r=await fetch(u);return r.json();}

// ---- profiles dropdown ----
let PROFS=[];
(async()=>{try{PROFS=await get('/api/profiles');
  $('prof').innerHTML=PROFS.map(p=>`<option value="${p.name}">${p.name} (${p.params} p)</option>`).join('');
 }catch(e){}})();

// ---- live ----
let live=false;
$('start').onclick=async()=>{
  const r=await post('/api/demo/start',{policy:$('pol').value,
    profile:$('prof').value,hz:parseFloat($('hz').value)||12,
    seed:parseInt($('seed').value)||0});
  $('startMsg').textContent=r.ok?`started · ${r.params} params`:('lỗi: '+r.error);
  $('startMsg').className=r.ok?'ok':'err';
  if(r.ok){live=true;$('feed').src='/frame.jpg';tick();}
};
$('stop').onclick=async()=>{await post('/api/demo/stop');live=false;};

async function tick(){
  try{
    const s=await get('/api/demo');
    $('dot').className='dot'+(s.running?' on':'');
    $('dotT').textContent=s.running?'demo running':'idle';
    const rows=[['episode',s.episodes??0],['steps',s.steps??0],['score',s.score??0],
      ['action',ACT[s.last_action]??s.last_action],
      ['reward (last)',s.last_reward??0],['episode reward',s.episode_reward??0],
      ['alive',s.alive?'yes':'DEAD'],
      ['horizon',s.horizon?('detected '+(s.horizon_conf??'')):'quiet'],
      ['profile',s.profile??'—'],['policy',s.policy??'—'],['params',s.params??'—']];
    $('kv').innerHTML=rows.map(([k,v])=>
      `<b>${k}</b><span class="${k==='alive'&&v==='DEAD'?'err':''}">${v}</span>`).join('');
    if(s.last_error)$('startMsg').textContent='lỗi: '+s.last_error;
  }catch(e){}
}
setInterval(()=>{if($('feed').src&&live)$('feed').src='/frame.jpg?t='+Date.now();},90);
setInterval(tick,700);

// ---- train ----
$('tstart').onclick=async()=>{
  const r=await post('/api/train/start',{steps:parseInt($('steps').value)||600});
  $('tmsg').textContent=r.ok?('running: '+r.cmd):('lỗi: '+r.error);
  $('tmsg').className=r.ok?'mut':'err';
};
$('tstop').onclick=async()=>{await post('/api/train/stop');};
setInterval(async()=>{
  if(!$('t-train').classList.contains('on'))return;
  const s=await get('/api/train?tail=200');
  $('dot').className='dot'+(s.running?' on':'');
  $('dotT').textContent=s.running?('training '+s.elapsed_s+'s'):(s.exit_code!==null?'finished':'idle');
  $('tlog').textContent=(s.lines||[]).join('\n')||'— chưa chạy —';
  $('tlog').scrollTop=$('tlog').scrollHeight;
},1200);

// ---- report ----
async function loadReport(){
  const r=await get('/api/report');
  if(!r.exists){$('rbody').innerHTML='<span class="mut">Chưa có report. Chạy tab Train trước.</span>';return;}
  $('rmeta').textContent='· bản '+(r.data.app_version||'?')+' · cập nhật '+r.mtime;
  const d=r.data,s=d.summary||{},recs=d.records||[];
  let h=`<p><b>Verdict:</b> <span class="warn">${d.verdict||'—'}</span></p>`;
  for(const kind of Object.keys(s)){
    if(typeof s[kind]!=='object'||!s[kind])continue;
    const g=s[kind];
    h+=`<h3 style="font-size:13px;margin:12px 0 6px">${kind}</h3><table><tr>
      <th>metric</th><th>n</th><th>mean</th><th>median</th><th>std</th><th>min</th>
      <th>max</th><th>ci95</th><th>best</th></tr>`;
    for(const m of Object.keys(g)){
      const v=g[m];if(!v||typeof v!=='object')continue;
      h+=`<tr><td>${m}</td><td class="n">${v.n}</td><td class="n">${v.mean}</td>
        <td class="n">${v.median}</td><td class="n">${v.std}</td>
        <td class="n">${v.min}</td><td class="n">${v.max}</td>
        <td class="n">${v.ci95}</td><td class="n">${v.best_single_run}</td></tr>`;
    }
    h+='</table>';
  }
  if(recs.length){
    h+='<h3 style="font-size:13px;margin:14px 0 6px">Episodes ('+recs.length+')</h3><table><tr>'+
      ['id','kind','survival_s','reward','steps','fps','act_p95_ms','inf_p95_ms','score']
      .map(c=>`<th>${c}</th>`).join('')+'</tr>';
    for(const e of recs.slice(-40))
      h+=`<tr><td class="n">${e.episode_id}</td><td>${e.kind}</td>
        <td class="n">${e.survival_s}</td><td class="n">${e.total_reward}</td>
        <td class="n">${e.steps}</td><td class="n">${e.fps}</td>
        <td class="n">${e.action_latency_p95_ms}</td>
        <td class="n">${e.inference_p95_ms}</td><td class="n">${e.score}</td></tr>`;
    h+='</table>';
  }
  $('rbody').innerHTML=h;
}

// ---- profiles ----
async function loadProfiles(){
  const ps=await get('/api/profiles');
  let h='<table><tr><th>profile</th><th>kind</th><th>params</th><th>budget</th>'+
        '<th>head</th><th>within budget</th></tr>';
  for(const p of ps)
    h+=`<tr><td><b>${p.name}</b></td><td>${p.kind}</td><td class="n">${p.params}</td>
      <td class="n">${p.budget||'—'}</td><td class="n">${p.head_hidden}</td>
      <td class="${p.within_budget?'ok':'err'}">${p.within_budget?'yes':'NO'}</td></tr>
      <tr><td colspan="6" class="mut" style="font-size:12px">${p.description}</td></tr>`;
  $('pbody').innerHTML=h+'</table>';
}

// ---- guide ----
async function loadGuide(){
  const g=await get('/api/guide');
  $('gbody').innerHTML=g.map(x=>`<div class="g"><h3>${x.title}
    <span class="badge ${/Windows/.test(x.where)?'win':''}">${x.where}</span></h3>
    <div class="mut">${x.detail}</div><code>${x.how}</code></div>`).join('');
}
tick();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"dashboard on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        # Ctrl-C: fall through to the finally block, which stops the live
        # demo, the training subprocess and the HTTP server.
        print("\ndashboard shutting down (Ctrl-C)", flush=True)
    finally:
        DEMO.stop()
        TRAIN.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
