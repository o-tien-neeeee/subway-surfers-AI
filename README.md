# Subway Surfers Research Bot (screen-capture RL agent)

> **Current version / Phiên bản hiện tại: `1.15.0`** — nguồn: `version.py` → `APP_VERSION`; lịch sử đầy đủ trong `DEEP_FIX_REPORT.md`. (README này được test buộc phải khớp `APP_VERSION` mỗi phiên.)

A personal, **black-box** UI-automation research bot that plays
[Poki Subway Surfers](https://poki.com/en/g/subway-surfers) using only screen
capture and normal keyboard/mouse input. No DOM access, no JavaScript
injection, no devtools, no memory reading, no network interception, no
anti-cheat tampering, no hidden game state.

Core stack: Python 3.11, PyTorch **CPU-only**, mss, pynput/pyautogui, Tkinter.
Target machine: Windows 64-bit, i5-7200U (2C/4T), 12 GB RAM, Intel HD 620,
Chrome.

> **Honesty first.** Everything in this repo runs and is unit-tested headless
> (449 tests, see §Testing). The bot has **not yet been run against the real
> Poki game** — no real-game benchmark numbers exist yet, so every real-game
> metric is labelled *not yet measured*. Nothing here is claimed to be
> "superhuman", "frame-perfect", or "production-ready"; those labels require
> the evaluation protocol in §Evaluation on the target machine.

---

## 0. What's new in this upgrade (v1.1)

Continuing from the first complete build, this round closes the remaining
specification gaps and hardens four subsystems — all verified by new tests
(340 → 403 → 449 after the deep-fix pass):

1. **Online best-model gating (§12).** `best_model.pth` used to be written
   only by behaviour cloning. Now the actor reports every finished episode
   (`survival_s`, `total_reward`) through shared counters and the learner
   updates `best_model.pth` whenever the **rolling mean over
   `rl.best_metric_window` (default 3) episodes** improves. The best value is
   persisted inside the checkpoint so a restart can never overwrite a
   historical best with a worse run. (`tests/test_best_model_tracking.py`)
2. **Capture geometry-drift detection (§6).** A resolution/DPI change or a
   window move silently shifts every calibrated coordinate. The capture
   process now re-checks the virtual screen against the calibrated reference
   every 5 s and warns (once per change) instead of failing silently.
   (`tests/test_capture_geometry.py`)
3. **Confidence-aware action cadence (§9).** The decision cadence now adapts
   to CPU load **and** horizon-detector confidence together: a brewing hazard
   (confidence ≥ `scheduler.fast_confidence`, default 0.75) selects the fast
   bound even under high load — safety outranks CPU thrift.
   (`tests/test_action_scheduler.py::TestConfidenceAwareCadence`)
4. **Statistical evaluation upgrade (§15).** Comparisons now include a
   tie-corrected **Mann-Whitney U** p-value and **bootstrap CIs** alongside
   the CI-overlap rule, plus an **adaptive target** derived from the measured
   baseline (baseline mean + 1σ, flagged provisional below 20 episodes).
   `python app.py --evaluate N --compare-baseline <report.json>` imports a
   saved report as the baseline set. (`tests/test_evaluation_stats.py`)

Also fixed on the way: the demo recorder's atomic `.npz` write silently
never created its temp file (numpy appends `.npz` to str filenames — now
writes through a file object, covered by tests); the demo recorder and the
actor no longer discard frames when the death anchor is not calibrated
(`ZonePreprocessor(require_anchor=False)`); demo episodes now embed absolute
screen geometry + keymap + calibration metadata.

---

## 1. Feasibility audit (pre-coding, requirement §3)

| # | Question | Audit finding | Engineering decision |
|---|----------|--------------|----------------------|
| A1 | Classic Atari CNN (Conv×3 → flatten → 512-dense) parameter count? | The flatten head alone is 7·7·64·512 ≈ **1.61 M params** (plus 3136·512 multiplies per forward pass) — dwarfing the convs and blowing both the 80 k StrictLite budget and CPU latency budget. | Rejected. Encoder → **global average pooling** → small dueling heads; depthwise-separable convs for StrictLite. Measured: 49 k / 96 k / 348 k params (§4). |
| A2 | BatchNorm with small online batches + replay training? | Unstable here: (a) replay batches are drawn from an old distribution while online data shifts; (b) **actor and learner are different processes** — inference uses running stats that the training process is updating, so identical weights give different features; (c) batch=32 online stats are noisy. | **GroupNorm(8)** everywhere (batch-size independent, identical train/eval behaviour). Enforced by `tests/test_models.py::test_no_batchnorm_anywhere`. |
| A3 | Inference every 4 frames vs "near-frame-perfect" play? | At 30 FPS, a 4-frame cadence is a **133 ms reaction floor** — Subway Surfers dodges routinely need < 300 ms, so 4 frames is viable but marginal; 1-frame decisions are impossible for this CPU while also training. | Adaptive cadence: **2–4 frames normal (load-dependent), 1 frame in danger** (horizon detector fires). Measured, not assumed: p50/p95 inference 1.3/2.0 ms (§4). |
| A4 | Training in a separate *thread* vs the Python GIL? | A learner **thread** would serialise torch backward passes with frame processing under the GIL — capture stalls guaranteed. | Learner is a separate **process** (`spawn`); weights flow through shared memory to an actor-local copy (version-guarded, ≤ 1.4 MB memcpy). |
| A5 | Replay-buffer persistence vs 12 GB RAM? | Naive storage (4×84×84 uint8 obs + next-obs per transition) ≈ 56 KB × 30 000 = **1.7 GB**, plus torch + Chrome — too close to the 4 GB working-set limit. | **Lazy frame store**: each unique 84×84 frame stored once (~7 KB), transitions reference frame ids; eviction-validated at sample time. ≈ 210–260 MB at 30 k transitions. Config warns above 1 GB (`BotConfig._replay_ram_estimate_gb`). |
| A6 | Raw pixel-difference rewards → reward hacking? | Yes: flashing score popups, coin sparkles and camera shake mint unlimited "novelty" reward; a policy can learn to *die spectacularly* for pixels. | Pixel-diff reward is **OFF by default**, tightly clipped (±0.01), normalised, and behind an ablation switch; a dedicated reward-hacking test (`test_reward_logic.py::TestPixelDiffAblation::test_reward_hacking_resistance`) proves bounded payout. Survival-time reward is primary. |
| A7 | Single colour anchor → false deaths? | Yes: hover highlights, score flashes, or a 1-frame animation over the anchor pixel cause false respawns (lost progress, stuck clicking). | 5×5 patch **median** RGB (outlier-robust), Euclidean distance vs calibrated baseline, **temporal debounce ladder** (POSSIBLE_EVENT → DEAD_CANDIDATE → DEAD_CONFIRMED at ≥3 consecutive frames), stability window (10 frames) before ALIVE resumes, and per-frame logging of raw distance + reason. Unit-tested with flash/noise cases. |
| A8 | Input backend reliability on Windows Chrome? | `pyautogui` key events go through SendInput — reliable, but its per-call `PAUSE` (default 0.1 s) would blow the latency budget, and stuck keys scroll pages. | Keys via **pynput** (press/release explicit, no auto-pause), mouse clicks via pyautogui with `FAILSAFE=True`, `PAUSE=0.01`. Guaranteed release: scheduled release + guardian thread + watchdog force-release. Dry-run backend for tests/CI. |
| A9 | Responsive browser geometry + DPI scaling? | Fixed absolute coordinates break when the window moves, resolution/DPI changes, or the page relayouts. | Calibration stores **fractions** of the virtual screen (plus DPI scale); region is revalidated (size caps, min 240×320); DPI-aware process on Windows; every geometry change is *detected and reported*, never silently ignored. |

### Contradictions found and resolved

1. **"Frame-perfect" vs 4-frame skip** — reframed as *measured latency targets*
   (p95 < 100 ms action latency, 20–30 FPS effective) enforced by the
   auto-downgrader, instead of unverifiable perfection claims.
2. **80 k parameter limit vs "maximum learning performance"** — kept as the
   *StrictLite* profile guarantee; BalancedCPU/QualityCPU exceed it **only
   when profiling passes** (§4). Default profile is StrictLite.
3. **"Inference every 4 frames" vs "decide every frame in danger"** — both:
   normal state 2–4 frames, danger state 1 frame (§Adaptive timing).
4. **Threading for the learner vs GIL** — process instead of thread (A4).
5. **Pixel-diff reward vs reward hacking** — disabled by default, clipped,
   ablation-gated (A6).
6. **Fast respawn vs safety** — respawn clicks are capped (interval 0.8 s,
   timeout 15 s → ERROR/PAUSED, never infinite clicking).
7. **OCR score reward** — not implemented as a reward: unvalidated OCR is a
   noise source. Score stays evaluation-only until validated with confidence
   thresholds (hook in `evaluation.py`).

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ GUI / main process (Tkinter)                                               │
│  calibration wizard (6 steps) · state machine · metrics panel · logs       │
│  F8 hotkey listener · preview grabber · demo recorder                      │
│  NEVER blocks on training; stays alive if any worker crashes               │
└───────┬──────────────────────────────┬─────────────────────────────────────┘
        │ spawn                        │ spawn
┌───────▼──────────┐          ┌────────▼──────────┐          ┌────────────────┐
│ capture process  │  ring    │ actor process     │ trans-Q  │ learner process│
│ mss / Synthetic  │─────────▶│ perception (zones,│─────────▶│ Double DQN +   │
│ 30 FPS paced,    │ shared   │ frame stack)      │ bounded  │ PER + Adam +   │
│ latest-wins,     │ memory   │ horizon detector  │ 128      │ checkpoints    │
│ drop-on-lap      │          │ death detector    │          │ BC pretraining │
└──────────────────┘          │ action scheduler  │◀─────────│ weight publish │
                              │ InferencePolicy   │ weights  │ (shared memory │
                              │ InputController   │ shared   │  + version)    │
                              │ (pynput/dry-run)  │          │                │
                              │ SafetyWatchdog    │◀─pause───│ (paused on     │
                              │ (thread, priority)│  event   │  death/respawn)│
                              └────────┬──────────┘          └────────────────┘
                                       │ keys/clicks (only exit to the OS)
                                  ▼ Chrome (Poki page)
Shared primitives (ipc.py): SharedFrameRing (bounded, lock-free),
SharedWeights (flat fp32 + version), SharedCounters, bounded queues,
events {stop, emergency, pause, pause_learning, death}.
```

**Synchronisation decisions**

* Frame transport: fixed-slot shared-memory ring, single writer (capture),
  single reader (actor), per-slot generation counters for tear-free reads;
  a slow reader drops stale frames *by construction* — memory is constant.
* Weights: learner is the **only writer**; actor copies out on version change
  (lock-free, retry-on-torn-copy).
* Counters: single-writer fields (`env_frame_id`, `action_step`, `episode_id`
  by the actor; `learner_update_step`, `beta`, `td_loss` by the learner).
* Events: `stop` (shutdown), `emergency` (F8), `pause` (gameplay+learner),
  `pause_learning` (learner only, during death/respawn), `death`.
* All queues are bounded; overflow drops and logs — nothing can grow RAM.

---

## 3. Repository layout

```
subway-surfers-AI/
├── app.py                    # entry point, process orchestrator, CLI
├── config.py                 # typed config tree + validation + JSON I/O
├── gui.py                    # Tkinter wizard (6 steps) + live metrics
├── capture_worker.py         # mss/synthetic capture process
├── input_controller.py       # safe keys/clicks, guardian, dry-run
├── safety_watchdog.py        # watchdog thread + F8 hotkey
├── environment.py            # actor loop, synthetic game, sync env
├── perception.py             # zones, anchor patch, frame stack
├── death_detector.py         # colour anchor + debounce + respawn ctl
├── horizon_detector.py       # frame-diff hazard detector
├── action_scheduler.py       # adaptive cadence, buffering, TTL
├── replay_buffer.py          # PER sum-tree, lazy frames, n-step, atomic IO
├── dataset.py                # demo validation, episode split, BC dataset
├── demonstration_recorder.py # human demo recording (npz)
├── models.py                 # StrictLite/BalancedCPU/QualityCPU nets
├── agent.py                  # Double DQN, BC epoch, InferencePolicy
├── learner_worker.py         # learner process (BC + online RL)
├── checkpoint_manager.py     # atomic best/latest/buffer + RNG states
├── metrics.py                # percentiles, FPS, counters, psutil
├── profiling.py              # params/FLOPs/latency/activation profiler
├── logging_utils.py          # logging, bounded queues, pickle integrity
├── ipc.py                    # shared ring/weights/counters/events
├── states.py                 # 12-state lifecycle machine
├── evaluation.py             # honest eval protocol + reports
├── evaluation_tool.py        # headless evaluation runner
├── requirements.txt · config.example.json · pytest.ini
└── tests/                    # 16 test files, 449 tests
```

---

## 4. Model profiles (exact, measured)

Parameters/FLOPs/latency measured with `python profiling.py`
(CPU, `torch.set_num_threads(1)`).

| profile | params (exact) | FLOPs (MACs) | act. mem (train) | inference p50/p95 | update @ batch 32 |
|---|---:|---:|---:|---:|---:|
| `strict_lite` | **49 154** (≤ 80 000 ✓) | 2 627 616 | 2.06 MB | 0.75 / 1.41 ms | 33.8 ms |
| `balanced_cpu` | **95 558** | 14 018 304 | 0.52 MB | 0.46 / 0.53 ms | 23.8 ms |
| `quality_cpu` | **348 134** | 23 603 200 | 0.73 MB | 0.73 / 0.89 ms | 36.7 ms |

Design per profile (`models.py`):
* **StrictLite** — 4 depthwise-separable blocks (32/48/64/128, stride 2/2/2/1)
  → GroupNorm → GAP → dueling heads (hidden 128).
* **BalancedCPU** — 3 convs (8×8 s4, 4×4 s2, 3×3) + GroupNorm → GAP → dueling
  heads (hidden 128).
* **QualityCPU** — 4 convs (8×8 s4 … 3×3) + GroupNorm → GAP → dueling heads
  (hidden 256). Enable only if `profiling.py` passes on the target machine.

> Latency numbers were measured on the **CI sandbox CPU, not the i5-7200U**.
> Re-run `python profiling.py` on the target machine before choosing a
> non-default profile; the auto-downgrader enforces the budget either way.
> (Curiosity worth knowing: depthwise-separable StrictLite is *not* faster
> than the dense BalancedCPU convs here — oneDNN optimises dense convolutions
  better — StrictLite wins on parameter/RAM guarantees, not raw speed.)

RL core: **Double DQN** + dueling heads + n-step returns (default 3) +
**Prioritized Experience Replay** (α=0.6, β 0.4→1.0), Huber loss, grad-clip
10, Adam 1e-4, hard target sync every 1000 updates, ε-greedy 1.0→0.05
linear over **150 000 env frames** (the word "step" is defined once:
ε decays on env frames observed by the actor, nothing else).

---

## 5. Windows setup

```powershell
git clone <this repo>; cd subway-surfers-AI
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only wheel
python -m compileall .          # must exit clean
python -m pytest -q             # 449 passed (headless, no display needed)
python profiling.py             # re-measure on THIS machine
python app.py                   # GUI
```

Requirements: Windows 10/11 64-bit, Chrome, Python 3.10–3.12. No CUDA.
The bot is offline after install (verified by
`tests/test_hygiene.py::test_no_network_calls_in_core`).

## 6. Calibration (the six GUI steps)

1. **Select region** — click-drag a rectangle over the game only (no browser
   UI). Live size/coords/DPI shown; Enter accepts, Esc cancels, R re-selects.
2. **Lock region** — live preview; must be ≥ 240×320 and under the pixel cap;
   stored as screen fractions + DPI scale. Re-select if geometry changed.
3. **Colour anchor** — click a stable UI element that is visible while ALIVE
   and disappears/changes on game-over. "Calibrate ALIVE (2 s)" samples the
   5×5 patch and accepts only if the per-channel std ≤ 6 (stability score
   shown). "Capture DEAD sample" can be taken later on the game-over screen.
4. **Respawn click** — click the restart button; stored relative to the
   region. "Test click" asks for confirmation first (never auto-clicks).
5. **Horizon** — slider 15–40 % (default 25 %), preview shows the split line.
6. **Train** — Start/Pause/Stop, ● Record demo, BC pretrain, ⛔ F8 emergency.
   Live: state, survival, episode, FPS/effective FPS, dropped frames,
   inference & action p95, ε, avg reward, TD loss, mean Q, buffer size,
   CPU/RAM, model profile.

Headless equivalents: `python app.py --headless --steps 600` (synthetic game,
dry-run input, same three-process pipeline).

## 7. Demonstrations & datasets

* Record: GUI → Step 6 → **● Record demo** (F9 stop). Saves
  `demos/episode_*.npz`: 84×84 uint8 ground frames, actions, timestamps,
  done, optional score/confidence/death-state + browser-geometry metadata.
  No duplicated frame stacks are stored.
* Validate: `python app.py --validate-demos demos` — missing-frame gaps,
  invalid actions, timestamp order, episode boundaries, class imbalance,
  and an **episode-level** train/val split (no frame leakage).
* Pretrain (BC): `python app.py --pretrain demos` or the GUI button
  (class-balanced cross-entropy on the advantage logits, DQfD-style).
  If fewer than 2 valid episodes exist, BC is **refused with a message** and
  the pipeline falls back to online RL with a warm-up phase — it never
  pretends pretraining happened.

## 8. Test commands

```bash
python -m compileall .     # clean compile of every file
python -m pytest -q        # 449 tests
python app.py --headless --steps 600       # end-to-end smoke (fake game)
python app.py --evaluate 5                 # headless eval + honest report
```

Test coverage highlights: horizon detection & debounce; colour-anchor death
ladder + false-positive flashes + respawn machine (timeout → FAILED, never
clicks forever); reward clipping + pending-hazard logic + no-future-leak +
reward-hacking bounds; action cadence/danger/expiry/duplicates + confidence-
aware cadence; key release after backend exceptions + guardian force-release
+ focus gate; PER maths, NaN protection, eviction validation, persistence +
corruption recovery; checkpoint atomicity, best-model policy, RNG-state round
trip, online best-model gating + restart protection; Mann-Whitney/bootstrap/
adaptive-target statistics; capture geometry drift; recorder metadata +
atomic npz writes; lifecycle state machine; process startup/emergency-stop/
idempotent shutdown; GUI-survives-worker-crash; stale-frame dropping;
headless multi-process training pipeline; no bare excepts / silent passes /
stubs / network calls.

## 9. Profiling commands

```bash
python profiling.py                 # params/FLOPs/activation/latency + updates
python app.py --headless --steps 900 --report runs/headless_report.json
python app.py --evaluate 20         # per-episode stats + system profile
```

## 10. Evaluation protocol & claims policy

1. Record a **baseline** (human episodes via the demo recorder, or scripted
   baseline headless via `evaluation_tool.record_human_baseline_headless`).
2. Evaluate the trained policy on **separate** episodes (ε fixed 0.05,
   learning paused), 20–50 episodes where practical.
3. Reports (JSON + Markdown under `runs/`) contain mean/median/std/min/max/
   95 % CI per metric, failure modes, and a **verdict**. The verdict says
   "beats baseline" only when the 95 % CIs do not overlap **and** the
   tie-corrected Mann-Whitney U p-value is < 0.05; suggestive-but-unproven
   and not-improved outcomes are labelled as such.
4. An **adaptive target** is derived from the measured baseline
   (baseline mean + 1σ) and marked *provisional* until the baseline has
   ≥ 20 episodes — no invented fixed targets.
5. Compare a saved baseline report against a fresh evaluation with
   `python app.py --evaluate 20 --compare-baseline runs/human_baseline.json`.
6. Training performance, evaluation performance, human baseline, best single
   run and typical run are reported as separate numbers; "superhuman" is
   never printed as a claim, only as an explicitly unmet requirement.

## 11. Known limitations

* **No real-game measurements yet.** Detector thresholds (horizon diff,
  death colour distance) are principled defaults that MUST be re-calibrated
  via the wizard on the real game; their real-game precision/recall is
  pending.
* Score OCR is not implemented (score = 0 / "not detectable" in the GUI)
  until validated — score is evaluation-only anyway.
* Chrome focus detection depends on window-title matching
  (`input.browser_title_hint`); verify on your setup.
* The synthetic game is a *machinery test*, not a Subway Surfers simulator:
  passing it proves the pipeline runs, nothing about real-game skill.
* Windows-specific paths (DPI awareness, pyautogui failsafe) are exercised
  best-effort on Linux CI; run the suite on the target machine before play.
* `quality_cpu` default-off; measured numbers here are from a faster CI CPU
  than the i5-7200U — always re-profile on the target machine.
* Latency budget (p95 < 100 ms) is enforced by auto-downgrade; if even
  StrictLite can't hold 20 FPS, the bot logs a loud violation rather than
  silently degrading further.

## 12. Truthful checklist

**Verified by code + tests (CI, headless)**

- [x] CPU-only torch (`device="cpu"` asserted; threads pinned to 1)
- [x] StrictLite ≤ 80 000 params (49 154, exact-count test)
- [x] GroupNorm, no BatchNorm (test-enforced)
- [x] Parameter/FLOPs/activation/latency profiling per profile
- [x] Bounded frame transport, stale-frame dropping, drop counters
- [x] Horizon detector (score/detection/confidence/ts/frame-id + debounce)
- [x] Death detector ladder + flash false-positive immunity + respawn
      machine with bounded clicking and unit-tested state transitions
- [x] Five discrete actions; configurable keys; TTL, cooldown, duplicate
      suppression, buffering; guaranteed key release (guardian + watchdog +
      exception paths); dry-run mode; focus gate
- [x] Adaptive cadence 2–4 frames / 1 in danger, from CPU load AND horizon
      confidence; separate counters
- [x] PER (α/β, IS weights, sum-tree, NaN guards, eviction validation),
      n-step returns, lazy uint8 frame storage, atomic persistence with
      sha256 + `.corrupt` quarantine
- [x] Survival reward (time-proportional), −10 death, +0.1 pending hazard
      (no future leak), clipping [−10, 1], pixel-diff ablation OFF +
      hacking test
- [x] Double DQN + dueling + target net + Huber + grad clip + ε schedule
      defined on env frames
- [x] BC pretraining with episode-level split, class weighting, per-action
      accuracy; refuses insufficient data instead of faking it
- [x] Checkpoints (best/latest/buffer) atomic + RNG states + git/config hash;
      best_model.pth gated online by the rolling episode metric and
      protected against regression across restarts
- [x] Capture-side geometry-drift detection (resolution/DPI change → warning)
- [x] Evaluation statistics: CI overlap + Mann-Whitney U + bootstrap CI +
      adaptive baseline-derived target; `--compare-baseline` merging
- [x] Lifecycle: 12 states, emergency stop, idempotent shutdown, GUI survives
      worker crashes, no stuck keys on any shutdown path
- [x] No bare excepts / `except Exception: pass` / TODO stubs / network
      calls (AST-enforced tests)
- [x] 449 automated tests green; `compileall` clean

**Requires the real Poki game on the target machine (pending)**

- [ ] Real-region capture quality, FPS and p95 action latency on the i5-7200U
- [ ] Horizon-detector precision/recall on real footage (threshold tuning)
- [ ] Colour-anchor death detection on the real game-over screen
      (including which UI element is truly stable)
- [ ] Respawn click reliability on the real button
- [ ] Chrome focus detection on the user's window setup
- [ ] Whether BC demos actually transfer to the real game
- [ ] Baseline vs trained evaluation, ablations, any performance claim
      (including "beats baseline", "superhuman")
- [ ] Long-run stability (hours), real death/respawn statistics

---

## Legal / ethical notes

Personal research only. Poki's Subway Surfers is embedded from SYBO; this
project does not modify the game, its traffic, or its internals — it watches
the screen and presses keys like a human. Respect the site's terms of
service; do not use this on leaderboards or competitive services.
