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
APP_VERSION = "1.24.0"

#: Human-readable notes for the current build, newest first.  Kept short on
#: purpose — the full reasoning lives in DEEP_FIX_REPORT.md.
CHANGELOG = [
    "1.24.0 — DEEP FIX MDP: 7 root cause khiến 'train 6000 episode chỉ +1s'. (1) **perception.py**: policy từng chỉ thấy 75% DƯỚI màn hình (cắt mất chân trời nơi vật cản xuất hiện) → giờ `policy_full_frame=True` cho CNN thấy TOÀN khung 84×84; dải horizon chỉ còn nuôi HorizonDetector. (2) **environment.py**: Atari-style frame-skip MDP (`rl.frame_skip=3`) — 1 quyết định/3 frame, action được giữ, reward cộng dồn thành MỘT transition, stack/n_step/gamma tính theo bước quyết định (tầm nhìn ~0.4s lịch sử thay vì 0.13s). (3) **rewards.py**: reward thành BỘ ĐẾM DƯƠNG (`death_penalty=0`, `reward_clip_min=0`) theo đúng công thức đã chứng minh của các bot DQN chơi được Subway Surfers; `hazard_bonus` (thưởng cho MỌI cú bấm → spam jumping/rolling) TẮT mặc định. (4) **obstacle_perception.py** (mới): lưới occupancy 3 làn × 5 mức xa-gần (CV thuần <1ms) sinh `clear_bonus` NHÂN QUẢ — chỉ thưởng khi vật cản thật sự đi qua làn người chơi mà không chết; state machine theo LÀN vì vật cản di chuyển. (5) **agent.py**: `epsilon_for_step()` — ε decay theo số QUYẾT ĐỊNH (`epsilon_decay_steps=100k`) thay vì frame môi trường. (6) **environment.py**: transition ship với obs TẠI THỜI ĐIỂM QUYẾT ĐỊNH (`_step_obs`), sửa lệch một bước trong credit assignment. (7) **demonstration_recorder.py**: `bc.label_backdate_ms=220` lùi nhãn cú bấm về frame người chơi NHÌN THẤY vật cản (trước đây dạy bot phản ứng muộn 200-300ms) + **HG-DAgger** (F10 khi bot chơi → ghi correction demo vào demos/dagger/, dataset gom đệ quy) để diệt compounding error. **Kèm theo**: config.example.json regenerate (bản cũ chứa đúng reward gây plateau: death_penalty=-10, alive_per_frame=0.02); dataset validate theo `policy_size` thay vì hardcode 84. **Tests**: 1125/1125 PASS (40 test mới trong tests/test_deep_research_fixes.py). Báo cáo: DEEP_RESEARCH_vi.md.",
    "1.23.0 — IBRL + SIL + EMA + AUTO-ENTROPY (Massive AI upgrade): 5 substantive breakthroughs từ 2024-2025 SOTA papers. (1) **sil.py** (mới, ~290 lines): Self-Imitation Learning (Oh et al. 2018) — replay good episodes weighted by (R-V(s))_+ advantage. SILBuffer + SILTrainer. 12 tests. (2) **ibrl.py** (mới, ~390 lines): Imitation Bootstrapped RL (Hu et al. ICLR 2024) — actor-proposal (BC + RL both propose; pick higher-Q) + bootstrap-proposal (max(Q(a_il), Q(a_rl)) cho TD target). 6.4× SOTA improvement trên hard exploration. 10 tests. (3) **ibrl_agent.py** (mới): IBRLDQNAgent — full integration of all 4 v1.23.0 modules + IBRL. 6 tests. (4) **dqfd_v2_agent.py** (mới, ~330 lines): DQfDv2Agent = DQfDAgent + SIL + EMA + auto-entropy + IBRL bootstrap. **v1.23.0 production agent trên SyntheticGame**. 4 tests. (5) **ema.py** (mới, ~120 lines): EMA of network weights cho evaluation (TD7 / Tarasov 2024 trick). 5 tests. (6) **avg_l1_norm.py** (mới, ~110 lines): TD7's AvgL1Norm — divide by mean(|x|) để bound Q-value scale. 6 tests. (7) **auto_entropy.py** (mới, ~140 lines): SAC-style auto-tuned entropy temperature (Haarnoja 2018) cho discrete actions. 5 tests. **Total**: 7 new modules, +109 new tests, 1072/1072 PASS in 2:20. **KPI re-verification**: audit_dqfd_v2.py → **30.00s stable 200 ep** (training + final EMA eval). v1.21.0 audit_bc_then_rl.py cũng vẫn pass 30s (no regression). **Architectural insight**: 1 shared network giữa BC + RL bị 'coupling' (BC muốn pin policy, RL muốn explore) — IBRL tách hẳn BC (frozen) và RL (trainable), chọn action tốt hơn ở runtime. SIL replay good episodes để agent 'biết' cái gì tốt.",
    "1.22.0 — DEEP FIX: critical DQfD backprop bug + 5 substantive issues từ senior-engineer audit. (1) **CRITICAL**: train_step của DQfDAgent cũ gọi super().train_step (đã làm optimizer.step) rồi mới thêm BC+margin backward → gradient bị ACCUMULATE, TD loss bị apply 2 lần, BC anchor thực sự KHÔNG được optimize. Fix: single forward/backward/step tổng loss = L_Q + λ·L_BC + λ_m·L_margin. (2) **disable_exploration_after_bc=True** (mới, mặc định): audit_bc_then_rl.py chứng minh ε=0.15 sau BC vẫn phá policy (30s → 14.6s). Giờ ε=0 hoàn toàn sau BC. (3) **bc_pretrain.py** mới: orchestrator build_dqfd_agent + pretrain_and_arm_dqfd + prefill_replay_with_demos (priority boost 100x để demo luôn được sample). (4) **expert_synthetic.py** mới: optimal policy cho SyntheticGame (mean 22.22s trên 30 seeds, max 30s) — 7-dim state Markovian đúng, có pre-emptive dodge + best-of-bad-lanes. (5) **learner_worker.py** tích hợp: khi cfg.bc.bc_pretrain=True, build DQfDAgent thay vì QR-DQN, gọi _pretrain_dqfd() với demo pre-fill. **Honest finding**: BC+DQfD trên SyntheticGame (visual 4×84×84) đạt ~12s mean (vs expert 22s) — visual encoder có quá nhiều params cho 5 demos. Recipe ĐÃ CHỨNG MINH trên LearnableEnv (30.00s stable 200ep) nhưng visual domain cần thêm dữ liệu. **New files**: dqfd_agent.py (production DQfD), bc_pretrain.py (orchestrator), expert_synthetic.py (SyntheticGame expert), audit_dqfd_integration.py (production audit), audit_synthetic_dqfd.py (SyntheticGame honest test), 5 test files (test_dqfd +3, test_bc_pretrain 6, test_expert_synthetic 9, test_disable_exploration 4, test_learner_dqfd_integration 4, test_bc_synthetic_audit 1). **Test counts**: 874/874 pass (1 slow).",
    "1.19.0 — RAINBOW CORE + RND CURIOSITY: deep research xác định 3 thành phần Rainbow có impact lớn nhất (multi-step learning, PER, distributional Q) — distributional đã có ở v1.18. v1.19 thêm: (1) NoisyNets thay ε-greedy (parameter-space noise, exploration tốt hơn trên long horizon), (2) n_step 3→5 (faster credit assignment, Rainbow paper: 'one of the two most crucial components'), (3) RND (Random Network Distillation) thêm intrinsic reward cho state novelty (Burda 2018 — giải quyết sparse-reward exploration). Default ON: distributional=True, noisy_nets=True, use_rnd=True, n_step=5. Code: noisy_nets.py (NoisyLinear factorised Gaussian), rnd.py (target+predictor pair, EMA normaliser), agent_distributional.py tích hợp attach_rnd + intrinsic reward injection. 33 tests mới (test_noisy_nets: 8, test_rnd: 8, RND integration: 2 + 15 existing). 750/750 tests pass.",
    "1.18.0 — DISTRIBUTIONAL Q (QR-DQN, 51 quantiles) + DENSE REWARD SHAPING: audit_pipeline.py chỉ ra 10 root cause khiến AI không học được (reward thưa, encoder quá nhỏ, death-polluted buffer, etc). 3 fix lớn: (1) alive_per_frame 0.02→0.5 (gradient ×25), death_penalty -10→-5 (bớt catastrophic), hazard_bonus 0.1→1.0. (2) SyntheticGame.step_with_reward thêm proximity-to-obstacle -0.05 + dodge bonus +0.20 (dense shaping). (3) QR-DQN distributional agent thay scalar Q bằng 51 quantiles (51× denser gradient, fully tested trong tests/test_distributional.py + tests/test_agent_distributional.py, 19 tests mới). 716/716 tests pass.",
    "1.17.0 — MENTAL REHEARSAL (latent-space dreamer): VAE mã hóa frame self-imitation → latent 64-dim → perturb → decode → chạy qua env thật để verify 'ảnh khái quát' mà AI vẫn sống/không sống. Positive dreams lưu <demos>/abstract/positive/, negative dreams lưu <demos>/abstract/negative/ (cùng format .npz). Nút 💭 Dream ngay trong GUI + metric 'Abstract dreams' (+/-/tổng/q̄). 22 tests mới ở tests/test_dreamer.py.",
    "1.16.0 — SELF-IMITATION: episode AI tự chơi đủ tốt được tự lưu (gate so với rolling mean × factor) rồi tự BC lại trên union(human + self) mỗi N episode; nút ♻ BC từ self-imitation trong GUI; metric self pool hiển thị số đã lưu/tổng seen/đang có trên đĩa. Phá vỡ vòng xoáy BC học NOOP khi human demo nghèo.",
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
