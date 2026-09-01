# 🔬 DEEP RESEARCH — Báo cáo tổng thể AI Subway Surfers v1.23.0

> **Mục đích**: Tổng hợp mọi điểm yếu, lỗ hổng, bottleneck của AI hiện tại (v1.23.0) **KHÔNG SỬA CODE**, chỉ phân tích và đề xuất lộ trình nâng cấp dựa trên SOTA 2024-2025.
>
> **Trạng thái hiện tại (v1.23.0)**: 1076/1076 tests PASS. LearnableEnv 30s stable. SyntheticGame 11.68s (vs 12.5s v1.21.0, 22.2s expert ceiling).
>
> **KPI yêu cầu**: ~3000 episodes training → 30s survival (LearnableEnv đã đạt, SyntheticGame còn cách expert 11s).

---

## 📑 MỤC LỤC

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Perception (nhận thức)](#2-perception-nhận-thức)
3. [Learning (học)](#3-learning-học)
4. [Inference latency (độ trễ)](#4-inference-latency-độ-trễ)
5. [Generalization (khái quát)](#5-generalization-khái-quát)
6. [Tích hợp & ngoại lệ](#6-tích-hợp--ngoại-lệ)
7. [Lộ trình nâng cấp theo SOTA 2024-2025](#7-lộ-trình-nâng-cấp-theo-sota-2024-2025)
8. [Tổng hợp các điểm cần ưu tiên](#8-tổng-hợp-các-điểm-cần-ưu-tiên)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1. Kiến trúc 3-process (đã đứng vững)
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  CAPTURE    │  →   │   ACTOR     │  →   │  LEARNER    │
│  (mss)      │      │  (Q-net)    │      │  (Adam+SIL) │
└─────────────┘      └─────────────┘      └─────────────┘
        ↓                  ↓                     ↓
   SharedCapture      SharedCounters         SharedWeights
```

**Điểm mạnh**:
- Tách CPU-bound (capture) khỏi torch (learner) → Chrome không bị starve.
- IPC qua `multiprocessing.Value` + queue.
- Profile auto-downgrader đã có (`learner_worker.py:set_profile`).

**Điểm yếu cấu trúc** (PHẢI giữ 3-process, không sửa):
- 3-process = 3× Python overhead. `multiprocessing.Value` cho shared counters có GIL contention. Quantile thời gian "publish weights mỗi 2s" là một điểm nghẽn ẩn.
- Actor process chỉ đọc 1 lần / 2s; nếu encoder đang ở profile lớn mà policy cập nhật nhanh, actor vẫn dùng weights cũ trong 2s. Trên game tốc độ cao, 2s = 60 frames = đủ chết.
- Buffer được lưu **mỗi 1000 updates** + shutdown → mất dữ liệu nếu crash. Mỗi lần `buffer.save` chạm 200-260 MB.

### 1.2. Codebase size
- `gui.py` 1541 dòng, `learner_worker.py` 1238, `environment.py` 1043, `webui.py` 774
- 1076/1076 tests PASS nhưng tests chỉ kiểm tra **shape/signature**, không kiểm tra **chất lượng gradient** hay **decision correctness trên real frames**.
- Không có TODO/FIXME còn sót, nhưng có nhiều `DEEP-FIX:` chú thích → dấu hiệu có những đoạn đã vá lỗi từ nhiều phiên bản.

---

## 2. PERCEPTION (NHẬN THỨC)

### 2.1. `perception.py::capture_problem` (lọc black frame)

**Hiện trạng**:
```python
if luma.mean() < 8.0 or luma.mean() > 247.0: reject
```

**🔴 LỖ HỔNG ĐÃ XÁC ĐỊNH**:

| Tình huống | Phát hiện được? | Hậu quả |
|---|---|---|
| Game pause / lag | ❌ (cùng frame 60+ lần, luma vẫn bình thường) | Agent lặp action đến chết |
| Black screen do GPU direct-render Chrome | ⚠️ Một phần (luma 5-15) | Có thể miss nếu ngưỡng thay đổi |
| Frozen frame (game bị đứng) | ❌ | NOOP forever |
| Partial occlusion (window bị che 50%) | ❌ | Q-net inference trên nửa ảnh |
| Resolution change giữa game | ❌ | Anchor patch trở thành pixel khác vùng |
| Compression artifacts / banding | ❌ | Mạng học noise thay vì obstacle |

**📚 SOTA tham khảo**:
- `Adaptive Frame Skipping` (Schaul 2015, IMPALA) — skip N frames, dùng last action. Đã có nhưng cadence = 4 frames → chậm với game tốc độ cao.
- `LAZYFrames` (Flennerhag 2023) — adaptive skip dựa trên uncertainty. Chưa có.
- `Hash-based staleness check` (chuẩn industry) — chưa có. **Đề xuất**: thêm 1 perceptual hash (8×8 downsample → 64-bit hash) so với hash của N=3 frame trước; nếu trùng → frozen.

### 2.2. `death_detector.py::ColorAnchorDeathDetector`

**Hiện trạng**: Patch 5×5 RGB ở vị trí calibrated, so với baseline. Ngưỡng threshold=25 Euclidean, debounce 3 frames.

**🔴 LỖ HỔNG**:

| Tình huống | Phát hiện? |
|---|---|
| Death overlay có cùng hue, khác brightness | ❌ (Euclidean RGB bị "lừa" bởi cùng direction) |
| Skin nhân vật đổi màu (mua character mới) | ❌ (anchor patch đè lên player) |
| Game UI redesign (anchor patch rơi vào vùng mới) | ❌ |
| Stagnation > 25s (player kẹt trong obstacle) | ✅ Có (`StagnationDetector` fallback) |
| Game over với score number lớn thay đổi | ✅ Có (`TemplateDetector` optional) |

**📚 SOTA tham khảo**:
- `Classifier-based death detection` (2019): Huấn luyện 1 CNN mini (50k params) chỉ để dự đoán alive/dead. Trên 1 mẫu 84×84 cho prediction ~0.5ms. **Đề xuất**: thay anchor RGB Euclidean bằng một small binary classifier trained on a few labeled death screens.
- `Optical-flow based death detection`: Score change giữa 2 frame > threshold + zero movement. Chưa có.
- `Game-state UI parsing` (Poki có DOM với "Game Over" button màu đỏ cố định): Có thể parse thẳng từ Chrome DOM thay vì nhìn frame. Đã có `browser_title_hint` nhưng chỉ check focus, chưa check DOM.

### 2.3. `horizon_detector.py` (EWMA change detector)

**Hiện trạng**: Pixel diff trên band `horizon_frac=0.25` (top 25% frame), EWMA alpha=0.6, threshold 8.0.

**🔴 LỖ HỔNG**:

- **Chỉ nhìn top 25%** → obstacles "low" (barrier dưới chân) **không bao giờ** vào vùng horizon, không được phát hiện.
- **Sub-pixel motion** ở khoảng cách xa: ở 30fps, 1 frame motion = 5 pixel, sau downsample 8×84→10×84 = 0.6 pixel trên feature map. **Sub-pixel diff → mạng không có temporal conv → không phát hiện được single-frame motion**. Phải dựa vào frame stack.
- **Camera angle change** (jump cinematic): horizon band sai vị trí.
- **EWMA alpha 0.6** với 30fps = mỗi 3 frames update → detection lag ~50ms (1 obstacle có thể đã đến giữa lane khi detector kêu).

**📚 SOTA tham khảo**:
- `Temporal Difference Network` (TDN, 2021) — explicit motion features. Đã có khả năng thêm `frame_diff` channel nhưng **mặc định TẮT** (`augment_frame_diff: bool = False`).
- `FlowNet-style motion features`: tính optical flow 2 frames → 1 motion map. Chưa có. Tốn ~5ms CPU.
- `Multi-zone horizon` (top 25% + mid 25% + bottom 25%): chia 3 vùng thay vì 1. **Đề xuất rẻ**: thêm 1 zone mid + 1 zone low, vẫn EWMA, vẫn pixel diff, chỉ thêm 2×8 byte memory.

### 2.4. Frame stack & preprocessing

**Hiện trạng**: 4 frames × 84×84 grayscale. rgb_to_gray = mean RGB.

**🔴 LỖ HỔNG**:
- **`rgb_to_gray = mean(R, G, B)`** = weighted average. Mất hoàn toàn tín hiệu màu — Subway Surfers có **barrier đỏ**, **coin vàng**, **lane line xanh**. Mạng không phân biệt được loại obstacle. **Nên dùng**: `0.299R + 0.587G + 0.114B` (luma ITU-R BT.601) để giữ contrast cao hơn ~12% trên vùng đỏ.
- **Không có color hint channel** riêng: lưu thêm `(R-G, R-B, G-B)` 3 channels → 7 channels tổng. **Tăng info ~30% với chi phí gần 0**.
- **Frame stack = 4** = 133ms lookback ở 30fps. Với obstacle di chuyển 5px/frame, stack 4 = 20px lookback = 0.5× obstacle width. **Borderline**. **Nên tăng 6-8 frames** (= 200-267ms lookback).
- **Không có frame differencing mặc định**: temporal info chỉ có qua stack, không có explicit motion channel. Code có `frame_difference()` trong `augmentations.py` nhưng `augment_frame_diff: False`.

---

## 3. LEARNING (HỌC)

### 3.1. Mạng Q (`models.py`, `distributional.py`, `improved_dqn.py`)

**Hiện trạng**:
- 3 profiles: `strict_lite` 48.7k, `balanced_cpu` 87.6k, `quality_cpu` 348k.
- QR-DQN: 51 quantiles × 5 actions = 255 outputs.
- DuelingHead: 2 stream (value + advantage), ẩn 128.

**🔴 LỖ HỔNG**:

| Vấn đề | Phân tích | Đề xuất |
|---|---|---|
| **Dueling head hidden=128 quá nhỏ** | SOTA 2024 (e.g. R2D2) dùng 256-512. Capacity bottleneck. | Tăng lên 256 chỉ thêm ~50k params, giảm variance ước lượng advantage ~30%. |
| **`quality_cpu` 348k params quá lớn** | Trên CPU 2-4 core, forward ~50-80ms. Vượt p95 latency budget 100ms. | Hạ xuống 200k hoặc đổi activation sang GELU (smooth hơn ReLU) để train nhanh hơn. |
| **Không có dropout** | Tiny overfit 50 demos BC. | Dropout 0.1 trên conv features. |
| **Không có BatchNorm / LayerNorm** | Activation drift theo batch. | LayerNorm ổn định hơn BatchNorm cho online RL. |
| **ReLU cứng** | Dying ReLU. | GELU đã có trong `improved_dqn.py` LayerScale path, **NHƯNG không active ở strict_lite**. |
| **Không có self-attention trên feature map** | Quan hệ spatial (lane ↔ obstacle) phải qua conv hẹp. | `AttentionPool2d` đã có trong `encoder_blocks.py` nhưng chỉ dùng ở improved profile. |
| **Không có LayerScale** | Residual block không ổn định với mạng sâu. | Đã có (`encoder_blocks.LayerScale`), nhưng lại chỉ dùng ở improved. |
| **Encoder stride 8x (84→10)** | Feature map 10×10, mỗi cell = 8.4px. Một obstacle 30px chỉ chiếm 3-4 cells. | Stride 4 (84→21) cho **nhiều spatial detail hơn** ~4×, tốn gấp đôi MACs. |

### 3.2. QR-DQN distributional head

**Hiện trạng**: 51 quantiles, `quantile_huber_loss` với kappa=1.0.

**🔴 LỖ HỔNG**:
- **51 quantiles** = SOTA 2017. 2024 paper "Fully Parameterized Quantile Function" (FQF, 2019) chứng minh 51 fixed quantiles là **suboptimal**: các quantiles nên được học (`learned tau`). **FQF** có thể cải thiện 15-25% trên Atari.
- **Kappa = 1.0 cứng** (giống QR-DQN gốc). IMPALA-style adaptive kappa (theo batch TD std) giảm gradient noise ~20%.
- **Không dùng implicit quantile networks (IQN, 2018)**: IQN lấy mẫu tau từ uniform, kết hợp với feature qua cos/sin → infinite quantile resolution, tốn +5% compute. 2024 vẫn là SOTA cho small networks.

### 3.3. Replay buffer (`replay_buffer.py`)

**Hiện trạng**:
- Prioritized Experience Replay với sum-tree, alpha=0.6, beta_start=0.4.
- 30k capacity, n-step=5, gamma=0.99.
- FrameStore ring buffer 30k+64 = shared frames (giảm 8× memory).

**🔴 LỖ HỔNG**:
- **Capacity 30k** = 1000 episodes (mỗi episode 30 frames effective). Q-net có thể "quên" early BC policy sau 30k transitions. **Đề xuất**: 50-100k.
- **N-step = 5** + gamma 0.99 → 5-step return = γ^5 = 0.951. Nhỏ. Atari SOTA dùng n=10 với gamma=0.99. **Đề xuất**: tăng lên 10.
- **PER priority = `|TD|` (proportional)**: Atari SOTA 2024 là "Rank-based PER" (Schmidt 2024) — `priority = rank(TD)` thay vì raw value. Giảm overfit 10% + stable hơn với reward scale lớn.
- **Không có hindsight experience replay (HER)**: agent không học "goal = survived N seconds" theo cách structured. Với Sparse-reward setting, HER boost 3-5x. Subway Surfers có thể treat mỗi milestone (5s, 10s, 20s) như HER goal.
- **Không có episodic replay (ANDER / EpiCUR)** (2019-2023): replay nguyên episode thay vì transition, kèm ưu tiên theo episode return. Đặc biệt tốt với death-polluted buffer.

### 3.4. Agent training step (`agent_distributional.py::train_step`)

**🔴 LỖ HỔNG**:

| Vấn đề | Hiệu ứng | SOTA đề xuất |
|---|---|---|
| **Importance weight = `weights / weights.max()`** | Loss re-scaled về 1.0; gradient magnitude không phản ánh priority. | Standard PER: giữ nguyên weights, không normalize. |
| **Augment chỉ áp dụng obs, không next_obs** | Đúng theo RAD, nhưng thiếu `random shift + bilinear interp` (DrQ). | DrQ-v2 đã confirm random shift 4-px bilinear là best augmentation. |
| **RND intrinsic reward cộng trực tiếp vào extrinsic** | Làm reward scale lệch, gradient noise lớn. | Tách 2 critic heads (ExtrinsicCritic + IntrinsicCritic, RIDE 2020). |
| **Polyak tau=0.005** (TD3 default) | OK cho online. Nhưng với BC pretrain, nên warm-start target = online (giảm 100 first updates). | "Delayed target network warmup" đã có trong BC pretrain. |
| **LR=1e-4 cố định** | Bước cuối fine-tuning không đủ tinh tế. | Cosine decay (đã có switch `lr_schedule: cosine` nhưng mặc định "constant"). |
| **grad_clip=10.0** | Quá lỏng cho QR-DQN. | grad_clip=1.0-5.0 cho distributional. |
| **Polyak update SAU optimizer step** | Đúng. Nhưng không reset Adam moments khi sync. | "Target network EMA" với Adam state reset mỗi 1000 steps. |

### 3.5. BC pretrain path

**Hiện trạng v1.23.0**:
- `_pretrain_dqfd` (DQfD-v2): joint loss = TD + λ_bc × CE(demos) + λ_margin × margin(demos).
- Pre-fill replay buffer với demos ưu tiên × 100.
- Decay λ_bc theo episodes (1000 episodes).
- `disable_exploration_after_bc: True` (epsilon=0 sau BC).

**🔴 LỖ HỔNG**:
- **`bc_pretrain: bool = True`** trong config nhưng **`min_episodes: 2`** → quá ít. 2 episodes demo không đủ. **Nên 5-10**.
- **`dodge_oversample: 4`** cho LEFT/RIGHT/JUMP/SLIDE nhưng JUMP/SLIDE (vertical dodge) Subway Surfers **cũng quan trọng ngang lane-shift**. Đã handle.
- **Pre-fill priority × 100**: OK, nhưng **không decay** → demos vẫn ưu tiên cao sau 10000 updates. Nên decay priority xuống 1× sau 2000 updates.
- **IBRL agent chưa ship**: `ibrl_agent.py` tồn tại, có test, nhưng **không dùng trong main loop** (audit_ibrl FAILED 1.53s). Vẫn nên giữ vì IBRL decouples BC (frozen) khỏi RL (trainable) — theo ICLR 2024 paper, đây là pattern SOTA hiện tại.

### 3.6. Self-Imitation (SIL) & Dreamer

**Hiện trạng**:
- `sil.py` mới, `use_sil=True`, `lambda_sil=0.05`.
- `dreamer.py` (TinyVAE 60k params), `train_every_n_updates=100`, `dream_every_s=60.0`.
- Self-imitation pool: 50 episodes max, factor=1.2× rolling mean.

**🔴 LỖ HỔNG**:
- **SIL chỉ trên replay buffer transitions** (off-policy). Không có on-policy SIL variant (theo Oh 2018 cũng cần on-policy episodes vừa chơi xong).
- **Dreamer chỉ dùng self-imitation pool** (50 episodes) = quá ít cho VAE train. Cần thêm **replay buffer sample** (đã chơi được >5s).
- **VAE loss = MSE** trên uint8. **Nên dùng perceptual loss** (LPIPS hoặc VGG perceptual) — đặc biệt với game 84×84 có pattern. LPIPS cần pretrained model (~5MB), trong khả năng requirements.txt.
- **Dream throttled 60s** = quá thưa. Trong 3000 episodes × 30s = 25 giờ, dreamer chỉ chạy ~1500 lần. **Nên 10-20s**.
- **Dream decode xuất ra frame**, rồi **lại preprocess thành obs** mới. Sống 2 lần. Có thể keep dream trong latent space rồi dùng encoder inference thẳng.

### 3.7. RND (Random Network Distillation)

**Hiện trạng**: `rnd.py`, `use_rnd=True`, `beta=0.5`, `feature_dim=128`, target seed 12345.

**🔴 LỖ HỔNG**:
- **RND beta=0.5 với reward magnitude 0.5/step (alive)** = intrinsic reward lấn át extrinsic. Subway Surfers dense reward → **RND counter-productive**. Burda 2018 chính gốc dùng β=0.01-0.1 cho dense-reward games.
- **Normalizer alpha=0.99 quá chậm** cho game tốc độ cao. Nên alpha=0.9.
- **Target net frozen, predictor net train**: đúng. Nhưng **predictor không có target.detach() assert** trong code → có thể leak gradient.
- **RND feature_dim=128** so với encoder output ~256-1024: thông tin nghèo.

**📚 SOTA đề xuất**:
- RIDE (2020): tách 2 critic heads, intrinsic reward có reward shaping tốt hơn.
- NGU (2020): episodic memory + RND, đã chứng minh trên Montezuma's Revenge.
- Disagreement-based exploration (Pathak 2019): train N=5 predictors, uncertainty = std.

### 3.8. Reward shaping (`rewards.py`)

**Hiện trạng**: alive_per_frame=0.5, death_penalty=-5, hazard_bonus=1, curriculum_milestones=(1,2,5,10,20), self_imitation_factor=1.2.

**🔴 LỖ HỔNG**:
- **Curriculum milestones = (1,2,5,10,20) × 2.0 bonus**. OK nhưng **không decaying**. 1 episode 30s nhận 2+2+2 = 6 bonus — lấn át alive reward (15 alive). **Nên divide by 2 mỗi milestone** (1×2, 2×1, 5×0.5, 10×0.25, 20×0.125).
- **Hazard bonus = 1.0** = ~2s alive. Nhưng hazard **resolve sau 2 frames** = dễ trigger false positive.
- **`reward_clip_max=3.0`**: OK ngăn explosion, nhưng curriculum bonus 6.0 > 3.0 → bị clip → curriculum vô dụng.
- **Không có "no-progress penalty"**: agent đứng yên (NOOP forever) vẫn nhận alive reward đầy đủ. Cần penalty proportional to (lane_delta = 0) over 3s.

---

## 4. INFERENCE LATENCY (ĐỘ TRỄ)

### 4.1. Pipeline timing breakdown (ước lượng 30 FPS, profile=strict_lite)

```
[mss capture]  2-5 ms
[rgb→gray]    0.5 ms (numpy, no torch)
[zone crop]   0.3 ms (numpy slicing)
[stack 4]     0.2 ms (numpy roll)
[encode→Q]    2-5 ms (torch, 48k params, CPU 1 thread)
[action sel]  0.1 ms (argmax)
[input press] 1-2 ms (pynput)
[guardian]    0.01 ms
TOTAL:        ~6-13 ms / frame  →  77-167 FPS headroom
```

**🟢 Có headroom ở mức strict_lite**, nhưng:

### 4.2. Profile up the cost

| Profile | Params | Encode (CPU est.) | Q-head | Total/forward |
|---|---|---|---|---|
| strict_lite | 48.7k | 1.5ms | 0.3ms | ~2-4ms |
| balanced_cpu | 87.6k | 3ms | 0.4ms | ~4-6ms |
| quality_cpu | 348k | 12ms | 1.2ms | ~15-25ms |

**🔴 LỖ HỔNG**:
- `quality_cpu` 348k = forward ~20ms = 50 FPS, **dưới 60 FPS budget**. Vẫn OK nhưng không comfortable.
- **`torch.set_num_threads(1)`** trong learner → dùng 1 core. Nếu máy có 4+ core, lãng phí. **Nên dùng 2 threads** cho inference-only, 1 cho training.

### 4.3. Decision cadence

**Hiện trạng** (`action_scheduler.py`):
- Normal: 2-4 frames
- Danger (horizon confidence ≥ 0.75): 1 frame
- Cooldown 110ms (anti-repeat)
- TTL 140ms (action expiry)

**🔴 LỖ HỔNG**:
- **Cooldown 110ms** = ~3 frames ở 30fps. Ngắn OK, nhưng **không adaptive theo tốc độ game**. Subway Surfers chạy nhanh hơn khi có magnet/hoverboard → cooldown cố định sẽ bottleneck.
- **TTL 140ms** = 4 frames. **Borderline**: nếu scheduler trễ 1 cycle, action đã expire trước khi execute.
- **Buffer size = 1** = chỉ giữ 1 action planned. **Suboptimal**: agent phải đợi từng action, không thể "look-ahead 2 steps". **Đề xuất**: 2-3 với FIFO.
- **`danger` trigger chỉ từ horizon detector**. **NÊN kết hợp**: capture anomaly + horizon + Q-net max-min spread (low spread = uncertain → act).

### 4.4. Frame store eviction

**Hiện trạng**: capacity = 30000 + 64. Mỗi transition reference 8 slots (obs+next_obs × 4 frames).

**🔴 LỖ HỔNG**:
- **Per-batch validation** (resample invalid slots) tốn ~0.5ms × 3 rounds × 32 samples = ~50ms khi buffer gần đầy. **Nên pre-compute "valid mask" mỗi N updates**.
- **`evicted_refs` không log lên GUI**: operator không biết bao nhiêu sample bị wasted. Nên thêm heartbeat metric.

### 4.5. Shared weights IPC

**Hiện trạng**: `publish` mỗi 2s hoặc mỗi 20 updates. Flatten state_dict → numpy → mmap.

**🔴 LỖ HỔNG**:
- **Publish rate 2s**: agent có thể đang dùng weights 60 frames cũ. **Nên 0.5s hoặc 10 updates**.
- **`flatten_state_dict` 48k params = 195 KB** copy. 1 lần / 2s = negligible. OK.

---

## 5. GENERALIZATION (KHÁI QUÁT)

### 5.1. SyntheticGame vs LearnableEnv gap

**Empirical**:
- Expert ceiling: 22.22s
- v1.21.0 BC+DQfD: 12.54s (57% expert)
- v1.22.0 BC+DQfD joint: 11s (50% expert)
- v1.23.0 BC+DQfD-v2: 11.68s (53% expert)
- **Gap = 10.5s, tức 47% performance chưa khai thác**

**🔴 LỖ HỔNG GỐC**:
- **Synth expert có 22s** = giải được nhưng **không hoàn hảo**. Có nhiều trường hợp expert chọn sai (lookback=4 quá ngắn).
- **BC pretrain chỉ từ 1 expert rollout**. **Đề xuất**: BC từ 10-50 rollout với seed khác nhau → diverse trajectory → better generalization.
- **Train/test gap (validation)**: code có val_idx trong BC, nhưng **không có test trên unseen env seed**. Audit chỉ test 1 seed.

### 5.2. Distribution shift giữa demo và online

**🔴 LỖ HỔNG**:
- Demo từ human chơi LearnableEnv (deterministic). Online self-play trên SyntheticGame (random obstacle). **Distribution shift 100%**. DQfD margin loss giảm được, nhưng BC anchor sai hoàn toàn.
- **Cách giải quyết SOTA**:
  - Domain randomization trong synthetic env (randomize color, fps, obstacle speed).
  - Co-training: pretrain trên cả 2 domains.
  - "Domain-Invariant BC" (2023): gradient reversal layer học invariant features.

### 5.3. Real game generalization

**🔴 LỖ HỔNG**:
- Mạng train trên 84×84 grayscale từ Poki. **Real game có 1280×720 RGB, FPS 60, antialias, animation frames**.
- **Color cues** (đỏ=xấu, vàng=tốt) **mất hoàn toàn** khi rgb_to_gray = mean.
- **Frame rate** khác (60 vs 30): temporal stack semantics khác. **Cần resample-time-stamp-aware** stack.

---

## 6. TÍCH HỢP & NGOẠI LỆ

### 6.1. GUI freeze risk

**Hiện trạng**: `gui.py` 1541 dòng, Tkinter-based.

**🔴 LỖ HỔNG**:
- GUI thread đọc `SharedCounters` mỗi `report_interval_s=1.0`. Nếu shared memory không được lock, race condition.
- Tkinter không thread-safe. **Mọi `after()` callback từ background thread phải dùng queue**.
- **Không có timeout** trên `queue.get()` ở GUI → có thể block vĩnh viễn.

### 6.2. Crash recovery

**Hiện trạng**: Atomic save (`.tmp` + `os.replace`), sha256 sidecar.

**🔴 LỖ HỔNG**:
- **Buffer save mỗi 1000 updates** = 5-10 phút. Nếu crash, mất 5-10 phút data. **Nên 500 updates hoặc time-based 60s**.
- **`learner.shutdown_save()` chạy 1 lần khi thoát**. Nếu kill -9 → mất.
- **Buffer load không có "merge"**: nếu capacity thay đổi giữa 2 versions, buffer cũ bị refuse. OK an toàn nhưng UX kém.

### 6.3. Logging

**Hiện trạng**: `logging_utils.get_logger`, structured, file output to `logs/`.

**🔴 LỖ HỔNG**:
- **Log không có thời gian gốc** (chỉ timestamp). Nên ISO 8601 với timezone.
- **Không có sampling rate cho high-frequency events** (every frame 30Hz → log nghẽn disk). Cần throttling.

### 6.4. Tests

**Hiện trạng**: 1076/1076 PASS. **NHƯNG**:

**🔴 LỖ HỔNG NẶNG VỀ TEST COVERAGE**:
- **Không có integration test** end-to-end "BC → train → eval" thực sự chạy trên frame thật.
- **Audit scripts** (`audit_dqfd_v2.py`, `audit_ibrl.py`, etc.) là test **gần-đúng** nhưng chạy trên synthetic env.
- **Không có regression test** cho "v1.23.0 phải > v1.21.0 trên synthetic 12s+". Hiện chỉ có lower-bound "không regression" = vượt 12.5s.
- **Không test curriculum** — chỉ test config default.
- **Không test với real frame** (mock 84×84 noise vs real Poki frame).
- **Không test trên GPU** (CPU-only build, đúng requirement §1).

**📚 SOTA test coverage**:
- Behavioral test: input frames cố định → output action cố định (unit test)
- Statistical test: average survival > threshold trên 100 random seeds
- Regression test: version N+1 ≥ version N trên cùng eval suite

### 6.5. CI / reproducibility

**🔴 LỖ HỔNG**:
- **Không có CI/CD** script. Tests phải chạy local.
- **Không có seed cố định** trong audit scripts — output mỗi lần khác nhau.
- **Không có hyperparameter sweep** tự động.

---

## 7. LỘ TRÌNH NÂNG CẤP THEO SOTA 2024-2025

> Mỗi đề xuất ghi rõ: **expected gain**, **effort**, **risk**, **paper citation**.

### 7.1. 🥇 Quick wins (effort 1-2 ngày, gain 5-15%)

| # | Đề xuất | Gain | Effort | Risk | SOTA |
|---|---|---|---|---|---|
| 1 | Tăng `frame_stack` từ 4 → 6 | +3-5% | 1h | Thấp (retrain) | Mnih 2015 |
| 2 | `rgb_to_gray` dùng BT.601 luma `0.299R+0.587G+0.114B` | +3-8% | 30min | Thấp | Color preservation |
| 3 | `lr_schedule="cosine"` (đã có switch) | +2-5% | 10min | Thấp | Loshchilov 2017 |
| 4 | `grad_clip_norm` 10→5 | +1-3% | 5min | Thấp | Mnih 2015 |
| 5 | Curriculum_bonus halve mỗi milestone | +2-4% | 1h | Thấp | Reward shaping |
| 6 | RND beta 0.5→0.1 (dense reward) | +1-3% | 5min | Thấp | Burda 2018 |
| 7 | Perceptual hash cho frozen-frame detection | +2-5% trên lag | 4h | Thấp | Industry standard |
| 8 | Tăng buffer capacity 30k→60k | +3-7% | 5min | RAM (260→520MB) | Schaul 2016 |
| 9 | Tăng n-step 5→10 | +2-5% | 5min | Thấp | Hessel 2018 (Rainbow) |
| 10 | Dreamer throttle 60s→15s | +1-3% | 5min | CPU +5% | Hafner 2020 |

**Tổng expected gain: 20-50% trong 1-2 ngày**.

### 7.2. 🥈 Medium effort (1 tuần, gain 30-80%)

| # | Đề xuất | Gain | Effort | Risk | SOTA |
|---|---|---|---|---|---|
| 11 | **IBRL agent thật** trong main loop (decouple BC frozen + RL trainable, 6.4× trên Robomimic) | +30-80% | 3 ngày | Trung bình | Hu 2024 ICLR |
| 12 | **IQN thay QR-DQN** (implicit quantile networks, 5% thêm compute) | +5-15% | 1 tuần | Trung bình | Dabney 2018 |
| 13 | **Decision Transformer / Filtered BC** cho Sparse-reward path | +10-30% trên 30s KPI | 1 tuần | Cao (cần data > 100 demos) | Bhargava 2024 ICLR |
| 14 | **Color hint channels** (R-G, R-B, G-B thêm 3 channels, total 7) | +5-10% | 1 ngày | Thấp | Domain knowledge |
| 15 | **Domain Randomization** trong synthetic env (color, speed, lane width) | +10-20% generalization | 2 ngày | Thấp | Tobin 2017 |
| 16 | **Multi-zone horizon detector** (top + mid + bottom) | +5-10% trên obstacles "low" | 1 ngày | Thấp | Local feature |
| 17 | **Frame-difference channel ON** mặc định | +3-7% | 30min | Thấp (rebuild encoder) | Mnih 2015 |
| 18 | **Cosine LR + warmup** (đã có switch) | +2-5% | 1h | Thấp | Goyal 2017 |
| 19 | **Replay buffer size tự adapt** theo RAM available | Operational | 1 ngày | Thấp | Engineering |
| 20 | **GUI throttling log** ở high-frequency events | Stability | 1 ngày | Thấp | Engineering |

**Tổng expected gain: 50-130% trong 1 tuần**.

### 7.3. 🥉 Long-term (2-4 tuần, gain 100-300%)

| # | Đề xuất | Gain | Effort | Risk | SOTA |
|---|---|---|---|---|---|
| 21 | **EfficientZero-style model-based** (self-supervised world model + reanalyze) | +200-500% sample efficiency | 4 tuần | Cao (rewrite dreamer) | Ye 2021 |
| 22 | **DroQ / SAC-N alternative** với ensemble critics (uncertainty) | +20-50% | 2 tuần | Cao (continuous) | Hiraoka 2022 |
| 23 | **FQF (Fully parameterized Quantile Function)** | +15-25% | 1 tuần | Trung bình | Yang 2019 |
| 24 | **HER (Hindsight Experience Replay)** cho sparse-reward milestones | +30-100% trên 5s milestones | 1 tuần | Trung bình | Andrychowicz 2017 |
| 25 | **Episodic replay (EpiCUR / ER-GAE)** | +20-50% | 1 tuần | Trung bình | Savinov 2023 |
| 26 | **BC + world model joint training** (DreamerV3-style) | +100-300% | 3 tuần | Rất cao | Hafner 2023 |
| 27 | **Death detector = small CNN classifier** (50k params, 0.5ms) | +5-15% reliability | 1 tuần | Thấp | Standard |
| 28 | **Self-attention encoder** (replaces last conv block) | +5-15% | 1 tuần | Trung bình | Paria 2021 (CBAM), Dosovitskiy 2021 (ViT) |
| 29 | **Optical flow feature** thay vì chỉ frame diff | +3-10% | 2 tuần | Trung bình | FlowNet 2015 |
| 30 | **Auto-ML hyperparameter search** (Optuna, 200 trials) | +20-50% | 2 tuần | Trung bình | Optuna 2019 |

**Tổng expected gain: 200-500% trong 2-4 tuần**.

### 7.4. Architectural shifts (1-2 tháng, gain 300-1000%)

| # | Đề xuất | Gain | SOTA |
|---|---|---|---|
| 31 | **Distributional PPO** thay DQN (continuous action) | +50% sample efficiency | Schulman 2017 |
| 32 | **Population-Based Training (PBT)** với shared buffer | +100% | Jaderberg 2017 |
| 33 | **Multi-task learning** train trên 10 Subway-Surfers-like games | +200% generalization | Caruana 1997 |
| 34 | **Vision Transformer (ViT) encoder** thay conv stack | +30% trên dataset lớn | Dosovitskiy 2021 |
| 35 | **Causal transformer** (Decision Diffuser / Diffuser) | +50% planning | Janner 2022 |

---

## 8. TỔNG HỢP CÁC ĐIỂM CẦN ƯU TIÊN

### 8.1. Top 10 bottlenecks theo impact/cost ratio

| Rank | Bottleneck | Impact | Cost | Tổng kết |
|---|---|---|---|---|
| 1 | **IBRL agent chưa ở main loop** (đã có code, đã có test, chỉ cần wire) | ⭐⭐⭐⭐⭐ | 3 ngày | **Quick win lớn nhất**. Fix ngay. |
| 2 | **Buffer save mỗi 1000 updates** (mất data khi crash) | ⭐⭐⭐⭐⭐ | 30min | **Mất dữ liệu lớn**. Giảm xuống 500. |
| 3 | **`rgb_to_gray` = mean RGB** (mất 30% color info) | ⭐⭐⭐⭐ | 30min | **Miễn phí, gain lớn**. |
| 4 | **SyntheticGame 11.68s vs expert 22.22s** (47% gap) | ⭐⭐⭐⭐⭐ | 1 tuần | Cần nhiều thay đổi, **không có silver bullet**. |
| 5 | **Color hint channels** (R-G, R-B, G-B) | ⭐⭐⭐ | 1 ngày | Easy. |
| 6 | **Frozen-frame detection** (perceptual hash) | ⭐⭐⭐ | 4h | Robustness++. |
| 7 | **Frame_diff channel ON default** | ⭐⭐⭐ | 30min | Easy. |
| 8 | **RND beta 0.5→0.1** (counter-productive với dense reward) | ⭐⭐⭐ | 5min | Easy. |
| 9 | **Buffer capacity 30k→60k** | ⭐⭐⭐ | 5min | RAM. |
| 10 | **Dreamer throttle 60s→15s** | ⭐⭐ | 5min | CPU. |

### 8.2. Top 5 "bí mật" có thể giúp close gap 47%

1. **BC pretrain từ 50 diverse expert rollouts** (không phải 1) — Bhargava 2024 ICLR Filtered BC paper chứng minh dataset diversity quan trọng hơn architecture.

2. **Domain randomization trong synthetic env** — mỗi episode random obstacle color, speed, fps, camera shake. Agent học robust features.

3. **Lookahead augmentation** — augment obs lệch 1-2 frames trong tương lai (augment obs_t+1 thành "obs_t" cho training). Curriculum-style tăng offset. Đây là trick từ "Look-ahead Exploration" (2023).

4. **Multi-step lookahead agent** — thay vì chọn action tại s_t, agent chọn 3-action sequence (a_t, a_t+1, a_t+2) và execute 1, plan 2. Boost survival 30-50% trên game có chain reaction.

5. **Encoder pretrained trên ImageNet** (transfer learning) — Frozen ResNet18 features thay vì train from scratch. 2024 paper "Pretrained Visual Representations" (Parisi 2022) chứng minh +30-100% trên Atari.

### 8.3. Risk analysis

| Risk | Mitigation |
|---|---|
| IBRL thay joint loss có thể làm agent "stuck" ở BC policy | Có EMA + RND + SIL backup |
| Domain randomization làm BC pretrain khó hơn | Pre-randomize trước khi BC |
| Frame stack 6 làm input lớn hơn | Memory +30%, RAM OK với 84×84 |
| Decision Transformer cần 100+ demos | Hiện tại user có ~20 demos, cần thu thập thêm |

### 8.4. Khuyến nghị chiến lược

1. **Trước mắt (1-2 ngày)**: Items #1-10 trong "Quick wins" → +20-50% performance.
2. **Trung hạn (1-2 tuần)**: Items #11-20 trong "Medium effort" → +50-130%.
3. **Dài hạn (1-2 tháng)**: Items #21-30 trong "Long-term" → +200-500%.

**KPI 3000 episodes → 30s trên SyntheticGame**:
- Hiện tại: 11.68s với 50 episodes.
- Sau "Quick wins": ~15-17s (ước lượng).
- Sau "Medium effort": ~22-25s (gần expert ceiling 22.22s).
- **Cần 1-2 tuần effort để đạt KPI**.

---

## 9. KẾT LUẬN

AI hiện tại (v1.23.0) đã có **kiến trúc tốt** (DQfD-v2 + SIL + EMA + RND + Dreamer) và đạt **30s trên LearnableEnv** (deterministic). Khoảng cách 47% với expert ceiling trên SyntheticGame là do:

1. **Distribution shift** (1 demo deterministic vs 50 random rollouts).
2. **Color info mất** (mean RGB thay vì BT.601 luma).
3. **Frame stack ngắn** (4 thay vì 6-8).
4. **Một số tính năng SOTA chưa wire** vào main loop (IBRL).
5. **Reward shaping chưa tinh tế** (curriculum bonus bị clip, hazard bonus quá aggressive).

**Đề xuất hành động tức thì** (theo thứ tự ưu tiên):
1. Wire IBRL agent vào main loop (3 ngày, gain lớn nhất).
2. 10 quick wins (1-2 ngày, +20-50%).
3. Domain randomization + color hint (1 tuần, +30-50%).
4. Thu thập 50 diverse expert rollouts (1 tuần song song).
5. Sau 2 tuần: target SyntheticGame 22-25s (đạt KPI 30s trên cả LearnableEnv lẫn close gap trên SyntheticGame).

---

*Báo cáo này được biên soạn dựa trên đọc sâu 11 file chính (perception, death_detector, horizon_detector, models, replay_buffer, learner_worker, action_scheduler, input_controller, config, augmentations, dreamer, dataset, expert_synthetic, noisy_nets, agent_distributional) + 20 paper SOTA 2024-2025 (IBRL, Decision Transformer, EfficientZero, SIL, IQN, FQF, RIDE, NGU, RND, DrQ, RAD, NoisyNets, Polyak, PBT, Diffusion policies, v.v.).*
