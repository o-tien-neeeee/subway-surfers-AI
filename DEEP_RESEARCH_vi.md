# DEEP RESEARCH — Vì sao AI không học được Subway Surfers, và đã sửa gì (v1.24.0)

> Báo cáo này ghi lại **chẩn đoán** (dựa trên việc đọc trực tiếp mã nguồn repo này),
> **bằng chứng thực chiến** từ các project Subway Surfers AI công khai, và **các fix đã
> triển khai** kèm số liệu kiểm chứng. Toàn bộ số đo repo (số test, giá trị mặc định,
> tên file/hàm) được lấy bằng lệnh thật trong phiên làm việc này.

---

## 0. Triệu chứng

> "Tôi đã train 6000 episode nhưng chỉ tiến triển đúng ~1 giây trung bình so với episode 1."

Đây **không** phải dấu hiệu "model nhỏ quá" hay "máy yếu quá". Đây là dấu hiệu kinh điển của
**tín hiệu học ≈ 0**: mạng có học, nhưng gradient không phân biệt được hành động tốt và xấu,
nên policy đi ngang. Repo này ở trạng thái `v1.23.0` trước khi sửa có **7 lỗi thiết kế MDP
phối hợp** gây đúng triệu chứng đó.

Trạng thái kiểm chứng trước/sau (đo bằng `pytest`):

| | Trước (baseline `25da791`) | Sau (v1.24.0) |
|---|---|---|
| Test suite | **1077 passed** / 247 s | **1125 passed** / 207 s |
| Test mới cho các fix | 0 | 40 (`tests/test_deep_research_fixes.py`) |

---

## 1. Bảy nguyên nhân gốc (kèm bằng chứng trong code)

### 1.1 AI **mù chân trời** — lỗi nặng nhất

`perception.py` (trước khi sửa) cắt khung hình làm hai: dải trên (`horizon_frac = 0.25`) đưa
cho `HorizonDetector`, phần **dưới** (`image[split:]`) resize 84×84 đưa cho CNN. Nghĩa là
**chính vùng vật cản mới xuất hiện bị loại khỏi observation**. Trong Subway Surfers tàu/hàng
rào hiện ra ở chân trời và cần ~1–2 s để tới người chơi; policy chỉ "thấy" vật cản khi nó đã
ở sát → không kịp né thứ nó chưa từng nhìn thấy.

*Fix:* `PerceptionConfig.policy_full_frame = True` + `obs_size = 84`; `ZonePreprocessor.process`
lấy **toàn bộ region** làm observation, dải chân trời chỉ còn là đầu vào của detector.
`ZoneResult.policy_gray` là tên chuẩn; `ground_gray` giữ làm alias để không phá call site cũ.

### 1.2 Reward: mọi episode ngắn đều **âm như nhau**

Trước khi sửa: `alive_per_frame` dương nhưng `death_penalty = -5` (và `-10` ở bản cũ),
`reward_clip_min = -10`. Với episode 3–10 s, tổng return luôn là một số âm gần bằng nhau
→ TD gradient không có gì để phân biệt "né đẹp" với "né ẩu".

*Fix:* reward thành **bộ đếm dương** (`death_penalty = 0.0`, `reward_clip_min = 0.0` mặc định).
Chết vẫn là **terminal transition** — `(1-done)` đã triệt bootstrap, không cần hình phạt.
Đường cũ vẫn bật lại được bằng `death_penalty = -5/-10` (có test giữ).

### 1.3 `hazard_bonus` **thưởng cho việc bấm bừa**

`PendingHazardTracker` trả `hazard_bonus` khi detector chân trời nháy rồi người chơi **có bấm
một phím né** trong vài frame sau — kể cả cú bấm không né gì. Policy tối ưu của hàm này là
**spam phím**: đúng câu "agent learnt to keep jumping or rolling" mà các project DQN tự thú.

*Fix:* `reward.use_hazard_bonus = False` mặc định, thay bằng `clear_bonus` **nhân quả**
(chỉ trả khi vật cản thật sự đã đi qua làn người chơi mà không chết — xem 1.4).

### 1.4 Không có tín hiệu "cú né đó THÀNH CÔNG"

Repo có `alive`, `death`, `hazard`, `curriculum`, `pixel_diff` — nhưng không có thành phần nào
nói "vật cản vừa đi qua và bạn còn sống". Đó là tín hiệu dày, đúng nhân quả, rẻ nhất có thể có.

*Fix:* module mới `obstacle_perception.py` — lưới occupancy `depth_rows × lanes` (mặc định 5×3).
Hai đường sinh dữ liệu, một state machine:
- `ObstacleTracker.occupancy_from_gray()` — CV thuần (Sobel density) cho game thật, <1 ms/frame;
- `occupancy_from_game_state()` / `snapshot_from_game_state()` — dùng state thật của
  `SyntheticGame` cho headless (không dùng proxy pixel khi đã có ground truth).

State machine theo **làn** (không theo ô), vì vật cản *di chuyển*: mỗi ô chỉ bị chiếm 1–2 frame
nên luật "chiếm N frame liên tiếp ở cùng một ô" không bao giờ xác nhận được gì. Thay vào đó:
đếm số frame liên tiếp có vật cản ở các hàng gần (`near_rows`) → latch → khi làn trống trở lại
mà latch đã bật và **người chơi còn sống** → `cleared += 1`.

### 1.5 MDP không phải frame-skip: `gamma`/`n_step` tính theo **frame**, không theo **quyết định**

Trước khi sửa, actor ship **một transition mỗi frame** với `self._last_action` — trong khi
`ActionScheduler` chỉ quyết định mỗi 2–4 frame. Hệ quả:
- một hành động bị gán nhãn cho 2–4 transition liên tiếp (action→Q credit sai);
- `gamma = 0.99` trên frame 30 FPS ⇒ tầm nhìn hiệu dụng ~3.3 s, không với tới cái chết xảy ra
  5–8 s sau quyết định né;
- frame stack 4 @30 FPS chỉ chứa **0.13 s** lịch sử chuyển động.

*Fix:* `rl.frame_skip = 3` (mặc định). Một **agent step** = `frame_skip` frame, action được giữ,
reward các frame **cộng dồn** thành một transition, stack push **một lần mỗi quyết định**
(⇒ 4 quyết định ≈ 0.4 s lịch sử), `n_step` và `gamma` tính theo bước quyết định.
Swipe của Subway Surfers là **tap**, nên giữ action 3 frame không lặp lại động tác.

### 1.6 ε-greedy tắt trước khi học được gì

`epsilon_decay_frames` đếm **frame môi trường**, nên lịch khám phá đổi nghĩa mỗi khi đổi FPS hay
cadence; ở bản cũ nó về ~`epsilon_end` rất sớm và khoá cứng policy mù ở chế độ tham lam.

*Fix:* `agent.epsilon_for_step()` + `effective_epsilon_for_step()`, decay theo **số quyết định**
(`rl.epsilon_decay_steps = 100_000`). `epsilon_for_frame` giữ nguyên cho ablation.
Luật "sau BC thì ε=0" (`disable_exploration_after_bc`) vẫn được giữ.

### 1.7 Transition bị **lệch một bước**

`_ship_transition` đẩy `self.stack.get()` **tại thời điểm ship** — tức observation đã chứa các
frame do chính hành động đó sinh ra. Cặp `(obs, action)` vì thế lệch một bước so với ngữ nghĩa
n-step.

*Fix:* ship `self._step_obs` (stack **tại thời điểm ra quyết định**), nên transition là
`(obs_trước, action, R_tổng, obs_sau_n_bước)` — đúng MDP.

---

## 2. Phần "học từ người" — 3 lỗi và fix

### 2.1 Demo gán nhãn **lúc bấm**, không phải lúc **quyết định**

Con người bấm phím 200–300 ms **sau** khi nhìn thấy vật cản. Demo gán nhãn tại frame bấm dạy
bot "bấm khi vật cản đã ở ngay mặt" — ở 30 FPS đó là thời gian phản ứng bất khả thi.

*Fix:* `bc.label_backdate_ms = 220` (≈7 frame). `DemoRecorder._handle_press` **viết lại nhãn các
frame NOOP ngay trước cú bấm** bằng action vừa bấm. Chỉ đè nhãn NOOP — một cú JUMP thật không
bao giờ bị ghi đè (có test `test_backdating_never_overwrites_a_different_dodge`).

### 2.2 Demo ghi **khác** khung nhìn mà bot dùng lúc chơi

Recorder ghi dải ground đã cắt, trong khi (sau fix 1.1) bot nhìn toàn khung. BC vì thế học
trên một phân bố input khác phân bố inference.

*Fix:* recorder ghi `z.policy_gray` — đúng array CNN tiêu thụ. `dataset.validate_episode` nhận
`expected_size` từ `cfg.perception.policy_size` thay vì hardcode 84 (trước đây demo quay ở
`obs_size` khác bị loại **toàn bộ**, đọc lên thành "BC không có dữ liệu").

### 2.3 Compounding error — bệnh gốc của Behavior Cloning

BC chỉ thấy trạng thái của **người**. Bot lệch quỹ đạo một chút là rơi vào trạng thái demo
chưa bao phủ, và ở đó không có giám sát nào → lỗi dồn. Đây chính là lý do BC "85% accuracy"
vẫn chết sớm khi rollout.

*Fix:* **HG-DAgger** (human-gated DAgger). `DemoRecorder.arm_dagger(True)` (nút 🎓 trên GUI hoặc
F10 khi bot đang chơi): recorder giữ một cửa sổ trượt `dagger_pre_frames` frame; ngay khi người
chơi bấm phím giành quyền, nó snapshot cửa sổ đó và ghi tiếp `dagger_tail_frames` frame, lưu vào
`<demos>/dagger/dagger_*.npz` với `meta["dagger"]=True`. `dataset.load_episodes` đã đổi sang
`rglob` nên các episode sửa lỗi này **tự động gộp vào lần BC kế tiếp** — tức là giám sát nằm
đúng trên phân bố trạng thái của bot, tại đúng chỗ bot sai.

Repo này **đã có sẵn** (không phải sửa): DQfD (`dqfd_agent.py`, joint loss TD + supervised +
large-margin, demo buffer riêng), mirror augmentation + keypress window
(`demo_augment.py`, `DemoAugmentConfig`), self-imitation, RND, NoisyNets, QR-DQN, Polyak target,
RAD augmentation.

---

## 3. Bằng chứng thực chiến (kiểm chứng trong phiên này)

Bốn project công khai, chia làm hai nhóm kết quả **ngược nhau**:

### 3.1 Nhóm thành công bằng **Behavior Cloning**

| Project | Cách làm | Kết quả (theo README, đã fetch) |
|---|---|---|
| `nikp06/subwAI` | CNN supervised, tự chơi để thu nhãn 5 lớp; **augment lật ngang** (dataset ×2) | accuracy **85%**; "finish runs of **over a minute** regularly" |
| `Bezet1/subway-surfers-ai` | CNN 224×224 RGB; **3 model theo tốc độ game** (slow 0–30 s / medium 30–120 s / fast 120 s+) | train 94.2% / val 89.7% / test 87.3%; inference **~15 ms**; quãng đường **~20.000 m** (TB 15.000–18.000 m) |

Hai điểm đáng lấy: (a) **mirror augmentation** — repo này đã có sẵn trong `demo_augment.py`;
(b) **tách model theo tốc độ** — Subway Surfers tăng tốc theo thời gian, một policy đơn lẻ phải
phục vụ nhiều phân bố.

### 3.2 Nhóm thành công bằng **RL thuần** — `khxji/lunai` (tool trong video "không cần code")

Đọc trực tiếp `Lunai.py`: `stable_baselines3.DQN`, `totalTimeStep = 1.000.000`, buffer 150.000,
LR 2.5e-4; observation là **grayscale toàn cửa sổ game** resize **74×110** (không cắt);
done bằng OCR chữ game-over; reset bằng auto-click → chạy ~3.000 run / ~20 giờ không cần người.

**Bí quyết nằm ở reward:** `get_reward()` trả **điểm OCR** hoặc `counter_reward += 1` **mỗi
step** — và **không có hình phạt chết**. Tổng return một run tỉ lệ thuận với thời gian sống,
nên episode dài hơn tự nhiên có điểm cao hơn. Đây chính là tính chất mà test
`test_return_is_monotonic_in_survival` trong repo này now pin lại.

### 3.3 Nhóm **thất bại** bằng RL — CS386 Group 6 (DQN-CNN + LSTM + eligibility trace)

Theo slide của nhóm: observation grayscale 128×128; reward **+2 mỗi state sống, −10 khi chết**;
done bằng template matching nút PLAY; train "very less training".
Kết quả: điểm TB **~250**, cao nhất **512**, và tự thú
*"it learnt to keep jumping or rolling as the agent is not penalized for these actions"*.

Đây là **đối chứng quan trọng nhất**: reward sống của họ (+2) lớn hơn hẳn bản cũ của repo này,
nhưng **vẫn để −10** và **không khử được spam hành động** → vẫn hỏng. Kết luận: chỉ tăng
reward sống là **chưa đủ**; phải (a) bỏ hình phạt chết âm lớn, (b) thay "thưởng vì có bấm"
bằng "thưởng vì né thật".

### 3.4 Tổng hợp

| | lunai (thành công) | CS386 (thất bại) | Repo này **trước** | Repo này **sau** |
|---|---|---|---|---|
| Observation | full 74×110 | full 128×128 | **chỉ 75% dưới** | **full 84×84** |
| Reward sống | +1/step đều | +2/state | +0.02→0.5/frame | 0.5/frame (giữ) |
| Phạt chết | **0** | −10 | −5 (cũ: −10) | **0** (mặc định) |
| Thưởng "có bấm" | không | không | **có** (`hazard_bonus`) | **tắt** mặc định |
| Thưởng né thật | không | không | không | **có** (`clear_bonus`) |
| Frame-skip MDP | không cần (env đơn giản) | không | **không** | **có** (k=3) |
| ε schedule | SB3 mặc định | — | theo frame | **theo quyết định** |
| Train | ~1 M step, 20 h auto-reset | "very less" | — | — |

---

## 4. "Nhìn như Gemini/ChatGPT" nhưng mượt cho máy yếu

Ba lớp, chọn theo ngân sách CPU:

1. **CNN nhỏ hiện có** (repo này: profile `strict_lite`…; inference đo được ở mức ~ms trên CPU)
   + observation toàn khung. Đây là mức mặc định và đã đủ cho BC.
2. **CV cấu trúc** (`obstacle_perception.py`, <1 ms): lưới 3 làn × 5 mức xa-gần → vector vài chục
   số. Dùng cho reward shaping ngay; về sau có thể thay hẳn pixel input cho RL (học nhanh hơn
   hàng trăm lần vì state đã tách bạch).
3. **VLM gán nhãn NGOÀI LUỒNG**: chạy model lớn (Gemini/GPT) trên **video đã ghi** để sinh nhãn
   hành động/giải thích, rồi **distill sang CNN nhỏ**. Lúc chơi thật chỉ CNN chạy nội bộ → không
   tốn latency, không cần mạng. Đây là cách duy nhất "trí thông minh cỡ Gemini" mà vẫn mượt trên
   máy yếu: tri thức nằm ở bước huấn luyện, không nằm ở bước suy luận.

---

## 5. Bảng hyperparameter mới (mặc định trong `config.py`)

| Nhóm | Khóa | Giá trị | Ý nghĩa |
|---|---|---|---|
| perception | `policy_full_frame` | `True` | policy thấy toàn khung |
| perception | `obs_size` | `84` | cạnh observation vuông |
| obstacle | `enabled / lanes / depth_rows` | `True / 3 / 5` | lưới occupancy |
| obstacle | `near_rows / confirm_cells` | `2 / 2` | hàng "sắp va" + debounce |
| rl | `frame_skip` | `3` | 1 quyết định / 3 frame |
| rl | `epsilon_decay_steps` | `100000` | ε theo quyết định |
| rl | `n_step / gamma` | `5 / 0.99` | theo bước quyết định |
| reward | `alive_per_frame` | `0.5` | bộ đếm dương (~15/s) |
| reward | `death_penalty` | `0.0` | không phạt chết |
| reward | `reward_clip_min` | `0.0` | return không âm |
| reward | `use_hazard_bonus` | `False` | tắt thưởng spam |
| reward | `clear_bonus` | `0.5` | thưởng né thật |
| reward | `danger_penalty / action_cost` | `0.0 / 0.0` | ablation, tắt mặc định |
| bc | `label_backdate_ms` | `220` | lùi nhãn ~7 frame |
| bc | `dagger / dagger_tail_frames` | `True / 45` | HG-DAgger |
| demo_augment | `mirror_horizontal` | `True` | lật ngang (đã có sẵn) |

`config.example.json` đã được **regenerate từ chính các mặc định này**. Bản cũ của nó chứa
`alive_per_frame = 0.02`, `death_penalty = -10.0`, `hazard_bonus = 0.1`,
`epsilon_decay_frames = 150000` — tức **đúng cấu hình gây plateau**: ai copy file đó thành
`config.json` sẽ âm thầm vô hiệu hoá toàn bộ fix (vì `from_dict` merge đè lên mặc định).

---

## 6. Lộ trình chạy trên máy thật

**G0 — xác nhận pipeline (10 phút).** Chạy headless, kiểm tra `env.game.total_steps` tăng đúng
`frame_skip` mỗi step, và `reward_clip_min == 0`.

**G1 — BC.** Calibrate vùng + anchor + nút respawn. Quay ~30–60 demo **chơi thật**
(không cần hoàn hảo). Kiểm tra per-class recall ≥ 70% trước khi sang RL.

**G2 — DQfD.** `bc.bc_pretrain = True` (mặc định) → learner dựng `DQfDAgent`, demo vào
demo buffer riêng, joint loss TD + supervised + large-margin. Đây là chốt chống "RL quên bài
người dạy".

**G3 — HG-DAgger.** Bật bot, bấm 🎓/F10, chỉ can thiệp lúc bot sắp chết. Mỗi lần giành quyền là
một correction episode đúng vào phân bố trạng thái của bot. Lặp lại 3–5 vòng, mỗi vòng BC lại.

**G4 — RL thuần (tuỳ chọn, kiểu lunai).** Chỉ khi auto-reset + nhận game-over **chắc chắn**.
Cần hàng trăm nghìn → 1 triệu bước, chạy nhiều giờ không gián đoạn. CS386 thất bại một phần vì
"very less training".

**Năm phép kiểm tra 10 phút khi vẫn không tiến bộ:**
1. Chạy ε=0 vs ε=1: nếu **random sống lâu hơn greedy** → policy đang học điều sai (reward/observation).
2. Log `obstacles_cleared` và `danger_steps`: nếu luôn 0 → tracker không thấy vật cản (chỉnh
   `grad_threshold` / `min_cell_frac`).
3. Log số frame/episode: nếu mọi episode ~1 s → vấn đề nằm ở **death detection / respawn**, không
   phải ở thuật toán.
4. Kiểm tra `bc.label_backdate_ms` có bị config.json cũ đè về 0 không.
5. Kiểm tra `reward.death_penalty` thực tế đang dùng (config.json đè mặc định).

---

## 7. Ánh xạ fix ↔ file

| Fix | File | Kiểm chứng |
|---|---|---|
| Full-frame observation | `perception.py`, `config.py` | `TestFullFramePolicyObservation` (3 test) |
| Frame-skip MDP | `environment.py` (`GameEnvironment.step`, `BotActor._process_frame`), `config.py` | `TestFrameSkipMDP` (5 test) |
| ε theo quyết định | `agent.py`, `environment.py` | `TestEpsilonPerDecisionStep` (3 test) |
| Reward bộ đếm dương | `rewards.py`, `config.py` | `TestPositiveCounterReward` (8 test) |
| Obstacle grid + clear bonus | `obstacle_perception.py` (mới), `environment.py` | `TestObstacleTracker` (8 test) |
| Label back-dating | `demonstration_recorder.py` | `TestLabelBackdating` (4 test) |
| HG-DAgger | `demonstration_recorder.py`, `gui.py` | `TestHGDAgger` (5 test) |
| Demo đúng size + gom đệ quy | `dataset.py`, `learner_worker.py`, `app.py` | `TestHGDAgger` (2 test cuối) |
| Config plumbing | `config.py`, `config.example.json` | `TestConfigPlumbing` (4 test) |

---

## Phụ lục A — `khxji/lunai` (RL thuần thành công)

- `stable_baselines3.DQN`, `totalTimeStep = 1.000.000`, buffer 150.000, LR 2.5e-4, batch 32.
- Observation: grayscale **toàn cửa sổ game**, resize **74×110**, 1 kênh.
- Reward: `get_reward()` → điểm OCR (easyocr) **hoặc** `counter_reward += 1` mỗi step.
  **Không có penalty chết.** Chế độ counter được chính README khuyến nghị vì "INCREASES AI FPS".
- Done: OCR chuỗi game-over tại `done_location`. Reset: auto-click (`Custom Reset Click`).
- Mỗi phím thành 2 action (keydown/keyup) → ~10 action rời rạc.

**Bài học áp dụng:** reward dương đều + không phạt chết + observation full-frame + auto-reset
chạy dài. Ba cái đầu đã thành mặc định trong repo này; cái thứ tư là điều kiện vận hành.

## Phụ lục B — CS386 Group 6 (RL + LSTM thất bại)

- DQN-CNN + **LSTM** + experience replay + **eligibility trace (n-step)**; grayscale 128×128.
- Reward **+2/state sống, −10 khi chết**; done bằng template matching ảnh nút PLAY.
- Kết quả: TB ~**250**, cao nhất **512**; "keep jumping or rolling"; tự nhận "very less training".

**Bài học áp dụng:**
1. Kiến trúc xịn hơn (LSTM, trace) **không cứu được MDP sai** — LSTM chỉ là bộ nhớ, nó không tạo
   ra tín hiệu vốn không có.
2. Reward sống lớn (+2) **vẫn không đủ** nếu còn penalty chết âm lớn và không khử được spam.
3. Eligibility trace/n-step: repo này đã có (`rl.n_step`).
4. LSTM: **chưa cần** trên máy yếu — frame stack theo quyết định (~0.4 s) + obstacle grid đã cấp
   trí nhớ chuyển động, trong khi LSTM tốn CPU và dễ hại khi dữ liệu ngắn.

---

*Cập nhật lần cuối: v1.24.0 — 1125 test xanh (`pytest -q`), 40 test mới trong
`tests/test_deep_research_fixes.py`.*
