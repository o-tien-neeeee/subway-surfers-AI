# BÁO CÁO DEEP RESEARCH — Subway Surfers AI

**Ngày:** 2026-09-01 · **Phiên:** research thuần (không sửa code)
**Câu hỏi gốc:** Train 6000 episode mà thời gian sống trung bình chỉ tăng ~1 giây so với episode 1.
Tại sao? Làm thế nào để AI "nhìn" tốt như Gemini/Claude/ChatGPT nhưng chạy mượt trên máy yếu
(i5-7200U, 2C/4T, không GPU)? Học từ người (demo) nên làm thế nào cho đúng?

---

## 0. Tóm tắt cho người vội (TL;DR)

1. **6000 episode ≈ 18–45 nghìn quyết định thật, không phải 6000 "lần học".** Con số này
   **nhỏ hơn 100–1000 lần** mức tối thiểu mà học tăng cường (RL) từ pixel thô cần
   (DQN gốc Atari: ~200 triệu frame; các project Subway Surfers thành công ngoài thực tế
   **đều không dùng RL from scratch** — xem §4.1).

2. **Đây gần như chắc chắn KHÔNG phải lỗi "model quá nhỏ" hay "máy quá yếu".** Mạng 49k–348k
   param của bạn thừa sức giải bài toán này (project thành công chỉ dùng CNN nhỏ, inference
   15ms trên CPU). Vấn đề nằm ở **thiết kế bài toán (MDP)**: observation, nhịp quyết định,
   reward, và cách gán nhãn demo — có **ít nhất 7 lỗi thiết kế phối hợp** khiến tín hiệu
   học gần như bằng 0. Quan trọng nhất:

   - **Observation cắt mất đường chân trời** — nơi vật cản xuất hiện. AI không thể né thứ
     nó không bao giờ nhìn thấy (`perception.py`: `ground = image[split:, :]` với split=25%).
   - **Reward sống +0.02/frame bị chết −10:** episode phải sống >16.7 giây mới hòa vốn;
     episode ngắn (3–10s) đều có tổng reward **âm như nhau** → mọi hành vi tệ đều "giống
     hệt nhau" trong mắt bộ học. Tín hiệu phân biệt hành động tốt/xấu gần như bằng 0.
   - **Hệ số chiết khấu γ=0.99 tính theo FRAME (30 FPS):** tầm nhìn hiệu dụng của Q-value
     chỉ ~3.3 giây (100 frame). Cái chết xảy ra 5–8 giây sau một quyết định né → tín hiệu
     chết không truyền ngược về được quyết định gây ra nó.
   - **ε (exploration) decay theo env-frame (150k frame ≈ 800 episode):** trước khi agent
     kịp học được gì, ε đã tụt còn 0.05 và policy tham lam "mù" bị **khóa cứng** — đúng
     pattern "plateau sau vài trăm episode".
   - **Transition nhãn action bị lệch/sai nhịp:** quyết định chỉ diễn ra mỗi 2–4 frame,
     nhưng reward/transition được ghi cho MỌI frame với action lặp lại từ frame trước
     (`environment.py._process_frame` dùng `self._last_action`); n-step (n=3) gộp reward
     3 frame (~100ms) nhưng gán cho 1 action. Học bị nhiễu và lệch nhân quả.
   - **Hazard bonus +0.1 (lớn gấp ~17 lần reward sống/frame)** được thưởng cho BẤT KỲ cú
     bấm nút nào khi horizon-detector kêu → dạy agent "thấy động là bấm bừa" (đúng triệu
     chứng project RL khác gặp: *"learn to keep jumping or rolling"*).
   - **Demo của người gán nhãn theo phím ĐANG GIỮ** (`demonstration_recorder.py`,
     `_current_action`), không phải thời điểm RA QUYẾT ĐỊNH, và không hiệu chỉnh độ trễ
     phản ứng (~200–300ms) → nhãn trễ, BC học "phản ứng muộn" và học cả các frame animation
     thừa.

3. **Bằng chứng từ thế giới thực (rất quan trọng):** Mọi project Subway Surfers AI công khai
   chạy được đều theo một trong hai hướng:
   - **Học bắt chước người (Behavior Cloning / imitation):** CNN nhỏ học từ ảnh người chơi
     gán nhãn → chơi được >1 phút thường xuyên ([nikp06/subwAI], 85% acc, sau augment lật
     ngang còn chắc hơn) hoặc **20.000m liên tục** ([Bezet1/subway-surfers-ai], 3 model theo
     3 mức tốc độ, train/val/test 94/90/87%, inference ~15ms).
   - **RL from pixel thì thất bại đúng kiểu của bạn:** [m4n4n-j/subway-surfers-AI-main]
     dùng DQN+CNN, kết quả: *"despite very less training... learnt to keep jumping or
     rolling"* — tập nhảy/lăn vô tội vạ, điểm thấp.

4. **Hướng đi đúng cho máy yếu, theo thứ tự ưu tiên:**
   **(A) Sửa MDP trước (observation toàn cảnh + bước thời gian nhất quán + reward lại
   + ε/γ lại)** — không sửa thì thuật toán nào cũng vô dụng.
   **(B) BC từ demo người làm trục chính**, ghi demo ĐÚNG cách (gán nhãn theo thời điểm
   bấm, dãn nhãn ngược về thời điểm vật cản còn ở xa, augment lật ngang, nhiều mức tốc độ),
   rồi **DAgger/HG-DAgger** (người chỉ can thiệp khi bot sắp chết) để gom đúng phân bố
   trạng thái bot gặp.
   **(C) DQfD** (demo nằm vĩnh viễn trong replay + large-margin loss) nếu vẫn muốn RL tinh
   chỉnh.
   **(D) Tùy chọn "AI nhìn như Gemini":** dùng VLM mạnh (Gemini/Claude/Qwen...) chạy NGOÀI
   LUỒNG CHƠI để gán nhãn hàng loạt video chơi game, rồi chưng cất (distill) xuống CNN nhỏ
   chạy nội bộ — đã có paper 2025–2026 chứng minh pipeline này (DGC, VLM-annotated datasets).
   **(E) Tùy chọn mượt nhất trên CPU yếu:** perception cổ điển (mask màu/HSV phát hiện tàu,
   hàng rào theo 3 làn + trạng thái nhảy/cuộn của nhân vật) → vector trạng thái nhỏ → MLP/
   rule; RL trên vector học nhanh hơn pixel 100–1000 lần và CPU gần như miễn phí.

---

## 1. Chẩn đoán: vì sao 6000 episode = +1 giây?

### 1.1 Trước hết — quy mô dữ liệu thật

| Đại lượng | Giá trị của bạn |
|---|---|
| Episode trung bình khi mới chạy | ~5–10 s (chết ở cụm vật cản đầu tiên) |
| Frame/episode (30 FPS, cadence quyết định 2–4 frame) | 150–300 frame, ~40–150 **quyết định** |
| 6000 episode | ~0.5–1.8 triệu frame, **~180k–900k transition** |
| Update learner (cap 20 update/s, nhưng actor chết nhiều + pause lúc respawn) | thực tế thường chỉ bằng ~0.3–1× số transition |
| ε-decay hoàn tất sau | 150.000 env-frame ≈ **episode 600–1.000** |

So sánh: DQN gốc Atari dùng **200 triệu frame**, Rainbow/Agent57 cũng cỡ đó; các thuật toán
sample-efficient tốt (DreamerV3, DART Atari-100k) vẫn cần **100k BƯỚC QUYẾT ĐỊNH** (tức
~400k frame với frame-skip 4) và đó là trên Atari với reward rõ ràng. Bài toán của bạn có
reward yếu hơn nhiều. ⇒ Bản thân "6000 episode" không hề lớn; nó nằm trong vùng "chưa kịp
học gì" đối với RL pixel thô — **và đặc biệt vô vọng nếu MDP bị lỗi** (phần dưới).

### 1.2 Bảy lỗi thiết kế phối hợp (mỗi lỗi đều đủ để làm learning phẳng)

#### Lỗi #1 — OBSERVATION KHÔNG CHỨA THÔNG TIN NÉ (nghiêm trọng nhất)

`perception.py::ZonePreprocessor.process`:
```python
split  = round(region_h * 0.25)        # horizon_frac = 0.25
ground = image[split:, :, :]           # <-- CNN chỉ thấy 75% DƯỚI màn hình
ground_gray = resize(gray(ground), 84x84)
```
Vật cản trong Subway Surfers **xuất hiện ở chân trời rồi mới lao xuống** ([fandom wiki]:
"The player must... avoid randomly generated obstacles"; [poki]: "Runs start slow, then
fill the screen with obstacles in seconds... you're reacting before you can see what's
coming"). Tức là trong ~1–2 giây đầu của mỗi vật cản, thông tin né chỉ tồn tại ở **25%
trên** — phần đó được dùng cho `HorizonDetector` (diff thô 40×40) chứ **không được đưa vào
CNN**. Khi vật cản đã rơi vào vùng "ground" (75% dưới) thì thường chỉ còn 0.5–1.5 giây để
phản ứng; tốc độ game còn **tăng dần và đạt max vào phút thứ 3** ([fandom]: "It takes
exactly 3 minutes to reach full speed").

Hệ quả: CNN phải học "dự báo tương lai" từ phần cuối quỹ đạo — bài toán gần như không giải
được bằng 4 frame chồng (xem Lỗi #6), và reward sống không thể gán cho quyết định đúng vì
quyết định đáng lẽ phải xảy ra TRƯỚC KHI vật cản vào observation.

Thêm: ảnh grayscale 84×84 làm mất **màu sắc** — trong Subway Surfers, tàu (đỏ/vàng/...),
hàng rào thấp (vượt bằng nhảy), chắn ngang trên cao (luồn bằng cuộn) phân biệt mạnh bằng
**màu + hình dạng**. Grayscale làm loại tàu vs loại chắn rất dễ lẫn.

#### Lỗi #2 — REWARD KHÔNG PHÂN BIỆT ĐƯỢC HÀNH ĐỘNG TỐT/XẤU

`config.py::RewardConfig`: sống +0.02/frame (~0.6/s), chết **−10**, hazard +0.1,
clip [−10, +1].

- Episode T giây (T < 16.7s) có tổng reward ≈ 0.6T − 10 < 0. T=3s → −8.2; T=8s → −5.2;
  T=15s → −1. **Mọi episode ngắn đều âm đậm, chênh lệch giữa chết sớm/chết muộn chỉ là
  vài phần trăm của biên độ reward** (±10). Huber loss + grad-clip 10 + LR 1e-4 → gradient
  bị chi phối bởi cột "−10" nhiễu (chết do bay màu, do focus, do respawn hụt...) chứ không
  phải bởi khác biệt nhỏ giữa các chiến lược.
- Survival reward là **reward trễ pha (delayed)**, lại không có shaping tiềm năng
  (potential-based) cho "né thành công một vật cản". Mỗi lần né xong một tàu KHÔNG được
  cộng gì cụ thể; chỉ có hazard +0.1 thô (xem Lỗi #4).
- Vì reward sống theo **thời gian thật** (`alive_reward` dùng `ts = monotonic clock`), khi
  máy giật/lag (máy yếu, Chrome ngốn CPU), reward/giây dao động theo hiệu năng máy ⇒ môi
  trường phi tĩnh (non-stationary) — policy học lúc được thưởng vì... máy mượt.

**Bài học từ lý thuyết/practice:** reward sống đơn thuần trong endless runner tạo ra
cực trị cục bộ "bấm càng ít càng tốt" (NOOP an toàn ngắn hạn); còn thưởng dày cho bấm nút
sẽ tạo cực trị "bấm liên tục" (đúng như project RL m4n4n-j ghi nhận: agent học nhảy/cuộn
liên miên vì không bị phạt). Cần shaping mật độ, đúng nhân quả: cộng khi **vật cản trôi qua
người an toàn**, phạt khi vào trạng thái nguy hiểm, phạt hành động vô nghĩa.

#### Lỗi #3 — TẦM NHÌN CHIẾT KHẤU QUÁ NGẮN (γ theo frame 30 FPS)

γ = 0.99/frame @30FPS ⇒ horizon hiệu dụng 1/(1−γ) = **100 frame ≈ 3.3 giây**.
Quyết định né (đổi làn/nhảy) xảy ra khi vật cản còn cách ~1–2.5s; cái chết vì quyết định
sai thường đến sau đó 1–2s nữa. Với n-step=3 (100ms) + γ^n bootstrap, tín hiệu −10 tại
time-of-death bị nhân với 0.99^(khoảng cách) — cách 5s (150 frame) thì hệ số ≈ 0.22, cách
8s ≈ 0.05. Tức là **Q-value của hành động gây chết hầu như không bị trừ**.

Cách chuẩn (Atari): frame-skip 4, γ=0.99 theo **BƯỚC QUYẾT ĐỊNH** → horizon ~100 quyết
định × 4 frame × 1/15FPS ≈ **26 giây**. Bạn đang có ~3.3s — ngắn hơn cả một pha né.

#### Lỗi #4 — HAZARD BONUS DẠY ĐIỀU SAI

`rewards.py::PendingHazardTracker`: khi horizon detector (frame-diff trên 40×40) báo động,
nếu ~2 frame sau "hết động" VÀ action lúc đó ∈ {LEFT, RIGHT, JUMP, SLIDE} ⇒ +0.1.

- Horizon-diff kêu liên tục vì **mặt đất/hoa văn cuộn, tàu chạy, coin lấp lánh** — không
  phân biệt được vật cản thật với mọi thứ khác chuyển động.
- Thưởng cho MỌI cú bấm khi có "động" ⇒ tối ưu cục bộ: cứ bấm nút bất kỳ. Mỗi lần "né hụt
  mà may không chết" vẫn được thưởng.
- +0.1 so với +0.02 sống/frame ⇒ một cú bấm may mắn ≈ 17 frame sống; agent hợp lý sẽ bấm
  spam. Đây chính là cơ chế tạo ra "jumping/rolling bot".

#### Lỗi #5 — ε-GREEDY DECAY TRƯỚC KHI HỌC GÌ → KHÓA POLICY MÙ

ε: 1.0 → 0.05 tuyến tính trên **150.000 env-frame**. Episode ~150–300 frame ⇒ ε chạm 0.05
sau ~500–1.000 episode. Trong 500 episode đầu, buffer toàn transition của policy ngẫu nhiên
(80% random vì ε cao), update lại ít (Lỗi #1 khiến tín hiệu yếu), nên khi ε tụt, policy
tham lam gần như là **policy ngẫu nhiên có hệ thống** — và từ đó hầu như không còn thăm dò
nữa (5%). Cải tiến tiếp theo gần như không thể xảy ra vì: (a) không thăm dò ⇒ không thấy
trạng thái mới; (b) PER ưu tiên lỗi TD lớn từ dữ liệu cũ; (c) best-model gate khóa model
"may mắn nhất 3 episode". Đây là công thức kinh điển cho **đường học phẳng từ rất sớm** —
khớp 100% với mô tả "+1 giây".

Khuyến nghị mẫu chung cho DQN (Mnih 2015; Nature DQN): ε 1.0→0.1 trên **1 triệu frame**
(hoặc ≥ 10% tổng frame đời train), và giữ ε_eval = 0.05. Quan trọng hơn: **đừng decay
theo env-frame khi cadence động** — hãy decay theo **số quyết định** hoặc theo update, và
bắt đầu bằng BC (ε không cần 1.0 nếu policy đã biết chơi cơ bản).

#### Lỗi #6 — NHỊP QUYẾT ĐỊNH, FRAME STACK VÀ GÁN ACTION KHÔNG NHẤT QUÁN

Một agent kiểu Atari chuẩn: mỗi "bước" = lặp action trong k frame liên tiếp (frame-skip
k=4), quyết định 1 lần, stack 4 frame **theo bước** (≈0.27s lịch sử @15FPS quyết định),
transition (s_k, a, r, s_{k+1}) sạch sẽ.

Trong code hiện tại (`environment.py.BotActor._process_frame`):
- Quyết định chỉ khi `scheduler.on_frame()` (mỗi 2–4 frame, hoặc danger);
- Nhưng MỌI frame đều: `stack.push`, `reward_calc.step(action=self._last_action, ...)`,
  `_ship_transition(action=self._last_action, ...)`;
- Tức 2–3 frame/quyết định được gán **nhãn action cũ**, trong khi game đã nhận action mới
  (lane đổi tức thì, nhảy bắt đầu) từ frame trước; reward cho frame này lại thuộc hệ quả
  của action MỚI.
- N-step (n=3) gộp reward 3 frame (~100ms @30FPS) gán cho 1 action — nhưng 100ms này có
  thể chứa 0 hoặc 2 hành động khác nhau tùi cadence.
- Frame stack 4 ở 30 FPS = **0.13 giây lịch sử** (Atari: 4×4/60 ≈ 0.27s). Vận tốc/tốc độ
  lao tới của vật cản ước lượng từ 4 frame cách nhau 33ms rất nhiễu; lại không thấy chân
  trời (Lỗi #1) nên tín hiệu chuyển động của vật cản xa = 0.

Kết quả: dữ liệu huấn luyện không tương ứng với MDP mà policy sẽ gặp lúc greedy
(train/action mismatch) — DQN không thể hội tụ đúng trên dữ liệu này dù mạng to bao nhiêu.

#### Lỗi #7 — DEMO NGƯỜI GÁN NHÃN SAI THỜI ĐIỂM + KHÔNG CÓ DAgger

`demonstration_recorder.py`: mỗi frame gán `_current_action` = phím **đang giữ**.
- Subway Surfers kích hoạt hành động tại thời điểm **bấm** (tap/keydown), lane change tức
  thì; giữ phím ~60–110ms. Nhãn "JUMP" do đó chỉ phủ ~2–4 frame quanh cú bấm, rồi các frame
  bay trên không (300–500ms) lại là NOOP.
- Người chơi bấm nút **trước khi va chạm ~0.2–0.6s** (phản ứng người ~200–300ms + dự
  đoán). Frame ngay trước cú bấm (vật cản còn ở tầm xa, chính là frame mà bot CẦN ra quyết
  định) lại mang nhãn NOOP ⇒ BC học "đợi đến sát nơi mới bấm" → toàn trễ.
- Không có hiệu chỉnh độ trễ, không gán nhãn theo **sự kiện bấm dãn ngược** (label
  backdating: gán action cho frame cách thời điểm bấm ~k frame, theo tốc độ game), không
  lọc frame menu/transition, không augment lật trái–phải (subwAI cho thấy mirroring tăng
  robustness rõ rệt), không tách mức tốc độ (Bezet1 dùng 3 model slow/medium/fast vì game
  đổi tốc độ → phân bố ảnh thay đổi mạnh).
- BC hiện tại train 8 epoch LR 1e-3 trên 49k-param net với vài demo ngắn → overfit nhanh;
  sau đó RL online với ε=1.0 đạp đổ BC (catastrophic forgetting) vì demo không nằm trong
  replay (DQfD chưa có margin loss thực sự — README ghi "DQfD-style" nhưng code chỉ có CE).

#### Lỗi vận hành (cần kiểm chứng trên máy thật, có thể âm thầm phá mọi thứ)

- **Focus gate / keymap:** nếu `focus_gate()` fail hoặc phím gửi không tới game (Poki cần
  click vào canvas lần đầu, hoặc trang nuốt phím khi có overlay), agent "chơi" nhưng game
  không nhận → episode chết ở cùng một mốc, học mãi không tiến. README cũng liệt kê
  "respawn/reliability chưa kiểm chứng trên game thật". **Test bắt buộc:** bật log, xác
  nhận 100% cú bấm xuất hiện trong game (ví dụ đếm phản hồi hình ảnh: nhân vật đổi làn).
- **Death detector giả:** anchor màu sai/nhấp nháy ⇒ chết giả giữa run (cắt episode thành
  các đoạn 3–5s, reward −10 oan) — dữ liệu toàn nhãn chết sai.
- **Respawn không ăn:** sau chết không restart được ⇒ màn hình game-over bị coi là "sống",
  các frame tĩnh bị đẩy vào buffer.
- **Synthetic game ≠ game thật:** toàn bộ 403 test chỉ chứng minh pipeline chạy;
  `SyntheticGame` có nhiễu/khó rất khác game thật (vật cản hình chữ nhật đặc màu, spawn
  đều). Đừng dùng kết quả headless để kỳ vọng ngoài đời.

### 1.3 Chẩn đoán phân biệt: nên kiểm tra gì trước (10 phút, không cần code mới)

1. Xem GUI/log: phân phối **thời gian sống từng episode** theo thời gian train. Nếu tất cả
   episode đều chết ở ~3–8s và phương tích gần như không đổi ⇒ khớp Lỗi #5 (policy bị khóa).
2. Xem `suppressed_duplicates`, `expired`, `danger_overrides`, tần suất action thực thi
   (`action_step/giây`). Nếu action thực thi ~0 ⇒ lỗi focus/keymap.
3. Xem log death detector: `distance` lúc chết thật vs lúc chơi bình thường; số lần
   "death confirmed" khi bạn đang xem và biết là chưa chết ⇒ chết giả.
4. Chạy thử 1 episode với ε = 0.0 (greedy sau BC) và 1 episode ε = 1.0 (random): nếu
   survival gần như nhau ⇒ policy chẳng học được biểu diễn hữu ích (Lỗi #1/#6); nếu ε=1
   sống LÂU HƠN greedy ⇒ policy đang học điều SAI (Lỗi #2/#4).
5. Lưu 30–60 frame observation thực tế (ảnh 84×84 mà CNN nhìn) ngay trước một cái chết:
   nếu không nhìn thấy vật cản trong 5 frame cuối ⇒ xác nhận Lỗi #1.

---

## 2. Giải mã game Subway Surfers (bài toán thật là gì?)

Nguồn: [Wikipedia], [Subway Surfers Fandom wiki], [Poki official page], [runsubway.com].

- **Thể loại:** endless runner 3 làn đường, camera theo sau trên cao. Chạy tự động về phía
  trước, người chơi chỉ ra 4 lệnh: **trái / phải (đổi làn, tức thì), lên (nhảy), xuống
  (cuộn/lăn)** + NOOP. (Đúng action space 5 của bạn.)
- **Vật cản chính:**
  - **Tàu hỏa (train):** đứng yên hoặc chạy tới, chiếm 1 làn, cao → phải **đổi làn** (không
    nhảy qua được tàu thường; nhảy được chướng ngại thấp).
  - **Hàng rào chắn thấp (low barrier):** nhảy QUA (jump).
  - **Chắn ngang trên cao (high barrier / bar):** luồn DƯỚI bằng cuộn (roll/slide).
  - **Tường hầm / chướng ngại kép:** cần đổi làn phối hợp; đôi khi 2 làn bị chặn ⇒ phải
    chọn làn còn trống từ xa.
- **Tốc độ:** bắt đầu chậm, **tăng dần, đạt max vào đúng phút thứ 3** của mỗi run
  ([fandom]: "It takes exactly 3 minutes to reach full speed when running") ⇒ phân bố hình
  ảnh và thời gian phản ứng thay đổi liên tục; đây là lý do Bezet1 chia 3 model theo
  slow/medium/fast.
- **Cơ chế phụ:** coin (không bắt buộc để sống), power-up (jetpack = bay bất tử vài giây,
  magnet, sneakers nhảy cao, 2× multiplier), hoverboard (double-tap/space = chịu được 1
  va chạm), revive bằng key/ad. Với mục tiêu "sống lâu", bot nên **bỏ qua coin/powerup ở
  giai đoạn đầu** và có thể học dùng hoverboard sau.
- **Điều khiển bản web (Poki):** mũi tên trái/phải/lên/xuống; swipe trên mobile; Space =
  hoverboard. Đổi làn bằng một cú nhấn (tap), không cần giữ; nhảy/cuộn là animation hằng
  ~0.4–0.7s, trong đó có thể đổi làn giữa không trung (advanced).
- **Mấu chốt tri nhận:** trò chơi về bản chất là **"đọc" vật cản ở 3 làn × 3–5 mức khoảng
  cách từ xa tới gần** rồi chọn 1 trong 5 lệnh. Đây là bài toán phân lớp theo không-thời
  gian đơn giản hơn Atari nhiều — nhưng BẮT BUỘC nhìn được chân trời và có lịch sử đủ dài
  để ước lượng tốc độ lao tới.

Suy ra thiết kế đầu vào tối thiểu: vùng capture toàn bộ khung game (không cắt 25% trên),
≥ 4–8 frame lịch sử (hoặc tốt hơn: tính sẵn "bản đồ chiếm lĩnh" 3 làn × n mức khoảng cách
bằng CV — §5), màu sắc giữ lại (hoặc mask màu đặc thù), và chu kỳ quyết định cố định
(đề xuất 5–8 quyết định/giây, action lặp tới kỳ quyết định tiếp theo).

---

## 3. PHẦN NHÌN (Vision) — "nhìn như Gemini" nhưng chạy trên máy yếu

### 3.1 Sự thật cần nói thẳng

Gemini/Claude/ChatGPT "nhìn hình xịn" vì chúng là mô hình hàng tỷ tham số chạy trên data
center GPU. **Không thể** chạy model cỡ đó real-time trên i5-7200U. Nhưng đích đến của bạn
không phải là một VLM tổng quát — nó là một bộ phân loại 5 hành động trên một game cố định.
Bài toán này "dễ" hơn ~6 bậc độ lớn, và có ba cách để vừa thông minh vừa nhẹ:

1. **Nhìn thông minh hơn, không phải mạng to hơn** — tận dụng cấu trúc game (3 làn, phối
   cảnh cố định, vật cản đặc màu): thị giác máy tính cổ điển (HSV mask, optical flow,
   background subtraction) trích "bản đồ vật cản" vài chục số; mạng chỉ học Q/action trên
   vector đó. Rẻ nhất, nhanh nhất, sample-efficient nhất.
2. **CNN nhỏ học từ ảnh full-frame** (như subwAI/Bezet1): input ~120–224px RGB, 3–5 conv
   + pooling + dense nhỏ; inference 5–15ms CPU là đủ. Mạng của bạn (49k–348k param) thuộc
   lớp này — đủ năng lực, vấn đề là INPUT và TRAINING, không phải size.
3. **VLM làm thầy NGOÀI LUỒNG, CNN nhỏ làm tròN TRONG LUỒNG** (distillation): dùng
   Gemini/Claude/Qwen... gán nhãn/mô tả hàng chục nghìn frame từ video chơi game (chạy 1
   lần, không real-time), rồi dạy CNN nhỏ. Tại inference chỉ có CNN nhỏ ⇒ mượt. Đây là
   hướng "AI lớn truyền hồn cho AI nhỏ" đang là research nóng 2025–2026 (§6).

### 3.2 So sánh các kiến trúc thị giác nhẹ cho CPU (định lượng)

| Hướng | Params | Inference CPU (tham khảo) | Cần dữ liệu | Ghi chú |
|---|---|---|---|---|
| CV trích feature → MLP 2 lớp | ~5–20k | <1ms | cực ít (RL vài trăm episode là chơi được) | bền, debug được, thấy được "AI đang nghĩ gì" |
| CNN nhỏ 3–4 conv + GAP (hiện tại) | 50–350k | 0.5–2ms (đo trong repo) | trung bình (BC vài chục–vài trăm nghìn frame) | đủ cho bài này; sửa input + training |
| MobileNetV3-Small / EfficientNet-Lite | ~1–2.5M | ~10–18ms | nhiều hơn (học sâu) | chỉ cần nếu muốn full RGB 224px |
| TinyViT/MobileViT | ~1–3M | ~15–30ms | nhiều | chưa cần thiết ở bài này |
| VLM thật (Gemini nano/local 1–3B) | ≥1B | ≥500ms–vài giây | không train | KHÔNG dùng real-time; chỉ offline làm thầy |

Tài liệu nền: DRL-TinyEdge đo lường MobileNetV3/EfficientNet trên thiết bị edge
([MDPI Future Internet 2026]); MiniConv (2025) cho thấy encoder visual cực nhỏ chạy
on-device vẫn cạnh tranh được với full-CNN cho visual RL; DGC (CVPR 2025) chưng cất biểu
diễn từ VLM vào agent VRL nhỏ rồi **vứt VLM đi lúc chơi** — chi phí train cao, chi phí
chơi bằng agent nhỏ.

### 3.3 Khuyến nghị cụ thể cho tầm nhìn (xếp theo tỷ lệ lợi ích/công sức)

1. **Bỏ cắt chân trời:** CNN nhận TOÀN BỘ khung game (resize 84–120px), hoặc dùng 2 vùng:
   full-frame grayscale/RGB 84–100px cho policy + vùng chân trời độ phân giải cao hơn cho
   bộ phát hiện sớm. Giữ RGB (3 kênh) — đắt hơn 3× RAM observation nhưng rẻ về compute và
   giữ thông tin màu; nếu muốn giữ 1 kênh, dùng **mask màu đặc trưng** (HSV) làm kênh thay
   cho grayscale thô.
2. **Tăng lịch sử thời gian:** stack 6–8 frame ở cadence quyết định (không phải 4 frame ở
   30FPS), hoặc thêm kênh "frame difference" (motion) làm kênh thứ 5.
3. **Tùy chọn mạnh nhất cho máy yếu — structured perception:**
   - Hiệu chỉnh vùng 3 làn (theo % chiều rộng, như cách bạn đã làm với anchor).
   - Mỗi làn × 4–6 mức khoảng cách (xa→gần, lưới trên ảnh phối cảnh): tại mỗi ô tính
     "mức chiếm chỗ" bằng diff với nền tham chiếu (hoặc mask HSV cho màu tàu/hàng rào).
   - Trạng thái: ma trận 3×(4–6) occupancy + 1 cờ nhảy/đứng/cuộn của nhân vật (theo dõi
     vị trí dọc của nhân vật qua các frame) + tốc độ ước lượng (từ vận tốc cuộn hoa văn).
   - Vector ~30–60 số → MLP 64–124 hidden → 5 Q-values. RL trên cái này học trong hàng
     trăm episode, CPU <0.2ms/quyết định. Tham khảo tinh thần: các hệ thống CV chơi game
     bằng sampling hàng ngang 3 mức xa/trung/gần ([CSU thesis, 2019]) — thô nhưng chạy
     real-time trên CPU và né được vật cản; học bằng MLP trên occupancy grid sẽ chắc hơn.
4. **Death/respawn detection:** ngoài anchor màu, thêm template matching cho màn
   game-over (bạn đã có hook `gameover_template_path`) và OCR nhẹ cho điểm (chỉ để đánh
   giá, không làm reward — đúng định hướng hiện tại).
5. **Data augmentation cho BC/RL-from-demo:** lật trái–phải (đảo luôn nhãn LEFT↔RIGHT),
   dịch nhẹ ± vài % (giả lập sai lệch calibration), thay đổi độ sáng/contrast (tối game
   theo city theme). subwAI báo cáo mirroring giúp model chắc hơn rõ rệt.

---

## 4. PHẦN HỌC (Learning) — tại sao RL pixel thô thất bại và thay bằng gì

### 4.1 Bằng chứng từ các project cùng bài toán

| Project | Phương pháp | Kết quả | Bài học |
|---|---|---|---|
| [Bezet1/subway-surfers-ai] | **BC thuần** (chơi tay → ảnh gán nhãn → CNN), 224×224 RGB, AdamW 1e-4, **3 model slow/medium/fast** | train 94.2% / val 89.7% / test 87.3%, inference ~15ms, **chơi đều 20.000m** | BC + chia theo tốc độ + dữ liệu người là đủ |
| [nikp06/subwAI] | **BC thuần** CNN (conv+avgpool+dense+dropout), ảnh phân folder theo action; **augment lật ngang** | ~85% acc, **run >1 phút đều**, xử lý an toàn vật cản thường; còn khám phá ra glitch | mirroring tăng robustness; BC không cần RL vẫn chạy ổn |
| [m4n4n-j/subway-surfers-AI-main] | **DQN + CNN + experience replay + LSTM + eligibility trace** (RL from scratch, emulator) | điểm TB ~250 (max 512), *"learnt to keep jumping or rolling as the agent is not penalized for these actions"* | RL pixel thô **suy ra chiến lược rác** đúng cơ chế Lỗi #2/#4 của bạn |

Kết luận thực nghiệm rất rõ: **bài toán này được giải bằng imitation learning, không phải
bằng RL from-scratch pixel** — ít nhất với ngân sách máy/thời gian của cá nhân. RL chỉ nên
vào vai trò tinh chỉnh (fine-tune) sau khi policy đã chơi được cơ bản.

### 4.2 Pipeline học đúng cho bài này (đề xuất)

```
Giai đoạn 0 — Sửa MDP (bắt buộc, xem §5 checklist)
Giai đoạn 1 — Gom demo người ĐÚNG CÁCH (10–30 phút chơi, nhiều tốc độ)
Giai đoạn 2 — Behavior Cloning (pretrain) tới val-acc ≥ 80%
Giai đoạn 3 — DAgger / HG-DAgger: bot chơi, người chỉ can thiệp lúc nguy hiểm
              (human-gated), gom state mà bot thực sự gặp → train lại
Giai đoạn 4 — (tùy chọn) DQfD fine-tune: demo ở replay vĩnh viễn + margin loss;
              hoặc RL trên structured-state (rẻ)
Giai đoạn 5 — Đánh giá thống nhất (protocol bạn đã có: Mann-Whitney, bootstrap CI)
```

#### Giai đoạn 2 — Behavior Cloning: làm đúng chi tiết

- **Ghi nhãn theo SỰ KIỆN, không theo phím giữ:** tại thời điểm keydown, ghi (frame_id,
  action, ts). Sau đó gán nhãn cho frame bằng cách **dãn ngược** (backdate): action được
  gán cho cửa sổ [t_press − Δ_back, t_press] với Δ_back ~150–350ms tùy tốc độ (xa hơn khi
  game nhanh — vì người ra quyết định sớm hơn), các frame còn lại NOOP. Cửa sổ này chính
  là khoảng "ra quyết định". Có thể ước lượng Δ_back từ dữ liệu: khoảng cách từ vật cản lúc
  bấm (dùng CV xem vật cản ở mức nào) → gán nhãn cho frame nơi vật cản ở mức đó.
- **Cân bằng lớp nghiêm túc:** NOOP chiếm 80–90% frame. Dùng inverse-frequency weighting
  (bạn đã có `inverse_sqrt`), nhưng tốt hơn nữa: **không train trên mọi frame** — lấy mẫu
  100% frame có action + ~20–30% frame NOOP; hoặc dùng class-balanced focal loss.
- **Chia train/val theo EPISODE** (bạn đã làm đúng), augment lật ngang (bắt buộc), sáng/
  tối, dịch nhỏ.
- **Input: full-frame RGB, ≥ 84–120px, stack 4** (stack giúp phân biệt nhảy/đứng/cuộn).
- Theo dõi **per-class recall**, đặc biệt JUMP/SLIDE (thường thấp); đừng tin accuracy tổng.
- Early stop theo val; lưu best theo val chứ không theo train.

#### Giai đoạn 3 — DAgger / HG-DAgger (vũ khí bí mật cho "học từ người")

Vấn đề kinh điển của BC: **compounding error / distribution shift** — bot chỉ học trạng
thái người chơi gặp; lúc tự chơi, sai một ly đi vào trạng thái chưa từng thấy ⇒ sai lầm
dây chuyền ([Imitation learning guide]; Ross et al. DAgger; [HG-DAgger, arXiv:1810.02890]).

Cách làm hợp với hệ thống của bạn (gần như không cần lý thuyết mới):
1. Bot tự chơi (policy từ BC). Người ngồi xem, **chỉ cầm quyền khi bot sắp húc** (giống
   HG-DAgger "human-gated intervention" — Kelly et al. 2018: người điều khiển toàn phần
   suốt pha can thiệp, không nhãn sau lag — tránh nhãn kém chất lượng do độ trễ).
2. Toàn bộ frame trong lúc can thiệp + vài trăm ms sau đó được gán action người và đẩy vào
   dataset (đây chính là phân bố trạng thái lỗi của bot — dữ liệu quý nhất).
3. Train lại (BC trên dataset gộp), lặp lại 3–6 vòng. Paper cho thấy HG-DAgger/DAgger giảm
   lỗi từ O(T²) của BC về O(T) và thực nghiệm trên lái xe vượt trội rõ.
4. Biến thể tiết kiệm sức người hơn: **RND-DAgger** — chỉ hỏi người khi trạng thái là
   "out-of-distribution" theo độ bất ngờ của một mạng ngẫu nhiên (RND uncertainty); hoặc
   can thiệp theo độ lệch giữa 2–3 model (ensemble disagreement). Bạn đã có sẵn recording
   pipeline (`demonstration_recorder`) — chỉ cần thêm chế độ "record while bot plays,
   người override".

#### Giai đoạn 4 — DQfD nếu vẫn muốn RL tinh chỉnh

[DQfD, Hester et al. AAAI 2018]: demo nằm **vĩnh viễn** trong replay (không bị evict),
pretrain bằng tổng 4 loss (1-step TD + n-step TD + **large-margin classification loss** ép
Q(action_chuyên_gia) ≥ Q(action_khác) + margin + L2), lúc online sample cả demo lẫn data
mới, priority của demo được cộng bonus ε_d. Code hiện tại của bạn mới có CE loss kiểu BC,
chưa có margin loss và demo không vào replay ⇒ thêm đúng 2 thứ này là thành DQfD thật.
Lưu ý: sau BC, ε khởi điểm nên thấp (0.1–0.3) thay vì 1.0, và LR RL nhỏ hơn (1e-4 giữ
nguyên) để không xóa trắng BC.

Nếu dùng structured-state (§3.3 #3), RL (Double DQN hiện có) sẽ học cực nhanh trên vector
nhỏ — có thể không cần DQfD nữa; BC trên state vector cũng cực rẻ.

### 4.3 Sửa reward & hyperparameters (nếu dùng RL/DQfD)

| Thông số | Hiện tại | Đề xuất | Lý do |
|---|---|---|---|
| Bước thời gian | frame 30FPS, cadence 2–4 lẫn lộn | **frame-skip cố định k=3–4** (quyết định ~7–10/s), action lặp cả kỳ; transition theo kỳ | MDP nhất quán; horizon γ=0.99 ⇒ ~20–27s |
| γ | 0.99/frame 30FPS (H≈3.3s) | 0.99 **theo kỳ quyết định** (H≈25s) | tín hiệu chết truy về được quyết định |
| Reward sống | +0.02/frame (0.6/s) | +1.0/s sống (chuẩn hóa theo kỳ), clip ± nhỏ | scale reward ≈ [−1, 1]/bước |
| Phạt chết | −10 | **−5 ~ −10 nhưng quy về cùng thang** (≈ −10s sống) | tương quan sống/chết hợp lý |
| Né thành công | không có | **+0.5–1.0 mỗi vật cản vượt qua an toàn** (phát hiện bằng CV/occupancy grid khi vật cản đi ngang người mà không va chạm) | shaping mật độ, đúng nhân quả |
| Vào trạng thái nguy hiểm | không có | **−0.2 khi ô ngay trước người bị chiếm** (potential-based shaping, bất biến chính sách) | tín hiệu sớm |
| Hành động vô ích | không phạt | **−0.05** mỗi cú bấm nhảy/cuộn/đổi làn khi trước mặt trống (chống spam) | diệt "jumping bot" |
| Hazard bonus cũ | +0.1 cho mọi cú bấm khi diff | **BỎ** hoặc thay bằng shaping ở trên | phần thưởng sai nhân quả |
| ε decay | 1.0→0.05 / 150k env-frame (~800 ep) | nếu from scratch: ≥1M frame; nếu sau BC: 0.3→0.05/200–400k frame; decay theo **số kỳ quyết định** | tránh khóa policy mù |
| n-step | 3 frame (~0.1s) | 3–5 **kỳ quyết định** (~0.4–0.7s) | propagate nhanh hơn |
| Target sync | hard/1000 update | giữ nguyên; cân nhắc soft τ=0.005 | ổn định |
| Exploration thêm | không | **NoisyNet** (thay ε-greedy, thăm dò theo tham số — cực hợp action ít) hoặc RND bonus cho trạng thái hiếm (nhưng tắt RND khi đã biết chơi) | thăm dò sâu, không ngẫu nhiên nông |

Nguyên tắc chung (reward hacking audit của bạn đã có tinh thần này — áp dụng triệt để
hơn): mọi shaping phải **potential-based** hoặc đo bằng bộ cảm biến trạng thái độc lập
(CV occupancy), không từ pixel-diff UI.

### 4.4 Tận dụng "teacher lớn" theo cách rẻ: VLM gán nhãn offline (tùy chọn nâng cao)

Pipeline từ các paper 2025–2026 ([DGC, CVPR 2025]; [VLM-annotated video game datasets,
2026]; [VLM-annotated conditioned agent, 2026]):

1. Quay hàng chục giờ video Subway Surfers (gameplay có sẵn trên YouTube/TikTok hoặc tự
   chơi) — không cần đồng bộ phím.
2. VLM mạnh (Gemini/Claude/GPT/Qwen...) xem từng đoạn 0.5–1s, trả lời câu hỏi cố định:
   "Ở frame này người chơi nên bấm gì? (trái/phải/nhảy/cuộn/không) vì sao? Vật cản gần
   nhất ở làn nào, loại gì?" → bộ nhãn dày + mô tả. Paper 2026 cho thấy VLM gán reward/
   action cho video game (Trackmania) đủ tốt để train offline RL, và RAM (Reward
   Annotation Model) học tổng hợp nhiều VLM trên một tập con người gán để tăng chất lượng.
3. Train CNN nhỏ của bạn trên bộ nhãn này (behavior cloning offline), distill cả phân phối
   soft-label (temperature distillation) nếu VLM trả xác suất.
4. Lúc chơi: chỉ CNN nhỏ, VLM tắt hoàn toàn ⇒ mượt trên máy yếu.
   ⚠️ Lưu ý bản quyền/ToS khi dùng video ngoài; dùng cho nghiên cứu cá nhân.

Đây là cách "hợp đồng" sức nhìn của model lớn vào model nhỏ mà không cần VLM real-time.

---

## 5. Lộ trình triển khai đề xuất (thứ tự, kỳ vọng, phép đo)

Mỗi giai đoạn có **tiêu chí thành công đo được** — không tin cảm tính.

**G0 — Sửa nền tảng MDP (1–2 buồng):**
- Capture TOÀN khung game vào policy (bỏ cắt 25% trên); RGB hoặc mask-kênh.
- Frame-skip cố định (k=3), quyết định 1 lần/kỳ, action giữ cả kỳ; transition ghi 1 lần/kỳ
  với (s_kỳ, a, tổng reward kỳ, s_kỳ+1); stack theo kỳ.
- Reward theo bảng §4.3 (bỏ hazard cũ, thêm né-thành-công nếu có occupancy; chưa có CV thì
  tạm dùng survival scale lại + phạt chết cân đối).
- γ theo kỳ; ε decay chậm/hoặc thấp sau BC.
- **Verify:** 30 frame obs trước khi chết phải nhìn rõ vật cản; log action khớp phản hồi
  trong game; episode giả lập headless có learning curve đi lên (synthetic game phải được
  nâng cấp độ khó gần thật, hoặc tốt hơn: test trên replay video thật).

**G1 — Demo người chuẩn (0.5–1 buồng):**
- Chế độ ghi: nhãn sự kiện keydown + timestamp, backdate Δ theo tốc độ; tách level tốc độ
  (0–30s, 30–90s, >90s); 15–30 run tốt của bạn; tự động bỏ frame menu/death.
- **Verify:** xem lại video có overlay nhãn — nhãn phải nằm ở đoạn vật cản CÒN XA.

**G2 — BC (0.5 buồng):**
- Full-frame RGB 96–120px, stack 4, CNN hiện tại (đủ), augment mirror/sáng/dịch,
  class-balanced, early stop theo val; target per-class recall ≥ 70% mọi lớp.
- **Verify:** cho bot chơi ε=0, sống trung bình ≥ 20–30s (subwAI/Bezet1 đạt >60s với BC).

**G3 — DAgger/HG-DAgger (vài buồng chơi kèm can thiệp):**
- 3–6 vòng, mỗi vòng 15–30 phút: bot chơi, người override lúc nguy hiểm; train lại.
- **Verify:** tần suất can thiệp/phút giảm dần; survival tăng từng vòng (vẽ đường cong).

**G4 — Tinh chỉnh (tùy chọn):** DQfD (margin loss + demo trong replay) hoặc RL trên
structured-state; NoisyNet exploration.
**Verify theo protocol sẵn có:** 20–50 episode, ε=0.05, bootstrap CI + Mann-Whitney vs
baseline người — đúng §10 README, không tuyên bố "thắng" khi chưa qua ngưỡng.

**Song song — structured perception (nếu muốn mượt/tin cậy tuyệt đối):** occupancy grid
3 làn × 5 mức + trạng thái nhân vật → MLP; đây là đường ít rủi ro nhất cho máy yếu và cho
phép reward shaping "né thành công" chính xác.

---

## 6. Tổng kết: trả lời thẳng 3 câu hỏi của bạn

1. **"Train 6000 episode chỉ +1s — vì sao?"** Không phải do model nhỏ hay máy yếu. Do
   MDP hỏng ở 7 điểm phối hợp: (1) observation không thấy chân trời = không thấy gì để né;
   (2) reward scale làm mọi episode tệ như nhau, không có tín hiệu mật độ đúng nhân quả;
   (3) horizon chiết khấu 3.3s không truyền được cái chết về quyết định; (4) hazard bonus
   dạy bấm nút bừa; (5) ε decay xong ở episode ~800 khóa cứng policy mù; (6) transition/
   action gán sai nhịp frame-skip, stack quá ngắn; (7) demo gán nhãn theo phím giữ, trễ
   ~200–300ms, không DAgger. Bằng chứng đối chiếu: project RL cùng bài ngoài đời cũng hỏng
   đúng kiểu này (jumping/rolling spam), còn các project thành công đều dùng BC từ người.

2. **"Nhìn như Gemini/ChatGPT/Claude nhưng mượt trên máy yếu" — thế nào?"** Không chạy
   model lớn real-time. Ba lựa chọn: (a) CV cấu trúc (mask 3 làn → vector nhỏ → MLP,
   <1ms, học cực nhanh); (b) CNN nhỏ full-frame RGB (mạng bạn đang có THỪA sức, chỉ cần
   sửa input: toàn khung hình, giữ màu, đủ lịch sử); (c) VLM lớn làm thầy OFFLINE gán nhãn
   video → distill sang CNN nhỏ, lúc chơi chỉ CNN (DGC 2025). Chơi game cố định 5 hành
   động không cần trí thông minh tổng quát của Gemini — cần đúng thông tin, đúng nhãn.

3. **"Học từ người thế nào cho đúng?"** BC là đường chính và đã được kiểm chứng ngoài thực
   tế (85–90% acc, chạy >1 phút–20.000m). Nhưng phải: ghi nhãn theo thời điểm ra quyết
   định (backdate theo khoảng cách vật cản), giữ màu + toàn khung, augment lật ngang, cân
   bằng lớp, đo per-class recall, chia tốc độ. Sau đó DAgger/HG-DAgger (người chỉ can thiệp
   lúc sắp chết) để diệt compounding error — đây là bước tạo khác biệt giữa "BC chạy được
   vài chục giây" và "chơi bền". Muốn chắc hơn nữa: DQfD giữ demo trong replay + margin
   loss để RL không quỹa bài người dạy.

---

## Tài liệu tham khảo chính

**Game mechanics:**
- [Wikipedia — Subway Surfers](https://en.wikipedia.org/wiki/Subway_Surfers)
- [Subway Surfers Fandom Wiki](https://subwaysurf.fandom.com/wiki/Subway_Surfers) (tăng tốc
  đạt max ở phút thứ 3, cơ chế obstacle/revive)
- [Poki — Subway Surfers (trang chính, điều khiển, mô tả tốc độ)](https://poki.com/en/g/subway-surfers)

**Project thực chiến cùng bài:**
- [Bezet1/subway-surfers-ai](https://github.com/Bezet1/subway-surfers-ai) — BC, 3 model
  theo tốc độ, 87–94% acc, 20.000m, ~15ms/inference
- [nikp06/subwAI](https://github.com/nikp06/subwAI) — BC + augment lật ngang, >1 phút/run
- [m4n4n-j/subway-surfers-AI-main](https://github.com/m4n4n-j/subway-surfers-AI-main) —
  DQN from scratch thất bại kiểu "jumping/rolling spam" (đối chứng quan trọng)

**Imitation learning:**
- Ross, Gordon, Bagnall — *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning* (DAgger), 2011
- [Kelly et al. — HG-DAgger: Interactive Imitation Learning with Human Experts,
  arXiv:1810.02890](https://arxiv.org/abs/1810.02890)
- [Imitation Learning from Demonstrations guide (BC/DAgger/GAIL, compounding error)](https://robocloud-dashboard.vercel.app/learn/blog/imitation-learning)
- RND-DAgger (active imitation với RND uncertainty), ICLR 2025 (dẫn trong trang
  emergentmind HG-DAgger)

**RL từ demo / reward / exploration:**
- Hester et al. — *Deep Q-learning from Demonstrations (DQfD)*, AAAI 2018;
  [DI-engine DQfD docs](https://di-engine-docs.readthedocs.io/en/latest/12_policies/dqfd.html)
  (4 loss: 1-step, n-step, large-margin, L2; demo priority bonus)
- Mnih et al. — *Human-level control through deep RL* (Nature DQN 2015) — ε-decay 1M
  frame, frame-skip 4, γ theo bước
- [Burda et al. — Exploration by Random Network Distillation (RND)](https://www.researchgate.net/publication/328627326_Exploration_by_Random_Network_Distillation)
- [Potential-Based Reward Shaping revisited (2025), arXiv:2502.01307](https://arxiv.org/html/2502.01307v1)
- [Survey: Dealing with Sparse Rewards in RL, arXiv:1910.09281](https://arxiv.org/pdf/1910.09281)

**Vision nhẹ & distillation từ VLM:**
- [DRL-TinyEdge (MobileNetV3/EfficientNet edge latency), MDPI Future Internet 2026](https://www.mdpi.com/1999-5903/18/1/31)
- [MiniConv: Tiny on-device visual encoders for RL, 2025](https://arxiv.org/html/2512.19726)
- [Xu et al. — DGC: VLMs-Guided Representation Distillation for Efficient VRL, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_VLMs-Guided_Representation_Distillation_for_Efficient_Vision-Based_Reinforcement_Learning_CVPR_2025_paper.pdf)
- [VLMs for Videogame Data Annotation (2026)](https://arxiv.org/html/2608.05949v1)
- [Training a Conditioned Video Game Agent on a VLM Annotated Dataset (2026)](https://arxiv.org/html/2608.05954v1)
- [Using Computer Vision Techniques to Play an Existing Video Game (CSU thesis, 2019)](https://scholarworks.calstate.edu/downloads/7w62f8544)
  (sampling 3 hàng xa/trung/gần để né chướng ngại trên CPU)

---
*Báo cáo này phân tích trên mã nguồn tại commit hiện tại (403 tests, pipeline headless)
và các nguồn công khai. Mọi con số triển khai (acc, khoảng cách, ngưỡng) cần được kiểm
chứng lại trên game thật/Poki theo đúng evaluation protocol của repo.*
