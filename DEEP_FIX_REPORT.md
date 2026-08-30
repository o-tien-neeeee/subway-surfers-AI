# DEEP FIX ALL — Báo cáo rà soát & sửa lỗi tận gốc

**Repo:** `o-tien-neeeee/subway-surfers-AI` · **Branch:** `arena/01a0506f-subway-surfers-ai`
**Phạm vi quét:** 26 file `.py` nguồn + 16 file test = **10.700 dòng** (đọc toàn bộ, kể cả comment/tên biến)
**Diff:** 19 file code/test, **+2.181 / −233 dòng** (+ báo cáo này 426 dòng), 82 chú thích `DEEP-FIX`

## Kết quả kiểm chứng (chạy thật, không suy đoán)

| | Trước | Sau |
|---|---|---|
| `pytest tests` | **1 failed, 388 passed** (`-x`) — 403 test thu thập | **473 passed, 0 failed** trong 79 s |
| Test treo | `test_gui_side_survives_actor_crash` **treo 600 s** rồi bị pytest-timeout giết | qua trong ~55 s (cả cụm multiprocess) |
| Test hồi quy mới | — | **+46** (41 trong `tests/test_deep_fix.py`, +5 test đã có) |
| Test mới **fail trên code gốc** | — | **34/41** (7 còn lại là test "chốt"/bằng chứng, liệt kê ở §3.3) |
| `python -m compileall` | clean | clean |
| `app.py --headless --steps 1500` | — | `HEADLESS SMOKE TEST: PASS`, 0 error, workers thoát sạch, `best_model.pth` + `.sha256` sinh ra sau **đúng cửa sổ 3 episode** |
| `app.py --profile-models` / `--evaluate 3` | — | chạy được, số liệu hợp lệ |

---

# PHẦN 1 — BẢNG CHẨN ĐOÁN

Mức độ: 🔴 nghiêm trọng (hỏng chức năng / treo / hỏng dữ liệu huấn luyện) · 🟠 cao · 🟡 trung bình · ⚪ thấp/dọn dẹp

## 🔴 Nghiêm trọng

| # | Loại lỗi | Vị trí | Mô tả | Hậu quả | Gốc rễ — vì sao tác giả sai? |
|---|---|---|---|---|---|
| 1 | Luồng điều khiển / IPC chết | `environment.py:737` (`_downgrade`) → `learner_worker.py` (`add_transitions`) | Auto-downgrader gửi `{"__cmd__": "set_profile"}` vào **`transition_q`**, nhưng learner chỉ đọc lệnh từ `cmd_q`; `add_transitions()` lặng lẽ vứt mọi thứ không phải `NStepTransition`. `grep -rn "__cmd__"` trên cả repo ra **đúng 1 kết quả: chính dòng ghi**. | Learner vẫn train profile nặng trong khi actor đã đổi sang profile nhẹ → `unflatten_into()` đọc **tiền tố** vector trọng số → actor nạp trọng số vô nghĩa. **Đã đo:** conv `(4,1,3,3)` của strict_lite bị ghi đè bằng những byte đầu của conv `(48,4,8,8)` quality_cpu, **không một dòng log lỗi**. | Tác giả viết kênh thông báo theo trí nhớ ("actor nói với learner qua transition queue"), rồi **không bao giờ grep lại phía nhận**. Comment `# 2. transitions (also accepts the actor's profile-change notice)` mô tả một tính năng chưa từng tồn tại — comment viết theo ý định, không theo code. |
| 2 | Logic thuật toán | `learner_worker.py` `train_one()` | `update_priorities(indices, td_error_abs_mean * ones(N))` — gán **một số vô hướng (trung bình batch)** cho toàn bộ index. | PER sụp đổ về uniform trong khi vẫn trả giá sum-tree + importance-sampling correction. **Đã đo:** sau 1 update, cả 32 slot có chung một giá trị `2.19244182`. | `train_step()` chỉ trả `td_error_abs_mean` (scalar). Thay vì sửa contract của `train_step`, tác giả "vá" bằng cách broadcast — vì `np.ones(N)` làm code **trông** như đang xử lý per-sample. Đây là vá đắp điển hình: đúng kiểu, sai ngữ nghĩa. |
| 3 | Deadlock / tài nguyên | `logging_utils.drain()` ← `app.py:_drain_all_queues`, `gui._poll_metrics` | `mp.Queue` ghi mỗi item = 4 byte độ dài + payload. Worker bị `terminate()` giữa chừng để lại header không payload; **tiến trình cha cũng giữ đầu ghi của queue** nên pipe không bao giờ EOF → `Connection._recv()` chặn vĩnh viễn. | `shutdown()` treo, **Tkinter đơ cứng** — đúng lời hứa "GUI sống sót khi worker chết" bị phá. **Đã chứng minh 2 lần:** test treo 600 s + repro tối thiểu cha/con (exit 124). | Tác giả tin `get_nowait()` = "không chặn". Đúng khi pipe lành; sai khi pipe cụt. Giả định ngầm: "process chết thì pipe đóng" — nhưng chính cha đang giữ đầu ghi. |
| 4 | Thiếu code / chức năng chết | `demonstration_recorder.DemoRecorder.tick()` / `.read_frame` | `read_frame` được truyền vào constructor và **không bao giờ được gọi**. `grep -rn "\.tick("` → chỉ có `FpsMeter.tick` và test của chính recorder. GUI nối một `_RingReader` vào rồi **không ai bơm**. | Mọi episode demo lưu **0 frame** → `stop()` trả `None` → "nothing saved". **Behaviour cloning không có nguồn dữ liệu nào cả**, trong khi README mô tả Phase-1 BC như một phần của pipeline. | Tác giả thiết kế 2 mô hình (pull: `read_frame`, push: `tick`), nối mô hình pull ở GUI, viết unit test cho mô hình push, rồi **quên viết vòng bơm nối hai cái**. Test pass vì test tự gọi `tick()` — vùng mù kinh điển của unit test: nó kiểm tra API, không kiểm tra wiring. |
| 5 | Hiệu năng / busy-spin | `learner_worker.learner_main()` | Vòng `while` **không có một lệnh sleep nào**. | **Đã đo A/B:** learner chiếm **100% một nhân** khi không thể train (0 update); sau fix còn **4%**. Trên i5-7200U 2C/4T chạy kèm Chrome, đây chính là starvation mà thiết kế "tách process để Chrome còn headroom" muốn tránh. | Tác giả đã cap tốc độ *update* bằng `can_train(now)` và cho rằng thế là đủ — nhưng cap tốc độ công việc ≠ cap tốc độ vòng lặp. |

## 🟠 Cao

| # | Loại lỗi | Vị trí | Mô tả | Hậu quả | Gốc rễ |
|---|---|---|---|---|---|
| 6 | Vòng đời tài nguyên | `demonstration_recorder.KeyboardTap.stop()` | `self._listener = None`; `start()` sau đó là no-op. | **Đã đo (stub listener):** episode 1 ghi phím thật, episode 2..N **ghi toàn NOOP** mà không báo gì. Dataset BC trông hợp lệ nhưng dạy policy không bao giờ né. | Tác giả coi `stop()` là "tắt" chứ không phải "huỷ". pynput `Listener` là **single-use** — giả định này không được kiểm tra. |
| 7 | Trạng thái episode | `environment.GameEnvironment.respawn()` | Chỉ reset `game` + `detector`; `reward_calc._episode_dead` vẫn `True`. | **Đã đo:** death #1 → `−10.0`, death #2 → `0.0`. **Mọi episode sau episode đầu trong headless train/eval không có tín hiệu chết.** | `_episode_dead` là cờ "once per episode" nhưng `respawn()` — chính là ranh giới episode — không reset nó. Cờ nằm ở object khác với object đổi trạng thái. |
| 8 | Phương pháp đánh giá | `evaluation_tool.run_headless_evaluation()` | Cả N episode dùng chung `cfg.seed` cho **cả game lẫn policy**. | **Đã đo A/B:** code gốc → 3 episode đều **79 steps, reward −8.42, std = 0.0000**; "survival" chỉ dao động vì **nhiễu đồng hồ máy** (0.62/0.84/0.82 s cho cùng 79 bước). Mann-Whitney p, bootstrap CI, "std" trong báo cáo được tính trên n bản sao của 1 mẫu. Sau fix: 134/130/85 steps, std ≠ 0. | Tác giả giả định `SyntheticGame` tự ngẫu nhiên hoá. Thực tế `GameEnvironment.__init__` truyền thẳng `cfg.seed`. Hệ quả: **toàn bộ giao thức thống kê §15 trông chặt chẽ nhưng vô nghĩa.** |
| 9 | Thứ tự thao tác | `checkpoint_manager.save_model(which="best")` | `self.best_metric = metric` chạy **trước** khi ghi file. | **Đã đo:** ép `OSError` → `best_metric = 10.0` mà `best_model.pth` không tồn tại; vì `best_metric` được persist vào checkpoint sau, **best đó mất vĩnh viễn** trong khi code vẫn tin là có. | Tác giả tối ưu hoá cho luồng happy-path và coi "ghi file chắc chắn thành công". Không có nguyên tắc "commit state sau khi side-effect thành công". |
| 10 | Kiểm tra biên / kiểu lỗi | `replay_buffer.PrioritizedReplayBuffer.load()` | `buf.frames.frames[:] = payload["frames"]` sau khi clamp `cap`. | **Đã đo:** `ValueError: could not broadcast input array from shape (4064,84,84) into shape (564,84,84)` — **không phải `CorruptFileError`**, nên người vận hành chỉ thấy một thông báo numpy khó hiểu thay vì "capacity trong config không khớp buffer đã lưu". | Tác giả xử lý hướng `cap > cfg.capacity` (clamp) mà **không xét hướng ngược lại** (`cap < cfg.capacity`), và không hình dung "người dùng giảm `per.capacity` sau khi đã train". |
| 11 | Nhất quán chỉ mục ↔ dữ liệu | `replay_buffer.sample()` | Sau `max_replace_rounds`, slot vẫn invalid được thay bằng `transitions[0]` nhưng **`leaf_idx` giữ index cũ**. | `probs`/`weights` tính cho transition khác với transition thực sự được train; `update_priorities(indices, td)` ghi TD error của `transitions[0]` vào **slot không liên quan**. Ngoài ra transition invalid được dùng nguyên → **pixel đã bị evict vẫn vào mạng**, mâu thuẫn chính docstring của module. | Tác giả viết fallback "để không crash" (`t if t is not None else transitions[0]`) mà không nhận ra `indices` là **đầu ra** chứ không chỉ là đầu vào. |
| 12 | Race condition (không atomic) | `ipc.SharedCounters` — `environment._end_episode` ↔ `learner_worker._poll_episode_metric` | 3 biến shared (`done_id`, `survival_s`, `reward`) ghi/đọc không thứ tự; cả hai bên đều chạm `done_id` **trước**. | Learner có thể ghép id của episode N với survival của episode N−1 → `best_model.pth` được gate bằng **một con số chưa từng được đo**. **Đã tái hiện:** 8.747 lần rách / 373.679 lần đọc. | Tác giả coi 3 phép gán liên tiếp là "một bản ghi". Với biến shared rời rạc thì không phải vậy — cần seqlock 3 pha (invalidate → payload → commit), và **bản sửa đầu tiên của tôi cũng sai ở chỗ này, xem §3.4**. |
| 13 | Cơ chế integrity chết | `checkpoint_manager._atomic_torch_save` / `load_model` | Ghi sidecar `.sha256` cho mọi model checkpoint, **không bao giờ đọc lại**. `_git_hash()` spawn subprocess `git` mỗi lần save, kể cả khi best-save bị từ chối. | File `.pth` bị bit-rot/cắt cụt vẫn được `torch.load` như thường. Docstring "atomic + hash-verified" chỉ đúng một nửa (đường buffer pickle thì có verify). | Copy-paste pattern từ đường buffer pickle mà quên nửa đọc. Sidecar "có vẻ" đủ để trông như đã an toàn. |
| 14 | Reproducibility giả | `checkpoint_manager.capture_rng_states()` | Lưu `np.random.default_rng().bit_generator.state` — trạng thái của **một Generator mới tạo bên trong hàm**, không phải `Learner.rng`. `restore_rng_states()` cũng không khôi phục Generator. | Claim "reproducible resume" trong docstring **không khôi phục được gì** cho luồng lấy mẫu replay. | Tác giả cần "một numpy state" để điền vào dict cho đủ, nên tạo tạm một cái. Không ai đối chiếu "state này của object nào". |
| 15 | Trộn hai đồng hồ | `environment.BotActor._execute()` | `t0 = time.perf_counter()` nhưng `t_frame` là `time.monotonic()` **do tiến trình capture ghi**, rồi `t0 - t_frame`. | p95 action latency sai một offset tuỳ ý → **nuôi thẳng auto-downgrader**. `LatencyMeter.observe_ms()` còn **lặng lẽ bỏ mẫu âm**, nên offset âm không chỉ làm lệch mà còn xoá phần lớn phân phối. Trên Linux bug tiềm ẩn (**đã đo:** 2 hàm cùng gốc, lệch −1.3e-7 s); trên Windows tài liệu CPython xác nhận `monotonic()` *trước đây* dùng `GetTickCount64()` (~15.6 ms) còn `perf_counter()` dùng `QueryPerformanceCounter()` — **tôi không kiểm chứng được trên Windows vì sandbox là Linux**. | Python nêu rõ "reference point of the returned value is undefined". Tác giả coi `perf_counter` và `monotonic` là một, vì **trên máy dev Linux chúng trùng nhau** — bug chỉ lộ trên máy đích Windows. |
| 16 | Thứ tự kiểm tra | `death_detector.RespawnController.update()` | Timeout được xét **trước** kiểm tra hồi phục. | Game sống lại đúng frame hết hạn → báo `FAILED` → `_respawn_failed()` set **pause toàn cục**, đỗ cả bot dù respawn thành công. | "Deadline" được coi là điều kiện chặn tuyệt đối, trong khi "đã hồi phục" là một **sự kiện đã xảy ra** và phải thắng. |

## 🟡 Trung bình

| # | Loại | Vị trí | Mô tả / hậu quả | Gốc rễ |
|---|---|---|---|---|
| 17 | Mất dữ liệu | `learner_worker.set_profile()` | `self.buffer = self._load_or_fresh_buffer()` **vứt buffer đang sống** (mọi transition kể từ lần save định kỳ cuối). | Coi buffer là "thuộc về profile" trong khi nó chỉ chứa frame/action/reward/priority. |
| 18 | Race condition | `DemoRecorder.tick()` vs `stop()` | `_recording = False` **không phải barrier** cho tick đang chạy → `np.stack()` có thể thấy độ dài lệch nhau giữa các cột. | Coi cờ bool là cơ chế đồng bộ. |
| 19 | Không có đường hồi phục | `safety_watchdog._intervene()` | Set `events["pause"]` và **không bao giờ clear** → một cú khựng 2 s của Chrome dừng vĩnh viễn. | "An toàn = dừng" mà quên nửa "và phải chạy lại được". |
| 20 | Log flood | `gui._check_workers()` | Nhánh `not alive["actor"]` **không có guard trạng thái** → log + ghi status strip mỗi 150 ms (~7 dòng/s) mãi mãi. | Viết guard cho nhánh learner (`and state is RUNNING`) nhưng quên nhánh actor ngay bên dưới. |
| 21 | Không cô lập lỗi | `dataset.load_episode` / `validate_directory` | Thiếu key → `KeyError` trần thoát ra → **một file hỏng làm sập cả lượt BC**. **Đã kiểm chứng sau fix:** `good.npz` OK, `bad_missing/bad_zip/ragged` thành report `ok=False`, `zip(eps, reps)` vẫn thẳng hàng. | `Learner.pretrain` không có vòng cô lập per-file; người viết validator giả định mọi `.npz` đều do recorder của mình sinh ra. |
| 22 | Schema drift | `evaluation.EvaluationReport.load` | `EpisodeRecord(**r)` → `TypeError` khi có field lạ/thiếu; `merge_baseline` chỉ bắt `(OSError, ValueError)` → `--compare-baseline` với report bản cũ làm sập cả lượt đánh giá. | Serialise bằng `asdict()` nhưng deserialise bằng `**` — không có tầng tương thích. |
| 23 | Im lặng chấp nhận dữ liệu xấu | `ipc.SharedWeights.copy_out()` / `unflatten_into()` | Sau 8 lần retry trả về **bản copy có thể bị xé rách** với comment "log caller" — không caller nào log. `unflatten_into` không kiểm tra độ dài. | "Trường hợp cực hiếm" bị xử lý bằng cách đẩy trách nhiệm cho nơi gọi, rồi không ai nhận. |
| 24 | Config chết | `config.CaptureConfig.grab_timeout_s`, `config.PerfConfig.max_working_set_gb` | **Đã grep: mỗi cái xuất hiện đúng 1 lần — chính dòng khai báo.** README còn nói `max_working_set_gb` là ngân sách được "enforce". Người dùng đổi → không có gì xảy ra. | Thêm knob vì spec yêu cầu, nhưng không nối vào nơi thi hành. |
| 25 | Trộn quần thể thống kê | `evaluation.failure_modes()` | Đếm trên **mọi** record (train + eval + baseline) → "early_death_lt_5s" không nói gì về policy đang đánh giá. | `self.records` tiện tay hơn `of_kind("eval")`. |
| 26 | Đè file | `DemoRecorder.stop()` | `episode_%Y%m%d_%H%M%S` — độ phân giải 1 giây → 2 episode cùng giây đè nhau. | Chọn format timestamp cho "đẹp", không kiểm tra va chạm. |
| 27 | Bắt sai kiểu exception | `agent.load_payload()` | `except ValueError` trong khi torch ném **`RuntimeError`** cho mismatch parameter-group → nhánh "profile change" không bao giờ chạy. | Đoán kiểu exception thay vì đọc thông báo lỗi của torch. |
| 28 | Bịa transition | `environment._flush_transitions()` | (a) `stack.get()` ném `RuntimeError` nếu stack chưa seed → shutdown sạch thành báo crash. (b) `_env_ids` rỗng tạo env-id tuple 0 phần tử cho stack 4 frame. | Viết đường flush cho trường hợp "đang chạy bình thường", không xét "chết trước frame hợp lệ đầu tiên". |
| 29 | Sentinel va chạm giá trị hợp lệ | `BotActor._episode_start_ts = 0.0` | `0.0` vừa là "không có episode" vừa là một timestamp hợp lệ → `_end_episode()` return sớm, không publish episode. | Dùng giá trị magic thay vì `Optional`. |
| 30 | Alias bí mật | `agent.py` 4 chỗ `torch.from_numpy(x).float().div_(255.)` | `.float()` là no-op nếu input đã là float32 → `div_` **chia thẳng buffer numpy của người gọi** cho 255. Chỉ tiềm ẩn vì mọi producer đều xuất uint8. **Đã kiểm chứng sau fix.** | `torch.from_numpy` chia sẻ bộ nhớ; `.float()` chỉ copy khi dtype khác — chi tiết này dễ quên. |

## ⚪ Thấp / dọn dẹp (đã sửa)

| # | Vị trí | Nội dung |
|---|---|---|
| 31 | `ipc.py` | Xoá `torch_no_grad()`, `pack_frame_header()`, `unpack_frame_header()`, `monotonic()`, `struct`/`UINT64_MAX` — **không nơi nào import** (đã grep toàn repo). |
| 32 | `learner_worker.py` | `from config import PROFILE_ORDER` — import chết. |
| 33 | `replay_buffer.NStepBuilder._emit` | `reward=float(r) if not terminal else float(r)` — hai nhánh **giống hệt nhau**, đọc như một lựa chọn thật. |
| 34 | `checkpoint_manager._atomic_torch_save` | `data = torch.save(...)` — `torch.save` luôn trả `None`, `data` không dùng. |
| 35 | `input_controller.press_action` | `self._pressed[keyname] = (self._pressed[keyname][0], existing)` — gán lại đúng giá trị cũ. |
| 36 | `metrics.FpsMeter.fps(now=…)` | Tham số `now` được tính rồi **không dùng**. |
| 37 | `evaluation_tool` | `env = GameEnvironment(cfg, game=None) if ep == 0 else GameEnvironment(cfg)` — hai nhánh giống hệt nhau. |
| 38 | `environment._put_metrics` | Nhánh `else` (gói bằng `metrics_message`) không bao giờ chạy vì mọi caller đều truyền dict có sẵn `"type"`. |
| 39 | `rewards.step` | `hazard = 0.0` bị ghi đè ngay dòng sau. |
| 40 | `dataset.py` | `from typing import Iterator` — import chết. |
| 41 | `gui.py` | `_preview_full_size`, `_episode_stats_frames0` — gán rồi không đọc. |

---

## Những chỗ tôi **đọc kỹ và kết luận là ĐÚNG** (không sửa)

Trung thực đòi hỏi phải nói cả phần này, vì tôi đã nghi ngờ chúng trước:

- **`NStepBuilder`** — tôi nghi terminal drain nhân đôi death penalty. Vết tay từng nhánh với `buf=[e0,e1,e2]`, `e2.done=True`, γ=0.9: `terminal=(span==0)` nên `span=2` và `span=1` **không** cộng reward terminal; return = `r0+γr1` và `r1`, bootstrap qua `Q(obs_2)` — đúng chuẩn n-step; `span=0` mới `done=True`, `r=r2`. **Không có lỗi.** Test `dones == [False, False, True]` là đúng.
- **`SharedFrameRing.read_latest()`** — gen trước/sau copy bắt được cả trường hợp writer ghi đè slot 2 lần trong lúc đọc (gen +2 ≠ gen). Seqlock đúng.
- **`SumTree`** — `update_batch` ghi rác vào `tree[0]` nhưng `total()` đọc `tree[1]`; `min_leaf`/`_filled` nhất quán vì `update()` clamp về `1e-6` nên không leaf nào = 0.
- **`priorities[]` lưu giá trị đã alpha-power** — nhất quán giữa `add_nstep`, `sample`, `save`/`load` (`rebuild_from` không power lần hai). Đúng như comment tuyên bố.
- **`gui._poll_metrics` nhánh `actor_stats`** — tôi tưởng là dead code vì `_put_metrics` có nhánh `metrics_message`; đọc lại thì `_put_metrics` kiểm tra `if "type" in data` và `_maybe_report` **có** truyền `"type": "actor_stats"` → nhánh GUI **có** chạy. Nhánh chết là `else` của `_put_metrics` (mục 38).

---

# PHẦN 2 — MÃ NGUỒN ĐÃ SỬA

Toàn bộ repo gốc 10.700 dòng; dán lại hết vào báo cáo sẽ vô dụng. **Mã đã sửa nằm trên đĩa**, mỗi chỗ thay đổi đều có comment `# DEEP-FIX: <lý do>` (82 chỗ). Xem nhanh:

```bash
git diff --stat                     # 19 file, +2181 / -233 (chưa kể báo cáo này)
grep -rn "DEEP-FIX" *.py | wc -l    # 74  (+4 trong tests/)
```

Dưới đây là **6 diff trọng yếu nhất** (phần còn lại đọc trực tiếp theo `file:DEEP-FIX`).

### 2.1 — Auto-downgrade phải tới được learner (lỗi #1)

`environment.py` — `BotActor.__init__` nhận thêm `cmd_q`, `_downgrade` gửi đúng kênh:

```python
# DEEP-FIX: this notice used to go ONLY to transition_q, which the
# learner never inspects for commands -- Learner.add_transitions()
# drops anything that is not an NStepTransition, and the whole repo
# contained exactly one reference to "__cmd__" (this line).  The
# learner therefore kept training the heavy profile while the actor
# ran the light one, and because unflatten_into() consumed a blind
# prefix of the flat vector the actor silently loaded garbage weights
# (verified: a (4,1,3,3) conv overwritten from a (48,4,8,8) conv).
notice = {"__cmd__": "set_profile", "profile": lighter}
if self.cmd_q is not None:
    put_bounded(self.cmd_q, {"cmd": "set_profile", "profile": lighter})
put_bounded(self.transition_q, notice)   # redundant path; idempotent
```

`ipc.py` — lớp phòng thủ thứ hai: vector trọng số có **schema**, đọc sai là từ chối chứ không im lặng:

```python
def layout_fingerprint(module) -> str:
    desc = "|".join(",".join(str(int(d)) for d in tuple(p.shape))
                    for p in module.parameters())
    return hashlib.sha256(desc.encode("ascii")).hexdigest()[:32]

def unflatten_into(module, flat) -> int:
    # DEEP-FIX: this used to consume a blind prefix of ``flat`` with no
    # validation, which is exactly how a profile mismatch between the actor
    # and the learner turned into silent garbage weights.
    need = module_param_count(module)
    if vec.size < need:
        raise ValueError(f"weight vector too short for this module: have "
                         f"{vec.size}, need {need} (profile mismatch?)")
```

`agent.InferencePolicy.refresh_weights` so fingerprint (có cache) và **giữ trọng số tốt cuối cùng** khi lệch, log đúng một lần để không flood ở 30 Hz.

### 2.2 — PER phải nhận TD error từng mẫu (lỗi #2)

```python
# agent.DoubleDQNAgent.train_step — trả thêm vector per-sample
with torch.no_grad():
    td_abs = td_error.abs().detach().to(torch.float32).cpu().numpy()
td_abs = np.nan_to_num(td_abs, nan=0.0, posinf=0.0, neginf=0.0)
return {..., "td_errors": np.ascontiguousarray(td_abs, dtype=np.float64), ...}

# learner_worker.Learner.train_one
# DEEP-FIX: PER must be re-prioritised with the PER-SAMPLE TD error.
# The old code broadcast the batch MEAN over every sampled index, so
# after a single update all 32 slots shared one priority value and the
# prioritisation the whole buffer exists for was erased (measured:
# 1 distinct priority across the batch).
td = metrics.get("td_errors")
if td is None or np.asarray(td).shape[0] != len(batch["indices"]):
    td = metrics["td_error_abs_mean"] * np.ones(len(batch["indices"]))
self.buffer.update_priorities(batch["indices"], td)
```

`update_priorities` giờ **từ chối** vector TD lệch độ dài (thay vì mis-attribute im lặng) và **loại bỏ** index ngoài phạm vi (trước đây `-1` ghi xuyên vào slot cuối).

### 2.3 — Drain không được phép chặn (lỗi #3)

```python
def drain(queue, limit=256, timeout_s=None):
    # DEEP-FIX: ``multiprocessing.Queue`` is NOT safe to read after one of
    # its writer processes has been killed.  ... a SIGTERM in the middle
    # leaves the prefix without the payload.  ``get_nowait()`` then reads
    # the prefix and blocks inside ``Connection._recv()`` ... and because
    # the *parent* also holds the queue's write end open, no EOF is ever
    # delivered, so the read blocks forever.  Verified: the shutdown test
    # hung for 600 s here, and a minimal parent/child repro hangs too.
    if timeout_s is None:
        return _drain_inline(queue, limit)          # hot loop: zero overhead
    if _queue_is_poisoned(queue):
        return []                                   # đã cách ly → bỏ qua ngay
    ...  # chạy trên daemon thread, join(timeout_s); treo → cách ly queue
```

`timeout_s=None` (mặc định) giữ nguyên đường inline **không tốn chi phí** cho hot loop; chỉ đường GUI/shutdown trả giá thread. `app.shutdown()` cách ly **đúng những queue mà worker bị terminate có thể ghi** (không cách ly bừa cả 4 kênh).

### 2.4 — Demo recorder thực sự ghi (lỗi #4 + #6 + #18 + #26)

```python
def pump(self, max_frames: int = 4) -> int:
    # DEEP-FIX: ``read_frame`` was stored by __init__ and never called by
    # anything in the repository ... every recorded episode had 0 frames
    # and stop() always reported "nothing saved": behaviour cloning had no
    # data source at all.
```
`gui._tick` gọi `self._pump_demo_recorder()` (main thread, đúng luật Tkinter); `KeyboardTap._build_listener()` dựng lại listener đã `stop()`; `tick()` append **trong lock**; `stop()` snapshot trong lock + kiểm tra độ dài các cột; tên file thêm bộ đếm chống đè.

### 2.5 — Learner không còn đốt 100% CPU (lỗi #5)

```python
# DEEP-FIX: this loop had no idle throttle at all, so a learner that could
# not train yet (buffer below warmup_transitions, or paused for a
# death/respawn) spun at 100% of one core doing nothing.  Measured on a
# 2-core box: 100% CPU, 0 updates, 60% system load.
if not did_work:
    time.sleep(0.005)
```

### 2.6 — `save_model(which="best")` commit state sau side-effect (lỗi #9 + #13)

```python
try:
    self._atomic_torch_save(payload, path)
except (OSError, RuntimeError, ValueError) as exc:
    ...; return None
if which == "best":
    # DEEP-FIX: the gate used to be advanced BEFORE the write, so a failed
    # save left best_metric pointing at a value with no file behind it.
    # Verified: a forced OSError left best_metric=10.0 with no best_model.pth.
    self.best_metric = metric
```
Kèm theo: `load_model` **verify sidecar** và cách ly file lệch hash thành `*.corrupt`; `_git_hash()` memo hoá; `capture_rng_states(generator=…)` / `restore_rng_states(…, generator=…)` chụp và khôi phục **đúng** Generator lấy mẫu replay.

---

# PHẦN 3 — GIẢI THÍCH CHUYÊN SÂU + TEST CASES + SUY LUẬN CỦA TÔI

## 3.1 Ba nguyên mẫu lỗi chi phối cả bảng chẩn đoán

Đọc 30 lỗi riêng lẻ sẽ không nhớ gì. Chúng quy về **3 kiểu tư duy sai lặp lại**:

**(A) "Kênh đã nối" ≠ "có người nghe."** Lỗi #1, #4, #13, #24, #31, #32.
Tác giả nối một đầu và tin rằng đầu kia tồn tại. Dấu hiệu nhận dạng rất ổn định: **một constant/sidecar/knob chỉ xuất hiện đúng một lần trong toàn repo**. Đó là lý do tôi chạy `grep -rn` cho từng cái tên khả nghi thay vì tin vào docstring — cả 4 phát hiện (#1 `__cmd__`, #4 `.tick(`, #24 hai knob, #13 sidecar) đều ra từ đúng một lệnh grep.

**(B) "Trông đúng kiểu" thay cho "đúng ngữ nghĩa."** Lỗi #2, #11, #14, #33, #35, #37.
`np.ones(N)` làm lời gọi `update_priorities` **có hình dạng** của một update per-sample. `capture_rng_states` trả về một dict **có hình dạng** của một snapshot RNG. `data = torch.save(...)` **có hình dạng** của một phép ghi có kiểm tra. Cách phát hiện: với mỗi lời gọi, hỏi *"giá trị này đến từ object nào, và ai tiêu thụ nó?"* — không phải *"kiểu có khớp không?"*.

**(C) Giả định đúng trên máy dev, sai trên máy đích.** Lỗi #15, #8, #6, #19.
`perf_counter`/`monotonic` trùng gốc trên Linux; `SyntheticGame` "chắc là ngẫu nhiên"; pynput `Listener` "tắt rồi bật lại được"; pause "an toàn thì dừng". Đây là nhóm nguy hiểm nhất vì **mọi test headless đều pass**.

## 3.2 Suy luận của tôi (My Reasoning) — theo đúng mạch đã đi

> **Tôi đọc `environment.py:737` và tự hỏi:** "ai đọc `__cmd__`?" Tôi grep `__cmd__` trên cả repo. Một kết quả — chính dòng ghi. Từ đó tôi *không* kết luận ngay "đây là dead code", vì có thể learner xử lý gián tiếp; tôi đọc `learner_main`: `drain(cmd_q)` cho lệnh, `drain(transition_q)` cho transition, và `add_transitions` chỉ nhận `NStepTransition`. Vậy dict bị vứt **im lặng**. Câu hỏi tiếp theo mới là câu quan trọng: *"nếu learner không đổi profile thì chuyện gì xảy ra?"* — và đó là lúc tôi thấy `SharedWeights` sized cho `PROFILE_ORDER[-1]` còn `unflatten_into` đọc tiền tố. Tôi viết script đo và thấy conv `(4,1,3,3)` bị ghi đè từ `(48,4,8,8)` mà không một exception nào. Một dòng dead code → hỏng trọng số âm thầm.

> **Tôi để ý `train_one` gọi `update_priorities` với `td_error_abs_mean * np.ones(N)` và tự hỏi:** "tại sao phải nhân với `ones`?" Nếu td là vector thì `* ones` thừa; nếu là scalar thì `* ones` là cách **biến scalar thành vector cho vừa chữ ký hàm**. Tác giả đã đổi hình dạng dữ liệu để khớp API thay vì đổi API để trả đúng dữ liệu. Tôi xác nhận bằng cách gọi `train_step` thật và kiểm tra dict trả về: chỉ có `td_error_abs_mean`, không có `td_errors`.

> **Tôi đọc `drain()` và tự hỏi:** "`get_nowait()` có bao giờ chặn không?" Phản xạ đầu tiên là "không — đó là nghĩa của `_nowait`". Nhưng tôi nhớ `mp.Queue` đọc 2 pha (độ dài, rồi payload), và `_nowait` chỉ áp dụng cho pha đầu. Tôi viết repro cha/con tối thiểu: con `put` blob 56 KB rồi bị `terminate()`, cha drain → **exit 124** (phải giết). Sau đó tôi đối chiếu với test đang treo: cùng stack `connection.py:_recv`. Khớp.

> **Tôi đọc `DemoRecorder.__init__` thấy `self.read_frame = read_frame` và tự hỏi:** "ai gọi cái này?" Đây chính là dạng câu hỏi mà prompt yêu cầu ("biến A không được dùng, có thể tác giả định dùng nhưng quên"). Grep `.tick(` → chỉ `FpsMeter` và test. Kết luận: recorder có **hai** API, GUI nối một cái, test test cái kia, và **không có gì nối chúng**. Tôi cố ý không "sửa" bằng cách gọi `read_frame` trong `stop()` — vì như thế cả episode sẽ được đọc một lần lúc kết thúc, sai mô hình latest-wins. Tôi thêm `pump()` và nối vào Tk loop, là nơi duy nhất được chạm widget.

> **Tôi nghi `NStepBuilder` nhân đôi death penalty — và tôi đã SAI.** Tôi đọc `_emit` và tưởng `if terminal: r += γ^span * r_span` chạy cho mọi span trong terminal drain. Viết ra từng bước với `buf=[e0,e1,e2]` mới thấy `terminal=(span==0)`, nên `span=2/1` **không** đi vào nhánh đó. Tôi bỏ phát hiện này. **Ghi lại ở đây vì một báo cáo chỉ liệt kê lỗi mình tìm ra, không kể lỗi mình tưởng ra, là báo cáo không đáng tin.**

> **Số 50–68% CPU đầu tiên của tôi là SAI.** Tôi đo `psutil.cpu_percent(interval=4.0)` chỉ 2–3 s sau khi process start, nên cửa sổ đo **bao gồm cả lúc learner cấp phát replay buffer 212 MB + SumTree**. Tôi làm lại bằng A/B có kiểm soát (cùng script, chỉ bật/tắt throttle) → **4% vs 100%**. Bài học: một con số đơn lẻ không phải bằng chứng; phải có对照组.

> **Tôi nghi `_end_episode` trộn đồng hồ giữa hai process** (`_episode_start_ts` là `monotonic` của capture, so với `monotonic` của actor). Tra lại: `CLOCK_MONOTONIC` là system-wide trên Linux, và Windows cũng vậy → phép trừ **có nghĩa**. Tôi không sửa, chỉ clamp ≥ 0 và ghi chú. Ngược lại `perf_counter` vs `monotonic` **trong cùng một process** thì Python tự tuyên bố là không so sánh được. Phân biệt được hai trường hợp này quan trọng hơn là "thấy đồng hồ khác nhau thì báo lỗi".

## 3.3 Test cases (bắt buộc ≥ 5) — `tests/test_deep_fix.py` (44 test) + `tests/test_tk_api_surface.py` (10 test) + `tests/test_input_controller.py` (26 test)

**Đã kiểm chứng hai chiều:** chạy trên code đã sửa → **41/41 pass**; chạy trên code gốc (`git stash` 17 file nguồn) → **34/41 fail**.

7 test pass trên **cả hai** tree, và tôi nêu rõ từng cái để bạn không hiểu nhầm là chúng chứng minh bug:

| Test pass trên cả hai | Vai trò thật của nó |
|---|---|
| `test_distinct_td_errors_give_distinct_priorities` | Chốt rằng `update_priorities` **vốn đã** đúng khi nhận vector — bug nằm ở **caller** (mục #2), không phải ở hàm này. |
| `test_learner_accepts_an_inline_profile_command` | Chốt kênh `cmd_q` của learner vốn đã hoạt động — bug là actor **không gửi** vào đó (mục #1). |
| `test_id_first_order_is_demonstrably_tearable` | **Bằng chứng** thứ tự ghi là load-bearing: tự tay ghi theo thứ tự CŨ và buộc reader thấy payload cũ. Chạy trên cả hai tree vì nó cố tình mô phỏng code cũ. |
| `test_round_trip_preserves_transitions` | Chốt save/load buffer đúng khi capacity khớp — bug chỉ lộ khi capacity **lệch** (mục #10). |
| `test_corrupt_checkpoint_is_quarantined_on_load` | Hành vi `*.corrupt` **đã có sẵn** cho file `.pth` hỏng; test này chống hồi quy, không phải sửa bug. |
| `test_episodes_are_not_identical` | 2 episode với seed **khác nhau** thì khác nhau — hiển nhiên đúng. Bug thật là **tool** dùng chung `cfg.seed`, nên tôi thêm test mức nguồn `test_evaluation_tool_derives_a_per_episode_seed` (fail trên code gốc). |
| `test_timeout_still_fails_when_never_recovered` | Chốt nhánh timeout **vẫn** hoạt động sau khi tôi đổi thứ tự kiểm tra (mục #16) — chống "sửa quá tay". |


| # | Test | Loại | Assert chính |
|---|---|---|---|
| 1 | `TestPerSamplePriorities::test_learner_writes_per_sample_priorities` | happy path | sau 1 `train_one()`, các priority được cập nhật có **> 1 giá trị phân biệt** |
| 2 | `::test_mismatched_td_length_is_rejected` | edge | `update_priorities(8 index, 7 td)` → `ValueError`, không mis-attribute |
| 3 | `::test_out_of_range_indices_are_dropped_not_wrapped` | edge (tấn công/biến dạng) | index `-1` **không** ghi xuyên vào slot cuối |
| 4 | `TestWeightLayoutGuard::test_mismatched_layout_is_refused` | edge | publish trọng số quality_cpu → actor strict_lite **từ chối**, trọng số giữ nguyên |
| 5 | `::test_nonfinite_weights_are_never_published` | edge (dữ liệu độc hại) | vector NaN → `ValueError`, actor không bị đầu độc |
| 6a | `TestEpisodeTelemetry::test_publish_invalidates_before_writing_the_payload` | **race** (xác định) | ghi đầu tiên = `(id, 0)`, ghi cuối = `(id, N)`, payload nằm giữa |
| 6b | `::test_reader_never_pairs_a_new_id_with_an_old_payload` | **race** (hammer) | 20.000 publish + reader song song (switch interval 1e-6), **> 100 lần đọc thật** → **0 lần rách** |
| 6c | `::test_id_first_order_is_demonstrably_tearable` | **race** (bằng chứng) | tiêm interleaving vào thứ tự CŨ → **bắt buộc** thấy payload cũ |
| 7 | `TestBufferPersistence::test_capacity_mismatch_is_a_corrupt_file_error` | edge | `CorruptFileError` có thông điệp `per.capacity`, không phải `ValueError` numpy |
| 8 | `::test_sample_never_skews_indices_and_transitions` | edge | evict một dải frame → mọi sample trả về có index **mô tả đúng** dữ liệu, **và** frame còn cư trú |
| 9 | `TestCheckpointIntegrity::test_best_metric_does_not_advance_on_a_failed_write` | edge | ép `OSError` → `best_metric is None`, và metric tốt hơn sau đó **vẫn** ghi được |
| 10 | `::test_corrupt_checkpoint_is_quarantined_on_load` | edge (bit-rot) | lật 1 byte → `load` trả `None`, file bị đổi thành `.corrupt` |
| 11 | `::test_rng_state_round_trips_the_sampling_generator` | happy path | Generator khác khôi phục state → **cùng dãy số** |
| 12 | `TestDemoRecorder::test_listener_is_rebuilt_after_stop` | edge | `stop()` rồi `start()` → listener mới, không phải no-op |
| 13 | `::test_pump_records_frames_from_the_reader` | happy path | `pump(5)` → file `.npz` có **5 frame** |
| 14 | `::test_same_second_episodes_do_not_overwrite` | edge | 3 episode cùng giây → **3 đường dẫn khác nhau**, cả 3 tồn tại |
| 15 | `TestGuardedDrain::test_poisoned_queue_is_skipped` | edge (deadlock) | queue treo 30 s → lần 1 trả `[]` trong < 5 s, lần 2 trả `[]` trong < 0.1 s |
| 16 | `TestEpisodeBoundaries::test_respawn_restores_the_death_penalty` | happy path | death #1 **và** #2 đều `−10.0` |
| 17 | `::test_flush_transitions_survives_an_unseeded_stack` | edge (shutdown sớm) | `_flush_transitions()` **không ném**, pending về 0, không ship gì |
| 18 | `TestEvaluation::test_evaluation_tool_derives_a_per_episode_seed` + `test_episodes_are_not_identical` | happy path | runner sinh seed theo episode; 2 episode **khác nhau** |
| 19 | `::test_report_load_tolerates_unknown_fields` | edge (schema drift + bản ghi rác) | field lạ + record không phải dict → bỏ qua 1, giữ 1, không crash |
| 20 | `TestDatasetIsolation::test_one_bad_file_does_not_abort_the_load` | edge | 1 tốt + 2 hỏng → load 1, 2 report `ok=False`, `zip` vẫn thẳng hàng |
| 21 | `TestNoAliasing::test_float32_input_is_not_mutated_in_place` | edge | input float32 **không bị sửa** tại chỗ, output vẫn `[0,1]` |
| 22 | `TestRespawnOrdering::test_recovery_at_the_deadline_is_not_reported_as_failure` | edge (biên thời gian) | hồi phục đúng lúc hết hạn → `RECOVERED`, không phải `FAILED` |

Ngoài ra: `tests/test_shutdown.py::test_gui_side_survives_actor_crash` (test có sẵn) — **trước: treo 600 s; sau: pass.**

## 3.4 LỖI TRONG CHÍNH BẢN SỬA CỦA TÔI — và cách một test bắt được nó

Phần này quan trọng không kém bảng chẩn đoán, vì nó là ví dụ cụ thể của "tự phản biện" mà bạn yêu cầu.

**Bản sửa đầu tiên của tôi SAI.** Với race episode-telemetry (mục #12) tôi chỉ đổi **thứ tự ghi**: payload trước, id sau, và cho reader đọc id → payload → đọc lại id để so. Tôi tin đó là seqlock. Nó không phải.

Trace lại bằng tay với reader/writer xen kẽ:

```
reader: id1 = 2217                       (đọc trước khi publish mới bắt đầu)
writer: survival = 2218                  (payload mới)
writer: reward   = 2218                  (id VẪN là 2217 -- chưa commit)
reader: survival = 2218                  <-- payload MỚI
reader: id2 = 2217  → bằng id1 → "nhất quán" → TRẢ VỀ {id: 2217, survival: 2218}
```

Reader đã ghép id của episode N với số liệu của episode N+1 — **đúng cái bug tôi định sửa**, chỉ đổi chiều. Ghi "id sau cùng" là *release marker*, không phải seqlock; seqlock thật phải **làm hỏng bản ghi trước khi đụng vào payload**.

**Tôi phát hiện ra không phải bằng đọc code, mà bằng siết test.** Test gốc `test_reader_never_pairs_a_new_id_with_an_old_payload` pass trên **cả hai** tree — tôi tưởng nó là bằng chứng. Khi tôi thêm hai assertion chống pass-rỗng (`writer_error is None`, `reads > 100`), nó **fail ngay trên code đã sửa** với:

```
E  AssertionError: torn episode telemetry: ['id=2217 survival=2218.0',
                                            'id=2337 survival=2338.0', ...]
```

Hai lần test của tôi đã che giấu lỗi thay vì bắt lỗi, và cả hai lần đều do cùng một nguyên nhân:
1. **Lần 1 — pass rỗng:** writer ném `AttributeError` (helper chưa tồn tại trên tree gốc), `finally` set `stop`, reader thoát ngay với 0 lần đọc → `mismatches` rỗng → "pass".
2. **Lần 2 — không có interleaving:** với switch interval mặc định 5 ms, writer chạy hết 4000 lần publish trong **một** lát GIL; reader không được chạy lần nào. Tôi phải hạ `sys.setswitchinterval(1e-6)` và cho writer `sleep(0)` định kỳ.

Và một phát hiện thứ ba: test "chứng minh race" đầu tiên của tôi **skip** với `no thread interleaving observed in 115212 reads` — nghĩa là race này **không tái lập được bằng may rủi** trong tiến trình dưới GIL (dù một script độc lập của tôi đo được 8.747 lần rách / 373.679 lần đọc). Test dựa vào may rủi là test vô dụng, nên tôi đổi sang **tiêm** interleaving: proxy buộc reader nhường GIL ngay sau khi đọc id, writer nhường giữa id và payload. Giờ nó xác định: thứ tự cũ **bắt buộc** lộ payload cũ.

**Bản sửa đúng** (`ipc.py:publish_episode_result`) — ba pha:

```python
# 1. invalidate -- 0 là marker "không có episode / đang ghi" mà
#    read_episode_result() đã coi là "không có gì để báo".
self.last_episode_done_id.value = 0
# 2. payload
self.last_episode_survival_s.value = float(survival_s)
self.last_episode_reward.value = float(total_reward)
# 3. commit -- ghi id cuối cùng mới làm bản ghi trở nên khả kiến.
self.last_episode_done_id.value = int(episode_id)
```

Giờ mọi reader có cửa sổ đọc chồng lấn pha ghi đều thấy `0` ở **một trong hai** lần đọc id → trả `None` → lần poll sau bắt lại. Không có đường nào để "id không đổi mà payload đã đổi".

**Kiểm chứng sau khi sửa:**
- `tests/test_deep_fix.py` → **41/41 pass**, trong đó test hammer đọc > 100 episode thật và **0 lần rách**.
- Test thứ tự (`test_publish_invalidates_before_writing_the_payload`) xác định: ghi đầu tiên phải là `(last_episode_done_id, 0)`, ghi cuối phải là `(last_episode_done_id, 7)`.
- Toàn suite → **473 passed, 0 failed**.
- Chạy thật `app.py --headless --steps 1500`: `new best model (survival_s rolling=5.50, window=[3.97, 7.37, 5.17])` — cửa sổ 3 episode đầy đủ, `best_model.pth` + `.sha256` sinh ra, và cửa sổ invalidate (micro-giây) **không làm mất episode nào**: `runs/headless_report.json` ghi đủ **12/12 episode** (id 1–12, `survival_s` mean 3.50 s, std 1.86).

**Bài học tôi rút ra:** một test race chỉ có giá trị khi nó chứng minh được rằng nó *đã chạy*. `assert reads > 100` là assertion rẻ nhất và quan trọng nhất trong cả file.

## 3.5 Thay đổi hành vi có chủ đích (phải khai báo rõ)

**Một** thay đổi không phải sửa lỗi thuần tuý: `best_model.pth` giờ **chỉ được ghi khi cửa sổ `rl.best_metric_window` đã đầy**.

- **Lý do:** README viết "rolling mean over `rl.best_metric_window` (default 3) episodes", nhưng code cũ mở gate ngay episode **đầu tiên** — một "rolling mean" của 1 mẫu. Một ván may 3 giây trở thành `best_model.pth` và mọi ván sau phải vượt qua nó.
- **Đánh đổi:** nếu run chết sau 2 episode thì chưa có `best_model.pth`. Chấp nhận được vì `latest_model.pth` vẫn có và `_load_checkpoint` đã fallback `latest → best`.
- **Tôi đã sửa 3 test có sẵn** trong `tests/test_best_model_tracking.py` để khớp ngữ nghĩa mới, mỗi chỗ có `# DEEP-FIX` giải thích, và thêm `test_window_of_one_still_saves_immediately` để khẳng định `best_metric_window=1` vẫn lưu ngay. Nếu bạn muốn giữ hành vi cũ, đây là **một khối code duy nhất** (`window_full` trong `_poll_episode_metric`) — đảo lại là xong.

---

# PHẦN 4 — MƠ HỒ & GIẢ ĐỊNH

Prompt yêu cầu nêu rõ chỗ mơ hồ. Đây là những chỗ tôi phải chọn, cùng phương án cho giả định còn lại:

| Chỗ mơ hồ | Giả định tôi chọn | Nếu giả định kia đúng thì sao |
|---|---|---|
| **`__cmd__` trên `transition_q` là tính năng hay lỗi?** | Lỗi — vì không có người đọc và actor không có `cmd_q`. | Nếu là tính năng "cố ý": tôi vẫn giữ đường `transition_q` và **thêm** xử lý phía learner, nên cả hai cách đều chạy. Không có hành vi nào bị mất. |
| **`_flush_transitions(final=True)` nên ship gì lúc shutdown?** | Huỷ phần pending nếu stack chưa seed; tham số `final` giữ nguyên chữ ký. | Nếu muốn ship transition terminal "nhân tạo" để đóng episode: cần reward/`done` có ý nghĩa, không phải `NOOP`/`0.0`. Tôi **không** bịa dữ liệu huấn luyện. |
| **Watchdog có được tự clear pause?** | Có, nhưng **chỉ** pause do chính nó set (`_stall_paused`), và phải có frame mới liên tục `stall_recovery_s=1.5 s`. | Nếu muốn "pause là phải người bấm Resume": đặt `stall_recovery_s` rất lớn; nhánh `if not self._stall_paused` đảm bảo pause của người dùng **không bao giờ** bị đụng. |
| **`per.capacity` đổi sau khi train: truncate hay từ chối?** | **Từ chối** bằng `CorruptFileError` kèm đúng con số cần đặt lại. | Nếu muốn truncate: phải xây lại frame-store + sum-tree và chấp nhận mất tham chiếu n-step — tôi coi "im lặng mất lịch sử" tệ hơn "báo lỗi rõ ràng". |
| **Đồng hồ trên Windows** | Coi việc trừ hai API đồng hồ là lỗi bất kể build nào (Python nói rõ reference point là undefined). | **Không kiểm chứng được ở đây** — sandbox là Linux. Tôi đo trên Linux: 2 hàm cùng `CLOCK_MONOTONIC`, lệch −1.3e-7 s. Tài liệu CPython xác nhận trên Windows `monotonic()` *trước đây* dùng `GetTickCount64()` (~15.6 ms) còn `perf_counter()` dùng `QueryPerformanceCounter()`, và chúng chỉ mới được hợp nhất ở bản CPython gần đây. Cần chạy `time.get_clock_info()` trên máy đích để chốt. |
| **`focus_gate()` trả `True` khi không xác minh được focus** | **Giữ nguyên** (không phải bug, là thiết kế cho headless) nhưng đã nêu ở mục 5. | Nếu muốn nghiêm: thêm cờ `input.require_verified_focus`, mặc định `False`. |

---

# PHẦN 5 — NHỮNG GÌ TÔI **KHÔNG** SỬA, VÀ TẠI SAO

Nói rõ để bạn không tưởng là đã xong:

1. **`StagnationDetector` và `TemplateDetector` chưa được nối vào đâu cả** (`DeathConfig.use_stagnation_fallback`, `gameover_template_path`, `respawn_template_path` cũng chết). Đây là **tính năng chưa hoàn thiện**, không phải bug — nối vào cần quyết định nghiệp vụ (fallback có quyền trigger respawn không? xung đột với debounce ladder thế nào?). Tôi để nguyên và nêu ra.
2. **`SharedWeights` vẫn là seqlock không có memory barrier.** Trên x86 (máy đích Windows) store có thứ tự nên an toàn; trên ARM thì không. Repo chỉ nhắm Windows/x86 nên tôi không đổi — nhưng đây là ràng buộc nền tảng cần ghi nhớ.
3. **`estimate_activation_memory_mb` đếm trùng** vì hook cả `nn.Sequential` con lẫn `Conv2d`/`GroupNorm`/`ReLU` bên trong. Docstring đã tự nhận là "estimate for comparison". Sửa sẽ đổi con số trong README §4 mà không đổi hành vi nào — tôi để nguyên.
4. **Đồng bộ trạng thái Pause giữa GUI và `events["pause"]`** vẫn lệch: GUI suy ra trạng thái từ `sm.state`, còn watchdog set event trực tiếp. Sửa đúng cần một nguồn sự thật duy nhất cho pause — là một refactor, không phải fix. Đã giảm thiểu bằng `_stall_paused`.
5. **Không có test chạy trên Windows thật, không có display, không có Chrome, không có pynput/pyautogui** trong sandbox này. Mọi phát hiện liên quan nền tảng (mục 15, focus gate, DPI) là **suy luận từ code + tài liệu**, không phải đo trên máy đích.
6. **Chưa chạy bot với game Poki thật** — giống như README đã khai báo trung thực từ trước. Mọi số liệu ở đây là headless/synthetic.

---

# PHẦN 6 — CÁCH TỰ KIỂM CHỨNG

```bash
cd /home/user/subway-surfers-AI

# 1. Toàn bộ test (473)
OMP_NUM_THREADS=1 python3 -m pytest tests -q --timeout=400        # 473 passed

# 2. Test hồi quy mới, và chứng minh chúng bắt được bug cũ
OMP_NUM_THREADS=1 python3 -m pytest tests/test_deep_fix.py -q     # 41 passed
# (lưu ý: đừng chạy khi đã `git add` -- khi đó `git diff --name-only` rỗng và
#  `git stash push --` sẽ nuốt luôn file test mới; tôi đã mắc đúng lỗi này)
git reset -q
git stash push -q -- $(git diff --name-only | grep -v '^tests/')
OMP_NUM_THREADS=1 python3 -m pytest tests/test_deep_fix.py -q     # 34 failed, 7 passed
git stash pop

# 3. Smoke test end-to-end đa tiến trình
OMP_NUM_THREADS=1 python3 app.py --headless --steps 1500          # HEADLESS SMOKE TEST: PASS

# 4. Xem toàn bộ chỗ sửa
grep -rn "DEEP-FIX" *.py tests/*.py | wc -l                       # 82
git diff --stat
```

**Môi trường đã dựng để chạy được test:** `torch 2.13.0`, `numpy 2.4.6`, `opencv-python-headless 5.0.0`, `pytest 9.1.1`, `psutil`, `pytest-timeout` (cài bằng `pip --break-system-packages`; `download.pytorch.org` bị chặn trong sandbox nên torch lấy từ PyPI — bản `+cu130` nhưng code assert `device.type == "cpu"` nên vẫn chạy CPU-only).

---

# PHẦN 7 — VÒNG 2: 4 LỖI PHÁT HIỆN KHI CHẠY TRÊN MÁY THẬT (Windows)

Bạn chạy commit `28f25f8` trên máy Windows và báo "vẫn dính cả đống lỗi".
Cả 4 lỗi dưới đây **đều là lỗi thật**, và điều quan trọng hơn: **cả 4 đều
nằm đúng vào điểm mù của vòng 1** — những đường code chỉ chạy khi có màn hình
thật, bàn phím thật, hoặc process con thật. Suite 449 test xanh trên Linux CI
nhưng không hề chạm tới chúng.

## Trước hết: tại sao vòng 1 bỏ sót?

| Lỗi | Vì sao test không bắt được |
|---|---|
| `winfovrootheight` | Không test nào tạo Tk root — CI không có display. Toàn bộ `gui.py` (923 dòng) được thực thi **0 lần**. |
| `backend_name == "auto"` | Test cũ chỉ xanh **vì** pynput luôn fail trên Linux. Nó pass ở CI và fail ở Windows — cùng một test, hai kết quả. |
| DPI awareness gọi quá muộn | Chỉ có tác dụng (hoặc thất bại) trên Windows có scaling. |
| `Thread._stop` bị shadow | Crash xảy ra trong `_bootstrap` của **process con**, *sau khi* log của ta đã báo shutdown sạch. |

Đây không phải 4 lỗi rời rạc — chúng là **một lớp lỗi**: *code không được
thực thi thì không được kiểm chứng*. Phần này sửa lỗi **và** đóng điểm mù.

---

## 7.1 🔴 `gui.py:294` — `winfovrootheight()` (thiếu dấu gạch dưới)

**Triệu chứng (log của bạn):**
```
AttributeError: '_tkinter.tkapp' object has no attribute 'winfovrootheight'
  File "gui.py", line 144, in _accept
  File "gui.py", line 294, in _region_accepted
```

**Nguồn gốc:** lỗi này **có sẵn từ commit gốc `948f7c3`, dòng 294** — không
phải do vòng 1 gây ra. Tôi đã xác minh bằng `git show 948f7c3:gui.py`.

**Gốc rễ (suy luận về tư duy người viết):** dòng 293 ngay trên viết
`winfo_vrootwidth()` **đúng**, dòng 294 viết `winfovrootheight()` **sai**.
Người viết gõ cặp width/height liên tiếp, tay trượt mất hai dấu `_` ở dòng
thứ hai. Vì dòng trên đúng nên khi đọc lướt cả cặp trông vẫn "hợp lý" —
đây chính là lý do tôi, khi "đọc từng dòng", vẫn bỏ qua: **mắt so khớp mẫu
(một cặp width/height) chứ không so khớp từng ký tự.**

**Hậu quả:** không chỉ Windows. Bấm "Select region" → kéo chọn → thả chuột
→ **crash trên mọi hệ điều hành**. Không thể hiệu chuẩn vùng bắt màn hình,
tức là không thể chạy bot. Lặp lại 3 lần trong log của bạn = 3 lần bấm thử.

**Đã sửa:** `winfo_vrootheight()`.

**Chống tái diễn — `tests/test_tk_api_surface.py` (mới, 10 test).** Vấn đề:
sandbox này không có `tkinter`, nên không thể `import gui`. Test do đó
**chỉ parse `gui.py` bằng `ast`** và đối chiếu mọi tên thuộc tính Tk với API
thật, lấy từ **nguồn CPython 3.13** (tải qua `gh api`, không phải trí nhớ):

- Ưu tiên 1: module `tkinter` đang cài (chạy trên máy bạn, có thẩm quyền).
- Ưu tiên 2: danh sách vendored từ CPython 3.13, gồm **toàn bộ họ
  `winfo_*`/`wm_*` (80 tên)** — để một lỗi gõ sai mới trong họ này bị bắt
  dù hôm nay `gui.py` chưa dùng.

Hai chi tiết kỹ thuật đáng ghi lại, vì chúng quyết định test có false
positive hay không:
- tkinter định nghĩa **alias** dạng `geometry = wm_geometry` và **alias nối
  tiếp** `pack = configure = config = pack_configure`. Regex naive bỏ sót cả
  hai → báo nhầm `geometry`, `pack`, `title` là không tồn tại.
- Test **tự kiểm chứng**: nó chèn lại đúng lỗi cũ vào bản sao và yêu cầu bộ
  quét phải bắt được. Một linter pass trên input hỏng thì tệ hơn không có
  linter.

**Đã kiểm chứng:** chèn lại lỗi → **4 test fail**; sửa → pass.

## 7.2 🟠 `input_controller.py` — `backend_name` không bao giờ resolve

```python
self.backend_name = backend        # "auto"  ← đây là YÊU CẦU, không phải câu trả lời
...
except Exception:
    self.backend_name = "dry_run"  # chỉ nhánh THẤT BẠI được ghi
```

**Gốc rễ:** tác giả gán `backend_name` từ tham số để thuộc tính luôn tồn tại,
rồi chỉ cập nhật ở nhánh degrade — vì điều tác giả nghĩ tới là *"ghi nhận
khi bị hạ cấp"*, không phải *"ghi nhận backend nào đang sống"*. Trên máy tác
giả điều này vô hại, vì **không có chỗ nào đọc trường này** ngoài hai phép
so `== "dry_run"` — mà hai phép so đó lại *tình cờ* cho kết quả đúng. Đây
là **trường chẩn đoán chỉ ghi không đọc (write-only diagnostic)**: lỗi nằm
yên cho tới khi có người đầu tiên đọc nó, và người đó là test này.

**Đã sửa:** mọi đường `return` trong `_make_backend` đều gán tên backend cụ
thể. Kèm theo:
- **Validate input** — `backend="pynpt"` (gõ sai) trước đây rơi thẳng vào
  nhánh `auto` và **âm thầm bấm phím thật** dưới một cái tên không ai yêu
  cầu. Nay raise `ValueError`.
- `requested_backend` giữ lại yêu cầu gốc, và `is_dry_run` thay cho chuỗi
  magic lặp ở 2 chỗ.
- `environment.py` **log ra** requested vs resolved — để trường này thực sự
  được *đọc*, không tái diễn tình trạng write-only. Đã thấy dòng này chạy
  thật: `input backend: dry_run`.

## 7.3 🟠 `gui.py` — DPI awareness được gọi **sau khi** đã tạo cửa sổ

```python
def set_dpi_awareness() -> float:
    """... return the Tk scale factor."""   # ← nói dối
    ...
    return 1.0                               # ← mọi nhánh đều trả 1.0 cứng
```

Ba lỗi trong một hàm:

1. **Docstring nói dối.** Hứa trả "Tk scale factor", nhưng trả `1.0` cứng ở
   *mọi* đường — kể cả đường thất bại. Caller nào tin nó sẽ coi màn hình
   150% thành 100%.
2. **Gọi quá muộn.** Windows chỉ chấp nhận `SetProcessDpiAwareness`
   **trước khi cửa sổ đầu tiên được tạo**. Hàm này chỉ được gọi từ
   `_region_accepted` (dòng 287) — tức là **sau** `tk.Tk()` (dòng 942), sau
   khi user đã kéo xong vùng chọn. Lúc đó Windows trả `E_ACCESSDENIED`
   (0x80070005) và `except Exception` **nuốt mất**.
3. **Không kiểm tra mã trả về.** `SetProcessDpiAwareness` báo lỗi bằng
   HRESULT chứ không raise, nên code cũ vẫn đặt `DPI_AWARENESS_SET = True`
   dù Windows đã từ chối.

**Hậu quả (chỉ trên Windows có scaling — đúng loại máy của bạn):** process ở
chế độ DPI-unaware, Tk báo toạ độ **ảo hoá** còn `mss` bắt ảnh theo pixel
**vật lý**. Vùng bạn kéo chọn **không phải** vùng bị capture. Không có thông
báo nào.

**Đã sửa:** gọi trước `tk.Tk()`; trả `bool` thật; phân biệt `E_ACCESSDENIED`
và log rõ; `_region_accepted` cảnh báo khi tỉ lệ lệch.

## 7.4 🔴 `safety_watchdog.py:57` — ghi đè `threading.Thread._stop`

```python
class SafetyWatchdog(threading.Thread):
    def __init__(self, ...):
        self._stop = threading.Event()   # ← đè lên method nội bộ của Thread
```

CPython's `Thread` có **method nội bộ tên `_stop`**. Gán thuộc tính cùng tên
làm nó biến mất. Khi `join()` hoàn tất, `Thread._wait_for_tstate_lock()` gọi
`self._stop()` → **`TypeError: 'Event' object is not callable`**.

**Tôi tìm ra lỗi này bằng cách chạy thật `python app.py --headless`**, chứ
không phải bằng đọc:
```
Process actor-worker:
Traceback (most recent call last):
  File "app.py", line 173, in _actor_entry
    watchdog.join(timeout=1.0)
  File "/usr/lib/python3.11/threading.py", line 1134, in _wait_for_tstate_lock
    self._stop()
TypeError: 'Event' object is not callable
```
**Mọi lần shutdown đều crash như vậy.** Lý do suite vẫn xanh: exception xảy
ra trong `_bootstrap` của process con, *sau khi* log chính đã in "actor loop
end" và "capture stop" — trông như một lần tắt sạch sẽ.

**Đã sửa:** `_stop` → `_stop_event`. Đã kiểm chứng trước/sau bằng chính lệnh
trên: traceback biến mất.

**Chống tái diễn:** test cấu trúc duyệt mọi lớp con của `threading.Thread`
trong repo và báo nếu có thuộc tính nào trùng tên với thuộc tính của
`Thread` — bắt cả lớp lỗi, không chỉ riêng `_stop`.

---

## 7.5 Hai lỗi "trông như lỗi" mà tôi đã kiểm tra và **kết luận là không phải**

- **`perception.py:189: undefined name 'torch'`** (pyflakes báo). Không phải
  lỗi: `"torch.Tensor"` là **PEP 563 string annotation** (`from __future__
  import annotations`), không bao giờ được evaluate; `import torch` nằm ngay
  trong thân hàm; tác giả đã đánh dấu `# noqa: F821`.
- **`environment.py:247-248: 'hz'/'dr' assigned but never used`**. Không phải
  lỗi: `_ingest` chỉ được gọi từ `reset()` (đã xác minh: đúng 1 call site),
  ở đó kết quả horizon/death cố ý bỏ đi. Chính hai lời gọi đó trong `step()`
  thì **có** dùng.

## 7.6 Hai lỗi trong **chính test của tôi**, tự bắt được

Ghi lại vì chúng đáng tin hơn là giấu đi:

1. `assert n.value.value == 1.0` — **`True == 1.0` là `True` trong Python**,
   nên mọi `return True` đều bị coi là "hằng số 1.0". Đã sửa bằng cách kiểm
   tra cả `isinstance(..., bool)`.
2. `assert "scale = set_dpi_awareness()" not in src` — fail vì **chính comment
   DEEP-FIX giải thích lỗi cũ có chứa chuỗi đó**. Đã chuyển sang assert trên
   AST thay vì trên văn bản.

## 7.7 Kiểm chứng (chạy thật)

| Kiểm chứng | Trước | Sau |
|---|---|---|
| `pytest -q` (toàn bộ) | 1 failed | **473 passed, 0 failed** (78.7 s) |
| `python app.py --headless --steps 300 --dry-run` | `TypeError: 'Event' object is not callable` khi shutdown | sạch, không traceback |
| Test mới chạy trên **source gốc** (`git show HEAD:`) | — | **17/17 fail** — chứng minh không vacuous |
| Quét toàn repo: `self.X` không được gán trong lớp/lớp cha | — | **0** (lỗi `winfovrootheight` là duy nhất của lớp này) |
| `python -m compileall` toàn bộ module first-party | — | OK |
| `python app.py --help`, `python profiling.py` | — | OK |

> **Lưu ý trung thực:** sandbox này không có `tkinter`, không có display, và
> không phải Windows. `gui.py` **không thể import được ở đây**. Mọi khẳng
> định về `gui.py` đều dựa trên phân tích AST + đối chiếu API CPython 3.13,
> **không** dựa trên việc chạy GUI. Ba lỗi 7.1/7.3 cần bạn xác nhận lại trên
> máy Windows thật.

---

# PHẦN 8 — VÒNG 3: hazard expiry + Dashboard để bạn **dùng & xem được**

## 8.1 🟠 `rewards.py` — hazard tồn tại lâu không bao giờ hết hạn (và trả thưởng sai thời điểm)

`register` được gọi ở **mỗi frame** horizon detector cháy, và nó từng làm
`self._current.opened_ts = horizon.ts` — ghi đè đúng cái field mà cả hai chỗ
kiểm tra hết hạn (`on_frame` và `expire_old`) đọc.

**Đã đo trước khi sửa:** 10 s nguy hiểm liên tục với `hazard_expiry_s=1.2`
→ `expired=0`, event vẫn mở. Tệ hơn: khi màn hình yên, event cũ resolve và
trả `hazard_bonus` cho action được ghi ở **frame 0 — tức 300 frame / 10 s
trước đó**. Đó là trả thưởng cho cú né một chướng ngại vật mà agent đã đi qua
mười giây trước — đúng thứ mà module này hứa là không làm ("no future
leakage").

**Đã sửa:** `opened_ts` giờ bất biến (là mốc deadline); thêm `last_seen_ts`
để giữ ngữ nghĩa "kéo dài cửa sổ nguy hiểm" mà không phá hết hạn;
`action_frame_id` ghi lại vị trí action được trả thưởng để chẩn đoán;
`stats()` thêm `hazards_open_age_frames` để một event bị kẹt hiện ra trong log.

**Test mới:** `tests/test_deep_fix.py::TestHazardExpiry` (4 test). Trên source
gốc: 3 fail, 1 pass (happy-path) — đúng kỳ vọng.

## 8.2 🖥️ `webui.py` — dashboard để preview, vì GUI Tk không thể hiện trong trình duyệt

Bạn nói "nhiều phần tôi không biết cách dùng và ko preview". GUI Tk thật cần
desktop Windows + Chrome + bàn phím, **không thể chạy trong preview**. Nên
`webui.py` phục vụ những phần **có thể chạy ở bất cứ đâu** qua HTTP thuần
stdlib (không cần cài gì):

- **Live**: `GameEnvironment` thật (perception + horizon + death detector +
  reward) chạy trên `SyntheticGame`, điều khiển bởi `InferencePolicy` thật,
  stream JPEG để bạn **xem bot chơi**.
- **Train**: chạy `app.py --headless` thật (đa tiến trình, shared memory) và
  tail log trực tiếp.
- **Report**: render `runs/headless_report.json`.
- **Profiles**: số param đo thật của từng profile + so budget.
- **Guide**: 7 mục giải thích phần nào chạy ở đâu; chỉ mục 7 (calibration GUI)
  cần Windows thật.

Chạy: `python webui.py --port 8000`. Đã kiểm chứng mọi endpoint bằng curl
(`health`, `profiles`, `demo start/stop`, `frame.jpg` trả ảnh thật, `train`
chạy ra `HEADLESS SMOKE TEST: PASS` với `workers_exited_cleanly: true`).

**Tôn trọng guard:** tôi bỏ `urllib` khỏi webui (parse query bằng tay) và bỏ
mọi `except: pass` để `tests/test_hygiene.py` vẫn xanh — vì webui giờ nằm
trong phạm vi quét của nó.

## 8.3 Kiểm chứng vòng 3

| Kiểm chứng | Kết quả |
|---|---|
| `pytest -q` toàn bộ | **481 passed, 0 failed** (81.4 s) — trước: 473 |
| `tests/test_hygiene.py` (quét cả webui.py) | 190 passed |
| `python webui.py` + curl mọi endpoint | OK; `frame.jpg` = ảnh game thật |
| `app.py --headless` qua dashboard | `HEADLESS SMOKE TEST: PASS` |

> **Trung thực:** sandbox này không có display/Windows, nên phần calibration
> GUI (mục 7) vẫn chưa thể chạy ở đây. Dashboard cho bạn xem Live/Train/Report
> — những phần thực sự chạy được không cần desktop.

---

# PHẦN 9 — VÒNG 4: preview đen thui + version hiển thị khắp nơi

Bạn gửi ảnh: bước 3 preview **đen thui**, anchor `RGB=(37,37,35) std=0.00
[ACCEPTED]`, và yêu cầu *"mỗi lần chỉnh sửa thì tăng version lên"*.

## 9.1 🔴 Vì sao preview đen và vì sao capture đen vẫn được [ACCEPTED]

Hai lỗi cộng hưởng:

1. **Ảnh preview không bao giờ được vẽ lên canvas bước 3.** Preview chỉ hiện ở
   `preview_label` (bước 2); `anchor_canvas` (bước 3) giữ nguyên nền `grey15`
   (gần đen) trống trơn. Nên bạn chọn anchor **mò trên một hộp đen**.
2. **Cổng chấp nhận là `std <= 6` ("ổn định") — nhưng một capture đen/đồng
   nhất lại là thứ *ổn định nhất* (std=0.00).** Heuristic "ổn định = tốt" bị
   đảo ngược khi capture hỏng: màn hình đen pass cổng và được `[ACCEPTED]`.
   Tác giả giả định "pixel ổn định là UI element tốt", nhưng quên rằng capture
   hỏng cũng ổn định tuyệt đối.

## 9.2 Đã sửa

- **Vẽ preview sống lên `anchor_canvas`** (cùng frame với bước 2) để bạn **nhìn
  thấy game** khi chọn anchor; thêm **vòng đỏ đánh dấu** vị trí bạn click.
- **Đọc độ sáng (luma) trực tiếp trên preview**: `capture luma X/255 — OK /
  BLACK! / WHITE?` — capture hỏng hiện ra ngay lập tức.
- **Cổng chấp nhận mới**: `ok = (std<=6) AND không đen AND không phẳng`.
  Capture đen → `[BLACK CAPTURE — not usable]`; màu phẳng → `[FLAT COLOUR]`.
- **Helper mới trong `perception.py`**: `mean_luma`, `is_black_frame`,
  `capture_problem` (chẩn đoán nguyên liệu: game bị che / sai vùng / hardware
  acceleration của Chrome), `is_degenerate_patch`.
- **Hệ thống version** (`version.py`, một nguồn duy nhất): title cửa sổ Tk,
  dòng khởi động `app.py`, header + `/api/health` của dashboard, và
  `app_version` trong `runs/headless_report.json`. Hiện tại **v1.5.0**. Quy
  tắc: bump `APP_VERSION` ở mỗi lần hành vi thay đổi.

## 9.3 Kiểm chứng

| Kiểm chứng | Kết quả |
|---|---|
| `pytest -q` | **490 passed, 0 failed** (83.4 s) — trước: 481 |
| `app.py --headless` | in `Subway Surfers Research Bot v1.5.0`; report có `app_version: 1.5.0` |
| Test guard capture đen (`TestBlackCaptureGuard`, 5 test) | pass; chứng minh patch đen có std=0 nhưng bị từ chối |
| Smoke | `HEADLESS SMOKE TEST: PASS`, `workers_exited_cleanly: true` |

> **Trung thực:** sandbox không có display, nên tôi không thể tự nhìn thấy cửa
> sổ Tk. Phần vẽ preview lên canvas được kiểm chứng bằng đọc code + test logic
> perception; bạn cần chạy `python app.py` trên Windows để thấy preview có hình.
> Nếu vẫn đen: tắt **hardware acceleration** của Chrome, đảm bảo game không bị
> che, và chọn lại vùng — dashboard/report giờ sẽ nói rõ lý do thay vì im lặng.
