TEOPARD BOT 3.2 — INTRADAY
==========================

Teopard Bot là bot Telegram phân tích tín hiệu crypto futures (Binance USDT-M, đòn bẩy 20x),
gọi LLM qua OpenRouter, lưu SQLite, chạy trên Railway. Hai chế độ:
- Manual: user chọn symbol rồi chọn INTRADAY (4H/1H/15m) hoặc SWING (1D/1W/1M).
- Auto Scan: mỗi chu kỳ quét (theo nến đóng) gọi Planner trực tiếp; NO TRADE thì không gửi,
  còn lại gửi ngay. 2 lần quét liên tiếp cùng hướng LONG/SHORT thì tự bỏ qua 2 chu kỳ kế tiếp
  để đỡ tốn chi phí.

NGUYÊN TẮC KIẾN TRÚC
--------------------
- Python chỉ ĐO, không KẾT LUẬN: dựng packet số đo khách quan (ATR/VWAP/EMA/rng/cl%),
  không gán nhãn xu hướng, không tự chọn hướng, không tự sửa Entry/SL/TP.
- Model tự kết luận hướng, Entry/SL/TP, điều kiện kích hoạt.
- Python có vai trò thứ hai là KIỂM TRA SỐ HỌC sau khi model trả lời
  (plan_validator.py): thứ tự giá, R:R sau phí, khoảng cách SL theo ATR,
  khoảng cách thanh lý, độ sát vùng Entry, số trích dẫn khớp packet.
  Sai thì trả lại model sửa tối đa 1 lần; vẫn sai thì loại, không gửi user.
- Mode INTRADAY chạy JSON: model trả 1 object JSON, Python render lại văn bản
  đúng định dạng cũ nên tracker, /history, /stats chạy nguyên không đổi.
- Model chỉ nhận dữ liệu thị trường hiện tại. History, Auto Scan log và evaluation
  data không được đưa lại vào prompt.
- Chỉ dùng nến đã đóng để kết luận outcome.

PIPELINE INTRADAY (mode "short")
--------------------------------
[1] DỮ LIỆU   4H/1H/15m (+1d,1w mức tham chiếu), funding, OI, long/short, BTC
[2] ĐO        chỉ báo + khoảng cách %/ATR + mức tham chiếu + thanh lý ước tính 20x (Python)
[3] ANALYST   1 model, 1 lần gọi, trả JSON theo schema cố định
[4] KIỂM TRA  validate_plan → sai trả lại model sửa tối đa 1 lần → vẫn sai thì loại
[5] RENDER    JSON → văn bản tiếng Việt đúng định dạng cũ (parse/tracker cũ chạy nguyên)
[6] GỬI + LƯU + TRACK (MFE/MAE, chấm bằng nến 5m)
Lần sửa lỗi trong Auto Scan cũng reserve 1 lượt quota Planner.

EVALUATION TRACKING
-------------------
- Auto Scan bị chặn sớm (quota/thiếu dữ liệu Binance): chỉ lưu log nhẹ, không gọi Planner.
- Khi Planner được gọi: lưu full market packet đã nén, output Planner và output public.
- Theo dõi Entry, SL, TP1, TP2, MFE và MAE.
- Sau khi SL bị chạm, tracker tiếp tục quan sát để phân biệt sai hướng với SL quá sát:
  - SL_THEN_TP1
  - SL_THEN_ENTRY_RECOVERED
  - SL_HIT_UNRESOLVED

LỆNH USER THƯỜNG DÙNG
---------------------
/start, /help, /listsymbols, /history, /stats
/autoscanon BTC, /autoscanoff, /autoscanstatus, /autoscanlog

LỆNH ADMIN THƯỜNG DÙNG
----------------------
/exportdb        Tạo SQLite snapshot nhất quán và gửi qua Telegram
/adduser, /removeuser, /listusers, /setlimit, /resetusage
/addsymbol, /removesymbol, /checknow
Lệnh bảo trì gõ tay (không hiện menu): /dashboard, /dashboardall, /historyall, /statsall, /clearhistory.

REPLAY (đo thay vì đoán)
------------------------
`replay_compare.py` chạy lại agent trên nến quá khứ, chấm kết quả bằng nến 5m,
so với 2 baseline (ngẫu nhiên, luôn LONG):
  python replay_compare.py --symbols BTCUSDT --from 2026-09-01 --to 2026-10-01 --step-hours 4 --max-calls 50 --out replay_out.csv
Ước tính số lượt gọi trước khi chạy; vượt --max-calls phải thêm --yes.
--dry-run không gọi model (chỉ test ống dẫn); --no-derivatives bỏ khối phái sinh.

DATABASE
--------
Railway dùng DB_PATH=/data/bot.db trên volume.
Không commit bot.db, bot_export*.db, *.db-wal, *.db-shm, __pycache__, .pytest_cache.
/exportdb: Admin gửi /exportdb trong Telegram, bot tạo snapshot bằng SQLite Backup API,
gửi file bot_export.db rồi xóa file tạm.

TEST
----
  python -m pytest tests -q
Test gồm: chỉ báo/packet intraday, validator từng quy tắc, render khứ hồi
(parse lại đúng số với giá lớn và giá nhỏ), mock luồng Manual/Auto Scan
(kế hoạch chưa qua validate_plan không bao giờ gửi), mode SWING chạy nguyên.

BIẾN MÔI TRƯỜNG MỚI (giá trị khởi điểm, chỉnh sau khi đo replay)
-----------------------------------------------------------------
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| LEVERAGE | 20 | Đòn bẩy cho ước tính thanh lý |
| LIQ_MMR_PCT | 0.5 | Ký quỹ duy trì ước tính (%), nên đối chiếu bậc ký quỹ thực từng cặp |
| FEE_TAKER_PCT | 0.05 | Phí taker mỗi chiều (%) |
| MIN_RR | 1.5 | R:R tối thiểu sau phí cho TP1 |
| SL_ATR_MIN / SL_ATR_MAX | 0.6 / 3.0 | Khoảng cách Entry→SL theo atr14_1h |
| LIQ_SL_MULT | 2.0 | Khoảng cách thanh lý tối thiểu so với khoảng cách SL |
| ENTRY_READY_ATR15 | 0.25 | Độ lệch cho phép khi READY_TO_ENTER (theo atr14_15m) |
| ENTRY_WAIT_ATR1H | 2.0 | Entry tối đa cách giá khi SETUP_WAITING_TRIGGER |
| CITE_REL_TOL | 0.005 | Sai số tương đối cho số trích dẫn dan_chung |
| INTRADAY_FETCH_4H/1H/15M | 300 | Số nến tải mỗi khung (warm-up chỉ báo) |
| INTRADAY_DISPLAY | 4H:30, 1H:36, 15m:32 | Số nến đã đóng hiển thị mỗi khung |
| PLANNER_REASONING_EFFORT | high | Mức suy luận (token reasoning dùng chung hạn mức output) |

LỊCH SỬ THAY ĐỔI SO VỚI PROMPT NÂNG CẤP
----------------------------------------
- Bỏ tùy chọn OPENROUTER_PROVIDER_ORDER: không gửi trường provider, để OpenRouter
  tự load-balance mặc định (yêu cầu trực tiếp, lệch so với prompt gốc).
- Nhãn hiển thị đổi SCALP → INTRADAY; tên nội bộ mode vẫn giữ "short" (không đổi schema DB).

VERSION
-------
Release hiện tại: 3.2
- 1.1, 1.2...: nâng cấp nhỏ hoặc sửa lỗi.
- 2.0, 3.0...: thay đổi kiến trúc lớn.
Version thực tế bot hiển thị (Telegram, DB) lấy từ biến Railway BOT_VERSION — sửa README này chỉ để tài liệu khớp, không ảnh hưởng bot chạy thật.
