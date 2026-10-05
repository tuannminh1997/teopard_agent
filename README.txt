TEOPARD BOT 3.2 — FUTURES
==========================

Teopard Bot là bot Telegram phân tích thị trường crypto qua LLM trên OpenRouter, lưu SQLite và chạy trên Railway.
Futures dùng Binance USDT-M với cấu hình đòn bẩy 20x; Spot dùng Binance Spot không đòn bẩy. Hai chế độ:
- Manual: user chọn symbol rồi chọn FUTURES (4H/1H/15m) hoặc SPOT (1W/1D/4H).
- Auto Scan: mỗi phiên (FUTURES/SPOT độc lập) quét theo nến đóng; NO TRADE không gửi
  và không lưu, chỉ lệnh trade được gửi + lưu vào /autoscanlog*. 2 lần quét liên tiếp
  cùng hướng thì tự bỏ qua 2 chu kỳ kế tiếp để đỡ tốn chi phí.

NGUYÊN TẮC KIẾN TRÚC
--------------------
- Futures có hai trạng thái TRADE (LONG/SHORT, vào được ngay) hoặc NO_TRADE. Spot chỉ BUY hoặc NO_TRADE. Không có trạng thái chờ trigger.
- Python chỉ ĐO, không KẾT LUẬN: dựng packet số đo khách quan (ATR/VWAP/EMA/rng/cl% và độ sâu sổ lệnh công khai),
  không gán nhãn xu hướng, không tự chọn hướng, không tự sửa Entry/SL/TP.
- Model tự kết luận hướng, Entry/SL/TP, điều kiện kích hoạt.
- FUTURES dùng `plan_validator.py` để kiểm tra thứ tự giá, R:R, khoảng cách SL,
  khoảng cách Entry và dẫn chứng; sai thì cho model sửa tối đa 1 lần.
- SPOT chỉ lấy ticker/nến OHLCV từ Binance Spot và gửi nguyên output của model ở Manual và Auto Scan, không sanitize, thêm giá,
  sửa format hay chặn theo mức Entry/SL/TP. Nếu các mức parse được thì bot lưu chúng
  để tracker theo dõi; không parse được vẫn gửi, nhưng không tạo bản ghi theo dõi.
- FUTURES chạy JSON: model trả 1 object JSON, bot trả NGUYÊN JSON cho caller
  (agent dịch vụ gọi API lấy trực tiếp để đặt lệnh Binance); tracker đọc JSON khi lưu.
- Model chỉ nhận dữ liệu thị trường hiện tại. History, Auto Scan log và evaluation
  data không được đưa lại vào prompt.
- Mỗi lần phân tích chỉ đưa dữ liệu thị trường của symbol được yêu cầu vào packet.
- Chỉ dùng nến đã đóng để kết luận outcome.

PIPELINE FUTURES (mode "futures")
--------------------------------
[1] DỮ LIỆU   4H/1H/15m (30/48/64 nến, chia 2 đoạn: đoạn cũ rút gọn + 24 nến gần nhất đủ cột)
              + mức tham chiếu ngày/tuần, funding, OI và long/short của symbol
[2] ĐO        chỉ báo + khoảng cách %/ATR + mức tham chiếu (Python) — packet KHÔNG in
              khối thanh lý; thay bằng quy tắc % giá MAX_SL_PCT; phái sinh in key=value
              khớp facts để model trích dẫn; dòng LIVE chỉ in trong bảng từng khung (không lặp)
[3] ANALYST   1 model, 1 lần gọi, trả JSON theo schema cố định (không còn trường trạng thái)
[4] KIỂM TRA  validate_plan → sai trả lại model sửa tối đa 1 lần → vẫn sai thì loại
[5] RENDER    JSON → văn bản tiếng Việt đúng định dạng cũ (parse/tracker cũ chạy nguyên)
[6] GỬI + LƯU + TRACK (MFE/MAE, chấm bằng nến 5m)
Lần sửa lỗi trong Auto Scan cũng reserve 1 lượt quota Planner.

PIPELINE SPOT (mode "spot")
-----------------------------
- Dùng riêng `analyze_system_prompt_spot.txt`; mode lưu trong DB là `spot`. DB cũ được migrate từ `long`/`swing` sang `spot` khi khởi tạo.
- Chỉ lấy ticker và nến OHLCV từ Binance Spot cho chính symbol đó. Không lấy funding, OI, tỷ lệ long/short hay dữ liệu Futures.
- Khung phân tích theo thứ tự: 1W bối cảnh → 1D cấu trúc/vùng → 4H trigger. Tải 300 nến mỗi khung; packet hiển thị 30/48/64 nến đã đóng (4H/1D/1W), nến cũ rút gọn và 24 nến mới nhất có thêm vol_ratio/takerBuy%/CVD.
- Prompt hướng model lần lượt đọc bối cảnh, cấu trúc, trigger, phản biện mua hay đứng ngoài, lập vùng mua và quyết định BUY hoặc NO TRADE.
- Manual và Auto Scan chuyển nguyên output model tới user. Python chỉ thử đọc hướng và giá để lưu/tracker nếu đủ trường; lỗi định dạng hoặc mức giá không đọc được không chặn hay sửa nội dung gửi.
- Auto Scan SPOT: NO TRADE không gửi (áp dụng cho cả 2 market); tin gửi đi là nguyên kết quả JSON của Planner, không sửa Entry/SL/TP.

ĐẶT LỆNH TỰ ĐỘNG (binance_executor.py)
----------------------------------------
- Bật phiên: /onfutu ETH hoặc /onspot ETH → bot hỏi "Bạn có muốn tự động hóa
  việc đặt lệnh không?" → [Có, cần thêm API key] [Không].
  + "Có": nhập API key → Secret (lưu trong bảng user_api_keys, mã hóa Fernet bằng env
    DATA_ENCRYPTION_KEY, loại khỏi bản /exportdb) → số lượng → đòn bẩy (futures).
  + "Không": chỉ gửi tín hiệu (order_status='no_auto').
- Khi Planner trả lệnh trade: đặt LIMIT entry (giá hiện tại, positionSide theo chế độ
  hedge/one-way tự detect) + TP/SL qua POST /fapi/v1/algoOrder (triggerPrice) TRƯỚC khi
  khớp; spot thì chờ khớp rồi gắn OCO. id lấy theo plan_id: futu-eth-1 (entry -e, TP -tp, SL -sl).
- Gửi user tin kèm block "🤖 ĐÃ ĐẶT LỆNH TỰ ĐỘNG" (plan_id, orderId, algoId, qty, leverage).
- /autoscanlogfutu | /autoscanlogspot liệt kê TOÀN BỘ lệnh phiên theo plan (không giới hạn 5).
- /offfutu ETH | /offspot ETH: tắt phiên + xóa lịch sử lệnh phiên;
  lệnh đã đặt trên Binance vẫn giữ nguyên (tự hủy trên GUI nếu muốn).

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
/onfutu ETH, /onspot ETH, /offfutu ETH, /offspot ETH
/autoscanstatus, /autoscanlogfutu, /autoscanlogspot

LỆNH ADMIN THƯỜNG DÙNG
----------------------
/exportdb        Tạo SQLite snapshot nhất quán và gửi qua Telegram
/adduser, /removeuser, /listusers, /setlimit, /resetusage
/addsymbol, /removesymbol, /checknow
Lệnh bảo trì gõ tay (không hiện menu): /dashboard, /dashboardall, /historyall, /statsall, /clearhistory.

DATABASE
--------
Railway dùng DB_PATH=/data/bot.db trên volume.
Không commit bot.db, bot_export*.db, *.db-wal, *.db-shm, __pycache__, .pytest_cache.
/exportdb: Admin gửi /exportdb trong Telegram, bot tạo snapshot bằng SQLite Backup API,
gửi file bot_export.db rồi xóa file tạm.

TEST
----
  python -m pytest tests -q
Test gồm: chỉ báo/packet futures, validator từng quy tắc, render khứ hồi
(parse lại đúng số với giá lớn và giá nhỏ), mock luồng Manual/Auto Scan
(kế hoạch FUTURES chưa qua validate_plan không gửi; SPOT trả JSON model theo cùng schema Futures).

BIẾN MÔI TRƯỜNG (ngưỡng kiểm tra Futures)
-------------------------------------------------------------
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| MIN_RR | 1.0 | R:R tối thiểu cho TP1 |
| SL_ATR_MIN / SL_ATR_MAX | 0.6 / 3.0 | Khoảng cách Entry→SL theo atr14_1h |
| MAX_SL_PCT | 2.0 | Khoảng cách Entry→SL tối đa theo % giá (thay quy tắc thanh lý) |
| ENTRY_READY_ATR15 | 0.25 | Giá được cách vùng Entry tối đa (theo atr14_15m) |
| CITE_REL_TOL | 0.005 | Sai số tương đối cho số trích dẫn dan_chung |
| FUTURES_FETCH_4H/1H/15M | 300 | Số nến Binance Futures tải mỗi khung (warm-up chỉ báo) |
| FUTURES_DISPLAY | 4H:30, 1H:48, 15m:64 | Số nến đã đóng hiển thị mỗi khung |
| FUTURES_FULL_COLS_N | 24 | Nến gần nhất mỗi khung in đủ cột vr/tb%/rng/cl% (cũ hơn rút gọn) |
| SPOT_FETCH_4H / SPOT_FETCH_1D / SPOT_FETCH_1W | 300 | Số nến Binance Spot tải mỗi khung để làm nóng chỉ báo |
| SPOT_DISPLAY | 4H:30, 1D:48, 1W:64 | Nến đóng hiển thị; nến cũ chỉ OHLC, 24 nến mới nhất thêm vol_ratio/takerBuy/CVD |
| SPOT_FULL_COLS_N | 24 | Số nến mới nhất mỗi khung có đủ cột vol_ratio/takerBuy/CVD |
| ORDERBOOK_DEPTH_LIMIT | 100 | Số mức bid/ask tối đa lấy từ endpoint riêng Futures/Spot; Python tính notional/imbalance theo dải quanh mid và các dải 0.05% tập trung notional lớn nhất |
| PLANNER_REASONING_EFFORT | high | Mức suy luận (token reasoning dùng chung hạn mức output) |
| DATA_ENCRYPTION_KEY | (bắt buộc trên Railway) | Khóa Fernet mã hóa API key của user trong DB |
| FUTURES_API_BASE | https://fapi.binance.com | Đổi sang https://demo-fapi.binance.com khi test demo |
| SPOT_API_BASE | https://api.binance.com | Base URL cho lệnh spot |

Đã XÓA (đợt 2): LEVERAGE, LIQ_MMR_PCT, LIQ_SL_MULT, ENTRY_WAIT_ATR1H.

LỊCH SỬ THAY ĐỔI
-----------------
- Đợt 2 (prompt vá): 2 trạng thái TRADE/NO_TRADE cho mọi mode; packet 30/48/64 nến
  chia 2 đoạn cột; bỏ khối thanh lý → MAX_SL_PCT; bỏ dòng LIVE trùng; phái sinh key=value;
  bỏ flash_note Auto Scan; bỏ STATUS_PARSE_ERROR (quy định không hợp lệ → NO_TRADE + log).
- Đợt 1: packet futures JSON; bỏ tùy chọn OPENROUTER_PROVIDER_ORDER (OpenRouter tự
  load-balance mặc định); mode trong DB dùng `futures`; migration tự đổi dữ liệu cũ từ `short`/`intraday`.
- Lưu ý cột debug predictions (market_snapshot, feature_snapshot, reasoning_summary,
  full_response, setup_status, mae/mfe, hold_hours, result_checked_at...) vẫn được GHI như
  bản gốc; setup_status giờ ghi TRADE/NO_TRADE. Dữ liệu legacy đọc qua
  normalize_decision_status().

VERSION
-------
Release hiện tại: 3.2
- 1.1, 1.2...: nâng cấp nhỏ hoặc sửa lỗi.
- 2.0, 3.0...: thay đổi kiến trúc lớn.
Version thực tế bot hiển thị (Telegram, DB) lấy từ biến Railway BOT_VERSION — sửa README này chỉ để tài liệu khớp, không ảnh hưởng bot chạy thật.
