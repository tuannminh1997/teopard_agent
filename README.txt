TEOPARD BOT 4.0 — FUTURES
==========================

Teopard Bot là bot Telegram phân tích thị trường crypto qua LLM trên OpenRouter, lưu SQLite và chạy trên Railway.
Futures dùng Binance USDT-M với đòn bẩy user nhập lúc bật phiên (1–125); Spot dùng Binance Spot không đòn bẩy. Hai chế độ:
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
- SPOT chỉ lấy ticker/nến OHLCV từ Binance Spot, không tự sửa các mức Entry/SL/TP
  (không đổi số, không chặn theo khoảng giá). Manual và Auto Scan hiển thị bản render
  cho user (kèm cắt khối Evidence), JSON thô trả qua khóa `json`. Các mức parse được
  thì bot lưu để tracker theo dõi; không parse được vẫn gửi, nhưng không tạo bản ghi theo dõi.
- FUTURES chạy JSON: model trả 1 object JSON. Manual và Auto Scan hiển thị bản render
  (Entry/SL/TP/Kích hoạt...) cho user, JSON thô trả qua khóa `json` để agent dịch vụ
  gọi API lấy trực tiếp đặt lệnh Binance. Tracker đọc JSON khi lưu.
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
- Manual và Auto Scan gửi user BẢN RENDER (render_plan_text + cắt Evidence) — không gửi
  nguyên JSON thô. Python chỉ thử đọc hướng và giá để lưu/tracker nếu đủ trường; lỗi
  định dạng hoặc mức giá không đọc được không chặn nội dung gửi và không tự sửa các mức.
- Auto Scan SPOT: NO TRADE không gửi (áp dụng cho cả 2 market); tin gửi đi là bản render
  của kết quả Planner, các mức Entry/SL/TP giữ nguyên số model trả về.

ĐẶT LỆNH TỰ ĐỘNG (binance_executor.py)
----------------------------------------
- Bật phiên: /onfutu ETH hoặc /onspot ETH → bot hỏi "Bạn có muốn tự động hóa
  việc đặt lệnh không?" → [Có, cần thêm API key] [Không].
  + "Có": nhập API key → Secret (lưu trong bảng user_api_keys, mã hóa Fernet bằng env
    DATA_ENCRYPTION_KEY, loại khỏi bản /exportdb) → số lượng → đòn bẩy (futures).
  + "Không": chỉ gửi tín hiệu (order_status='no_auto').
- Khi Planner trả lệnh trade: đặt LIMIT entry (giá hiện tại, positionSide theo chế độ
  hedge/one-way tự detect) + TP/SL TRƯỚC khi khớp.
  + Futures: POST /fapi/v1/algoOrder với algoType=CONDITIONAL, triggerPrice
    (endpoint hiện hành từ 12/2025 — ĐỪNG đổi về /fapi/v1/order, sàn trả -4120).
  + Spot: chờ entry khớp rồi gắn OCO qua POST /api/v3/orderList/oco
    (aboveType=LIMIT_MAKER + belowType=STOP_LOSS). Endpoint cũ POST /api/v3/order/oco
    đã deprecated 2024-04-02; chỉ dùng lại làm fallback khi HTTP 404/405 (endpoint
    không tồn tại nên chắc chắn không sinh OCO trùng).
  id lấy theo plan_id: futu-eth-77-1 (entry -e, TP -tp, SL -sl) — có user_id trong id
  để hai user không còn cùng plan_id; số thứ tự = MAX hậu tố + 1 nên không bị trùng
  sau khi executor bump plan_id.
- Demo symbol mapping: phân tích luôn lấy nến THẬT (fapi/api.binance.com). Khi
  FUTURES_API_BASE trỏ sang demo, bot map symbol sang symbol demo khớp giá live
  (ETHUSDT → ETHU nếu tồn tại trên demo) và đặt Entry/TP/SL GIỮ NGUYÊN từng số
  như plan — không re-anchor, không nhân tỷ lệ. Kiểm tra "plan hết hiệu lực"
  vẫn theo giá THẬT trước khi đặt.
- Gửi user tin kèm block "🤖 ĐÃ ĐẶT LỆNH TỰ ĐỘNG" NGẮN (chỉ Plan id + qty + đòn bẩy);
  orderId/algoId đầy đủ lưu DB và hiện trong /autoscanlog*.
- /autoscanlogfutu | /autoscanlogspot liệt kê TOÀN BỘ lệnh phiên theo plan (không giới hạn 5).
- Cửa sổ ngủ đêm (00:00–07:00 VN): autoscan tự TẮT và khi vào cửa sổ ngủ xóa TOÀN BỘ
  auto_scan_signals (lịch sử lệnh phiên của ngày cũ) — idempotent 1 lần/đêm theo ngày VN;
  predictions (lịch đánh giá) và lệnh đã đặt trên Binance giữ nguyên. 07:00 tự bật lại
  với log trống, next_session_plan_id bắt đầu lại từ 1.
  CHỈ wipe khi hủy lệnh treo thành công: thiếu key hoặc lỗi mạng thì GIỮ ledger +
  không set signals_wiped_day để tick sau thử lại (trước đây xóa blind làm mất mỗi
  đường về orderId trong khi lệnh còn treo trên sàn).
- Hủy lệnh TREO khi autoscan TẮT (mọi lý do: cửa sổ ngủ đêm 00:00–07:00, /offfutu,
  /offspot): hủy MỌI lệnh chưa khớp theo ledger auto_scan_signals (entry còn MỞ + TP/SL
  đi kèm, re-check entry trước khi gỡ TP/SL để không bao giờ làm position trần); entry
  ĐÃ KHỚP HẲN → giữ nguyên toàn bộ (position + TP/SL bảo vệ); entry khớp HẦN → hủy phần
  chưa khớp, giữ TP/SL. Thiếu key → không hủy được, BÁO CỤ THỂ cho user và GIỮ các dòng
  ledger đã đặt lệnh. Lệnh mòn theo ngày (rule 3 ngày) đã bỏ.
- Gỡ API key (/autoscanstatus → Đổi/Gỡ → "xóa"): hủy lệnh TREO TRƯỚC trong khi key còn,
  rồi mới xóa key. Đổi thứ tự từng làm lệnh treo thành mồ côi vì bot hết key để ký lệnh hủy.
- /autoscanstatus hiện trạng thái từng phiên + dòng "API key: Đã Thêm / Chưa thêm /
  ⚠️ KHÔNG đọc được" (trạng thái thứ 3 = dòng key tồn tại nhưng DATA_ENCRYPTION_KEY đã
  đổi nên không giải mã được → mọi lần đặt lệnh sẽ abort) và nút
  Thêm API Key / Đổi-Gỡ API Key (gõ "xóa" hoặc gửi tin trống để gỡ key).
  Luồng nhập key/qty hết hạn sau 10 phút để tin nhắn thường của user không bị coi là API key.
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
/start, /whoami, /help, /listsymbols
/history (10 lệnh gần nhất, mọi coin — KHÔNG nhận tham số symbol)
/onfutu ETH, /onspot ETH, /offfutu ETH, /offspot ETH
/autoscanstatus, /autoscanlogfutu, /autoscanlogspot

LỆNH ADMIN THƯỜNG DÙNG
----------------------
/exportdb        Tạo SQLite snapshot nhất quán và gửi qua Telegram
/adduser, /removeuser, /listusers, /setlimit, /resetusage
/addsymbol, /removesymbol
Lệnh bảo trì gõ tay (không hiện menu): /clearhistory (cần `/clearhistory CONFIRM`).

ĐÃ GỠ Ở 4.0
------------
/stats, /statsall, /dashboard, /dashboardall, /historyall, /checknow — cùng helper
format_stats() và các biến thể tham số `/history <symbol>`; help.txt/README đã cập nhật
theo. Admin không còn bản xem toàn hệ thống, /history của admin cũng chỉ hiện lệnh của
chính admin.

DATABASE
--------
Railway dùng DB_PATH=/data/bot.db trên volume.
Không commit bot.db, bot_export*.db, *.db-wal, *.db-shm, __pycache__, .pytest_cache.
/exportdb: Admin gửi /exportdb trong Telegram, bot tạo snapshot bằng SQLite Backup API,
gửi file bot_export.db rồi xóa file tạm. Bảng user_api_keys bị XÓA khỏi snapshot trước
khi gửi (API key mã hóa Fernet không rời khỏi máy).
- predictions: mỗi user giữ 10 dòng terminal mới nhất (/history) — lệnh đang mở
  (PENDING_ENTRY/ENTRY_FILLED) KHÔNG bị prune để job auto-check còn thấy vị thế.
- Index đã gỡ ở 4.0 (không query nào dùng): idx_predictions_user_symbol_mode_id,
  idx_auto_scan_settings_enabled, idx_eval_source_phase.
- Bảng analysis_snapshots không còn được ghi (chỉ được xóa khi cleanup) — giữ lại vì
  là bảng legacy; test_old_schema_preserved.py pin không được DROP.

TEST
----
  python -m pytest tests -q
Test gồm: chỉ báo/packet futures, validator từng quy tắc, render khứ hồi
(parse lại đúng số với giá lớn và giá nhỏ), mock luồng Manual/Auto Scan
(kế hoạch FUTURES chưa qua validate_plan không gửi; SPOT trả JSON model theo cùng schema Futures),
và tests/test_v4_fixes.py — regression cho các fix an toàn tiền/ledger của 4.0
(lỗi mạng không bỏ qua cleanup, hủy phần entry còn lại khi khớp nửa, gỡ TP algo mồ côi,
ledger không bị xóa khi lệnh còn trên sàn, prune không xóa lệnh đang mở, qty "0,97",
endpoint OCO spot, claim slot nguyên tử, plan_id unique).

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
| AUTOSCAN_SYMBOL_TIMEOUT_SECONDS | 600 | Watchdog: giới hạn thời gian MỖI symbol trong 1 cycle. Vượt quá thì bỏ chờ, ghi log timeout và chuyển symbol tiếp — tránh 1 symbol treo giữ lock làm autoscan chết tới khi restart |
| AUTOSCAN_SCHEDULER_TICK_SECONDS | 60 | Chu kỳ đánh thức job autoscan (mặc định code 60; .env Railway đang đặt 3600) |

Đã XÓA (đợt 2): LEVERAGE, LIQ_MMR_PCT, LIQ_SL_MULT, ENTRY_WAIT_ATR1H.

LỊCH SỬ THAY ĐỔI
-----------------
- Đợt 3b (4.0 — rà soát endpoint + chống chạy trùng):
  * Spot OCO: đổi POST /api/v3/order/oco (deprecated 2024-04-02) sang POST /api/v3/orderList/oco
    với aboveType/belowType; fallback endpoint cũ CHỈ khi HTTP 404/405 (không fallback theo
    lỗi nghiệp vụ vì có thể sinh OCO trùng). Sửa luôn tham số endpoint cũ (limitClientOrderId
    thay newClientOrderId, bỏ timeInForce) và tp_leg_order_id trước đây luôn None.
  * plan_id unique toàn cục: thêm user_id (futu-eth-77-1) + số thứ tự = MAX hậu tố + 1
    (COUNT(*)+1 sinh lại đúng id vừa bị executor bump → trùng).
  * Claim slot nguyên tử bằng BEGIN IMMEDIATE trên DB: lock in-memory chỉ bảo vệ 1 process,
    2 process chung bot.db vẫn qua cửa check-then-set rồi cùng đặt lệnh trùng.
  * Watchdog AUTOSCAN_SYMBOL_TIMEOUT_SECONDS (600s) cho MỖI symbol: cycle treo từng giữ lock
    mãi, mọi tick sau bị bỏ qua âm thầm cho tới khi restart.
  * Dọn: xóa bot_export.db sót trong thư mục gốc, bỏ 2 key demo chết trong .env.
  * Futures algoOrder (POST /fapi/v1/algoOrder, algoType=CONDITIONAL, triggerPrice) đã tra
    docs và XÁC NHẬN đúng — endpoint hiện hành từ 12/2025, không đổi.
- Đợt 3 (4.0 — rà soát luồng + dọn code):
  * An toàn tiền: lỗi mạng gói thành ExecutorError(-1000) để cleanup chạy + cancel best-effort
    khi timeout lúc gửi entry; hủy phần entry CHƯA khớp khi khớp nửa trước khi đóng position;
    gỡ TP algo mồ côi khi SL fail; spot hủy entry không khớp sau 30s + gắn OCO cho phần đã
    khớp; check MIN_NOTIONAL cho spot; chặn leverage 0 (đang dùng đòn bẩy cũ trên sàn).
  * Ledger: không xóa dòng auto_scan_signals khi Telegram gửi fail mà lệnh đã đặt;
    update_signal_orders scope theo prediction_id (trước đây WHERE plan_id ghi đè chéo user);
    /off* và gỡ key GIỮ dòng placed khi không hủy được; wipe đêm chỉ chạy khi hủy thành công;
    claim slot NGAY lúc bắt đầu cycle (chống đặt trùng khi restart giữa chừng).
  * Prune không xóa prediction còn đang mở.
  * Quota: hoàn lượt khi Planner trả rác (spot) hoặc repair không ra JSON (futures) — trước
    đây chỉ futures hoàn ở lỗi transport.
  * UX/sai số: qty "0,97" từng bị đọc thành 97; state nhập key hết hạn sau 10 phút;
    /autoscanstatus hiện trạng thái key không giải mã được; cảnh báo khi bot không xóa được
    tin chứa key; tắt trend-skip state khi tắt phiên + DELETE thiếu filter mode.
  * Cửa sổ ngủ: SLEEP==WAKE từng làm autoscan tắt vĩnh viễn.
  * Gỡ lệnh: /stats, /statsall, /dashboard, /dashboardall, /historyall, /checknow;
    /history bỏ tham số symbol và hiện 10 lệnh mọi coin.
  * Dọn: bỏ hàm chết (get_funding_rate_context, get_open_interest_context,
    build_futures_context_block, _guarded_no_trade_output, ensure_current_price_line,
    normalize_decision_status, binance_executor.current_price, get_auto_scan_logs),
    3 index chết, migrate_mode_values gọi thừa, ô ctx chết (oi/long_short/fear_greed),
    key trả về không ai đọc (candidate_id, admin_messages, last_log).
- Đợt 2 (prompt vá): 2 trạng thái TRADE/NO_TRADE cho mọi mode; packet 30/48/64 nến
  chia 2 đoạn cột; bỏ khối thanh lý → MAX_SL_PCT; bỏ dòng LIVE trùng; phái sinh key=value;
  bỏ flash_note Auto Scan; bỏ STATUS_PARSE_ERROR (quy định không hợp lệ → NO_TRADE + log).
- Đợt 1: packet futures JSON; bỏ tùy chọn OPENROUTER_PROVIDER_ORDER (OpenRouter tự
  load-balance mặc định); mode trong DB dùng `futures`; migration tự đổi dữ liệu cũ từ `short`/`intraday`.
- Lưu ý cột debug predictions (market_snapshot, feature_snapshot, reasoning_summary,
  full_response, setup_status, mae/mfe, hold_hours, result_checked_at...) vẫn được GHI như
  bản gốc; setup_status giờ ghi TRADE/NO_TRADE.

VERSION
-------
Release hiện tại: 4.0
- 1.1, 1.2...: nâng cấp nhỏ hoặc sửa lỗi.
- 2.0, 3.0...: thay đổi kiến trúc lớn.
Version thực tế bot hiển thị (Telegram, DB) lấy từ biến Railway BOT_VERSION — sửa README này chỉ để tài liệu khớp, không ảnh hưởng bot chạy thật.
