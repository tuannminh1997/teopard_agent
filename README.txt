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

GIớI HẠN LƯỢT PHÂN TÍCH THỦ CÔNG (manual)
- Mặc định 10 lượt/ngày cho mỗi user, tự reset lúc 00:00 giờ Việt Nam
  (`DEFAULT_DAILY_LIMIT` trong `auth.py`, cột `daily_limit` của bảng `whitelist`).
- Trừ 1 lượt ngay khi bắt đầu phân tích, hoàn lại nếu phân tích/gửi tin thất bại
  (`increment_user_usage` / `decrement_user_usage`).
- Đổi hạn mức theo từng user: `/setlimit <user_id> <số lượt>`.
- Hết lượt thì bot báo "Bạn đã hết 10 lượt hôm nay" và không gọi LLM.
- Giới hạn này KHÁC với `AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY` (quota Auto Scan,
  mặc định 20 lượt) và được tính riêng, không dùng chung.


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
- Bật phiên: /onfutu ETH,BTC,XRP hoặc /onspot ETH,BTC,XRP (tối đa 3 coin) → bot hỏi "Bạn có muốn tự động hóa
  việc đặt lệnh không?" → [Có, cần thêm API key] [Không].
  + "Có": nhập API key → Secret (lưu trong bảng user_api_keys, mã hóa Fernet bằng env
    DATA_ENCRYPTION_KEY, loại khỏi bản /exportdb) → **số USDT dùng chung cho mỗi coin** → đòn bẩy (chỉ futures).
    Khi chọn nhiều coin, bot đọc filter/giá Binance riêng từng coin, hiển thị từng minimum
    và yêu cầu nhập ít nhất mức cao nhất trong danh sách.
  + "Không": chỉ gửi tín hiệu (order_status='no_auto').
- Khi Planner trả lệnh trade: đặt LIMIT entry (giá hiện tại, positionSide theo chế độ
  hedge/one-way tự detect) + TP/SL.
  + **Chia chốt lời khi plan có tp2**: TP1 chốt `TP1_SHARE` (mặc định 70%), TP2 chốt phần
    còn lại (30%), **SL luôn = 100% qty**. Số lượng tự làm tròn theo stepSize và tp2 = qty
    − tp1 nên tổng LUÔN đúng bằng qty (không rơi rớt, không bị sàn trả -1111 "Precision
    is over the maximum defined"). Không có tp2 → TP1 nhận toàn bộ qty.
    SL 100% là an toàn dù TP1 khớp trước: đã test trên demo-fapi (hedge mode) — algo
    STOP_MARKET với qty LỚN HƠN position vẫn được sàn chấp nhận, và có reduceOnly=true.
  + Futures: POST /fapi/v1/algoOrder với algoType=CONDITIONAL, triggerPrice
    (endpoint hiện hành từ 12/2025 — ĐỪNG đổi về /fapi/v1/order, sàn trả -4120).
  + Mỗi plan lưu cả `algoId` và `clientAlgoId` cho TP/TP2/SL. Khi một leg terminal,
    user-stream Futures map event về đúng plan rồi chỉ hủy các leg còn lại của plan đó;
    các plan khác cùng symbol không bị hủy nhầm. REST reconciliation chạy lại sau restart
    hoặc khi WebSocket mất kết nối.
  + One-way mode chặn mở plan ngược chiều khi symbol đang có net position; muốn LONG và
    SHORT đồng thời phải bật Hedge Mode.
  + Spot: **MARKET BUY** `POST /api/v3/order`, chờ `SPOT_OCO_DELAY_SECONDS` rồi đọc
    `executedQty` và gắn OCO. Không có tp2 thì 1 OCO; có tp2 thì 2 OCO riêng (70% ở
    TP1, 30% ở TP2). OCO luôn dùng số coin đã khớp thực tế sau phí.
    Endpoint OCO: `POST /api/v3/orderList/oco` (aboveType=LIMIT_MAKER + belowType=STOP_LOSS);
    endpoint cũ `POST /api/v3/order/oco` deprecated 2024-04-02 chỉ fallback khi HTTP 404/405.
  id lấy theo plan_id: futu-eth-77-1 (entry -e, TP -tp, TP2 -tp2, SL -sl) — có user_id trong
  id để hai user không còn cùng plan_id; số thứ tự = MAX hậu tố + 1 nên không bị trùng
  sau khi executor bump plan_id.
- Demo symbol mapping: phân tích luôn lấy nến THẬT (fapi/api.binance.com). Khi
  FUTURES_API_BASE trỏ sang demo, bot map symbol sang symbol demo khớp giá live
  (ETHUSDT → ETHU nếu tồn tại trên demo) và đặt Entry/TP/SL GIỮ NGUYÊN từng số
  như plan — không re-anchor, không nhân tỷ lệ. Kiểm tra "plan hết hiệu lực"
  vẫn theo giá THẬT trước khi đặt.
- Gửi user tin kèm block "🤖 ĐÃ ĐẶT LỆNH TỰ ĐỘNG" NGẮN (chỉ Plan id + giá trị vị thế USDT +
  đòn bẩy + số coin tương ứng); orderId/algoId đầy đủ lưu DB và hiện trong /autoscanlog*.
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
- /offfutu ETH | /offspot ETH: tắt riêng coin đó khỏi phiên, hủy lệnh treo của coin
  và xóa lịch sử coin;
  `/offfutu all` | `/offspot all`: tắt toàn bộ coin của đúng market;
  lệnh đã khớp/position đang được bảo vệ vẫn giữ nguyên.

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
/start, /whoami, /help
/history (10 lệnh gần nhất, mọi coin — KHÔNG nhận tham số symbol)
/onfutu ETH, /onspot ETH, /offfutu ETH, /offspot ETH
/offfutu all, /offspot all
/autoscanstatus, /autoscanlogfutu, /autoscanlogspot

LỆNH ADMIN THƯỜNG DÙNG
----------------------
/exportdb        Tạo SQLite snapshot nhất quán và gửi qua Telegram
/adduser, /removeuser, /listusers, /setlimit, /resetusage
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
| PLANNER_SUPPLIER | (trống) | Danh sách nhà cung cấp OpenRouter được phép chạy, phân tách bằng dấu phẩy, THEO THỨ TỰ ƯU TIÊN. Bot chỉ gửi model tới các nhà trong danh sách (`provider.only` + `allow_fallbacks=false`); nhà đầu timeout / mất mạng / 5xx / 429 thì tự chuyển sang nhà tiếp theo. Lỗi 4xx (key/model sai) thì báo ngay chứ không đổi nhà. Để trống = OpenRouter tự chọn provider như cũ. Ví dụ: `inference.net,Modal,Baseten` |
| FUTURES_ENTRY_TTL_SECONDS | 86400 | TTL của LIMIT entry futures. Hết hạn mà entry **chưa** khớp → hủy entry + TP/SL đi kèm. Entry đã khớp một phần/hết thì **không bao giờ** bị hủy. Chỉ hủy lệnh có clientOrderId do bot tạo (`{plan}-e`) — lệnh user tự đặt trên GUI không bao giờ bị đụng tới |
| SPOT_USER_STREAM_ENABLED | 0 | Bật WebSocket user stream spot. Chỉ là đường tăng tốc; REST reconciliation vẫn là nguồn sự thật |
| SPOT_RECONCILIATION_INTERVAL_SECONDS | — | Chu kỳ REST reconciliation spot (chạy trong `job_check_predictions`) |
| FUTURES_RECONCILIATION_INTERVAL_SECONDS | — | Chu kỳ REST reconciliation futures |
| TP1_SHARE | 0.7 | Tỷ lệ qty cho TP1 khi plan có tp2. Phải nằm trong `(0, 1]`; ngoài khoảng thì log cảnh báo và dùng lại 0.7 |
| DATA_ENCRYPTION_KEY | (bắt buộc trên Railway) | Khóa Fernet mã hóa API key của user trong DB |
| FUTURES_API_BASE | https://fapi.binance.com | Host Futures hiện tại: LIVE; đổi về `https://demo-fapi.binance.com` nếu cần quay lại Demo |
| SPOT_API_BASE | https://api.binance.com | Host Spot hiện tại: LIVE; đổi về `https://demo-api.binance.com` nếu cần quay lại Demo |

Biến Auto Scan theo market (futures và spot tách riêng, chỉnh độc lập).
Tên đọc theo thứ tự: `AUTOSCAN_FUTURES_<TEN>` / `AUTOSCAN_SPOT_<TEN>` → fallback `AUTOSCAN_<TEN>` cũ → default:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| AUTOSCAN_MODES | futures,spot | Market nào được quét (dùng chung) |
| AUTOSCAN_{FUTURES,SPOT}_INTERVAL_SECONDS | 3600 | Tần số quét (giây) — mỗi market có SLOT RIÊNG nên đặt 2 giá trị khác nhau được |
| AUTOSCAN_{FUTURES,SPOT}_CANDLE_CLOSE_DELAY_SECONDS | 10 | Chờ thêm N giây sau khi nến đóng rồi mới quét |
| AUTOSCAN_{FUTURES,SPOT}_SYMBOL_TIMEOUT_SECONDS | 600 | Watchdog cho MỖI symbol trong 1 cycle; vượt quá thì bỏ chờ, ghi log timeout, chuyển symbol tiếp |
| AUTOSCAN_{FUTURES,SPOT}_SEND_NO_TRADE | 0 | 1 = gửi cả tin NO TRADE (thường tắt cho khỏi spam) |
| AUTOSCAN_SCHEDULER_TICK_SECONDS | 60 | Chu kỳ đánh thức job autoscan (dùng chung; .env đặt 300) |
| AUTOSCAN_SLEEP_HOUR_VN / AUTOSCAN_WAKE_HOUR_VN | 0 / 7 | Cửa sổ ngủ đêm VN (dùng chung) |
| AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY | 20 | Quota gọi Planner/ngày trên MỖI user (dùng chung 2 market) |
| AUTOSCAN_LOG_RETENTION_DAYS | 14 | Số ngày giữ auto_scan_logs (dùng chung) |
| TP1_SHARE | 0.7 | Tỷ lệ TP1 chốt khi plan có tp2 (TP2 nhận phần còn lại) |
| SPOT_FILL_TIMEOUT_SECONDS | 30 | Thời gian tối đa chờ market entry cập nhật executedQty |
| SPOT_OCO_DELAY_SECONDS | 3 | Thời gian chờ sau MARKET BUY trước khi gắn OCO |
| FUTURES_USER_STREAM_ENABLED | 0 | Bật user-data WebSocket để xử lý TP/SL Futures theo plan realtime |

Lưu ý key khi chạy Demo/Live:
- **1 API key DÙNG CHUNG cho Futures lẫn Spot**, và cũng dùng chung giữa demo với live
  (Binance: 1 key trade được cả Spot + Futures của cùng tài khoản).
  - Demo: `FUTURES_API_BASE=https://demo-fapi.binance.com` + `SPOT_API_BASE=https://demo-api.binance.com`
    → kiểm chứng 1 key chạy được cả 2 host demo.
  - `testnet.binance.vision` là HỆ RIÊNG (key riêng, 401 với key demo) → KHÔNG dùng.
  - Key lưu theo `user_id` (bảng `user_api_keys`, bỏ cột `market`); nhập 1 lần qua
    /onfutu hoặc /onspot → "tự động đặt lệnh".
  - Hiện tại project đã chuyển sang production/live: `.env` bật hai host live. API key
    giao dịch vẫn nằm trong `bot.db` và phải là key live; thay key qua
    `/autoscanstatus → Đổi API Key`, không đặt key thật vào README hoặc source.
- Spot demo DÙNG CÙNG symbol với production (BTCUSDT...) nên không cần map symbol demo
  như futures (futures mới có ETHUSDT → ETHU).
- Giá spot demo khớp giá production (đã đo: lệch 0.0000%), nên phần phân tích vẫn đọc
  api.binance.com như cũ không gây lệch; exchangeInfo demo có PRICE_FILTER/LOT_SIZE/
  NOTIONAL và ocoAllowed=true → `_market_filters` parse y hệt production.

Đã XÓA (đợt 2): LEVERAGE, LIQ_MMR_PCT, LIQ_SL_MULT, ENTRY_WAIT_ATR1H.

LỊCH SỬ THAY ĐỔI
-----------------
- Đợt 3d (4.0 — 5 bug cuôi + SL mồ côi):
  * **SL không tự hủy khi TP khớp**: Binance KHÔNG tự hủy TP/SL khi position đóng (hạn chế
    nội bộ sàn, có nhiều thread xác nhận) → SL cũ còn treo sẽ đóng luôn vị thế MỚI user mở
    sau trên cùng symbol. Thêm 2 tầng: `_watch_futures_close_and_cleanup` (task nền poll 30s)
    + `ensure_no_futures_orphan_protection` (job 30', sống sót qua restart). Điều kiện gỡ
    phải đủ 2: entry không còn mở VÀ symbol không còn vị thế (thiếu 1 là không gỡ).
  * `/offfutu`, `/offspot` giờ **giữ `_AUTO_SCAN_RUN_LOCK`** suốt hủy → xóa ledger. Trước đây
    cycle đang chạy có thể đặt lệnh sau bước hủy và trước bước xóa → lệnh sống trên sàn mà
    dòng ledger đã mất, trong khi reply vẫn báo "đã xóa N lệnh".
  * Wipe đêm bỏ lỡ khi bot down qua đêm: điều kiện mở rộng "trong cửa sổ ngủ **HOẶC** đã qua
    giờ thức" — trước đây chỉ trong cửa sổ ngủ nên bot down 00:00–07:00 là không wipe bao giờ.
  * Startup prune dùng **chung 1 limit cho cả nhóm ẩn và nhóm hiển thị** → nhóm ẩn bị giữ 10
    thay vì `HIDDEN_LEARNING_RETENTION_LIMIT` (5). Đã chia `CASE is_hidden`.
  * `_AUTO_PENDING` chỉ key theo `user_id` → bắt đầu `/onfutu` tới giữa chừng rồi gõ `/onspot`
    sẽ **ghi đè im lặng**, số lượng nhập cho futures bật nhầm phiên spot. Giờ báo rõ đã hủy.
  * `_binance_get_with_retry` in trọn `params` khi lỗi → chặn key nếu sau này có ai gọi signed.
  * Test mới: 104 (thêm 6 case cho các fix trên).
- Đợt 4 (4.0 — TP chia 70/30, OTOCO cho spot, bộ test UT+IT):
  * **TP1 70% / TP2 30% / SL 100%** khi plan có tp2, cho cả futures lẫn spot.
    `split_tp_qty()` làm tròn theo stepSize và tp2 = qty − tp1 → tổng luôn đúng bằng qty
    (bug tiềm ẩn: 0.0066 với step 0.001 từng bị sàn trả -1111 "Precision is over the
    maximum defined"). Không có tp2 → TP1 nhận toàn bộ qty.
    Xác minh TRÊN DEMO: futures algo TP1=0.015, TP2=0.007, SL=0.022 với qty=0.022.
  * **Spot đặt SL+TP chung lệnh entry**: `POST /api/v3/orderList/otoco` (entry + TP + SL
    trong 1 request, exchange tự gắn khi khớp) khi plan không có tp2; có tp2 thì 2 OCO
    sau khi khớp (OCO chỉ có 1 pendingQuantity nên không tách 70/30 được).
    Xác minh TRÊN DEMO: OTOCO ra đủ 2 lệnh TP/SL, `protected: True`.
  * **Phí theo coin**: `spot_fee_rate()` đọc commissionRates từ /api/v3/account;
    `pendingQuantity` của OCO bị kẹp bởi `free_before + qty*(1-phí)`. Bug thật đã gặp trên
    demo: gửi qty đầy khi số dư coin < phí → exchange bỏ qua pending IM LẶNG → mua xong
    position trần, không có lỗi nào.
  * Safety net `ensure_spot_protection` hỏi TRÊN SÀN (`spot_pending_state`) thay vì tin cột
    tp_algo_id trong DB — tránh trường hợp DB nói "đã bảo vệ" nhưng sàn không có gì.
  * Biến mới: `TP1_SHARE`, `SPOT_FILL_TIMEOUT_SECONDS`.
  * Bộ test: `tests/test_it_autoscan_order.py` (8 IT ghép autoscann + đặt lệnh + hủy lệnh
    cho cả 2 market, mock ở ranh giới HTTP nên code executor chạy thật) + 17 UT mới
    trong `tests/test_v4_fixes.py`. Tổng 98 test.
- Đợt 3c (4.0 — cấu trúc 1 key + spot như futures, kiểm chứng trên Spot Demo thật):
  * 1 API key DÙNG CHUNG futures + spot (cả demo lẫn live): bảng `user_api_keys` bỏ cột
    `market` (migration tự gộp, giữ dòng mới nhất/user), `key_store` bỏ tham số market,
    UI nói rõ 1 key nhập 1 lần, /autoscanstatus hiện 1 dòng key + 1 nút.
  * Spot Demo: `SPOT_API_BASE=https://demo-api.binance.com` (cùng hệ demo với
    demo-fapi, cùng 1 key). `testnet.binance.vision` là hệ riêng → 401 với key demo.
  * Biến Auto Scan tách theo market: AUTOSCAN_{FUTURES,SPOT}_{INTERVAL_SECONDS,
    CANDLE_CLOSE_DELAY_SECONDS,SYMBOL_TIMEOUT_SECONDS,SEND_NO_TRADE} + SLOT RIÊNG từng
    market (state key `last_scan_slot:{market}`) → đặt tần số quét khác nhau được.
  * Luồng spot giống futures: bỏ hành vi hủy entry sau 30s (entry GTC treo chờ zone);
    gắn OCO khi khớp qua task nền `_watch_spot_fill_and_protect` + safety net
    `ensure_spot_protection` chạy cùng job auto-check (sống sót qua restart).
  * BUG THẬT tìm khi chạy demo (từng làm hỏng cả futures):
    - `signed_request` gửi mọi method != GET thành POST → MỌI lệnh hủy (DELETE) chưa từng
      hoạt động → "/off*" và wipe đêm không hủy được lệnh treo. Đã sửa dùng
      `requests.request(method, ...)`.
    - `DELETE /api/v3/orderList` thiếu `symbol` (bắt buộc theo docs) → hủy OCO spot fail.
    - Clock lệch server → Binance trả -1021 chặn toàn bộ lệnh. Đã đồng bộ server time
      và tự thử lại 1 lần.
    - OCO bán đúng qty lý thuyết nhưng sàn đã trừ phí 0.1% vào coin → -2010 insufficient
      balance. Đã đọc số free thật và làm tròn xuống theo stepSize.
    - `get_api_keys()` trả None (không raise) khi chưa có key → None lọt vào place_plan
      thành TypeError rồi bị báo nhầm "lỗi mạng".
  * Xác minh trên Spot Demo thật: đặt entry → khớp → OCO đủ 2 leg (TP LIMIT_MAKER +
    SL STOP_LOSS) → hủy OCO + bán lại được. 81 test unit pass.
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


## Script kiểm tra / chạy thử (operator)

| Script | Mục đích |
|---|---|
| `python test_real_api.py` | Kiểm tra 86 mục **gọi API THẬT** theo host hiện tại trong `.env` (live/demo) không cần key: filter, giá, `min_order_usdt`, invariant TP/SL, WebSocket base, biến env có thực sự được dùng |
| `python dryrun_orders.py` | Dry-run đặt lệnh futures + spot với **filter thật** lấy từ `exchangeInfo`. Không gửi lệnh, chỉ assert payload (164 kiểm tra) |
| `python verify_orders.py --market futures --qty 10 --leverage 20` | Đọc key từ `key_store`, kiểm tra key sống trên base đang chọn, in toàn bộ payload rồi **hỏi xác nhận** mới đặt |
| `python verify_orders.py --market spot --qty 270 --live` | Biến thể đặt lệnh thật (cần chạy nơi có `bot.db` chứa key, ví dụ Railway) |

`verify_orders.py` luôn **đọc lại từ sàn sau khi đặt** (position, openOrders, algo orders) để verify — không chỉ tin response lúc gửi.

Không đặt được lệnh ở local vì `.env` không chứa API key Binance (key lưu per-user trong DB).


## Chế độ DEMO / LIVE

Bot in banner lúc khởi động để không ai nhầm lẫn mình đang tiền ảo hay tiền thật:

```
[MARKET_MODE] DEMO — futures=https://demo-fapi.binance.com spot=https://demo-api.binance.com — lệnh chỉ dùng tiền ảo
[MARKET_MODE] ⚠️⚠️ LIVE — futures=https://fapi.binance.com spot=https://api.binance.com — lệnh sẽ dùng TIỀN THẬT
```

**Mặc định khi thiếu biến env là DEMO** — cố ý, để biến env bị xoá nhầm chỉ làm mất tiền ảo.
**Cấu hình hiện tại của workspace là LIVE**: `.env` bật `https://fapi.binance.com` và
`https://api.binance.com`, còn hai dòng Demo được comment. API key của bot được lưu mã hóa
trong DB; cần thay key live qua `/autoscanstatus` trước khi bật tự động đặt lệnh.
