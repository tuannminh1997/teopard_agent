# BÁO CÁO AUDIT — LUỒNG TỰ ĐẶT LỆNH BINANCE SPOT (Demo + Live)

Ngày audit: 2026-10-11
Phạm vi: auto đặt lệnh Spot cho Auto Scan (analyze.py + binance_executor.py + spot_user_stream.py)

---

## 1. SPOT DEMO (`SPOT_API_BASE=https://demo-api.binance.com`)

| Thành phần | Endpoint / Method | Parameter đã verify | Trạng thái |
|---|---|---|---|
| MARKET BUY | `POST /api/v3/order` | `symbol`, `side=BUY`, `type=MARKET`, `quantity`, `newClientOrderId` | PASS |
| Query order | `GET /api/v3/order` | `symbol`, `orderId` hoặc `origClientOrderId` | PASS |
| Tạo OCO | `POST /api/v3/orderList/oco` | `symbol`, `side=SELL`, `quantity`, `aboveType`, `abovePrice`, `aboveClientOrderId`, `belowType`, `belowStopPrice`, `belowClientOrderId`, `listClientOrderId` | PASS |
| Hủy OCO | `DELETE /api/v3/orderList` | `symbol`, `orderListId` hoặc `listClientOrderId` | PASS |
| Balance/fee | `GET /api/v3/account` | đọc `balances[].free`, `commissionRates` | PASS |
| Tra OCO | `GET /api/v3/orderList` | `symbol`, `origClientOrderId` (= listClientOrderId) | PASS |
| Listen key | `POST /api/v3/userDataStream` + `PUT` keepalive | `listenKey` | PASS |
| WebSocket | `wss://demo-stream.binance.com/ws/` | event `executionReport` / `listStatus` / `outboundAccountPosition` | PASS |

## 2. SPOT LIVE (`SPOT_API_BASE=https://api.binance.com`)

| Thành phần | Endpoint / Method | Parameter đã verify | Trạng thái |
|---|---|---|---|
| MARKET BUY | `POST /api/v3/order` | giống Demo (khác base URL) | PASS |
| Query order | `GET /api/v3/order` | giống Demo | PASS |
| Tạo OCO | `POST /api/v3/orderList/oco` | giống Demo | PASS |
| Hủy OCO | `DELETE /api/v3/orderList` | giống Demo | PASS |
| WebSocket | `wss://stream.binance.com:9443/ws/` | giống Demo | PASS |

Toàn bộ request lấy base từ `SPOT_API_BASE` (env) — **không có base URL nào hard-code**
trong `binance_executor.py`. `_market_filters()` luôn query `exchangeInfo` từ đúng base,
không cache filter cứng.

## 3. KHÁC BIỆT DEMO / LIVE

| Khía cạnh | Demo | Live | Xử lý trong code |
|---|---|---|---|
| REST base | `demo-api.binance.com` | `api.binance.com` | Đổi bằng env `SPOT_API_BASE`, code không phân biệt |
| WebSocket base | `demo-stream.binance.com` | `stream.binance.com:9443` | `spot_user_stream._ws_base()` map theo base |
| API key | key Demo Trading riêng | key production | 1 bộ key lưu/user (`key_store`); key Demo không dùng được trên Live |
| Symbol | thường trùng Live | — | Vẫn query `exchangeInfo` thật để lấy filter, không hard-code |
| Testnet | `testnet.binance.vision` (hệ riêng) | — | KHÔNG dùng — key khác hệ, sẽ 401 |

Không có khác biệt về tên/method/parameter giữa Demo và Live — đúng như tài liệu
Binance ("Demo Mode = production URL → demo URL").

## 4. LỖI PARAMETER ĐÃ SỬA

| Lỗi | Trước | Sau |
|---|---|---|
| Tạo entry futures/spot | `clientOrderId` (không tồn tại khi tạo order) | `newClientOrderId` |
| Hủy algo futures | `algoid` (sai case) | `algoId` |
| OCO thiếu mapping list | không gửi `listClientOrderId` | gửi `listClientOrderId` → tra/hủy được cả LIST |
| Sellable quantity | dùng `executedQty` / requested | `spot_fill_fees()`: executedQty − commission base-asset, cộng dồn MỌI fill |
| Phí BNB/USDT | coi là trừ base | chỉ trừ base khi `commissionAsset == baseAsset` |
| Partial fill | dùng requested qty | dùng `executedQty` thực tế |
| Free balance | không kẹp | `attach_spot_oco()` lấy `min(sellable, free)` rồi floor theo stepSize |
| Không đủ minQty | tạo OCO số lượng sai | raise `-4164`, KHÔNG tạo OCO, KHÔNG bán |
| TP2 dưới minNotional | tạo OCO sai | gộp về 1 OCO tại TP1 |
| Hủy OCO | chỉ theo `orderListId` | fallback tra `listClientOrderId` khi thiếu orderListId |
| Intent ledger Spot | không ghi trước MARKET BUY | ghi `market_submitting` → `filled` → `protection_submitting` → `protected` |
| Recovery Spot | không có | `recover_spot_intent_ledger()`: query entry cũ, chỉ tạo OCO thiếu |

### Trạng thái intent ledger Spot
`preparing → market_submitting → market_unknown (lỗi mạng) | filled → protection_submitting
→ protected | partially_protected → closed/canceled/expired/failed/recovery_required`

## 5. NHIỀU PLAN CÙNG SYMBOL

- Mỗi plan có bộ client ID riêng: `{plan_id}-e`, `{plan_id}-list`, `{plan_id}-tp`,
  `{plan_id}-sl`, `{plan_id}-list2`, `{plan_id}-tp2`, `{plan_id}-sl2`.
- `handle_spot_user_event()` map event → plan bằng `clientOrderId`/`listClientOrderId`.
  Khi OCO một plan khớp → **chỉ hủy OCO cùng plan đó**, không đụng plan khác.
- `cancel_pending_plan_orders()` hủy theo `orderListId` cụ thể, **không bao giờ hủy theo
  symbol tổng thể** (sẽ xóa nhầm OCO đang bảo vệ position thật của plan khác).
- Sellable qty luôn kẹp bởi `min(qty, free)` → plan sau không bán quá balance plan trước.

## 6. TRƯỜNG HỢP CHƯA TỰ ĐỘNG XỬ LÝ ĐƯỢC

1. **User hủy thủ công 1 leg trên GUI** — Binance không gửi event rõ ràng cho "leg bị hủy
   thủ công". Bot chỉ phát hiện qua reconciliation khi leg biến mất; nếu cả TP và SL đều
   bị hủy thủ công, coin sẽ trần cho tới khi `ensure_spot_protection` / recovery phát hiện.
2. **market_unknown mà entry không tồn tại** — đã đánh dấu `canceled`. Nếu nhưng response
   lost SAU khi FILLED thì entry tồn tại và bot tạo lại protection — đã xử lý được.
3. **`get_api_keys` lỗi/hết hạn key** — recovery đánh dấu `failed` cho user đó, không chặn
   user khác.
4. **Partially protected** — OCO-A tạo được, OCO-B lỗi: phần TP2 được đóng thị trường (đã
   xử lý), nhưng nếu đóng cũng thất bại → báo "VÀO GUI ĐÓNG NGAY" và cần can thiệp tay.
5. **DB write fail sau khi Binance accept** — request đã lên sàn nhưng DB không ghi được.
   Bot đánh dấu `recovery_required`; lần recovery sau sẽ query lại theo clientOrderId.
6. **WebSocket Binance chưa có event riêng cho "OCO do user tự hủy"** — chỉ có
   `executionReport` (cancel) cho từng leg. Đã map qua `listStatus` ALL_DONE, nhưng nếu
   event này rớt thì phải chờ REST reconciliation.

## 7. WEBSOCKET SPOT

- File mới: `spot_user_stream.py`
- Env thêm vào `.env`: `SPOT_USER_STREAM_ENABLED=1`,
  `SPOT_RECONCILIATION_INTERVAL_SECONDS=60` (bật cùng `job_check_predictions`)
- Xử lý: `executionReport`, `listStatus`, `outboundAccountPosition`, `listenKeyExpired`
- Dedup key: `(event_type, orderId/clientOrderId/orderListId, event_time)`
- Reconnect: exponential backoff 1s → 60s, daemon thread tự restart khi crash
- **Stream failure KHÔNG bao giờ làm fail order**: REST reconciliation
  (`ensure_spot_protection` + `ensure_spot_recovery`) vẫn chạy độc lập

## 8. TESTS

`tests/test_spot_order_flow.py` — 31 test, assert CHÍNH XÁC request payload:

- Spot Demo base URL / Spot Live base URL
- MARKET endpoint + `newClientOrderId` (và KHÔNG có `clientOrderId`)
- Query bằng `orderId` / `origClientOrderId`
- OCO endpoint mới + `aboveType`/`abovePrice`/`aboveClientOrderId`/
  `belowType`/`belowStopPrice`/`belowClientOrderId`/`listClientOrderId`
- Không nhầm `orderId` với `orderListId`; không dùng `price`/`stopPrice` (tên cũ)
- Cancel OCO bằng `orderListId` / `listClientOrderId`, KHÔNG hủy theo symbol
- Commission: base asset / quote asset / BNB / nhiều fills / partial fill
- requestedQty khác executedQty
- Free balance thấp hơn executedQty
- minQty / minNotional / rounding LOT_SIZE
- OCO duplicate / OCO-A thành công OCO-B lỗi
- TP1/TP2 2 OCO với client IDs riêng
- Plan A không hủy OCO Plan B (cả executionReport và listStatus)
- Balance dùng chung nhiều plan
- Timeout MARKET BUY → soft cancel theo clientOrderId
- Restart recovery: entry không tồn tại / tạo lại OCO thiếu / bỏ qua OCO đã có
- Manual cancel TP/SL
- WebSocket: dedup key, dedup cache, disabled by default, ws base demo vs live

Kết quả: **135/135 test PASS**, `py_compile` sạch.

## 9. BẢO MẬT

- `.env` và `*.db` đã nằm trong `.gitignore` — không commit được key/secret/DB.
- Không có secret nào bị log: `signed_request` chỉ log path, params in qua `safe_params`
  đã che `api_key`/`signature`/`secret`/`token`.
