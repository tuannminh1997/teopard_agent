# PROMPT NÂNG CẤP TEOPARD BOT: INTRADAY 4H / 1H / 15m

**Cách dùng:** copy toàn bộ phần nằm giữa `=== BẮT ĐẦU PROMPT ===` và `=== KẾT THÚC PROMPT ===` dán vào coding agent (mở sẵn thư mục repo `teopard_agent`). Prompt chia thành các giai đoạn, agent phải **dừng lại báo cáo sau Giai đoạn 0** để bạn duyệt rồi mới sửa code. Các con số/ngưỡng trong prompt là **giá trị khởi điểm**, đều đưa vào biến môi trường để chỉnh sau khi chạy thử.

---

=== BẮT ĐẦU PROMPT ===

# VAI TRÒ VÀ QUY TẮC LÀM VIỆC

Bạn là kỹ sư Python cẩn thận. Repo này là **Teopard Bot**: bot Telegram phân tích tín hiệu crypto futures (Binance USDT-M, đòn bẩy 20x), gọi LLM qua OpenRouter, lưu SQLite, chạy trên Railway. File trung tâm là `analyze.py` (~4.250 dòng).

Quy tắc bắt buộc:
1. **Đọc trước, sửa sau.** Không sửa dòng nào trước khi hoàn thành Giai đoạn 0 và tôi đồng ý.
2. Tạo nhánh git mới `intraday-4h-1h-15m`. Mỗi giai đoạn một commit riêng, message rõ ràng.
3. Sửa tối thiểu và có chủ đích. **Không đụng** chế độ `long` (SWING: 1D/1W/1M) và các hàm nó đang dùng; giữ nguyên các hàm cũ nếu chế độ `long` còn gọi chúng (thêm hàm mới thay vì sửa hàm cũ).
4. **Giữ nguyên tên chế độ nội bộ `"short"`** (không đổi schema DB, không migrate). Chỉ đổi các tham số bên trong. Nhãn hiển thị cho người dùng đổi từ SCALP thành INTRADAY ở bước cuối nếu không gây vỡ parse.
5. Không in, không commit API key/token. Không commit `*.db`, `*.db-wal`, `*.db-shm`, `__pycache__`.
6. Mọi ngưỡng/hằng số mới đọc từ biến môi trường kèm giá trị mặc định (bảng ở cuối).
7. Mỗi thay đổi hành vi phải có unit test (pytest). Chạy toàn bộ test trước khi commit.
8. Nếu một giả định trong prompt này mâu thuẫn với code thật, **dừng lại và hỏi tôi**, không tự đoán.
9. Đừng nhầm lẫn: model LLM trong dự án này mặc định là `deepseek/deepseek-v4-flash-0731` qua OpenRouter (không phải Claude). Đừng đổi model mặc định; chỉ làm cho nó cấu hình được và so sánh được (Giai đoạn 3).

# MỤC TIÊU VÀ NGUYÊN TẮC THIẾT KẾ

Vấn đề cần giải quyết: model phân tích **không chính xác** khi đọc dữ liệu thị trường (đọc sai số, xác định sai vùng giá, SL/TP không khớp bằng chứng). Packet hiện tại đưa quá nhiều chỉ báo trùng lặp (RSI6/12/24, MACD, Ichimoku, EMA 7/25/50 cùng chuỗi 8 nến mỗi loại, CVD tính lại từ 0) và validation gần như không kiểm tra gì.

Nguyên tắc (không được vi phạm):
- **Python chỉ ĐO, không KẾT LUẬN.** Python được tính số (ATR, khoảng cách theo % và theo ATR, VWAP, cao/thấp N nến, giá thanh lý ước tính). Python **không** được gắn nhãn xu hướng/regime/setup, **không** được chọn hướng, **không** được tính hay sửa Entry/SL/TP hộ model.
- **Model tự kết luận mọi thứ** (hướng, Entry/SL/TP, điều kiện kích hoạt).
- **Python có một vai trò thứ hai: KIỂM TRA SỐ HỌC sau khi model trả lời** (SL đúng phía, R:R sau phí, khoảng cách ATR, khoảng cách thanh lý, số trích dẫn khớp packet). Nếu sai, **trả lại cho model sửa** kèm danh sách lỗi; Python không bao giờ tự sửa số.
- Model chỉ nhận dữ liệu thị trường hiện tại (giữ nguyên nguyên tắc cũ: không đưa history/log/evaluation vào prompt).
- Chỉ dùng nến đã đóng để kết luận; nến đang chạy tách riêng và ghi rõ chưa xác nhận.

# KIẾN TRÚC ĐÍCH

```
Lịch quét (mặc định vẫn 1H, cấu hình được) / lệnh Manual
 → [1] DỮ LIỆU   4H, 1H, 15m (+1d, 1w cho mức tham chiếu), funding, OI, long/short, BTC
 → [2] ĐO LƯỜNG  chỉ báo + khoảng cách + mức tham chiếu + khối thanh lý 20x (Python, chỉ số)
 → [3] ANALYST   1 model, 1 lần gọi, trả JSON theo schema cố định
 → [4] KIỂM TRA SỐ HỌC (Python) → sai thì trả lại model sửa tối đa 1 lần → vẫn sai thì loại, không gửi user
 → [5] RENDER    JSON → văn bản tiếng Việt đúng định dạng cũ (để parse/tracker cũ chạy nguyên)
 → [6] GỬI TÍN HIỆU + LƯU + TRACK (MFE/MAE giữ nguyên, chấm bằng nến 5m)
```

Không còn prefilter/reviewer (đã bỏ từ trước). Giữ nguyên.

---

# GIAI ĐOẠN 0: ĐỌC VÀ BÁO CÁO (KHÔNG SỬA CODE)

Đọc kỹ và báo cáo ngắn gọn (dưới 1 trang), trích số dòng:
1. Mọi chỗ dùng `SHORT_TERM_TIMEFRAMES`, `_mode_frame_roles`, `_missing_critical_timeframes`, `collect_timeframe_data`, `add_indicators`, `build_feature_engineering_block`, `build_user_prompt`, `build_synchronized_decision_snapshot`, `build_feature_snapshot`.
2. `parse_prediction_from_output` và `sanitize_user_output`, `_strip_public_evidence_for_user`: liệt kê chính xác regex/định dạng văn bản mà chúng yêu cầu (nhãn dòng, dấu gạch `–` giữa Entry thấp và cao, kiểu số).
3. `analyze_symbol`, `prepare_analysis_context`, `auto_scan_symbol_for_user`: luồng gọi từ Manual và Auto Scan có dùng chung hàm dựng packet và gọi model không.
4. `evaluation_store.py`: giá trị hiện tại của `ENTRY_WAIT_HOURS`, `TRADE_MAX_HOLD_HOURS` cho mode `short`, và cách tracker chấm outcome.
5. `get_funding_rate_context`, `get_open_interest_context`, `get_long_short_ratio_context`, `get_btc_correlation_snapshot`: chúng trả về dữ liệu gì, khung thời gian nào.
6. Nơi tính `_calc_rr` và các lifecycle (`evaluate_prediction_lifecycle`, `_tp_sl_result`): xác nhận chúng không phụ thuộc vào khung 15m/1H cụ thể.
7. Danh sách chỗ có thể vỡ khi đổi khung `short` từ (1H, 4H, 1D) sang (4H, 1H, 15m).
8. Liệt kê mọi điểm mâu thuẫn giữa mô tả trong prompt này và code thật.

**DỪNG và chờ tôi xác nhận trước khi sang Giai đoạn 1.**

---

# GIAI ĐOẠN 1: DỮ LIỆU, CHỈ BÁO, PACKET

## 1.1 Khung thời gian và số nến

Mode `short` dùng 3 khung, đối xử theo vai trò (nhưng Python **không** ghi vai trò như kết luận vào packet):

| Khung | Vai trò (chỉ để thiết kế) | Số nến đã đóng hiển thị | Số nến tải (warm-up) |
|---|---|---|---|
| 4H | bối cảnh | 30 (5 ngày) | 300 |
| 1H | cấu trúc | 36 (1,5 ngày) | 300 |
| 15m | điểm vào | 32 (8 giờ) | 300 |

- Đặt trong config: `INTRADAY_TIMEFRAMES = {"4H": ("4h", 300), "1H": ("1h", 300), "15m": ("15m", 300)}` và `INTRADAY_DISPLAY = {"4H": 30, "1H": 36, "15m": 32}`. Cho phép ghi đè qua env.
- `_mode_frame_roles("short")` trả `("15m", "1H", "4H")` (nhỏ đến lớn, giữ quy ước cũ). Khi dựng packet, in **lớn đến nhỏ** (4H, 1H, 15m).
- Tải thêm để lấy mức tham chiếu: nến `1d` (limit 10) và `1w` (limit 3). Nếu tải thất bại, **bỏ khối mức tham chiếu**, không làm hỏng cả phân tích. Nếu thiếu một trong 3 khung chính thì giữ hành vi cũ (ép NO_TRADE).
- Nến "đã đóng" = bỏ nến cuối (nến đang chạy), dùng lại `_v50_closed_df`.

## 1.2 Chỉ báo (tạo hàm mới `add_indicators_intraday`, KHÔNG sửa `add_indicators` vì mode `long` còn dùng)

Giữ lại, mỗi mục đích đúng một chỉ báo:

| Khung | Chỉ báo |
|---|---|
| 4H | EMA50, EMA200, ATR14, ADX14 |
| 1H | EMA20, EMA50, ATR14, RSI14 |
| 15m | EMA20, EMA50, ATR14, RSI14, VWAP ngày |

Các cột theo từng nến đã đóng (đi cùng dòng nến): `vol_ratio`, `takerBuy%`, `rng`, `cl%`, và **EMA chủ đạo của khung** (4H: EMA50; 1H: EMA20; 15m: EMA20) cùng `vwap` (chỉ 15m).

**Bỏ hẳn khỏi packet:** MACD, Ichimoku, RSI6, RSI24, CVD, chuỗi 8 nến của từng chỉ báo, EMA7, EMA25. (Giữ hàm tính cũ nếu mode `long` còn dùng.)

Định nghĩa chính xác:
- ATR14: Wilder, `tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()`, `tr = max(H-L, |H-prevC|, |L-prevC|)`.
- EMA: dùng lại `calculate_ema` hiện có. RSI14: dùng lại `calculate_rsi`. ADX14: dùng lại `calculate_adx`.
- `vol_ratio`: giữ như cũ (volume nến chia trung bình 20 nến liền trước, không tính chính nó).
- `takerBuy%`: dùng lại `_taker_buy_ratio`.
- `rng` = (H − L) / ATR14 tại **chính nến đó** (không phải ATR hiện tại). Làm tròn 2 chữ số.
- `cl%` = (C − L) / (H − L) × 100; nếu H == L thì 50. Làm tròn số nguyên.
- VWAP ngày (chỉ khung 15m): neo tại 00:00 UTC (07:00 giờ VN), `sum(tp*vol)/sum(vol)` cộng dồn từ nến đầu ngày UTC, `tp = (H+L+C)/3`. Tính trên nến đã đóng. Các nến đầu ngày VWAP dựa trên ít dữ liệu, chấp nhận.

## 1.3 Định dạng packet (QUAN TRỌNG, làm đúng chính xác)

Viết hàm mới `build_intraday_packet(timeframe_data, ref_levels, derivs_ctx, btc_ctx, current_price) -> tuple[str, dict]`. Trả về **(text, facts)**, trong đó `facts` là dict phẳng `{tên_ref: số}` chứa mọi giá trị số model có thể trích dẫn.

Nguyên tắc định dạng:
- Bảng cột thẳng hàng, **tiêu đề cột ghi một lần** mỗi khung, không lặp tên trường theo từng nến.
- Nến đặt nhãn theo vị trí: `t0` là nến đã đóng gần nhất, `t-1` nến trước nó, v.v. Thứ tự in: **cũ nhất trước, `t0` ở cuối**.
- Thời gian chỉ ghi `MM-DD HH:MM` (giờ VN). Ghi múi giờ và năm **một lần** ở đầu packet.
- Số thập phân cố định theo cột; dùng lại `fmt` hiện có cho giá để nhất quán độ chính xác theo từng coin.
- Chỉ báo dạng giá trị đơn kèm khoảng cách tới giá hiện tại ngay trong dòng: `key=giá_trị [±x.x% | ±x.xatr]`. Dấu dương nghĩa là mức nằm **trên** giá hiện tại. Khoảng cách theo ATR dùng `atr14_1h` làm chuẩn (ghi rõ ở đầu packet).
- Không thêm bất kỳ nhãn kết luận nào ("uptrend", "quá mua", "quá xa", "regime", ...).

Ví dụ bố cục (số minh họa, phải sinh bằng code, không viết cứng):

```
OBJECTIVE_MARKET_PACKET
Symbol BTCUSDT | INTRADAY | tạo lúc 2026-10-01 14:15 | giờ VN (UTC+7) | năm 2026
Giá hiện tại: price=64880.0
Khoảng cách trong [ ]: (mức − giá)/giá theo %, và theo ATR 1H (atr14_1h). Dấu + = mức nằm trên giá.

== 4H: 30 nến đã đóng == cột: n | thời gian | O H L C | ema50 | vr | tb% | rng | cl%
t-29 | 09-26 14:00 | ... 
...
t0   | 10-01 12:00 | 64420 64990 64380 64880 | 63900 | 1.4 | 57 | 1.3 | 81
atr14_4h=1180.0 | adx14_4h=24.6 | ema50_4h=63900.0 [-1.5% | -1.9atr] | ema200_4h=61250.0 [-5.6% | -7.0atr]
LIVE 4H (nến chưa đóng, đã trôi 40%): O=64880 H=64950 L=64790 C=64910 V=1523.4

== 1H: 36 nến đã đóng == cột: n | thời gian | O H L C | ema20 | vr | tb% | rng | cl%
...
atr14_1h=520.0 | rsi14_1h=61.2 | ema20_1h=64310.0 [-0.9% | -1.1atr] | ema50_1h=63990.0 [-1.4% | -1.7atr]
LIVE 1H (...)

== 15m: 32 nến đã đóng == cột: n | thời gian | O H L C | ema20 | vwap | vr | tb% | rng | cl%
...
atr14_15m=185.0 | rsi14_15m=64.0 | ema20_15m=64720.0 [...] | ema50_15m=64590.0 [...] | vwap_15m=64650.0 [...]
LIVE 15m (...)

== MỨC THAM CHIẾU == key=giá [%, atr]
prev_day_high=... | prev_day_low=... | prev_day_close=... | today_open=... | today_high=... | today_low=...
prev_week_high=... | prev_week_low=...
hh_15m_12=... | ll_15m_12=... | hh_15m_24=... | ll_15m_24=... | hh_15m_48=... | ll_15m_48=...
hh_1h_24=... | ll_1h_24=... | hh_1h_48=... | ll_1h_48=...

== RỦI RO ĐÒN BẨY 20x ==
liq_long=... [-4.5%] | liq_short=... [+4.5%] | fee_roundtrip_pct=0.10

== PHÁI SINH ==
funding (4 kỳ, cũ→mới, %): ...
oi_chg_1h/4h/24h (%): ... | price_chg_1h/4h/24h (%): ...   (cùng cửa sổ để model tự so)
long_short_top=... | long_short_crowd=...
taker_buy_pct_1h=... | taker_buy_pct_4h=...

== BTC (chỉ khi symbol khác BTC) ==
btc_chg_1h_pct=... | btc_chg_4h_pct=... | btc_ema50_1h_dist=[...]
```

Quy ước tên `ref` (để model trích dẫn và Python kiểm tra):
- Giá trị trong khối chỉ báo/mức/phái sinh: **dùng đúng key in trong packet** (ví dụ `ema20_1h`, `prev_day_high`, `funding_last`).
- Giá trị theo nến: `<trường>_<khung>_t<k>`, trường ∈ {o, h, l, c, vr, tb, rng, cl}; ví dụ `h_15m_t-3`, `c_1h_t0`.
- `facts` phải chứa **mọi** ref hợp lệ ở trên.

## 1.4 Mức tham chiếu, rủi ro đòn bẩy, phái sinh

- **Mức tham chiếu:** từ nến `1d` (ngày UTC) và `1w` (tuần UTC). `prev_day_*` = nến 1d đã đóng gần nhất; `today_*` = nến 1d đang chạy (O, H, L đến hiện tại); `prev_week_*` = nến 1w đã đóng gần nhất. `hh_/ll_` = cao nhất/thấp nhất của N nến **đã đóng** gần nhất ở khung tương ứng.
- **Rủi ro đòn bẩy:** `liq_long = price * (1 - 1/LEVERAGE + MMR)`, `liq_short = price * (1 + 1/LEVERAGE - MMR)`, với `LEVERAGE=20`, `MMR` lấy từ env `LIQ_MMR_PCT` (mặc định 0.5, đơn vị %). Đây là **ước tính xấp xỉ** (isolated, chưa gồm phí/funding). Ghi rõ chữ "ước tính" trong tài liệu và log. `fee_roundtrip_pct = 2 * FEE_TAKER_PCT` (mặc định `FEE_TAKER_PCT=0.05`).
- **Phái sinh:** dùng lại các hàm hiện có cho funding, OI, long/short. Bổ sung `oi_chg_*` và `price_chg_*` trong **cùng 3 cửa sổ** (1h, 4h, 24h) để model tự so sánh; `taker_buy_pct_1h/4h` tính từ dữ liệu taker buy volume trong nến 1H đã có (1 nến và 4 nến). Nếu một nguồn lỗi thì **bỏ riêng dòng đó**, không lỗi toàn bộ.
- **BTC:** chỉ khi symbol khác BTC. Rút gọn `get_btc_correlation_snapshot` còn: % thay đổi giá 1H và 4H, khoảng cách tới EMA50 1H. Bỏ phần còn lại.

## 1.5 Cập nhật nơi gọi

- `build_user_prompt` và `build_synchronized_decision_snapshot` cho mode `short` dùng packet mới (nến thô nằm **trong** packet, bỏ khối RAW OHLCV riêng để tránh trùng).
- `build_feature_snapshot` (lưu DB) cập nhật cho khớp khung mới, vẫn **không** gửi cho model.
- Lưu `facts` cùng analysis snapshot để validator và replay dùng lại.

**Commit Giai đoạn 1.** Thêm test: (a) ATR/VWAP/rng/cl% đúng trên dữ liệu mẫu; (b) packet có đủ các khối, `facts` chứa mọi key được in; (c) thiếu `1d`/`1w` thì packet vẫn dựng được; (d) mode `long` vẫn chạy y như cũ.

---

# GIAI ĐOẠN 2: PROMPT, JSON, KIỂM TRA SỐ HỌC, RENDER

## 2.1 Thay toàn bộ `analyze_system_prompt.txt` bằng nội dung sau

Hàm `load_system_prompt` thay các placeholder `{MIN_RR}`, `{SL_ATR_MIN}`, `{SL_ATR_MAX}`, `{LIQ_SL_MULT}`, `{FEE_RT}` bằng giá trị cấu hình thực tế.

````text
Bạn là trader futures crypto nhiều năm kinh nghiệm, giao dịch intraday trên Binance USDT-M, đòn bẩy 20x (isolated), giữ lệnh từ vài giờ đến khoảng một ngày. Bạn tự phân tích và tự quyết định: có vào lệnh hay không, hướng nào, các mức giá, điều kiện kích hoạt. Không ai kết luận hộ bạn: packet chỉ chứa số đo.

PACKET CÓ GÌ
Ba khung 4H, 1H, 15m. Mỗi khung có bảng nến đã đóng (t0 là nến mới nhất, cũ nhất ở trên) với cột: O H L C, EMA chủ đạo của khung (4H: ema50; 1H và 15m: ema20), vr, tb%, rng, cl% (15m còn có vwap). Dưới bảng là các giá trị đơn: ATR14, các EMA, RSI14, ADX14 (4H), kèm [khoảng cách tới giá hiện tại theo % và theo ATR 1H]; dấu + nghĩa là mức nằm trên giá. Sau đó là MỨC THAM CHIẾU, RỦI RO ĐÒN BẨY, PHÁI SINH, và BTC nếu coin không phải BTC. Nến LIVE là nến đang chạy, chưa xác nhận.

ĐỌC TỪNG TRƯỜNG
- vr: khối lượng nến chia trung bình 20 nến liền trước. tb%: tỷ trọng mua chủ động trong nến (50 là cân bằng).
- rng: biên độ nến (cao trừ thấp) tính theo ATR14 tại chính nến đó. cl%: vị trí đóng cửa trong biên độ nến (0 là sát đáy, 100 là sát đỉnh). Hai giá trị này giúp bạn đọc thân nến và râu nến mà không cần tự tính.
- vwap: giá trung bình theo khối lượng, tính từ 00:00 UTC (07:00 giờ VN) của ngày hiện tại. Đầu ngày vwap dựa trên ít nến.
- ADX14: sức mạnh xu hướng, không cho biết hướng.
- liq_long, liq_short: giá thanh lý ước tính ở 20x (xấp xỉ, chưa gồm phí và funding).
- PHÁI SINH: funding (dương: phe long trả phí cho phe short), thay đổi OI so với thay đổi giá trong cùng cửa sổ, tỷ lệ long/short của top trader so với đám đông, taker_buy_pct. Đây là bối cảnh về phe nào đông và dễ bị quét, không phải tín hiệu độc lập.

CÁCH PHÂN TÍCH (phương pháp của một trader, bạn tự rút ra kết luận)
- Đi từ khung lớn xuống nhỏ: 4H cho biết bối cảnh và các mức lớn, 1H cho cấu trúc và vùng vào lệnh, 15m cho điểm kích hoạt và vị trí đặt SL.
- Đọc nến bằng nhiều dấu hiệu cùng lúc: thân nến và râu nến (rng, cl%), khối lượng (vr), phe chủ động (tb%), vị trí so với EMA và vwap. Một nến đơn lẻ ở 15m nhiều nhiễu, đừng kết luận từ một nến.
- Phá vỡ chỉ đáng tin khi nến đóng cửa vượt mức và giữ được; râu dài vượt mức rồi đóng lại bên trong là dấu hiệu bị từ chối.
- Giá ở giữa biên độ, các khung mâu thuẫn nhau, hoặc không có vùng rõ để đặt SL: NO TRADE là quyết định hợp lệ và thường đúng.
- Với đòn bẩy 20x, biên thanh lý chỉ khoảng 4 đến 5% và phí khứ hồi khoảng {FEE_RT}% notional. SL phải đặt sau điểm mà ý tưởng bị vô hiệu cộng một khoảng đệm theo ATR, không đặt sát ngay rìa vùng hay râu nến mà bạn tự nhận là có thể bị quét.
- Tự cân nhắc phái sinh và BTC khi chúng ủng hộ hoặc đi ngược ý tưởng của bạn, và nêu trong phần rủi ro.

TRẠNG THÁI
- READY_TO_ENTER: điều kiện vào lệnh đã xảy ra trong nến đã đóng, vào được ngay.
- SETUP_WAITING_TRIGGER: kế hoạch hợp lệ nhưng điều kiện vào lệnh chưa xảy ra.
- NO_TRADE: không vào lệnh.

KIỂM TRA SỐ HỌC (hệ thống làm sau khi bạn trả lời)
Một bộ kiểm tra bằng code sẽ tính lại các điều sau, không đánh giá quan điểm của bạn:
1. LONG: SL < Entry thấp ≤ Entry cao < TP1 (< TP2 nếu có). SHORT: ngược lại.
2. R:R của TP1 tính theo giữa vùng Entry, đã trừ phí khứ hồi {FEE_RT}%, phải ≥ {MIN_RR}.
3. Khoảng cách từ Entry tới SL nằm trong khoảng {SL_ATR_MIN} đến {SL_ATR_MAX} lần atr14_1h.
4. Khoảng cách từ Entry tới giá thanh lý phải ≥ {LIQ_SL_MULT} lần khoảng cách tới SL.
5. READY_TO_ENTER thì giá hiện tại phải nằm trong hoặc sát vùng Entry.
6. Mỗi số bạn trích dẫn trong "dan_chung" phải khớp đúng với packet.
Nếu sai, bạn sẽ nhận danh sách lỗi để sửa. Hãy tự tính trước khi trả lời. Nếu sau khi sửa kế hoạch không còn đạt, hãy đổi sang NO_TRADE thay vì giữ một kế hoạch bạn tự thấy không vững.

CÁCH TRÍCH DẪN
Mỗi mức giá bạn chọn (Entry, SL, TP1, TP2) phải kèm tối thiểu một dẫn chứng là số có thật trong packet, dưới dạng {"ref": "...", "value": ...}. Quy ước ref: giá trị trong khối chỉ báo, mức, phái sinh dùng đúng tên key in trong packet (ví dụ ema20_1h, prev_day_high); giá trị theo nến dùng <trường>_<khung>_t<k> với trường ∈ {o,h,l,c,vr,tb,rng,cl}, ví dụ h_15m_t-3, c_1h_t0. Tối đa 8 dẫn chứng.

OUTPUT
Chỉ trả về MỘT đối tượng JSON hợp lệ, không có markdown, không có chữ ngoài JSON. Các trường số là số JSON (dấu chấm thập phân, không dấu phẩy ngăn nghìn). Các trường chữ viết tiếng Việt tự nhiên, không dùng thuật ngữ tiếng Anh trong câu giải thích (viết "đỉnh thấp dần", "trượt giá", "kiểm tra lại", "phá vỡ giả").

Nếu vào lệnh:
{
  "quyet_dinh": "LONG" hoặc "SHORT",
  "trang_thai": "READY_TO_ENTER" hoặc "SETUP_WAITING_TRIGGER",
  "entry_thap": số, "entry_cao": số, "sl": số, "tp1": số, "tp2": số hoặc null,
  "kich_hoat": "điều kiện vào lệnh, cụ thể và kiểm chứng được trên nến đã đóng",
  "bang_chung": {"entry": "...", "sl": "...", "tp1": "...", "tp2": "... hoặc null"},
  "dan_chung": [{"ref": "...", "value": số}],
  "rui_ro": ["...", "..."],
  "do_tin_cay": số nguyên 0-100
}
Nếu không vào lệnh:
{
  "quyet_dinh": "NO_TRADE",
  "trang_thai": "NO_TRADE",
  "ly_do": "1-2 câu ngắn, nêu đúng lý do không vào lệnh"
}
````

## 2.2 Gọi model và định dạng đầu ra

- Trong `_openrouter_create_once`: thêm tham số `response_format={"type": "json_object"}` khi gọi planner. Nếu nhà cung cấp/model không hỗ trợ và trả lỗi 400, **thử lại không có response_format** và dùng `_extract_json_object` (đã có) để tách JSON.
- **Bỏ** `"provider": {"sort": "price"}` mặc định. Thay bằng env tùy chọn `OPENROUTER_PROVIDER_ORDER` (danh sách nhà cung cấp, dấu phẩy); nếu không đặt thì không gửi trường `provider`. Mục đích: kết quả ổn định giữa các lần gọi.
- Sửa comment/giá trị lệch: comment ghi "high" nhưng mặc định là `"max"`. Giữ biến env, đặt mặc định `PLANNER_REASONING_EFFORT="high"`, và ghi chú rằng token reasoning dùng chung hạn mức `PLANNER_MAX_OUTPUT_TOKENS` với câu trả lời.

## 2.3 Hàm kiểm tra số học (module mới `plan_validator.py`)

`validate_plan(plan: dict, facts: dict, cfg) -> list[str]` trả danh sách lỗi (tiếng Việt, ngắn, nêu số cụ thể). Rỗng nghĩa là hợp lệ. Python **không sửa** kế hoạch.

Các kiểm tra (ngưỡng từ env, xem bảng cuối):
1. `quyet_dinh` ∈ {LONG, SHORT, NO_TRADE}; `trang_thai` nhất quán với quyết định.
2. Với LONG/SHORT: có đủ `entry_thap, entry_cao, sl, tp1` là số hữu hạn; `entry_thap ≤ entry_cao`. Thứ tự giá đúng phía (LONG: `sl < entry_thap`, `entry_cao < tp1`, `tp2 > tp1`; SHORT: ngược lại).
3. R:R sau phí cho TP1: `entry_mid = (entry_thap+entry_cao)/2`; `reward = |tp1-entry_mid|/entry_mid*100 - fee_rt`; `risk = |entry_mid-sl|/entry_mid*100 + fee_rt`; yêu cầu `reward/risk ≥ MIN_RR`. Báo lỗi kèm giá trị tính được.
4. Khoảng cách SL theo ATR: `|entry_mid-sl| / atr14_1h` ∈ [`SL_ATR_MIN`, `SL_ATR_MAX`].
5. Thanh lý: khoảng cách `|entry_mid - liq|` (liq_long cho LONG, liq_short cho SHORT) ≥ `LIQ_SL_MULT` × `|entry_mid - sl|`.
6. Entry gần giá: nếu `READY_TO_ENTER` thì `price` nằm trong `[entry_thap, entry_cao]` hoặc cách vùng ≤ `ENTRY_READY_ATR15` × `atr14_15m`. Nếu `SETUP_WAITING_TRIGGER` thì khoảng cách từ giá tới vùng Entry ≤ `ENTRY_WAIT_ATR1H` × `atr14_1h`.
7. `dan_chung`: mỗi mục có `ref` tồn tại trong `facts`; `value` khớp `facts[ref]` trong sai số tương đối `CITE_REL_TOL` (mặc định 0.005). Mỗi mức Entry, SL, TP1 phải có ít nhất một dẫn chứng (số lượng dẫn chứng ≥ 3 khi vào lệnh). `ref` lạ là lỗi.
8. Dữ liệu gốc (`facts`) thiếu giá trị cần cho một kiểm tra thì **bỏ qua kiểm tra đó và ghi log**, không báo lỗi giả.

## 2.4 Vòng sửa lỗi

Trong `analyze_symbol` (dùng chung cho Manual và Auto Scan, xác nhận ở Giai đoạn 0):
1. Gọi model lần 1, parse JSON. Nếu không parse được thì coi như lỗi định dạng (dùng lại ý tưởng của `_repair_planner_format`).
2. `validate_plan`. Nếu có lỗi: gọi model **lần 2 duy nhất**, giữ nguyên hội thoại (system, user packet, câu trả lời trước) và thêm một lượt user nêu danh sách lỗi và yêu cầu: sửa kế hoạch cho đúng, hoặc đổi sang NO_TRADE. **Không đưa thêm dữ liệu hay gợi ý hướng.**
3. Nếu lần 2 vẫn lỗi: loại kế hoạch, **không gửi cho người dùng**, ghi bằng `log_hidden_rejection`/kết quả `REJECTED_PLAN` như cơ chế cũ, kèm danh sách lỗi.
4. Mỗi lần thực sự gọi model đều phải đi qua cơ chế hạn mức/quota hiện có (`reserve_auto_scan_glm_call` và tương đương); lần gọi sửa lỗi tính thêm 1 lần. Xác nhận lại ở Giai đoạn 0 rằng không phá quota.

## 2.5 Render JSON thành văn bản cũ

Viết `render_plan_text(plan, symbol, mode_label, current_price) -> str` sinh ra **đúng định dạng văn bản mà `parse_prediction_from_output` và `sanitize_user_output` đang yêu cầu** (các dòng `🎯`, `🏆 QUYẾT ĐỊNH`, `Trạng thái`, `Giá hiện tại`, `Entry: low–high`, `SL`, `TP1`, `TP2`, `Kích hoạt`, `Bằng chứng ...`, `⚠️ Rủi ro`). Số viết theo kiểu quốc tế (dấu phẩy ngăn nghìn, dấu chấm thập phân). Nhờ vậy tracker, lưu DB, lịch sử, stats không phải sửa.

Bắt buộc có unit test vòng khứ hồi: `parse_prediction_from_output(render_plan_text(plan))` cho lại đúng `direction, entry_low, entry_high, sl, tp1, tp2` của `plan` (test với giá lớn như 64,880.5 và giá nhỏ như 0.08423).

Nhãn người dùng thấy: đổi "SCALP" thành "INTRADAY" **chỉ ở lớp hiển thị**, và chỉ nếu Giai đoạn 0 xác nhận không có regex/DB nào khớp theo chữ SCALP; nếu có thì giữ nguyên và báo tôi.

**Commit Giai đoạn 2.** Test: validator (mỗi quy tắc một test đạt/không đạt), vòng sửa lỗi (mock model trả sai rồi đúng; sai hai lần thì loại), render khứ hồi, `load_system_prompt` thay đủ placeholder.

---

# GIAI ĐOẠN 3: ĐÁNH GIÁ VÀ CÔNG CỤ SO SÁNH

## 3.1 Tham số vòng đời cho mode `short`

Trong `evaluation_store.py` và `analyze.py` (sau khi Giai đoạn 0 xác nhận cách chúng được dùng):
- `RESULT_CHECK_INTERVAL["short"] = "5m"` (chấm chạm SL/TP bằng nến 5m để phân biệt thứ tự chính xác hơn).
- `ENTRY_WAIT_HOURS["short"] = 3` và `TRADE_MAX_HOLD_HOURS["short"] = 24` (giá trị khởi điểm cho khung 15m/1H).
- `CHECK_INTERVAL_HOURS["short"]` giữ hoặc đặt 0.25 nếu tracker chịu được; báo lại tôi trước khi đổi nếu ảnh hưởng tải.
- Lịch quét: **không đổi mặc định** (vẫn `AUTOSCAN_INTERVAL_SECONDS=3600`). Chỉ xác nhận đặt `900` thì scheduler (`_auto_scan_slot_info`) vẫn căn đúng theo nến 15m, và báo rõ chi phí gọi model tăng khoảng 4 lần. Không tự bật.

## 3.2 Công cụ phát lại `replay_compare.py` (rất quan trọng, để đo thay vì đoán)

Tạo script độc lập (không chạy trong bot) để **chạy lại agent trên dữ liệu quá khứ**:
- Tham số: `--symbols`, `--from`, `--to`, `--step-hours` (mặc định 4), `--model`, `--packet {full,compact}`, `--no-derivatives`, `--max-calls`, `--out`.
- Với mỗi mốc thời gian `as_of`: tải nến lịch sử đến `as_of` (tham số `endTime` của Binance klines), dựng packet bằng **đúng** hàm production (`build_intraday_packet` nhận dữ liệu đã cắt theo `as_of`), gọi model, kiểm tra bằng `validate_plan`, ghi kết quả.
- Chấm kết quả: với mỗi kế hoạch hợp lệ, duyệt nến 5m **sau** `as_of` đến hạn `TRADE_MAX_HOLD_HOURS`, xét khớp Entry, chạm SL hay TP1/TP2 trước (nếu cùng một nến 5m chạm cả hai, tính là SL trước để thận trọng), tính R sau phí. Tái sử dụng logic có sẵn (`_tp_sl_result`, `_calc_rr`, `evaluation_store`) nếu phù hợp; nếu không, viết lại gọn và test.
- Phái sinh lịch sử (OI, long/short) Binance chỉ giữ khoảng 30 ngày; funding có lịch sử dài hơn. Script phải xử lý thiếu dữ liệu bằng cách bỏ khối đó (cờ `--no-derivatives` bỏ hẳn để so sánh công bằng).
- Đầu ra: file CSV/JSON mỗi dòng một mốc (as_of, symbol, quyết định, lỗi validator nếu có, kết quả, R), và tóm tắt cuối: số lệnh, tỷ lệ NO_TRADE, tỷ lệ bị validator loại, winrate, expectancy (R trung bình sau phí), drawdown, **so sánh với 2 baseline**: (a) vào lệnh ngẫu nhiên cùng hướng ngẫu nhiên với SL/TP theo ATR giống kế hoạch; (b) luôn LONG theo hướng EMA50 4H, SL/TP theo ATR. Ước tính trước số lần gọi model và chi phí, yêu cầu xác nhận trước khi chạy nếu vượt `--max-calls`.
- Dùng script này để A/B: `--packet full` (packet cũ) so với `compact` (packet mới), và so các `--model` khác nhau trên cùng dữ liệu.

**Commit Giai đoạn 3.** Test chạy khô (`--dry-run`) với model giả.

---

# GIAI ĐOẠN 4: DỌN DẸP VÀ BÁO CÁO

- Thêm `__pycache__/`, `*.pyc`, `*.db`, `*.db-wal`, `*.db-shm` vào `.gitignore` và `git rm -r --cached __pycache__` (không xóa lịch sử).
- Sửa README: version khớp, mô tả kiến trúc mới (mục "KIẾN TRÚC ĐÍCH" ở trên), liệt kê biến môi trường mới.
- Chạy toàn bộ test. Chạy thử với model giả (mock) cả luồng Manual và Auto Scan, đảm bảo không còn đường nào gửi kế hoạch chưa qua `validate_plan`.
- **Báo cáo cuối** (ngắn): danh sách file đổi, hàm mới, biến môi trường mới, những gì cố ý **không** làm, mọi điểm còn nghi ngờ hoặc cần tôi quyết định, và câu lệnh để chạy `replay_compare.py` lần đầu.

---

# BẢNG BIẾN MÔI TRƯỜNG MỚI (giá trị khởi điểm, hãy chỉnh sau khi đo)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `LEVERAGE` | 20 | Đòn bẩy dùng cho ước tính thanh lý |
| `LIQ_MMR_PCT` | 0.5 | Tỷ lệ ký quỹ duy trì ước tính (%), nên đối chiếu bậc ký quỹ thực của từng cặp |
| `FEE_TAKER_PCT` | 0.05 | Phí taker mỗi chiều (%), cần kiểm tra biểu phí hiện hành của tài khoản |
| `MIN_RR` | 1.5 | R:R tối thiểu sau phí cho TP1 |
| `SL_ATR_MIN` / `SL_ATR_MAX` | 0.6 / 3.0 | Khoảng cách SL theo `atr14_1h` |
| `LIQ_SL_MULT` | 2.0 | Khoảng cách thanh lý tối thiểu so với khoảng cách SL |
| `ENTRY_READY_ATR15` | 0.25 | Độ lệch cho phép khi READY_TO_ENTER (theo `atr14_15m`) |
| `ENTRY_WAIT_ATR1H` | 2.0 | Entry tối đa cách giá khi SETUP_WAITING_TRIGGER (theo `atr14_1h`) |
| `CITE_REL_TOL` | 0.005 | Sai số tương đối cho số trích dẫn |
| `INTRADAY_DISPLAY` | 4H:30, 1H:36, 15m:32 | Số nến đã đóng hiển thị mỗi khung |
| `OPENROUTER_PROVIDER_ORDER` | (trống) | Ghim nhà cung cấp; trống thì không gửi trường `provider` |
| `PLANNER_REASONING_EFFORT` | high | Mức suy luận |

=== KẾT THÚC PROMPT ===
