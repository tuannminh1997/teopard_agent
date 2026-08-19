TEOPARD BOT 2.2
================

Teopard Bot là bot Telegram phân tích tín hiệu crypto theo hai chế độ:
- Manual: user chọn symbol và SCALP/SWING.
- Auto Scan: Flash prefilter → xác nhận bias → Flash planner → GPT reviewer.

NGUYÊN TẮC KIẾN TRÚC
--------------------
- Model chỉ nhận dữ liệu thị trường hiện tại.
- History, Auto Scan log và evaluation data không được đưa lại vào prompt.
- Python điều phối pipeline, parse format và lưu dữ liệu; không tự sửa Entry/SL/TP.
- Chỉ dùng nến đã đóng để kết luận outcome.

EVALUATION TRACKING
-------------------
- Auto Scan bị loại sớm: chỉ lưu metadata nhẹ.
- Khi Pro được gọi: lưu full market packet đã nén, output Pro, reviewer và output public.
- Theo dõi Entry, SL, TP1, TP2, MFE và MAE.
- Sau khi SL bị chạm, tracker tiếp tục quan sát để phân biệt sai hướng với SL quá sát:
  - SL_THEN_TP1
  - SL_THEN_ENTRY_RECOVERED
  - SL_HIT_UNRESOLVED

LỆNH USER THƯỜNG DÙNG
---------------------
/start
/help
/listsymbols
/history
/stats
/autoscanon BTC
/autoscanoff
/autoscanstatus
/autoscanlog

LỆNH ADMIN THƯỜNG DÙNG
----------------------
/exportdb        Tạo SQLite snapshot nhất quán và gửi qua Telegram
/adduser
/removeuser
/listusers
/setlimit
/resetusage
/addsymbol
/removesymbol
/checknow

Một số lệnh bảo trì vẫn có handler và có thể gõ tay, nhưng không hiện trong menu để tránh rối:
/dashboard, /dashboardall, /historyall, /statsall, /clearhistory.

DATABASE
--------
Railway dùng DB_PATH=/data/bot.db trên volume.
Không commit bot.db, bot_export*.db, *.db-wal hoặc *.db-shm lên GitHub.

EXPORT DATABASE
---------------
Admin gửi /exportdb trong Telegram. Bot tạo snapshot bằng SQLite Backup API, gửi file bot_export.db rồi xóa file tạm.

VERSION
-------
Release hiện tại: 2.2
- 1.1, 1.2...: nâng cấp nhỏ hoặc sửa lỗi.
- 2.0, 3.0...: thay đổi kiến trúc lớn.
Version thực tế bot hiển thị (Telegram, DB) lấy từ biến Railway BOT_VERSION — sửa README này chỉ để tài liệu khớp, không ảnh hưởng bot chạy thật.
