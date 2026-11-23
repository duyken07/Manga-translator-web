# PATCHES.md — khác biệt giữa `existing_app/` và source Desktop gốc

Nguyên tắc: **không đổi thuật toán**. Chỉ đổi những chỗ liên quan Qt. Desktop gốc (`manga_translator_final.zip`) giữ nguyên.

| File | Thay đổi |
|---|---|
| `app/qt_compat.py` | **File mới.** Giả lập `QObject/Signal/Slot/QRunnable/QThreadPool/QSettings/QStandardPaths`. Signal phát từ luồng khác được xếp hàng về một *luồng trạng thái* duy nhất (thay luồng GUI) → AppState không khoá vẫn an toàn như Desktop. |
| `app/config.py`, `app/services/{queue_manager,redraw_manager,page_export,page_detect,direct_paint,koharu_detector}.py`, `app/state/app_state.py` | Chỉ đổi dòng import: `from PySide6.QtCore import …` → `from app.qt_compat import …`. Không đổi dòng logic nào. |
| `app/ui/help_dialog.py` | **Rút gọn.** Chỉ giữ 5 khối dữ liệu hướng dẫn (`GUIDE`, `SETTINGS_ITEMS`, `TOOLS_ITEMS`, `FEATURE_ITEMS`, `EXTRA`) trích bằng AST từ file gốc, để `app_guide` (chatbot hướng dẫn) chạy không cần Qt. |
| `app/perf_monitor.py`, `app/ui/*` (còn lại), `app/assets/icons` | **Không copy** (chỉ phục vụ GUI Desktop). |

Kiểm chứng: `python -m pytest existing_app/tests` → **109 pass / 1 fail**, giống hệt baseline Desktop (fail có sẵn:
`test_chat_scenarios::test_search_grounding_and_fallback`).

## Hành vi cần biết khi dùng engine redraw
- Ở "đường tắt" (chỉ sửa chữ), `engine.sync_page` **không ghi lại `out.png`**; Desktop hiển thị/xuất bằng `compose_from_cache`/`render_page_image`.
  Vì vậy API `redraw-preview` ghép từ lớp (cùng hàm với export) thay vì phục vụ `out.png`.
- Khóa cache redraw = hash(đường dẫn ảnh, kích thước, mtime). Giữ ảnh ở đường dẫn ổn định (`data/projects/<id>/originals/`); di chuyển
  thư mục data sang máy/đường dẫn khác thì cache tự tính lại (không mất dữ liệu).
