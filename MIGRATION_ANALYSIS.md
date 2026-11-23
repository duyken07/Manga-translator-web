# MIGRATION_ANALYSIS.md — Manga Translator: Desktop (PySide6) → Web

Phạm vi: Phase 1 (Source Analysis). Mọi số liệu dưới đây lấy từ việc đọc và chạy thử source thật trong `manga_translator_final.zip`
(196 file, ~25.700 dòng Python), không dựa vào README.

## 1. Kết luận nhanh

- **Lõi AI/xử lý ảnh gần như sạch Qt.** 28/38 module không-UI import được khi *không có* PySide6 (đã kiểm chứng bằng import thử).
  `redraw/{engine,stage_a,typeset,lama_manga,scheduler}`, `ocr_engine`, `manga_ocr_pipeline`, `koharu_detector`, `strokes`, `magic_wand`,
  `find_replace`, `custom_glossary`, `gemini_models`, `chat_service`, `characters/*`, `models.py` → **tái sử dụng nguyên văn**.
- **10 module không-UI bị Qt chặn**, nhưng coupling nông và tách được (mục 3). Chỉ cần: (a) thay `app/config.py`, (b) thay các lớp
  `QThreadPool/Signal` bằng job manager, (c) viết lại `AppState` thành service không-Qt.
- **UI (`app/ui/**`, ~9.500 dòng) viết lại hoàn toàn** bằng React; chỉ dùng làm đặc tả hành vi.
- Baseline test của source gốc: **109 pass / 1 fail có sẵn** (`test_chat_scenarios::test_search_grounding_and_fallback`, lỗi do test
  dùng chung dict giữa 2 lần gọi, không liên quan port). Đây là mốc để so sánh sau khi port.

## 2. Những điểm khác/lệch so với đặc tả (cần xác nhận)

| # | Phát hiện trong source | Ảnh hưởng |
|---|---|---|
| 1 | Source có **detector thứ 3: `mayocream/koharu-yolo26s`** (`koharu_detector.py`, 925 dòng; bóng thoại + vùng chữ + mask + SFX), bật bằng `use_koharu`. Đặc tả chỉ nêu YOLOv8 + LaMa + Manga-OCR. | Model Manager phải quản lý 4 model, không phải 3. Bỏ koharu = mất tính năng (vi phạm nguyên tắc "không bỏ tính năng"). |
| 2 | **Zip không chứa `models/` và `fonts/`**. Font mặc định `fonts/000 WildWords2 TB Roman.ttf` và font hỗ trợ dấu tiếng Việt do người dùng tự có. | Typeset cần font ở server. Cần bạn cung cấp font (hoặc chốt bộ font có license phù hợp). |
| 3 | **`TORCH_INFER_LOCK`** (RLock toàn cục trong `threading_utils.py`) bao mọi suy luận torch (YOLO/koharu/OCR/LaMa). Nguyên nhân: OpenMP chồng nhau gây `Fatal Python error: Aborted`. | "Giới hạn inference đồng thời" thực tế = **1 trong mỗi process**. Không dùng ProcessPool nhân bản model; chạy **1 process Uvicorn (workers=1)**. Ghi vào quyết định kiến trúc. |
| 4 | Tên backend thật: `yolo_ocr_gemini` và `yolo_gemini_vision`; `mock` và `gemini_vision` là backend **đã bị gỡ** (`_REMOVED_BACKENDS`), `normalize_backend()` ánh xạ về mặc định. `MockTranslationService` còn trong code nhưng là test double. | Web chỉ cung cấp 2 mode thật; giữ `normalize_backend` để đọc project cũ. Không đưa mock vào production API. |
| 5 | `gemini_model` bị **ép về `DEFAULT_GEMINI_MODEL`** khi load settings ("model cố định"); fallback `FALLBACK_GEMINI_MODEL` nằm trong `gemini_models.py`. | Đặc tả muốn "cấu hình model trong Settings" — mâu thuẫn hành vi Desktop. Cần quyết định (mục 8). |
| 6 | Source có nhiều tính năng **ngoài danh sách đặc tả**: chatbot nhân vật (`chat_service`, `characters/`), phân tích ngữ pháp (`grammar_analysis`), retranslate dialog, `page_detect` (dò vùng chữ cả trang), `direct_paint` (vẽ/fill thẳng lên bản làm việc), `ocr_gemini_rescue`, undo/redo nét vẽ, nhân bản/sao chép khung chữ, `app_guide`. | Thêm vào checklist và API (mục 6); không được bỏ. |
| 7 | `Page.image_path` là **đường dẫn tuyệt đối**; redraw cache khóa theo đường dẫn ảnh (`engine.cache_dir_for`). Trang định danh bằng `page_name` (duy nhất trong project), không có page_id. | DB giữ cả `page_id` (cho API) lẫn `page_name` (cho tương thích JSON). Ảnh phải ở đường dẫn **ổn định** trên server để cache còn hiệu lực qua restart. |
| 8 | Redraw cache mặc định ở `QStandardPaths.AppDataLocation`. | Chuyển sang `storage/projects/{id}/redraw_cache/`. |

## 3. Bản đồ phụ thuộc Qt (10 module bị chặn) và chiến lược adapter

| Module | Qt dùng cho | Phần thuần Python tái dùng | Chiến lược |
|---|---|---|---|
| `config.py` | `QSettings` (lưu settings), `keyring`+QSettings fallback (API key), `QStandardPaths` (thư mục nhân vật) | Toàn bộ hằng số, `AppSettings` dataclass, `normalize_backend` | **Thay bằng bản không-Qt cùng tên/giao diện**: giữ hằng số + `AppSettings`, bỏ `_settings()`; settings lưu SQLite, API key mã hóa (Fernet) với khóa từ env/secret file. `translation_service` chỉ cần `DEFAULT_GEMINI_MODEL` và `BACKEND_*` từ module này → bản thay mở khóa nó. |
| `translation_service.py` | Chỉ gián tiếp qua `app.config` (1 dòng comment nhắc QThreadPool) | **Toàn bộ** (1.234 dòng: 3 service, context, glossary, honorific, retry/rate-limit) | Không sửa; chỉ cần config mới. |
| `queue_manager.py` | `QObject/Signal/QRunnable/QThreadPool` (26 điểm) | Thân `_PageTranslateTask.run`, logic tuần tự có ngữ cảnh/glossary (`enqueue_pages_sequential`), `translate_region`, `analyze_sentence`, `chat_reply` | Giữ logic; **viết `job_manager`** thay lớp Qt. Hai chế độ (song song `concurrency_limit=4` / tuần tự truyền glossary) phải giữ nguyên. |
| `redraw_manager.py` | `QThreadPool`, Signal (progress, page_layout, page_layers…), `QStandardPaths` | `snapshot_dialogues`, `snapshot_fills`, `engine.sync_page`, `RedrawScheduler` (đã thuần Python: dedupe/ưu tiên/generation) | `RedrawScheduler` + `engine.sync_page` dùng nguyên; thay vỏ Qt bằng worker trong job manager, phát sự kiện qua WebSocket. Giữ cơ chế **generation** để bỏ kết quả cũ. |
| `page_export.py` | Chỉ batch runner `PsdBatchTask/QThreadPool` (từ dòng 405) | `PageExportData`, `render_page_image`, `export_page_psd`, `load_text_blocks` | Dùng nguyên các hàm; thay batch runner bằng job. PSD cần `psd-tools` (chưa cài trong sandbox). |
| `page_detect.py` | `QObject/QRunnable/QThreadPool/Signal` | Thân task dò vùng chữ | Bọc lại bằng job manager. |
| `direct_paint.py` | `QRunnable/Signal` | Logic vẽ/fill trực tiếp lên bản làm việc | Bọc lại bằng job; bản làm việc lưu `storage/.../working/`. |
| `state/app_state.py` | `QObject` + ~40 signal; **chứa toàn bộ logic nghiệp vụ chỉnh sửa** (reorder, duplicate, undo stroke, visibility, redraw box freeze, text frame…) | ~60 phương thức mutation trên `Project/Page/Dialogue` | **Viết `ProjectService` không-Qt**, port từng phương thức giữ nguyên ngữ nghĩa; thay signal bằng sự kiện WS. Đây là phần port tốn công nhất của backend. |
| `perf_monitor.py` | `QObject/QTimer` (đo đứng UI) | — | **Bỏ** (không áp dụng cho server; thay bằng log/metrics). |
| `app_guide.py` | Import lười `app.ui.help_dialog` | Nội dung hướng dẫn | Trích nội dung sang dữ liệu tĩnh cho chatbot. |

`koharu_detector.py` đã có fallback không-Qt cho đường dẫn cache; chỉ cần truyền `model_dir`/cache từ Model Manager.

## 4. Luồng dữ liệu (đọc từ code)

**Dịch** — `TranslationService.translate_page(image_path, …context, glossary…) → Page(dialogues, glossary_updates)`
- Mode A `HybridOcrTranslationService`: detector (koharu và/hoặc YOLO `.pt`) → box → `manga_ocr_pipeline` OCR (confidence, `ocr_low_conf=0.60`,
  `ocr_gemini_rescue`) → Gemini lọc lỗi + dịch + romanization.
- Mode B `YoloGeminiVisionTranslationService`: YOLO box → đánh số lên ảnh → Gemini đọc + dịch → ánh xạ về box.
- Fallback model Gemini + retry rate-limit nằm trong `gemini_models.py` / `TranslationRateLimitError`.
- Tuần tự có ngữ cảnh: trang N dùng ngữ cảnh/glossary từ trang N-1 (`apply_translated_page` → `update_glossary`).

**Redraw** — `engine.sync_page(image_path, dialogues[dict], cache_root, opts, force_full, fills, relayout) → RedrawResult`
- Cache theo ảnh: `cache_dir_for(root, image_path)` chứa clean image, `layers.json`, bố cục; chữ ký `_block_sig/_erase_key/_opts_sig`
  quyết định chỉ-vẽ-lại-phần-đổi (**incremental**). Sửa chữ dịch ⇒ chỉ xếp chữ lại, không inpaint lại.
- `redraw_box` (khung neo) tách khỏi `box_location`; `text_frame`/`text_rotation` chỉ ảnh hưởng typeset; `redraw_gen` ép xóa lại.
- LaMa: `get_lama(opts)` cache theo `opts.lama_key()`; tiled, native-resolution (không resize 512), có inference lock.
- Layer cho viewer/PSD: `layers.json` + `compose_layers/compose_from_cache(hide_redraw, hide_text, strokes)`.

**Export** — `PageExportData.from_page(page)` → `render_page_image` (PNG/JPG) hoặc `export_page_psd(..., separate_layers, editable_text)`.
PSD thật: lớp `Anh goc` + từng vùng redraw/fill/nét vẽ + chữ dịch là **Type layer sửa được** (`psd_text`, template `.bin`). Tên lớp ASCII
(psd-tools lỗi với tên có dấu). Giới hạn 30.000px/cạnh. Lớp bị ẩn trong app giữ ẩn trong PSD.

**Lưu trữ hiện tại** — `Project.save_json` gộp theo `page_name`, giữ `glossary`, `grammar_cache`, `work_title`. Các trường `strokes`, `fills`,
`direct_edited` lưu trong page JSON; `redraw_path/layers_path/layout_*`, undo/redo chỉ ở RAM (suy ra lại từ cache).
**Giữ nguyên schema này** làm định dạng import/export; DB lưu thêm metadata.

## 5. Kiến trúc đề xuất (điều chỉnh so với đặc tả)

- `backend/existing_app/app/…`: bản sao **Qt-free** của các module tái dùng, chỉ vá những chỗ ở mục 3 (diff được ghi lại trong `PATCHES.md`).
  Desktop gốc giữ nguyên ngoài repo web.
- **1 process, ThreadPoolExecutor** cho job; không ProcessPool nhân bản model (mục 2 #3). Model Manager giữ singleton + `TORCH_INFER_LOCK`.
  Job loại IO/Gemini có thể song song (`concurrency_limit`), job torch bị tuần tự hóa bởi lock.
- Đồng bộ Desktop → Web cho mutation: `ProjectService` là nguồn sự thật; frontend chỉ gửi *thao tác* (op) thay vì ghi đè cả JSON, để
  không ghi đè nhau và hỗ trợ autosave/undo.
- Hệ tọa độ: mọi box/frame lưu theo **pixel ảnh gốc** (đúng như `Dialogue`); frontend dùng Konva với `scale` thuần hiển thị.
- Auth: token API tĩnh theo user (dependency FastAPI) + cột `owner_id` ở mọi bảng, chưa làm phân quyền chi tiết.
- Model Manager quản lý 4 model: `yolo-bubble` (`ogkalu/comic-speech-bubble-detector-yolov8m`, file `comic-speech-bubble-detector.pt`),
  `lama-manga` (`mayocream/lama-manga`, `lama-manga.safetensors`), `manga-ocr` (`mayocream/manga-ocr` → fallback `kha-white/manga-ocr-base`),
  `koharu-yolo26s` (`mayocream/koharu-yolo26s`: `model.safetensors`, `*.yaml`, `config.json`). Trạng thái: not_downloaded / downloading /
  downloaded / loading / ready / error.

## 6. API bổ sung so với đặc tả (để không mất tính năng)

`POST /api/pages/{id}/detect-regions` (page_detect) · `POST /api/pages/{id}/dialogues:reorder|duplicate|visibility|overrides` ·
`PUT /api/dialogues/{id}/box|frame|rotation` · `POST /api/pages/{id}/strokes` (+undo/redo/clear) · `POST /api/pages/{id}/fills` (+apply engine) ·
`POST /api/pages/{id}/magic-wand` · `POST /api/pages/{id}/direct-paint` · `GET /api/pages/{id}/layers` (+ layer PNG) ·
`POST /api/projects/{id}/find-replace` · `GET/PUT /api/projects/{id}/glossary` (+ custom glossary import) ·
`POST /api/pages/{id}/grammar` · `POST /api/projects/{id}/chat` + `/api/characters` · `POST /api/projects/{id}/retranslate` ·
`GET /api/fonts`.

## 7. Rủi ro

1. **Không thể kiểm chứng AI thật trong môi trường làm việc hiện tại**: sandbox của tôi chặn `huggingface.co`, không có GPU/weights/font,
   và `psd-tools` chưa cài. Tôi có thể viết + test logic, API, DB, job, UI; còn **so sánh pixel với Desktop bằng model thật phải chạy trên máy bạn**
   (tôi sẽ cung cấp script `tools/parity_check.py`). Không được tuyên bố "xong" cho phần AI nếu chưa chạy.
2. RAM: torch + LaMa + Manga-OCR + YOLO + koharu cùng lúc. Cần lazy load + `unload` theo thứ tự LRU; mặc định LaMa cpu.
3. `AppState` ~830 dòng logic dễ lệch ngữ nghĩa khi port → mỗi phương thức cần unit test trên cùng fixture với Desktop.
4. Gemini model names (`gemini-3.5-flash-lite`…) lấy từ source; tôi không xác minh được trong sandbox.
5. Upload ảnh lớn: `engine.imread` đọc cả ảnh vào RAM mỗi lần; giới hạn kích thước + giới hạn job đồng thời để tránh OOM.

## 8. Quyết định cần bạn xác nhận

1. Giữ **koharu-yolo26s** (khuyến nghị: giữ, vì là tính năng hiện có).
2. Settings Gemini model: giữ hành vi Desktop (model cố định + fallback) hay cho chọn model như đặc tả? (khuyến nghị: cho chọn từ danh sách, mặc định = Desktop).
3. Cung cấp **font** (WildWords2 + font có dấu tiếng Việt) hoặc cho phép tôi dùng font mã nguồn mở thay thế mặc định.
4. Triển khai mục tiêu: Linux có GPU hay chỉ CPU? Windows Server? (ảnh hưởng `lama_device`, `ocr_force_cpu`, hướng dẫn deploy).
5. Auth: token tĩnh một người dùng là đủ cho v1?

---
## Cập nhật sau Phase 2 (điều chỉnh kiến trúc so với mục 3 & 5)
Sau khi đo lại bề mặt Qt thực dùng (chỉ `QObject, Signal, Slot, QRunnable, QThreadPool, QStandardPaths, QSettings`), **không viết lại**
`AppState` / `queue_manager` / `redraw_manager` / `page_export` như dự kiến ban đầu. Thay vào đó `existing_app/app/qt_compat.py` giả lập đúng các
API đó (signal xếp hàng về một luồng trạng thái) và chỉ đổi dòng import → giữ 100% logic gốc. `job_manager` bao ngoài để lưu DB, phát WebSocket,
hủy/retry. Chi tiết: `backend/existing_app/PATCHES.md`.
Phát hiện thêm: engine ở "đường tắt" không ghi lại `out.png` → preview/export phải ghép từ lớp (`render_page_image`).
