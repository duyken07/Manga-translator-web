# MIGRATION_CHECKLIST.md

Ký hiệu: ☐ chưa làm · ◐ đang làm · ☑ xong **và có test chạy qua**. Không đánh ☑ nếu chỉ có UI/mock.
Cột "Nguồn" là module Desktop dùng làm chuẩn hành vi. "Loại": **R** = tái dùng nguyên, **A** = adapter/vá nhỏ, **W** = viết lại.

## Phase 1 — Source Analysis
- ☑ Giải nén, đọc module lõi, bản đồ phụ thuộc, kiểm chứng import không-Qt
- ☑ Baseline test Desktop (109 pass / 1 fail có sẵn)
- ☑ MIGRATION_ANALYSIS.md, MIGRATION_CHECKLIST.md
- ☑ Chốt 5 quyết định (giữ koharu · Gemini cố định+fallback · bạn cấp font · Windows Server · token tĩnh)

## Phase 2 — Backend Foundation
| | Hạng mục | Nguồn | Loại |
|---|---|---|---|
| ☑ | Copy module Qt-free vào `backend/existing_app`, ghi `PATCHES.md` | app/** (trừ ui) | R |
| ☑ | `config.py` không-Qt: dùng nguyên file gốc + `qt_compat.QSettings` (chỉ đổi import) | config.py | A |
| ☑ | Settings lưu SQLite; API key mã hóa, không trả về client, không log | config.py | W |
| ☑ | DB SQLAlchemy: project, page, dialogue(JSON), job, export, setting | models.py | W |
| ◐ | Lưu/nạp theo schema `Page.to_dict` đã round-trip qua restart (test); import/export file JSON ở Phase 6 — Import/Export JSON tương thích `Project.to_json_dict/load_json` (round-trip test) | models.py | R |
| ☑ | Storage: upload an toàn (validate magic bytes, giới hạn size, chống path traversal), tên ổn định | file_service.py | A |
| ☑ | Thay `ProjectService` bằng **dùng lại nguyên AppState** qua `qt_compat` + `ProjectRuntime`/`editing.py` (có test API) | app_state.py | A |
| ☑ | `model_manager.py`: 4 model, trạng thái, tiến trình tải, lazy load, lock, `MODEL_DIR` | threading_utils.py, koharu_detector.py | W |
| ☑ | `job_manager.py`: queue, concurrency, cancel, retry, persistence, WS progress | queue_manager.py, redraw_manager.py | W |
| ◐ | API: projects, pages, dialogues, settings, models, jobs, WS, translate, redraw ✔ — còn: strokes/fills/magic-wand/glossary/find-replace/chat/grammar/detect-regions (Phase 5) | — | W |
| ☑ | Auth token + CORS | — | W |

## Phase 3 — Frontend Foundation
| | Hạng mục | Nguồn |
|---|---|---|
| ☐ | Vite + React + TS + Tailwind + shadcn/ui, Zustand, TanStack Query | — |
| ☐ | Dashboard: upload nhiều ảnh/thư mục, drag-drop, natural sort, xóa, ngôn ngữ, mode, project | dashboard_view.py |
| ☐ | Workspace 3 cột, panel co giãn/thu gọn, dark/light | workspace_view.py, main_window.py |
| ☐ | Page list: thumbnail, trạng thái dịch/redraw/chỉnh sửa, tiến trình, thêm/xóa | page_list_panel.py |
| ☐ | Image viewer (Konva): zoom/pan/fit/100%, overlay box, chọn/kéo/resize, tọa độ ảnh gốc | image_viewer.py |
| ☐ | Dialogue editor: sửa gốc/dịch/romanization, thêm/xóa, kéo sắp xếp, ẩn/hiện, đồng bộ highlight | dialogue_table.py |
| ☐ | Phím tắt (Ctrl+Z/Y/F/H/D/E/G/T/[/], F3, Delete, Shift+Enter…), context menu, toast | main_window.py |
| ☐ | WebSocket tiến trình | queue_manager.py |

## Phase 4 — AI Integration (cần chạy trên máy có model thật)
| | Hạng mục | Nguồn | Loại |
|---|---|---|---|
| ☐ | YOLO bubble detect (`comic-speech-bubble-detector.pt`) | manga_ocr_pipeline.py | R |
| ☐ | koharu-yolo26s detect (bóng thoại + vùng chữ + SFX) | koharu_detector.py | R |
| ☐ | Manga-OCR + fallback `kha-white/manga-ocr-base` + confidence | ocr_engine.py, manga_ocr_pipeline.py | R |
| ☐ | Mode A: YOLO + OCR + Gemini (rescue bằng ảnh crop) | translation_service.py | R |
| ☐ | Mode B: YOLO + Gemini Vision | translation_service.py | R |
| ☐ | Dịch tuần tự có ngữ cảnh/glossary + chế độ song song | queue_manager.py | A |
| ☐ | Honorifics, romaji, reading order, fallback model, retry rate-limit | translation_service.py, gemini_models.py | R |
| ☐ | Dịch vùng crop (`translate-region`) | translate_region | A |
| ☐ | Retranslate (giữ chỉnh sửa tay) | apply_retranslated_page | W |
| ☐ | LaMa tiled, native-res, cache instance, lock | lama_manga.py, engine.get_lama | R |
| ☐ | OpenCV redraw + typeset + cache + incremental | engine.py, stage_a.py, typeset.py | R |
| ☐ | `tools/parity_check.py`: so ảnh Desktop vs Web cùng input/model/config | — | W |

## Phase 5 — Advanced Editor
| | Hạng mục | Nguồn |
|---|---|---|
| ☐ | Thoại thủ công, crop region → thêm thoại | manual_dialogue_dialog.py |
| ☐ | Per-dialogue: redraw lại vùng, redraw-only, no-redraw (chỉ chèn chữ), ẩn chữ, ẩn redraw | workspace_view.py |
| ☐ | Khung neo `redraw_box` vs `box_location`; đóng băng/đặt lại | app_state.py |
| ☐ | Khung chữ kéo/đổi cỡ/xoay, sửa chữ trực tiếp (double-click) | image_viewer.py |
| ☐ | Typeset: font, cỡ, màu, đậm, giãn dòng, căn lề, auto-size, override theo câu và theo lô | text_format_bar.py, typeset.py |
| ☐ | Bút/bút dạ/dạ quang, tẩy, undo/redo/clear, ẩn nét | pen_toolbar.py, strokes.py |
| ☐ | Fill content-aware (auto/lama/texture/opencv), bàn chải/tẩy, áp dụng/hủy | fill_toolbar.py, engine._apply_fills |
| ☐ | Magic wand (Shift/Alt/Ctrl, tolerance, xác nhận) | wand_toolbar.py, magic_wand.py |
| ☐ | Vẽ/Fill trực tiếp lên bản làm việc (`paint_direct`) | direct_paint.py |
| ☐ | Dò vùng chữ cả trang (`page_detect`) | page_detect.py |
| ☐ | Find & Replace (Ctrl+F/H, F3) | find_replace.py |
| ☐ | Custom glossary | custom_glossary.py |
| ☐ | Phân tích ngữ pháp + cache | grammar_analysis.py |
| ☐ | Chatbot nhân vật + scenario | chat_service.py, characters/ |
| ☐ | Preview: gốc / clean / bản dịch; xem lớp từng câu | engine.compose_from_cache |

## Phase 6 — Export
| | Hạng mục | Nguồn | Loại |
|---|---|---|---|
| ☐ | PNG / JPG (một hoặc nhiều trang) | page_export.render_page_image | R |
| ☐ | JSON (schema hiện tại) / TXT (`flatten_text`, "Page N") | models.py | R |
| ☐ | **PSD nhiều lớp thật**: Anh goc, redraw, fill, nét vẽ, Type layer chữ dịch | page_export.export_page_psd, psd_text.py | R |
| ☐ | Tùy chọn PSD: nén rle/zip/raw, chữ sửa được, tách lớp Fill/nét | main_window.py | R |
| ☐ | ZIP batch, job nền + tải về, dọn file hết hạn | PsdBatchTask | A |
| ☐ | Kiểm tra PSD bằng `psd-tools`: đếm lớp, loại lớp, tên ASCII | — | W |

## Phase 7 — Testing & Deployment
| | Hạng mục |
|---|---|
| ☐ | Chạy lại toàn bộ test Desktop trên module đã copy (mục tiêu: 109 pass, 1 fail có sẵn) |
| ☐ | Unit test ProjectService (từng mutation), model_manager (mock HF), job_manager (cancel/retry/giới hạn) |
| ☐ | API test (pytest + httpx), WS test |
| ☐ | UI test (Vitest + Playwright cho toạ độ box khi zoom/pan) |
| ☐ | Test persistence: restart server → project + cache còn nguyên, redraw không chạy lại |
| ☐ | Test "không tải model lại mỗi request" |
| ☐ | Kiểm tra backend chạy khi **không cài PySide6** |
| ☐ | Performance test (RAM đỉnh, thời gian/trang CPU) |
| ☐ | Deployment: Linux (systemd/Docker), Windows Server, reverse proxy, README, `.env.example` |

## Tiêu chí hoàn thành (từ đặc tả mục 15) — theo dõi
Mỗi mục chỉ đánh ☑ khi có bằng chứng chạy được (test hoặc log chạy thật), riêng các mục dùng **model thật** cần xác nhận từ máy có model:
☐ YOLO thật · ☐ Manga-OCR thật · ☐ Gemini dịch · ☐ LaMa thật · ☐ so sánh với Desktop cùng ảnh/model/config