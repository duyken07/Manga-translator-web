# MIGRATION_CHECKLIST.md

Ký hiệu: `☐` chưa làm · `◐` triển khai một phần / cần cấu hình bổ sung · `☑` xong và đã kiểm chứng trong môi trường hiện tại.

> **Lưu ý môi trường AI Studio (Node.js 22):** Thư mục `/backend` (Python FastAPI / PyTorch) đã bị xóa trong bước chuẩn hóa runtime Node.js ở lượt đầu. Ứng dụng hiện chạy hoàn chỉnh trên Node.js (`src/server.ts`) + React/Konva (`src/client/`) và hỗ trợ **2 chế độ**:
> 1. **Standalone Node.js + Gemini (`FASTAPI_BACKEND_URL` chưa đặt):** Đầy đủ CRUD Project/Page/Dialogue, upload ảnh/ZIP, Magic Bytes validation, Find & Replace, Glossary, Gemini Vision OCR/dịch thuật/phân tích ngữ pháp/chat qua `@google/genai`, WebSocket live progress, và xuất PNG/JPG/JSON/TXT/PSD nhiều lớp (`ag-psd`)/ZIP.
> 2. **Cloud Run FastAPI Proxy (`FASTAPI_BACKEND_URL` đã cấu hình):** Node.js proxy toàn bộ HTTP `/api/*` và WebSocket `/ws/projects/:id/progress` sang dịch vụ Python FastAPI độc lập cho các tác vụ PyTorch/OpenCV (YOLO, Koharu, Manga-OCR, LaMa inpainting, tải model Hugging Face).

## Phase 1 — Source Analysis & Architecture
- ☑ Phân tích module gốc, API contracts, schema `Project.to_json_dict` / `Page.to_dict`, và tài liệu tham chiếu giao diện
- ☑ Thiết kế kiến trúc 2 chế độ (Node.js Standalone + Cloud Run FastAPI HTTP/WS Proxy) — xem `deployment/cloud-run.md`
- ☑ Cấu hình biến môi trường (`.env.example`) và `metadata.json` (`MAJOR_CAPABILITY_SERVER_SIDE_GEMINI_API`)

## Phase 2 — Node.js Backend & Proxy Layer (`src/`)
- ☑ `src/server.ts`: Express 5 + Vite middleware (dev) / `dist/client` static (prod) trên `0.0.0.0:3000`
- ☑ Proxy HTTP `/api/*` & WebSocket `/ws/projects/:id/progress` sang `FASTAPI_BACKEND_URL` khi được cấu hình
- ☑ Projects CRUD + Find & Replace + Export JSON (`Project.to_json_dict`) / TXT (`flatten_text`) (đã test HTTP)
- ☑ Pages upload (PNG/JPG/WEBP/BMP + giải nén `.zip`), natural sort, kiểm tra magic bytes, giới hạn kích thước/pixel, chống path traversal (đã test HTTP)
- ☑ Dialogues CRUD (`POST`, `PATCH`, `DELETE`, `PUT /order`), strokes & fills persistence (đã test HTTP)
- ☑ Auth Bearer token (`MT_API_TOKEN` so sánh `timingSafeEqual`) & CORS cấu hình qua `MT_CORS_ORIGINS`
- ☑ `EventHub` & `jobManager`: WebSocket tiến trình real-time, reconnect replay (`since`), cancel/retry job (đã test WS)
- ◐ Lưu trữ hiện ở RAM (`Map`) khi chạy Node standalone; lưu SQLite/persistent disk nằm ở dịch vụ FastAPI ngoài (`FASTAPI_BACKEND_URL`)

## Phase 3 — Frontend Foundation (`src/client/`)
- ☑ React 19 + TypeScript + Vite + Tailwind CSS v4 + shadcn/ui + Zustand + TanStack Query
- ☑ Navigation shell: Desktop 3-zone top bar + Mobile bottom navigation (`Projects`, `Workspace`, `Models`, `Settings`) + Dark/Light/System theme
- ☑ Projects Dashboard: tạo/đổi tên/xóa project, kéo thả upload ảnh/ZIP, cấu hình ngôn ngữ & mode dịch, danh sách trang
- ☑ Workspace 3 cột (desktop) & bottom sheet (mobile): thanh trang, Konva canvas trung tâm, inspector 3 tab (`Dialogues`, `Typeset`, `Tools & Export`)
- ☑ Konva Image Viewer (`KonvaStage.tsx`): zoom/pan/fit/100%, vẽ hộp mới, crop dịch vùng, kéo/resize/xoay hộp thoại bằng `Transformer` theo tọa độ ảnh gốc
- ☑ Dialogue Editor: sửa gốc/dịch, romanization, sắp xếp thứ tự, ẩn chữ/ẩn redraw/no-redraw, đồng bộ chọn khung trên canvas
- ☑ Phím tắt (`Ctrl+Z`/`Y` Undo/Redo, `Ctrl+F` Tools, `Delete` xóa thoại, `F` Fit, `1` 100%) + `sonner` toast
- ☑ WebSocket kết nối `/ws/projects/:id/progress` cập nhật tiến trình job & invalidate cache tự động

## Phase 4 — AI Integration
- ☑ Server-side Gemini (`@google/genai`, `gemini-4-hydrogen`) cho dịch trang, dịch vùng (`translate-region`), dò vùng thoại (`detect-regions`), phân tích ngữ pháp (`grammar`), và chat nhân vật (`chat`)
- ☑ Chế độ dịch tuần tự kèm ngữ cảnh trang trước + Glossary + Honorifics
- ◐ Các model PyTorch/Hugging Face cục bộ (`comic-speech-bubble-detector-yolov8m`, `koharu-yolo26s`, `manga-ocr-base`, `lama-manga.safetensors`) và OpenCV/LaMa redraw yêu cầu kết nối `FASTAPI_BACKEND_URL` trên Cloud Run (trả `501` rõ ràng ở chế độ Node standalone)

## Phase 5 — Advanced Editor
- ☑ Thêm thoại thủ công & crop vùng dịch (`add_box`, `translate_region`)
- ☑ Per-dialogue flags (`hide_text`, `hide_redraw`, `no_redraw`), reset `text_frame` & `text_rotation`
- ☑ Typeset inspector: font, cỡ chữ, màu, đậm, giãn dòng, căn lề, áp dụng hàng loạt cho cả trang
- ☑ Công cụ vẽ & chỉnh sửa trên Konva: Pen, Highlighter, Eraser, Fill Brush, Magic Wand flood-box sampler, Undo/Redo stack, Clear
- ☑ Find & Replace toàn project, Custom Glossary, Grammar Breakdown modal, Translation Assistant Chat (`Assistant` & `Sengoku Yuno`)

## Phase 6 — Export
- ☑ Export PNG / JPG độ phân giải gốc qua offscreen Canvas (áp dụng redraw, fills, strokes, text formatting)
- ☑ Export JSON & TXT tương thích cấu trúc gốc
- ☑ Export **PSD nhiều lớp** (`ag-psd` `writePsd`, header `8BPS` đã kiểm chứng): `Anh goc`, `Redraw`, `Fills`, `Strokes`, `Text_<id>`
- ☑ Batch export ZIP (PNG / JPG / PSD) cho toàn bộ project qua `JSZip`

## Phase 7 — Verification & Deployment
- ☑ `vite build && tsc -p tsconfig.server.json` (`compile_applet`) pass không lỗi
- ☑ `tsc --noEmit` (`lint_applet`) pass không lỗi
- ☑ Smoke test HTTP + WebSocket trực tiếp trên cổng `3000` pass (`GET /`, `/api/health`, `/api/runtime`, CRUD project/page/dialogue, upload PNG + reject file sai định dạng, find-replace, JSON/TXT export, WS `hello` + `page_changed`)
- ☑ Tài liệu triển khai Cloud Run FastAPI (`deployment/cloud-run.md`)
- ☐ Chạy Python pytest baseline (`backend/` đã bị xóa trong runtime Node.js — cần chạy từ repo Python gốc khi triển khai Cloud Run)