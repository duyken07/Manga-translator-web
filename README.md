# Manga Translator — Web

Port của ứng dụng Desktop (PySide6) sang Web (FastAPI + React). **Trạng thái: Phase 1–2 xong (backend foundation); frontend chưa làm.**
Xem `MIGRATION_ANALYSIS.md` (phân tích), `MIGRATION_CHECKLIST.md` (tiến độ từng tính năng), `backend/existing_app/PATCHES.md`.

## Backend
```bash
cd backend && pip install -r requirements.txt && python run.py        # http://127.0.0.1:8000/docs
python -m pytest tests -q                                             # 26 test web
python -m pytest existing_app/tests -q                                # 109 pass / 1 fail có sẵn = baseline Desktop
```
- `existing_app/app`: module Desktop tái sử dụng (không Qt) + `qt_compat.py`.
- `webapp/`: FastAPI. `core/` (config, security, model_manager, job_manager, events, settings_store), `database/`, `storage/`,
  `adapters/runtime.py` (AppState + RedrawManager + TranslationQueueManager chạy trên luồng trạng thái), `adapters/editing.py`, `api/`.
- Xác thực: `Authorization: Bearer <token>`; WebSocket: `/ws/projects/{id}/progress?token=...`.

## Ghi chú API so với đặc tả
- Thoại: `/api/pages/{page_id}/dialogues[/{dialogue_id}]` (dialogue_id chỉ duy nhất trong một trang, nên cần page_id).
- `GET /api/pages/{id}/redraw-preview?kind=translated|clean|original` ghép từ lớp (đúng như Desktop hiển thị/xuất).
- Export (PNG/JPG/JSON/TXT/PSD/ZIP): Phase 6, chưa có.
