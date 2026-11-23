"""Quản lý job nền: trạng thái pending/processing/completed/failed/cancelled, tiến trình tổng + từng trang, hủy, lưu DB.

Việc nặng KHÔNG chạy trong request HTTP. Dịch / redraw dùng hàng đợi sẵn có của Desktop (TranslationQueueManager / RedrawManager, đã có
giới hạn đồng thời, hủy, retry rate-limit); JobManager theo dõi + phát tiến trình qua EventHub + lưu DB, và chạy các job "callable"
chung (export, tải model...) bằng ThreadPoolExecutor. Job còn dở khi server khởi động lại được đánh dấu failed ('server restarted')."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..database.models import JobRow
from ..database.session import session_scope
from .events import EventHub

log = logging.getLogger("jobs")
PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED = "pending", "processing", "completed", "failed", "cancelled"
FINAL = (COMPLETED, FAILED, CANCELLED)


def _snap(r: JobRow) -> dict:
    return {"id": r.id, "project_id": r.project_id, "kind": r.kind, "status": r.status, "total": r.total, "done": r.done,
            "failed": r.failed, "message": r.message, "pages": dict(r.pages or {}), "params": dict(r.params or {}),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None}


class JobManager:
    def __init__(self, hub: EventHub, workers: int = 2) -> None:
        self.hub = hub
        self._cancel_fns: Dict[str, Callable[[], None]] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")

    # ---------------------------------------------------------------- CRUD
    def create(self, project_id: Optional[str], kind: str, total: int = 0, params: Optional[dict] = None,
               page_ids: Optional[List[str]] = None) -> str:
        pages = {pid: {"status": PENDING, "message": ""} for pid in (page_ids or [])}
        with session_scope() as s:
            row = JobRow(project_id=project_id, kind=kind, total=total, params=params or {}, pages=pages)
            s.add(row)
            s.flush()
            jid, snap = row.id, _snap(row)
        self._emit(snap)
        return jid

    def get(self, job_id: str) -> Optional[dict]:
        with session_scope() as s:
            r = s.get(JobRow, job_id)
            return _snap(r) if r else None

    def list(self, project_id: Optional[str] = None, limit: int = 50) -> List[dict]:
        with session_scope() as s:
            q = s.query(JobRow)
            if project_id:
                q = q.filter(JobRow.project_id == project_id)
            return [_snap(r) for r in q.order_by(JobRow.created_at.desc()).limit(limit).all()]

    def _update(self, job_id: str, fn: Callable[[JobRow], None]) -> Optional[dict]:
        with self._lock, session_scope() as s:
            r = s.get(JobRow, job_id)
            if r is None or r.status in FINAL:
                return None
            fn(r)
            s.flush()
            snap = _snap(r)
        self._emit(snap)
        return snap

    def _emit(self, snap: dict) -> None:
        self.hub.publish(snap["project_id"] or "*", {"type": "job", "job": snap})

    # ---------------------------------------------------------------- vòng đời
    def start(self, job_id: str, message: str = "") -> None:
        def f(r: JobRow) -> None:
            r.status = PROCESSING
            r.message = message or r.message
        self._update(job_id, f)

    def set_total(self, job_id: str, total: int) -> None:
        self._update(job_id, lambda r: setattr(r, "total", total))

    def page(self, job_id: str, page_id: str, status: str, message: str = "", count: bool = True) -> None:
        """Trạng thái từng trang; count=True: tự cập nhật done/failed khi trang kết thúc (False: tiến trình do nơi khác báo)."""
        def f(r: JobRow) -> None:
            pages = dict(r.pages or {})
            prev = (pages.get(page_id) or {}).get("status")
            pages[page_id] = {"status": status, "message": message}
            r.pages = pages                       # gán lại để SQLAlchemy nhận biết thay đổi JSON
            if count and status in (COMPLETED, FAILED) and prev not in (COMPLETED, FAILED):
                r.done += 1
                if status == FAILED:
                    r.failed += 1
        self._update(job_id, f)

    def progress(self, job_id: str, done: int, total: Optional[int] = None, message: str = "") -> None:
        def f(r: JobRow) -> None:
            r.done = done
            if total is not None:
                r.total = total
            if message:
                r.message = message
        self._update(job_id, f)

    def finish(self, job_id: str, status: str = COMPLETED, message: str = "") -> None:
        def f(r: JobRow) -> None:
            r.status = status
            if message:
                r.message = message
            r.finished_at = datetime.now(timezone.utc)
        self._update(job_id, f)
        self._cancel_fns.pop(job_id, None)

    # ---------------------------------------------------------------- hủy
    def register_cancel(self, job_id: str, fn: Callable[[], None]) -> None:
        self._cancel_fns[job_id] = fn

    def cancel(self, job_id: str) -> bool:
        j = self.get(job_id)
        if not j or j["status"] in FINAL:
            return False
        fn = self._cancel_fns.get(job_id)
        if fn:
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("Hủy job %s lỗi", job_id)

        def f(r: JobRow) -> None:
            r.status = CANCELLED
            r.finished_at = datetime.now(timezone.utc)
            pages = {k: (v if v.get("status") in (COMPLETED, FAILED) else {**v, "status": CANCELLED}) for k, v in (r.pages or {}).items()}
            r.pages = pages
        self._update(job_id, f)
        return True

    # ---------------------------------------------------------------- job callable chung
    def submit(self, kind: str, fn: Callable[[str, "JobManager"], Any], project_id: Optional[str] = None,
               params: Optional[dict] = None, total: int = 1) -> str:
        jid = self.create(project_id, kind, total=total, params=params)

        def run() -> None:
            if (self.get(jid) or {}).get("status") != PENDING:
                return                                  # đã bị hủy trước khi chạy
            self.start(jid)
            try:
                res = fn(jid, self)
                self.finish(jid, COMPLETED, str(res) if isinstance(res, str) else "")
            except Exception as exc:  # noqa: BLE001
                log.exception("Job %s (%s) lỗi", jid, kind)
                self.finish(jid, FAILED, str(exc)[:500])

        self._pool.submit(run)
        return jid

    # ---------------------------------------------------------------- khởi động
    def recover_orphans(self) -> int:
        n = 0
        with session_scope() as s:
            for r in s.query(JobRow).filter(JobRow.status.in_((PENDING, PROCESSING))).all():
                r.status, r.message, r.finished_at = FAILED, "Server đã khởi động lại khi job đang chạy", datetime.now(timezone.utc)
                n += 1
        return n

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
