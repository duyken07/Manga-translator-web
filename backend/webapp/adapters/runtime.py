"""ProjectRuntime - adapter giữa FastAPI và các thành phần Desktop (AppState, RedrawManager, TranslationQueueManager).

Mỗi project đang mở có một runtime. Toàn bộ thay đổi trạng thái chạy trên "luồng trạng thái" (qt_compat.STATE_LOOP) - tương đương luồng
GUI của Qt - nên AppState (không khoá) an toàn như trên Desktop. Các hàm public gọi từ request phải đi qua `rt.run(fn, ...)`.

Phần "keo dán" bên dưới là bản port của MainWindow (_translate_pages, _request_redraw, _on_dialogues_changed, _connect_redraw_signals...),
giữ nguyên thứ tự và điều kiện; chỗ khác Desktop được ghi chú rõ (tải model tự động, job, sự kiện WebSocket, lưu DB).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.models import Dialogue, Page, Project, natural_sort_key
from app.qt_compat import STATE_LOOP, run_on_state_thread
from app.services import custom_glossary as _cg
from app.services.queue_manager import TranslationQueueManager
from app.services.redraw import engine as redraw_engine
from app.services.redraw_manager import RedrawManager
from app.state.app_state import AppState

from ..core import settings_store as ss
from ..core.events import EventHub
from ..core.job_manager import COMPLETED, FAILED, PROCESSING, JobManager
from ..core.model_manager import ERROR, NOT_DOWNLOADED, ModelManager
from ..database.models import PageRow, ProjectRow
from ..database.session import session_scope
from ..storage.files import ProjectPaths, project_paths
from .options import apply_legacy_configuration

log = logging.getLogger("runtime")

DEFAULT_GEMINI_MODEL = None     # đặt lười từ app.config để tránh import sớm


class RuntimeError_(Exception):
    status = 400


class RuntimeBusy(RuntimeError_):
    status = 409


class MissingApiKey(RuntimeError_):
    status = 400


class NotFound(RuntimeError_):
    status = 404


ServiceFactory = Callable[..., Any]


def default_service_factory(backend: str, api_key: str, model_name: str, yolo_model_path: str, enable_stray_text_removal: bool = False):
    from app.services.translation_service import get_translation_service
    return get_translation_service(backend, api_key, model_name, yolo_model_path, enable_stray_text_removal=enable_stray_text_removal)


class AppContext:
    def __init__(self, server_settings, models: ModelManager, jobs: JobManager, hub: EventHub,
                 service_factory: ServiceFactory = default_service_factory):
        self.server = server_settings
        self.models = models
        self.jobs = jobs
        self.hub = hub
        self.service_factory = service_factory
        self.registry = RuntimeRegistry(self)


class RuntimeRegistry:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._rts: Dict[str, "ProjectRuntime"] = {}
        self._lock = threading.Lock()

    def get(self, project_id: str) -> "ProjectRuntime":
        with self._lock:
            rt = self._rts.get(project_id)
            if rt is None:
                with session_scope() as s:
                    if s.get(ProjectRow, project_id) is None:
                        raise NotFound("Không tìm thấy project")
                rt = ProjectRuntime(project_id, self.ctx)
                run_on_state_thread(rt.load)
                self._rts[project_id] = rt
            return rt

    def drop(self, project_id: str) -> None:
        with self._lock:
            rt = self._rts.pop(project_id, None)
        if rt:
            run_on_state_thread(rt.shutdown)

    def refresh_all(self) -> None:
        for rt in list(self._rts.values()):
            run_on_state_thread(rt.refresh_options, True)

    def shutdown(self) -> None:
        for pid in list(self._rts):
            self.drop(pid)


# ---------------------------------------------------------------------------------------------------------------- overrides <-> lưu trữ
def _font_to_name(ov: dict) -> dict:
    """font_path tuyệt đối -> tên file (DB không chứa đường dẫn server)."""
    if ov and ov.get("font_path"):
        ov = dict(ov)
        ov["font_path"] = Path(str(ov["font_path"])).name
    return ov


def _font_to_abs(ov: dict) -> dict:
    if ov and ov.get("font_path"):
        ov = dict(ov)
        ab = ss.resolve_font(Path(str(ov["font_path"])).name)
        if ab:
            ov["font_path"] = ab
        else:
            ov.pop("font_path")            # font không còn trên server: dùng font mặc định thay vì lỗi
    return ov


class ProjectRuntime:
    def __init__(self, project_id: str, ctx: AppContext):
        self.id = project_id
        self.ctx = ctx
        self.paths: ProjectPaths = project_paths(project_id).ensure()
        self.state = AppState()
        self.redraw = RedrawManager(cache_root=str(self.paths.cache))
        self.queue: Optional[TranslationQueueManager] = None
        self.id_by_name: Dict[str, str] = {}
        self.dims: Dict[str, Tuple[int, int]] = {}
        self._redraw_pending: set = set()
        self._redraw_timer: Optional[threading.Timer] = None
        self._save_timer: Optional[threading.Timer] = None
        self._paint_quiet = False
        self._translate_job: Optional[str] = None
        self._translate_prev: Dict[int, str] = {}
        self._translate_ids: Dict[int, str] = {}
        self._translate_keep_manual: Optional[bool] = None
        self._redraw_job: Optional[str] = None
        self._lama_wait: set = set()
        self._closed = False
        self._connect()

    # ---------------------------------------------------------------------------------------------------------------- tiện ích
    def run(self, fn: Callable, *args, timeout: Optional[float] = 120.0):
        return run_on_state_thread(fn, *args, timeout=timeout)

    def post(self, fn: Callable, *args) -> None:
        STATE_LOOP.post(fn, *args)

    @property
    def project(self) -> Project:
        return self.state.project

    def name_of(self, page_id: str) -> Optional[str]:
        for n, i in self.id_by_name.items():
            if i == page_id:
                return n
        return None

    def index_of(self, page_id: str) -> int:
        name = self.name_of(page_id)
        for i, p in enumerate(self.project.pages):
            if p.page_name == name:
                return i
        raise NotFound("Không tìm thấy trang")

    def page_id_at(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.project.pages):
            return self.id_by_name.get(self.project.pages[index].page_name)
        return None

    def ids_to_indices(self, page_ids: Optional[List[str]]) -> List[int]:
        if not page_ids:
            return list(range(len(self.project.pages)))
        return sorted({self.index_of(pid) for pid in page_ids})

    def settings(self) -> ss.UserSettings:
        with session_scope() as s:
            row = s.get(ProjectRow, self.id)
            ov = dict(row.settings or {}) if row else {}
        return ss.effective_settings(ov)

    def emit(self, what: str, page_index: Optional[int] = None, **extra) -> None:
        pid = self.page_id_at(page_index) if page_index is not None else None
        self.ctx.hub.publish(self.id, {"type": "page_changed", "what": what, "page_id": pid, **extra})

    # ---------------------------------------------------------------------------------------------------------------- nạp / lưu
    def load(self) -> None:
        with session_scope() as s:
            row = s.get(ProjectRow, self.id)
            proj = Project(source_lang=row.source_lang, target_lang=row.target_lang, glossary=dict(row.glossary or {}),
                           grammar_cache=dict(row.grammar_cache or {}), work_title=row.work_title or "")
            for pr in sorted(row.pages, key=lambda r: natural_sort_key(r.page_name)):
                data = dict(pr.data or {})
                for d in data.get("dialogues", []):
                    if d.get("overrides"):
                        d["overrides"] = _font_to_abs(d["overrides"])
                orig = self.paths.originals / pr.page_name
                page = Page.from_dict({"page_name": pr.page_name, **data}, image_path=str(orig))
                page.status = pr.status or page.status
                page.error_message = pr.error_message
                if page.status == "translating":          # server từng tắt giữa chừng: trang chưa dịch xong
                    page.status = "done" if page.dialogues else "pending"
                work = self.paths.working / pr.page_name
                if page.direct_edited and work.is_file():
                    page.original_image_path, page.image_path = str(orig), str(work)
                rm = pr.redraw_meta or {}
                out, layers = self.paths.resolve_rel(rm.get("out", "")), self.paths.resolve_rel(rm.get("layers", ""))
                if out:
                    page.redraw_path, page.redraw_status = str(out), "done"
                if layers:
                    page.layers_path = str(layers)
                page.layout_boxes = {int(k): list(v) for k, v in (rm.get("layout_boxes") or {}).items()}
                page.layout_info = {int(k): v for k, v in (rm.get("layout_info") or {}).items()}
                proj.pages.append(page)
                self.id_by_name[pr.page_name] = pr.id
                self.dims[pr.id] = (pr.width, pr.height)
        self.state.load_project(proj)
        self.refresh_options()

    def schedule_save(self, delay: float = 1.0) -> None:
        if self._closed:
            return
        if self._save_timer:
            self._save_timer.cancel()
        self._save_timer = threading.Timer(delay, lambda: STATE_LOOP.post(self.persist))
        self._save_timer.daemon = True
        self._save_timer.start()

    def persist(self) -> None:
        """Ghi project vào SQLite (schema trang = Page.to_dict của Desktop). Chạy trên luồng trạng thái."""
        p = self.project
        with session_scope() as s:
            row = s.get(ProjectRow, self.id)
            if row is None:
                return
            row.source_lang, row.target_lang, row.work_title = p.source_lang, p.target_lang, p.work_title
            row.glossary, row.grammar_cache = dict(p.glossary), dict(p.grammar_cache)
            by_id = {r.id: r for r in row.pages}
            for page in p.pages:
                pr = by_id.get(self.id_by_name.get(page.page_name, ""))
                if pr is None:
                    continue
                d = page.to_dict()
                d.pop("page_name", None)
                for dd in d.get("dialogues", []):
                    if dd.get("overrides"):
                        dd["overrides"] = _font_to_name(dd["overrides"])
                pr.data, pr.status, pr.error_message = d, page.status, page.error_message
                rel = lambda x: self.paths.rel(x) if x and Path(x).is_file() else ""     # noqa: E731
                pr.redraw_meta = {"status": page.redraw_status, "out": rel(page.redraw_path), "layers": rel(page.layers_path),
                                  "layout_boxes": {str(k): v for k, v in page.layout_boxes.items()},
                                  "layout_info": {str(k): v for k, v in page.layout_info.items()}}
        self.state.mark_saved()

    def shutdown(self) -> None:
        self._closed = True
        for t in (self._save_timer, self._redraw_timer):
            if t:
                t.cancel()
        if self.queue:
            self.queue.cancel()
        self.redraw.reset()
        self.persist()

    # ---------------------------------------------------------------------------------------------------------------- cấu hình
    def refresh_options(self, rerender: bool = False) -> None:
        """Áp cài đặt hiện tại (như _redraw_options + set_options của Desktop). rerender: xếp lại chữ các trang đã có ảnh redraw."""
        st = self.settings()
        try:
            from app.services.redraw import typeset
            fd = self.ctx.server.font_dir
            if fd.is_dir() and ss.list_fonts():
                typeset.set_font_only_dir(str(fd))          # chỉ dùng font trên server (client không chọn được path)
        except Exception:  # noqa: BLE001
            log.warning("Không đặt được thư mục font", exc_info=True)
        self.redraw.set_options(apply_legacy_configuration(self.ctx.models, st))
        if rerender and st.enable_redraw:
            for i, page in enumerate(self.project.pages):
                if page.redraw_path:
                    self.request_redraw(i)

    # ---------------------------------------------------------------------------------------------------------------- nối signal
    def _connect(self) -> None:
        s, r = self.state, self.redraw
        s.dialogues_changed.connect(self._on_dialogues_changed)
        s.dirty_changed.connect(lambda dirty: self.schedule_save() if dirty else None)
        s.page_updated.connect(lambda i: (self.emit("page", i), self.schedule_save()))
        s.dialogues_changed.connect(lambda i: self.emit("dialogues", i))
        s.dialogue_meta_changed.connect(lambda i, d: self.emit("dialogues", i))
        s.dialogues_meta_changed.connect(lambda i, ids: self.emit("dialogues", i))
        s.dialogue_box_changed.connect(lambda i, d: self.emit("dialogues", i, dialogue_id=d))
        s.visibility_changed.connect(lambda i, d: self.emit("dialogues", i, dialogue_id=d))
        s.strokes_changed.connect(lambda i: self.emit("strokes", i))
        s.fills_changed.connect(lambda i: self.emit("fills", i))
        s.redraw_updated.connect(lambda i: (self.emit("redraw", i), self.schedule_save()))
        r.page_started.connect(lambda i: (s.mark_redraw(i, "running"), self._redraw_page_state(i, PROCESSING)))
        r.page_info.connect(s.set_layout_info)
        r.page_layers.connect(s.set_layers_meta)
        r.page_layout.connect(s.set_layout_boxes)
        r.page_redrawn.connect(self._on_redrawn)
        r.page_failed.connect(self._on_redraw_failed)
        r.warning.connect(lambda msg: self.ctx.hub.publish(self.id, {"type": "warning", "message": msg}))
        r.progress.connect(self._on_redraw_progress)
        r.idle.connect(self._on_redraw_idle)

    # ---------------------------------------------------------------------------------------------------------------- redraw
    def _redraw_page_state(self, index: int, status: str, message: str = "") -> None:
        pid = self.page_id_at(index)
        if self._redraw_job and pid:
            self.ctx.jobs.page(self._redraw_job, pid, status, message, count=False)

    def _on_redrawn(self, index: int, path: str) -> None:
        self.state.mark_redraw(index, "done", path)
        self._redraw_page_state(index, COMPLETED)

    def _on_redraw_failed(self, index: int, msg: str) -> None:
        self.state.mark_redraw(index, "error", message=msg)
        self._redraw_page_state(index, FAILED, msg)

    def _on_redraw_progress(self, done: int, total: int) -> None:
        if self._redraw_job:
            self.ctx.jobs.progress(self._redraw_job, done, total)

    def _on_redraw_idle(self) -> None:
        if self._redraw_job:
            self.ctx.jobs.finish(self._redraw_job, COMPLETED)
            self._redraw_job = None

    def _on_dialogues_changed(self, index: int) -> None:
        """Dịch xong / sửa chữ / thêm / xoá thoại -> đồng bộ ảnh redraw (chỉ phần chênh lệch). Debounce 250 ms như Desktop."""
        if not self.settings().enable_redraw or self._paint_quiet or self._closed:
            return
        if not (0 <= index < len(self.project.pages)):
            return
        page = self.project.pages[index]
        if page.redraw_path or page.has_redraw_content():
            self._redraw_pending.add(index)
            if self._redraw_timer:
                self._redraw_timer.cancel()
            self._redraw_timer = threading.Timer(0.25, lambda: STATE_LOOP.post(self._flush_redraw))
            self._redraw_timer.daemon = True
            self._redraw_timer.start()

    def _flush_redraw(self) -> None:
        pending, self._redraw_pending = sorted(self._redraw_pending), set()
        for index in pending:
            self.request_redraw(index)

    def request_redraw(self, index: int, force_full: bool = False, relayout: bool = False) -> bool:
        """Port của MainWindow._request_redraw. Khác Desktop: nếu LaMa chưa tải thì tải trước (nền) rồi mới redraw."""
        pages = self.project.pages
        if not (0 <= index < len(pages)):
            return False
        mm = self.ctx.models
        if self.settings().enable_redraw and mm.state("lama-manga") in (NOT_DOWNLOADED, "downloading"):
            self._lama_wait.add(index)
            self.state.mark_redraw(index, "queued", message="Đang tải model LaMa lần đầu…")
            self._wait_lama()
            return True
        if force_full:
            self.state.reset_redraw_boxes(index)
        self.state.freeze_redraw_boxes(index)
        if self._redraw_job is None:
            self._redraw_job = self.ctx.jobs.create(self.id, "redraw", total=0)
            self.ctx.jobs.start(self._redraw_job)
        if self.redraw.request(index, pages[index], force_full=force_full, front=(index == self.state.current_page_index), relayout=relayout):
            self.state.mark_redraw(index, "queued")
            pid = self.page_id_at(index)
            if pid:
                self.ctx.jobs.page(self._redraw_job, pid, "pending", count=False)
            return True
        return False

    def _wait_lama(self) -> None:
        if getattr(self, "_lama_thread", None) and self._lama_thread.is_alive():
            return

        def work() -> None:
            warn = ""
            try:
                self.ctx.models.download("lama-manga", wait=True)
            except Exception as exc:  # noqa: BLE001
                warn = f"Không tải được LaMa ({exc}); dùng OpenCV cho mọi vùng."
            self.post(self._lama_ready, warn)

        self._lama_thread = threading.Thread(target=work, daemon=True, name="lama-wait")
        self._lama_thread.start()

    def _lama_ready(self, warn: str) -> None:
        if warn:
            self.ctx.hub.publish(self.id, {"type": "warning", "message": warn})
        self.refresh_options()
        waiting, self._lama_wait = sorted(self._lama_wait), set()
        for i in waiting:
            if self.ctx.models.state("lama-manga") == NOT_DOWNLOADED or self.ctx.models.state("lama-manga") == ERROR:
                self.ctx.models.rescan()
            self.request_redraw(i) if self.ctx.models.state("lama-manga") not in (NOT_DOWNLOADED, "downloading") else self._redraw_direct(i)

    def _redraw_direct(self, index: int) -> None:       # LaMa không có: vẫn redraw bằng OpenCV (lama_path rỗng) như Desktop khi thiếu model
        self.state.freeze_redraw_boxes(index)
        if self.redraw.request(index, self.project.pages[index]):
            self.state.mark_redraw(index, "queued")

    def cancel_redraw(self) -> None:
        self.redraw.reset()
        for i, p in enumerate(self.project.pages):
            if p.redraw_status in ("queued", "running"):
                self.state.mark_redraw(i, "none" if not p.redraw_path else "done")
        if self._redraw_job:
            self.ctx.jobs.cancel(self._redraw_job)
            self._redraw_job = None

    # ---------------------------------------------------------------------------------------------------------------- dịch
    def _models_needed(self, st: ss.UserSettings) -> List[str]:
        need: List[str] = []
        if not (st.use_koharu and not st.koharu_use_yolo):
            need.append("yolo-bubble")
        if st.use_koharu:
            need.append("koharu-yolo26s")
        if st.translation_backend == "yolo_ocr_gemini":
            need.append("manga-ocr")
        return need

    def _check_idle(self) -> None:
        if self.queue is not None and self.queue.is_busy():
            raise RuntimeBusy("Đang có trang được dịch. Hãy đợi xong hoặc hủy job hiện tại.")
        if self._translate_job:
            raise RuntimeBusy("Đang có job dịch chạy.")

    def start_translate(self, page_ids: Optional[List[str]], keep_manual: Optional[bool] = None, no_redraw: bool = False) -> str:
        """Dịch các trang (None = tất cả). keep_manual != None -> kiểu 'Dịch lại trang' (apply_retranslated_page). Trả job_id.
        Chạy trên luồng trạng thái; việc tải/nạp model + khởi tạo service làm ở luồng phụ để không chặn."""
        self._check_idle()
        indices = self.ids_to_indices(page_ids)
        if not indices:
            raise RuntimeError_("Project chưa có trang")
        st = self.settings()
        api_key = ss.get_api_key()
        if not api_key:
            raise MissingApiKey("Chưa có Gemini API key. Hãy nhập trong Settings.")
        ids = [self.page_id_at(i) for i in indices]
        job = self.ctx.jobs.create(self.id, "translate", total=len(indices), params={"keep_manual": keep_manual}, page_ids=ids)
        self._translate_job = job
        self._translate_keep_manual = keep_manual
        self._translate_prev = {i: self.project.pages[i].status for i in indices}
        self._translate_ids = {i: pid for i, pid in zip(indices, ids)}
        self._translate_no_redraw = no_redraw
        self.ctx.jobs.start(job, "Đang chuẩn bị model…")
        self.ctx.jobs.register_cancel(job, lambda: self.run(self._cancel_translate))
        names = [self.project.pages[i].page_name for i in indices]
        threading.Thread(target=self._prepare_translate, args=(job, names, st, api_key), daemon=True, name="translate-prep").start()
        return job

    def _prepare_translate(self, job: str, names: List[str], st: ss.UserSettings, api_key: str) -> None:
        try:
            mm = self.ctx.models
            for sid in self._models_needed(st):
                self.ctx.jobs.progress(job, 0, message=f"Đang chuẩn bị model: {sid}")
                mm.ensure(sid)
            apply_legacy_configuration(mm, st)
            from app.config import DEFAULT_GEMINI_MODEL
            yolo = mm.path("yolo-bubble")
            service = self.ctx.service_factory(st.translation_backend, api_key, DEFAULT_GEMINI_MODEL, str(yolo) if yolo else "",
                                               enable_stray_text_removal=st.enable_stray_text_removal)
        except Exception as exc:  # noqa: BLE001
            log.exception("Chuẩn bị dịch lỗi")
            self.post(self._translate_aborted, job, f"Không chuẩn bị được dịch: {exc}")
            return
        self.post(self._begin_translate, job, names, st, service)

    def _translate_aborted(self, job: str, msg: str) -> None:
        self._translate_job = None
        self.ctx.jobs.finish(job, FAILED, msg)

    def _begin_translate(self, job: str, names: List[str], st: ss.UserSettings, service) -> None:
        if self._translate_job != job:                 # đã bị hủy trong lúc chuẩn bị
            return
        by_name = {p.page_name: i for i, p in enumerate(self.project.pages)}
        indices = sorted(by_name[n] for n in names if n in by_name)
        self._translate_ids = {i: self.id_by_name[self.project.pages[i].page_name] for i in indices}
        pages = [self.project.pages[i] for i in indices]          # pages và indices PHẢI tương ứng 1-1
        for i in indices:
            self.state.mark_page_translating(i)
        q = TranslationQueueManager(service, concurrency_limit=st.concurrency_limit)
        self.queue = q
        q.page_started.connect(lambda i: self.ctx.jobs.page(job, self._translate_ids.get(i, ""), PROCESSING))
        q.page_done.connect(self._on_page_translated)
        q.page_failed.connect(self._on_page_failed)
        q.all_finished.connect(self._on_translate_finished)
        self.ctx.jobs.progress(job, 0, len(indices), "Đang dịch…")
        self.ctx.jobs.start(job, "Đang dịch…")
        proj = self.project
        proj.source_lang, proj.target_lang = st.source_lang, st.target_lang
        base = self._translation_context(st)
        if self._translate_keep_manual is None and st.sequential_context_mode:
            q.enqueue_pages_sequential(pages, st.source_lang, st.target_lang, indices, initial_glossary=base["glossary"],
                                       keep_honorifics=base["keep_honorifics"])
        else:
            q.enqueue_pages(pages, st.source_lang, st.target_lang, indices, base_context=base)

    def _translation_context(self, st: ss.UserSettings) -> dict:
        """Port của MainWindow._build_translation_context: glossary tùy chỉnh ưu tiên hơn glossary tự tích lũy của project."""
        custom = {k: v for k, v in (st.custom_glossary or {}).items() if k and v}
        return {"glossary": {**self.project.glossary, **custom}, "recent_dialogues": [], "keep_honorifics": st.keep_honorifics}

    def _on_page_translated(self, index: int, page: Page) -> None:
        if self._translate_keep_manual is None:
            self.state.apply_translated_page(index, page)
        else:
            self.state.apply_retranslated_page(index, page, keep_manual=self._translate_keep_manual,
                                               no_redraw=getattr(self, "_translate_no_redraw", False))
        self.ctx.jobs.page(self._translate_job or "", self._translate_ids.get(index, ""), COMPLETED)

    def _on_page_failed(self, index: int, message: str) -> None:
        if self._translate_keep_manual is not None:
            self.state.restore_page_status(index, self._translate_prev.get(index, "pending"))   # dịch lại lỗi: trang vẫn dùng được
        self.state.mark_page_error(index, message)
        self.ctx.jobs.page(self._translate_job or "", self._translate_ids.get(index, ""), FAILED, message)

    def _on_translate_finished(self) -> None:
        job, self._translate_job = self._translate_job, None
        if job:
            j = self.ctx.jobs.get(job) or {}
            total, failed = j.get("total", 0), j.get("failed", 0)
            status = FAILED if total and failed >= total else COMPLETED
            self.ctx.jobs.finish(job, status, f"Dịch xong {total - failed}/{total} trang" + (f", {failed} trang lỗi" if failed else ""))
        self.persist()

    def _cancel_translate(self) -> None:
        if self.queue:
            self.queue.cancel()
        for i, prev in self._translate_prev.items():
            if 0 <= i < len(self.project.pages) and self.project.pages[i].status == "translating":
                self.state.restore_page_status(i, prev)
        self._translate_job = None


def _unused():  # pragma: no cover - giữ import có chủ đích cho mở rộng
    return natural_sort_key, _cg
