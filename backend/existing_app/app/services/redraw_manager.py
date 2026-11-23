"""
RedrawManager: chạy redraw (xoá chữ gốc + chèn bản dịch) nền cho từng trang, nối vào Qt.

Dùng chung 1 hàng đợi cho 3 trường hợp:
  * dịch hàng loạt  : mỗi trang dịch xong -> request() -> redraw tự động (không chặn luồng dịch);
  * thêm khung thoại: chỉ xoá chữ gốc ở vị trí mới rồi xếp lại chữ (engine.sync_page làm phần chênh lệch);
  * sửa / xoá thoại : xếp lại chữ, hoặc trả điểm ảnh gốc ở khung bị xoá.
Mọi thao tác nặng chạy trên 1 thread riêng; kết quả về main thread qua signal.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.qt_compat import QObject, QRunnable, QStandardPaths, QThreadPool, Signal, Slot

from app.models import Page
from app.services.redraw import engine
from app.services.redraw.scheduler import RedrawJob, RedrawScheduler

log = logging.getLogger("redraw_manager")


def default_cache_root() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    root = Path(base) if base else Path.home() / ".manga_translator"
    return str(root / "redraw_cache")


def snapshot_fills(page: Page) -> list[dict]:
    return [dict(id=f["id"], strokes=[dict(pts=[list(p) for p in s["pts"]], width=s["width"]) for s in f["strokes"]],
                 engine=f.get("engine", "auto"), hidden=bool(f.get("hidden")))
            for f in page.fills]


def snapshot_dialogues(page: Page) -> list[dict]:
    out = []
    for d in page.dialogues:
        if d.no_redraw and not d.translated_text.strip():
            continue  # khung 'chỉ chèn chữ' mà chưa có bản dịch: không có gì để vẽ
        overrides = dict(d.overrides)
        if d.text_frame and len(d.text_frame) == 4:
            x, y, w, h = d.text_frame
            overrides["region"] = [x, y, x + w, y + h]  # khung chữ do người dùng kéo: chữ xếp vừa và căn giữa trong đó
        out.append(dict(
            dialogue_id=d.dialogue_id,
            box_location=list(d.redraw_box or d.box_location),  # khung neo: kéo thả khung không đổi ảnh redraw
            original_text=d.original_text,
            translated_text=d.translated_text,
            overrides=overrides,
            hide_text=d.hide_text,
            hide_redraw=d.hide_redraw,
            redraw_only=d.redraw_only and not d.no_redraw,
            no_erase=d.no_redraw,  # không xoá chữ gốc, vẫn chèn chữ dịch
            text_rotation=d.text_rotation,
            redraw_gen=d.redraw_gen,
        ))
    return out


class _Signals(QObject):
    finished = Signal(int, int, object)  # page_index, generation, RedrawResult
    failed = Signal(int, int, str)  # page_index, generation, message


class _RedrawTask(QRunnable):
    def __init__(self, page_index: int, generation: int, job: RedrawJob, cache_root: str, opts: engine.RedrawOptions):
        super().__init__()
        self.page_index, self.generation, self.job = page_index, generation, job
        self.cache_root, self.opts = cache_root, opts
        self.signals = _Signals()

    @Slot()
    def run(self) -> None:
        try:
            res = engine.sync_page(
                self.job.image_path, self.job.dialogues, self.cache_root, self.opts, force_full=self.job.force_full,
                fills=self.job.fills, relayout=self.job.relayout,
            )
            self.signals.finished.emit(self.page_index, self.generation, res)
        except Exception as exc:  # noqa: BLE001 - không để 1 lỗi làm sập worker pool
            log.exception("Redraw trang idx=%s lỗi", self.page_index)
            self.signals.failed.emit(self.page_index, self.generation, str(exc))


class RedrawManager(QObject):
    page_started = Signal(int)
    page_info = Signal(int, object)  # page_index, {dialogue_id: {size, font, line_spacing, bold, color}}
    page_layers = Signal(int, str)  # page_index, đường dẫn layers.json (các lớp riêng từng câu)
    page_layout = Signal(int, object)  # page_index, {dialogue_id: [x, y, w, h]} khối chữ dịch (phát trước page_redrawn)
    page_redrawn = Signal(int, str)  # page_index, đường dẫn ảnh redraw
    page_failed = Signal(int, str)
    warning = Signal(str)  # cảnh báo (không lặp lại cùng 1 nội dung)
    progress = Signal(int, int)  # done, total
    idle = Signal()  # hàng đợi đã rỗng

    def __init__(self, cache_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.cache_root = cache_root or default_cache_root()
        self.options = engine.RedrawOptions()
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)
        # Redraw / Fill chạy code native nặng (OpenCV, torch). Luồng của QThreadPool mặc định chỉ có ~1 MB stack trên
        # Windows, dễ tràn stack và làm app tắt đột ngột -> cấp 32 MB.
        self._pool.setStackSize(32 * 1024 * 1024)
        self._sched = RedrawScheduler()
        self._tasks: dict[int, _RedrawTask] = {}  # giữ tham chiếu tới khi xong (QRunnable không có parent)
        self._generation = 0
        self._warned: set[str] = set()

    # -- cấu hình --------------------------------------------------------------
    def set_options(self, options: engine.RedrawOptions) -> None:
        self.options = options
        self._warned.clear()

    # -- yêu cầu --------------------------------------------------------------
    def request(self, page_index: int, page: Page, force_full: bool = False, front: bool = False,
                relayout: bool = False) -> bool:
        """Yêu cầu redraw 1 trang theo trạng thái thoại HIỆN TẠI. False nếu trang không có ảnh gốc."""
        if not page.image_path or not Path(page.image_path).is_file():
            return False
        job = RedrawJob(page.image_path, snapshot_dialogues(page), force_full, relayout, snapshot_fills(page))
        self._sched.submit(page_index, job, front=front)
        self._emit_progress()
        self._pump()
        return True

    def is_pending(self, page_index: int) -> bool:
        return self._sched.is_pending(page_index)

    def is_idle(self) -> bool:
        return self._sched.is_idle()

    def reset(self) -> None:
        """Bỏ hết việc đang chờ (VD: mở project mới). Việc đang chạy dở sẽ bị bỏ qua kết quả."""
        self._generation += 1
        self._sched.clear()
        self._pool.clear()

    def shutdown(self, timeout_ms: int = 15000) -> None:
        self.reset()
        self._pool.waitForDone(timeout_ms)

    # -- nội bộ ----------------------------------------------------------------
    def _pump(self) -> None:
        nxt = self._sched.take_next()
        if nxt is None:
            return
        page_index, job = nxt
        task = _RedrawTask(page_index, self._generation, job, self.cache_root, self.options)
        task.signals.finished.connect(self._on_finished)
        task.signals.failed.connect(self._on_failed)
        self._tasks[page_index] = task
        log.info("Redraw bắt đầu: trang idx=%s (thoại=%d, fill=%d, làm lại từ đầu=%s)",
                 page_index, len(job.dialogues), len(job.fills), job.force_full)
        self.page_started.emit(page_index)
        self._pool.start(task)

    def _emit_progress(self) -> None:
        done, total = self._sched.progress()
        self.progress.emit(done, total)

    def _complete(self, page_index: int) -> None:
        self._tasks.pop(page_index, None)
        all_done = self._sched.finish(page_index)
        done, total = self._sched.progress()
        self.progress.emit(done if not all_done else max(done, 1), total if not all_done else max(done, 1))
        if all_done:
            self.idle.emit()
        else:
            self._pump()

    def _on_finished(self, page_index: int, generation: int, res) -> None:
        if generation != self._generation:
            self._tasks.pop(page_index, None)
            return
        log.info("Redraw xong: trang idx=%s trong %.2fs (xoá %d, khôi phục %d)", page_index, res.seconds, res.erased, res.restored)
        for w in res.warnings:
            if w not in self._warned:
                self._warned.add(w)
                self.warning.emit(w)
        self.page_info.emit(page_index, res.info)
        self.page_layers.emit(page_index, res.layers_path)
        self.page_layout.emit(page_index, res.layout)
        self.page_redrawn.emit(page_index, res.out_path)
        self._complete(page_index)

    def _on_failed(self, page_index: int, generation: int, message: str) -> None:
        if generation != self._generation:
            self._tasks.pop(page_index, None)
            return
        self.page_failed.emit(page_index, message)
        self._complete(page_index)
