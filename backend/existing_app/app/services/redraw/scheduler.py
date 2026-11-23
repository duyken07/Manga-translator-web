"""
scheduler.py - hàng đợi redraw theo trang, KHÔNG phụ thuộc Qt (để test được).

  * mỗi lúc chỉ chạy 1 việc (LaMa/OpenCV đều ăn CPU, stage_a lại dùng biến toàn cục);
  * "mới nhất thắng": yêu cầu mới cho 1 trang đang chờ sẽ thay yêu cầu cũ (người dùng sửa liên tục
    1 câu thì chỉ redraw bản cuối); nếu trang đó đang chạy dở thì chạy lại 1 lần ngay sau khi xong;
  * force_full và relayout là "dính": nếu việc cũ yêu cầu làm lại từ đầu / xếp lại cả trang thì việc thay thế cũng vậy;
  * front=True đưa trang lên đầu hàng đợi (trang người dùng đang xem).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RedrawJob:
    image_path: str
    dialogues: list          # snapshot dict của các câu thoại (an toàn khi chạy ở thread khác)
    force_full: bool = False
    relayout: bool = False   # xếp lại cả trang + tự chỉnh khung khỏi đè (redraw hàng loạt); False = chỉ xếp lại khung vừa sửa
    fills: list = field(default_factory=list)   # vùng Content-Aware Fill của trang


class RedrawScheduler:
    def __init__(self) -> None:
        self._queue: "OrderedDict[int, RedrawJob]" = OrderedDict()
        self._rerun: dict[int, RedrawJob] = {}
        self._running: Optional[int] = None
        self._total = 0
        self._done = 0

    # -- đưa việc vào ----------------------------------------------------------
    def submit(self, page_index: int, job: RedrawJob, front: bool = False) -> None:
        if self._running == page_index:
            prev = self._rerun.get(page_index)
            if prev is None:
                self._total += 1
            job.force_full = job.force_full or bool(prev and prev.force_full)
            job.relayout = job.relayout or bool(prev and prev.relayout)
            self._rerun[page_index] = job
            return
        prev = self._queue.get(page_index)
        if prev is None:
            self._total += 1
        else:
            job.force_full = job.force_full or prev.force_full
            job.relayout = job.relayout or prev.relayout
        self._queue[page_index] = job
        if front:
            self._queue.move_to_end(page_index, last=False)

    # -- lấy việc / báo xong ---------------------------------------------------------
    def take_next(self) -> Optional[tuple[int, RedrawJob]]:
        if self._running is not None or not self._queue:
            return None
        page_index, job = self._queue.popitem(last=False)
        self._running = page_index
        return page_index, job

    def finish(self, page_index: int) -> bool:
        """Báo việc `page_index` đã xong. Trả về True nếu toàn bộ hàng đợi đã rỗng (rồi đặt lại bộ đếm)."""
        if self._running == page_index:
            self._running = None
        self._done += 1
        rerun = self._rerun.pop(page_index, None)
        if rerun is not None:
            self._queue[page_index] = rerun
            self._queue.move_to_end(page_index, last=False)
        if self.is_idle():
            self._total = self._done = 0
            return True
        return False

    # -- trạng thái ---------------------------------------------------------------
    def is_idle(self) -> bool:
        return self._running is None and not self._queue

    def is_pending(self, page_index: int) -> bool:
        return self._running == page_index or page_index in self._queue

    def progress(self) -> tuple[int, int]:
        return self._done, self._total

    def clear(self) -> None:
        self._queue.clear()
        self._rerun.clear()
        self._running = None
        self._total = self._done = 0
