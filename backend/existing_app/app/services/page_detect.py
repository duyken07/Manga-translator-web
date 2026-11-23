"""Dò vùng chữ của trang KHÔNG dịch: dùng cho "Chỉ redraw (không dịch)" ở những trang chưa có câu thoại nào.

Dùng đúng bộ dò như khi dịch (YOLO bóng thoại, koharu, hoặc kết hợp cả hai) nhưng không OCR, không gọi Gemini. Chạy ở luồng nền
(QThreadPool); kết quả về luồng giao diện qua signal. Mỗi vùng sau đó trở thành một câu "chỉ redraw" (xoá chữ gốc theo mask).
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from app.qt_compat import QObject, QRunnable, QThreadPool, Signal, Slot

log = logging.getLogger("page_detect")


def detect_page_regions(image_path: str, yolo_model_path: str = "", yolo_conf: float = 0.35) -> Tuple[List[List[int]], List[List[int]]]:
    """(vùng chữ / bóng thoại, SFX ngoài bóng): các hộp [x, y, w, h]. SFX chỉ có khi koharu bật + tuỳ chọn xoá SFX; còn lại rỗng."""
    boxes = detect_page_boxes(image_path, yolo_model_path, yolo_conf)
    sfx: List[List[int]] = []
    try:
        from app.services import koharu_detector as kd
        cfg = kd.current()
        if cfg.enabled and cfg.erase_sfx:
            sfx = kd.sfx_units(kd.get_layout(image_path, cfg), exclude_boxes=boxes)
    except Exception as exc:  # noqa: BLE001
        log.warning("Không lấy được SFX từ koharu: %s", exc)
    return boxes, sfx


def detect_page_boxes(image_path: str, yolo_model_path: str = "", yolo_conf: float = 0.35) -> List[List[int]]:
    """Các hộp [x, y, w, h] của vùng chữ / bóng thoại theo thứ tự đọc. Rỗng nếu không dò được gì."""
    from app.services import koharu_detector as kd
    from app.services.translation_service import _yolo_boxes_for_koharu_hybrid

    yolo_boxes = _yolo_boxes_for_koharu_hybrid(yolo_model_path, yolo_conf, image_path)   # None nếu koharu tắt / không có YOLO
    got = kd.units_for_page(image_path, yolo_boxes=yolo_boxes)                           # None nếu koharu tắt / lỗi
    if got is not None:
        return [[int(v) for v in u.box] for u in got[1]]
    if yolo_model_path and os.path.isfile(yolo_model_path):
        from app.services.manga_ocr_pipeline import get_bubble_boxes, get_yolo_model, sort_reading_order
        return sort_reading_order(get_bubble_boxes(get_yolo_model(yolo_model_path), image_path, conf=yolo_conf))
    return []


class _Job(QRunnable):
    def __init__(self, owner: "PageDetector", index: int, image_path: str, yolo_model_path: str, yolo_conf: float):
        super().__init__()
        self._o, self._i, self._p, self._m, self._c = owner, index, image_path, yolo_model_path, yolo_conf

    @Slot()
    def run(self) -> None:
        try:
            boxes, sfx = detect_page_regions(self._p, self._m, self._c)
            self._o.detected.emit(self._i, boxes, sfx)
        except Exception as exc:  # noqa: BLE001
            log.exception("Dò vùng chữ lỗi (%s)", self._p)
            self._o.failed.emit(self._i, str(exc))


class PageDetector(QObject):
    """Dò nền từng trang một (một luồng): detected(page_index, boxes, sfx_boxes) / failed(page_index, message)."""

    detected = Signal(int, list, list)   # page_index, hộp vùng chữ, hộp SFX
    failed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)   # model dò dùng chung: tuần tự cho nhẹ máy

    def request(self, index: int, image_path: str, yolo_model_path: str = "", yolo_conf: float = 0.35) -> None:
        self._pool.start(_Job(self, index, image_path, yolo_model_path, yolo_conf))
