"""Vẽ / Fill THẲNG lên ảnh (không tạo lớp), chỉ hiển thị và dùng trong app: FILE GỐC CỦA NGƯỜI DÙNG KHÔNG BAO GIỜ BỊ SỬA.

  * lần chỉnh đầu tiên của một trang, file gốc được chép sang một BẢN LÀM VIỆC trong thư mục dữ liệu của app (cùng tên file);
    Page.image_path trỏ sang bản này nên xem, dịch, redraw, xuất ảnh / PSD đều dùng ảnh đã chỉnh; Page.original_image_path giữ file gốc;
  * mỗi thao tác chụp trước một bản `undo_*` (giữ tối đa MAX_UNDO) -> Hoàn tác / Làm lại bằng file, không tốn RAM;
  * ghi file kiểu nguyên tử (ghi ra .tmp rồi os.replace) nên không bao giờ để lại ảnh hỏng nửa chừng.

Không phụ thuộc giao diện; việc nặng chạy trong QRunnable (xem DirectPaintTask) để không chặn UI.
Ảnh đổi nội dung (kích thước / mtime đổi) nên cache redraw (khoá theo đường dẫn + kích thước + mtime) tự vô hiệu và redraw chạy lại.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from app.qt_compat import QObject, QRunnable, Signal, Slot

log = logging.getLogger("direct_paint")

MAX_UNDO = 15
_LOCK = threading.RLock()


# --------------------------------------------------------------------------- đọc / ghi ảnh
def read_image(path: str) -> np.ndarray:
    """Đọc BGR, hỗ trợ đường dẫn Unicode trên Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise RuntimeError(f"Không đọc được ảnh: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    """Ghi đè ảnh theo ĐÚNG định dạng đuôi file (jpg chất lượng 95), kiểu nguyên tử."""
    p = Path(path)
    ext = p.suffix.lower() or ".png"
    params: list = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 98]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"Không mã hoá được ảnh ({ext})")
    tmp = p.with_name(p.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(tmp, p)


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


# --------------------------------------------------------------------------- bản làm việc + lịch sử hoàn tác bằng file
class DirectPaint:
    """Quản lý BẢN LÀM VIỆC của ảnh khi vẽ / fill trực tiếp: file gốc của người dùng KHÔNG bao giờ bị sửa.
    Bản làm việc: <root>/<hash đường dẫn gốc>/work/<đúng tên file gốc> (trùng tên nên tên khi xuất ảnh / PSD không đổi)."""

    def __init__(self, root) -> None:
        self.root = Path(root)

    @staticmethod
    def _key(path: str) -> str:
        return hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]

    def _dir(self, image_path: str, create: bool = True) -> Path:
        d = self.root / self._key(image_path)
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # -- bản làm việc (gọi trên luồng UI: chỉ chép một file ảnh)
    def work_path(self, original: str) -> Path:
        return self.root / self._key(original) / "work" / Path(original).name

    def ensure_working_copy(self, page) -> str:
        """Trang chưa có bản làm việc -> chép file gốc sang bản làm việc rồi trỏ page.image_path sang đó. Trả về đường dẫn đang dùng."""
        if page.original_image_path and page.image_path != page.original_image_path and Path(page.image_path).is_file():
            return page.image_path
        original = page.original_image_path or page.image_path
        work = self.work_path(original)
        work.parent.mkdir(parents=True, exist_ok=True)
        if not work.is_file():
            shutil.copyfile(original, work)
        page.original_image_path = original
        page.image_path = str(work)
        page.direct_edited = True
        return page.image_path

    def relink(self, page) -> None:
        """Mở lại project: trang từng vẽ / fill trực tiếp (direct_edited) và bản làm việc còn trên máy -> dùng lại bản đó."""
        if not (page.direct_edited and page.image_path) or page.original_image_path:
            return
        work = self.work_path(page.image_path)
        if work.is_file():
            page.original_image_path, page.image_path = page.image_path, str(work)

    @staticmethod
    def _stack(d: Path, kind: str) -> List[Path]:
        return sorted(d.glob(f"{kind}_*"), key=lambda f: f.name)

    @staticmethod
    def _name(kind: str, ext: str) -> str:
        return f"{kind}_{time.time_ns()}{ext}"

    # -- truy vấn (chạy được trên luồng UI: chỉ liệt kê file)
    def can_undo(self, image_path: str) -> bool:
        return bool(image_path) and bool(self._stack(self._dir(image_path, False), "undo"))

    def can_redo(self, image_path: str) -> bool:
        return bool(image_path) and bool(self._stack(self._dir(image_path, False), "redo"))

    # -- thao tác trên BẢN LÀM VIỆC (gọi trong luồng nền, tuần tự)
    def snapshot(self, image_path: str) -> None:
        """Chụp bản làm việc hiện tại TRƯỚC khi sửa: thêm một mốc hoàn tác, xoá lịch sử làm lại."""
        with _LOCK:
            p = Path(image_path)
            d = self._dir(image_path)
            shutil.copyfile(p, d / self._name("undo", p.suffix))
            for old in self._stack(d, "undo")[:-MAX_UNDO]:
                old.unlink(missing_ok=True)
            for f in self._stack(d, "redo"):
                f.unlink(missing_ok=True)

    def undo(self, image_path: str) -> bool:
        with _LOCK:
            p = Path(image_path)
            d = self._dir(image_path)
            st = self._stack(d, "undo")
            if not st:
                return False
            shutil.copyfile(p, d / self._name("redo", p.suffix))
            _copy_atomic(st[-1], p)
            st[-1].unlink(missing_ok=True)
            return True

    def redo(self, image_path: str) -> bool:
        with _LOCK:
            p = Path(image_path)
            d = self._dir(image_path)
            st = self._stack(d, "redo")
            if not st:
                return False
            shutil.copyfile(p, d / self._name("undo", p.suffix))
            _copy_atomic(st[-1], p)
            st[-1].unlink(missing_ok=True)
            return True

    def restore_original(self, work_path: str, original_path: str) -> bool:
        """Bỏ mọi chỉnh sửa trực tiếp: bản làm việc về đúng nội dung file gốc (vẫn hoàn tác được)."""
        with _LOCK:
            if not Path(original_path).is_file():
                return False
            self.snapshot(work_path)
            _copy_atomic(Path(original_path), Path(work_path))
            return True


# --------------------------------------------------------------------------- "nướng" nét vẽ / fill vào ảnh
def bake_strokes(image_path: str, strokes: list) -> None:
    from app.services.strokes import render_strokes
    img = read_image(image_path)
    write_image(image_path, render_strokes(img, strokes))


def bake_fill(image_path: str, strokes: list, engine_name: str, opts) -> str:
    """Vá vùng bôi ngay trên ảnh (cùng bộ máy với Fill dạng lớp: LaMa / patch-match / OpenCV). Trả về ghi chú (nếu có)."""
    from app.services.redraw import engine as eng
    img = read_image(image_path)
    lama, _err = eng.get_lama(opts)
    a = eng.make_args(opts, img.shape[0], lama is not None)
    res = eng._compute_fill(img, strokes, engine_name, a, lama)
    if res is None:
        return "Vùng bôi rỗng, không có gì để vá."
    eng._paste_bgra(img, res["bgra"], res["x"], res["y"])
    write_image(image_path, img)
    return res.get("note", "") or ""


# --------------------------------------------------------------------------- chạy nền
class _Signals(QObject):
    done = Signal(int, bool, str)   # page_index, ok, thông báo (lỗi hoặc ghi chú)


class DirectPaintTask(QRunnable):
    """Chạy `fn()` trong luồng nền (pool 1 luồng -> các thao tác trên cùng ảnh luôn tuần tự đúng thứ tự)."""

    def __init__(self, page_index: int, fn: Callable[[], Optional[str]]) -> None:
        super().__init__()
        self.page_index, self.fn = page_index, fn
        self.signals = _Signals()

    @Slot()
    def run(self) -> None:
        try:
            msg = self.fn() or ""
            self.signals.done.emit(self.page_index, True, msg)
        except Exception as exc:  # noqa: BLE001
            log.exception("Vẽ trực tiếp lỗi")
            self.signals.done.emit(self.page_index, False, str(exc))
