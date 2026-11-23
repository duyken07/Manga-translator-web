"""Tiện ích chống treo giao diện (xem các chú thích tại nơi dùng)."""
from __future__ import annotations

import os


def effective_threads(configured: int = 0, reserve: int = 2) -> int:
    """Số luồng cho torch. Người dùng đặt (>0) thì tôn trọng. Mặc định torch dùng TẤT CẢ lõi: khi YOLO / koharu / Manga-OCR / LaMa chạy
    (và có lúc hai thứ cùng lúc: dịch trang kế tiếp trong khi redraw trang trước) thì mọi lõi bị chiếm, luồng giao diện không được
    chạy -> Windows báo "Not Responding". Mặc định chừa `reserve` lõi cho giao diện + Python."""
    n = int(configured or 0)
    if n > 0:
        return n
    cpu = os.cpu_count() or 4
    return max(1, cpu - reserve) if cpu > 2 else 1


# --------------------------------------------------------------------------- khoá torch dùng chung
import logging
import threading
import traceback

# MỘT khoá cho MỌI suy luận torch trong app (YOLO / koharu / Manga-OCR / LaMa). Trước đây LaMa (luồng redraw) chạy song song với
# Manga-OCR beam-search (luồng dịch) mà không có khoá chung; hai vùng song song OpenMP của torch chồng nhau + set_num_threads gọi
# lúc luồng khác đang chạy là nguồn gây "Fatal Python error: Aborted" trên Windows.
TORCH_INFER_LOCK = threading.RLock()
_THREADS_LOCK = threading.Lock()
_THREADS_APPLIED = [False]


def apply_torch_threads(configured: int = 0, force: bool = False) -> None:
    """Đặt số luồng torch ĐÚNG MỘT LẦN và chỉ khi đang không có suy luận nào chạy. torch.set_num_threads không an toàn khi gọi
    lúc luồng khác đang ở trong vùng song song (OpenMP) -> có thể abort tiến trình."""
    if _THREADS_APPLIED[0] and not force:
        return
    with _THREADS_LOCK:
        if _THREADS_APPLIED[0] and not force:
            return
        import torch
        with TORCH_INFER_LOCK:          # chờ suy luận đang chạy xong rồi mới đổi
            torch.set_num_threads(effective_threads(configured))
        _THREADS_APPLIED[0] = True


def warn_if_ui_thread(what: str) -> None:
    """Gọi ở mọi chỗ chặn lâu (gọi mạng Gemini...). Nếu đang ở luồng giao diện -> ghi log kèm stack ĐẦY ĐỦ để tìm ra nơi gọi sai."""
    try:
        if threading.current_thread() is threading.main_thread():
            logging.getLogger("perf").warning(
                "%s đang chạy TRÊN LUỒNG GIAO DIỆN (sẽ gây khựng):\n%s", what, "".join(traceback.format_stack(limit=30)))
    except Exception:  # noqa: BLE001
        pass
