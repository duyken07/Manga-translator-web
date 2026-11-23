"""qt_compat.py - thay thế tối thiểu cho PySide6.QtCore, đủ để chạy AppState / TranslationQueueManager / RedrawManager /
page_export / page_detect / direct_paint của bản Desktop MÀ KHÔNG cần Qt.

Ngữ nghĩa giữ giống Qt:
  * Signal.emit từ luồng khác -> slot được XẾP HÀNG và chạy tuần tự trên "luồng trạng thái" duy nhất (tương đương luồng GUI).
    Nhờ vậy toàn bộ logic AppState (không khoá) vẫn an toàn như trên Desktop. Emit từ chính luồng trạng thái -> gọi thẳng.
  * QThreadPool: setMaxThreadCount / setStackSize / start / waitForDone / clear / activeThreadCount; luồng nền dùng stack lớn
    (code native OpenCV/torch cần > 1 MB mặc định).
  * QSettings / QStandardPaths: lưu JSON / đường dẫn thư mục dữ liệu (đổi bằng biến môi trường MANGA_APP_DATA).

Mọi thao tác của web API lên trạng thái dự án phải đi qua `run_on_state_thread(fn)` để giữ đúng mô hình một-luồng-ghi.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("qt_compat")

# --------------------------------------------------------------------------- luồng trạng thái (thay luồng GUI)


class _StateLoop:
    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Condition()

    def ensure_started(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="state-loop", daemon=True)
            self._thread.start()

    def on_loop_thread(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def post(self, fn: Callable, *args) -> None:
        self.ensure_started()
        with self._idle:
            self._pending += 1
        self._q.put((fn, args))

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, args = item
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 - một slot lỗi không được làm chết vòng lặp
                log.exception("Lỗi trong slot chạy trên luồng trạng thái")
            finally:
                with self._idle:
                    self._pending -= 1
                    if self._pending <= 0:
                        self._idle.notify_all()

    def call(self, fn: Callable, *args, timeout: Optional[float] = 60.0):
        """Chạy fn trên luồng trạng thái và chờ kết quả (gọi từ luồng khác, vd. request FastAPI)."""
        if self.on_loop_thread():
            return fn(*args)
        box: dict = {}
        done = threading.Event()

        def _w():
            try:
                box["r"] = fn(*args)
            except BaseException as e:  # noqa: BLE001
                box["e"] = e
            finally:
                done.set()

        self.post(_w)
        if not done.wait(timeout):
            raise TimeoutError("state thread không phản hồi")
        if "e" in box:
            raise box["e"]
        return box.get("r")

    def drain(self, timeout: float = 10.0) -> bool:
        """Chờ hàng đợi sự kiện rỗng (dùng trong test)."""
        end = time.monotonic() + timeout
        with self._idle:
            while self._pending > 0:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._idle.wait(left)
        return True


STATE_LOOP = _StateLoop()


def run_on_state_thread(fn: Callable, *args, timeout: Optional[float] = 60.0):
    return STATE_LOOP.call(fn, *args, timeout=timeout)


# --------------------------------------------------------------------------- Signal / Slot / QObject


class _BoundSignal:
    __slots__ = ("_slots", "_lock")

    def __init__(self) -> None:
        self._slots: list = []
        self._lock = threading.Lock()

    def connect(self, slot: Callable) -> None:
        with self._lock:
            self._slots.append(slot)

    def disconnect(self, slot: Optional[Callable] = None) -> None:
        with self._lock:
            if slot is None:
                self._slots.clear()
            else:
                self._slots = [s for s in self._slots if s is not slot and s != slot]

    def emit(self, *args) -> None:
        with self._lock:
            slots = list(self._slots)
        for s in slots:
            if STATE_LOOP.on_loop_thread():
                try:
                    s(*args)
                except Exception:  # noqa: BLE001
                    log.exception("Lỗi trong slot (gọi trực tiếp)")
            else:
                STATE_LOOP.post(s, *args)


class Signal:
    def __init__(self, *types: Any) -> None:
        self._name = ""

    def __set_name__(self, owner, name) -> None:
        self._name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        d = obj.__dict__
        key = "__sig_" + self._name
        b = d.get(key)
        if b is None:
            b = d[key] = _BoundSignal()
        return b


def Slot(*_a, **_k):  # noqa: N802 - giữ tên như Qt
    def deco(fn):
        return fn

    return deco


class QObject:
    def __init__(self, parent: Optional["QObject"] = None) -> None:
        self._parent = parent

    def parent(self):
        return self._parent


# --------------------------------------------------------------------------- QRunnable / QThreadPool

_STACK_LOCK = threading.Lock()


class QRunnable:
    def run(self) -> None:  # pragma: no cover - ghi đè ở lớp con
        raise NotImplementedError

    def setAutoDelete(self, _on: bool) -> None:  # noqa: N802
        pass


class QThreadPool:
    def __init__(self, parent: Any = None) -> None:
        self._max = max(1, (os.cpu_count() or 2))
        self._stack = 0
        self._q: deque = deque()
        self._cv = threading.Condition()
        self._workers: list = []
        self._active = 0

    def setMaxThreadCount(self, n: int) -> None:  # noqa: N802
        with self._cv:
            self._max = max(1, int(n))
            self._cv.notify_all()

    def setStackSize(self, n: int) -> None:  # noqa: N802
        self._stack = int(n)

    def maxThreadCount(self) -> int:  # noqa: N802
        return self._max

    def activeThreadCount(self) -> int:  # noqa: N802
        with self._cv:
            return self._active

    def clear(self) -> None:
        with self._cv:
            self._q.clear()
            self._cv.notify_all()

    def start(self, runnable: QRunnable) -> None:
        with self._cv:
            self._q.append(runnable)
            alive = [w for w in self._workers if w.is_alive()]
            self._workers = alive
            if len(alive) < self._max:
                self._spawn()
            self._cv.notify()

    def _spawn(self) -> None:
        t = threading.Thread(target=self._work, name="qpool-worker", daemon=True)
        with _STACK_LOCK:
            old = threading.stack_size()
            try:
                if self._stack:
                    threading.stack_size(max(self._stack, 32 * 1024))
                t.start()
            finally:
                threading.stack_size(old)
        self._workers.append(t)

    def _work(self) -> None:
        me = threading.current_thread()
        while True:
            with self._cv:
                while True:
                    if self._q and self._active < self._max:
                        task = self._q.popleft()
                        self._active += 1
                        break
                    if not self._q:
                        # không còn việc: luồng thoát (Qt cũng thu hồi luồng rảnh)
                        if me in self._workers:
                            self._workers.remove(me)
                        self._cv.notify_all()
                        return
                    self._cv.wait()
            try:
                task.run()
            except Exception:  # noqa: BLE001
                log.exception("Tác vụ nền lỗi chưa bắt")
            finally:
                with self._cv:
                    self._active -= 1
                    self._cv.notify_all()

    def waitForDone(self, msecs: int = -1) -> bool:  # noqa: N802
        end = None if msecs is None or msecs < 0 else time.monotonic() + msecs / 1000.0
        with self._cv:
            while self._q or self._active:
                if end is None:
                    self._cv.wait(0.1)
                else:
                    left = end - time.monotonic()
                    if left <= 0:
                        return False
                    self._cv.wait(min(left, 0.1))
        return True


# --------------------------------------------------------------------------- QStandardPaths / QSettings


class QStandardPaths:
    AppDataLocation = 1

    @staticmethod
    def writableLocation(_loc: int = 1) -> str:  # noqa: N802
        return os.environ.get("MANGA_APP_DATA") or str(Path.home() / ".manga_translator")


class QSettings:
    """Kho khoá-giá trị JSON. Đường dẫn file: MANGA_SETTINGS_FILE; không đặt -> chỉ trong bộ nhớ (web dùng SQLite riêng)."""

    _mem: dict = {}
    _lock = threading.Lock()

    def __init__(self, *_a, **_k) -> None:
        self._file = os.environ.get("MANGA_SETTINGS_FILE", "")
        if self._file and Path(self._file).is_file():
            try:
                with QSettings._lock:
                    QSettings._mem.update(json.loads(Path(self._file).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    def value(self, key: str, default: Any = None, type: Any = None):  # noqa: A002
        v = QSettings._mem.get(key, default)
        if type is bool and isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        if type is not None and v is not None and not isinstance(v, type):
            try:
                return type(v)
            except (TypeError, ValueError):
                return default
        return v

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802
        with QSettings._lock:
            QSettings._mem[key] = value
            if self._file:
                try:
                    Path(self._file).write_text(json.dumps(QSettings._mem, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    log.warning("Không ghi được settings: %s", self._file)

    def remove(self, key: str) -> None:
        with QSettings._lock:
            QSettings._mem.pop(key, None)

    def contains(self, key: str) -> bool:
        return key in QSettings._mem
