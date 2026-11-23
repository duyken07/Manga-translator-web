"""Hub sự kiện thời gian thực cho WebSocket. publish() an toàn từ MỌI luồng (luồng trạng thái, worker...)."""
from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional, Set


class EventHub:
    def __init__(self, backlog: int = 300) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subs: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._hist: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=backlog))
        self._seq = 0
        self._lock = threading.Lock()
        self._listeners: List[Callable[[str, dict], None]] = []     # đồng bộ, dùng cho test / log

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def add_listener(self, fn: Callable[[str, dict], None]) -> None:
        self._listeners.append(fn)

    def publish(self, project_id: str, event: dict) -> dict:
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "project_id": project_id, **event}
            self._hist[project_id].append(ev)
            subs = list(self._subs.get(project_id, ())) + list(self._subs.get("*", ()))
        for fn in self._listeners:
            try:
                fn(project_id, ev)
            except Exception:  # noqa: BLE001
                pass
        if self._loop is not None and subs:
            for q in subs:
                try:
                    self._loop.call_soon_threadsafe(q.put_nowait, ev)
                except RuntimeError:        # loop đã đóng
                    pass
        return ev

    def subscribe(self, project_id: str, since: int = 0) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subs[project_id].add(q)
            for ev in self._hist.get(project_id, ()):
                if ev["seq"] > since:
                    q.put_nowait(ev)
        return q

    def unsubscribe(self, project_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs[project_id].discard(q)
