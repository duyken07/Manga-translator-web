from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings
from .models import Base

_engine: Optional[Engine] = None
_Session: Optional[sessionmaker] = None


def init_db(url: Optional[str] = None) -> Engine:
    global _engine, _Session
    url = url or get_settings().database_url
    kw = {"connect_args": {"check_same_thread": False, "timeout": 30}} if url.startswith("sqlite") else {}
    _engine = create_engine(url, future=True, **kw)
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi_conn, _rec):            # WAL: đọc không chặn ghi; foreign key bật để cascade hoạt động
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False, future=True)
    return _engine


def dispose_db() -> None:
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = _Session = None


@contextmanager
def session_scope() -> Iterator[Session]:
    if _Session is None:
        init_db()
    s: Session = _Session()   # type: ignore[misc]
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    with session_scope() as s:
        yield s
