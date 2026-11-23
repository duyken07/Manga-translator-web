from __future__ import annotations

from typing import Tuple

from fastapi import HTTPException, Request

from ..adapters.runtime import AppContext, NotFound, ProjectRuntime, RuntimeError_
from ..database.models import PageRow
from ..database.session import session_scope


def ctx_of(request: Request) -> AppContext:
    return request.app.state.ctx


def runtime(request: Request, project_id: str) -> ProjectRuntime:
    try:
        return ctx_of(request).registry.get(project_id)
    except NotFound as e:
        raise HTTPException(404, str(e))


def page_runtime(request: Request, page_id: str) -> Tuple[ProjectRuntime, str]:
    with session_scope() as s:
        row = s.get(PageRow, page_id)
        pid = row.project_id if row else None
    if not pid:
        raise HTTPException(404, "Không tìm thấy trang")
    return runtime(request, pid), page_id


def http(e: RuntimeError_) -> HTTPException:
    return HTTPException(e.status, str(e))
