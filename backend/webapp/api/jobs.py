from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from ..adapters.runtime import RuntimeError_
from ..core.security import ws_token_ok
from ..database.models import ProjectRow
from ..database.session import session_scope
from .deps import ctx_of, http, runtime

router = APIRouter(tags=["jobs"])


@router.get("/api/projects/{project_id}/jobs")
def project_jobs(project_id: str, request: Request):
    return ctx_of(request).jobs.list(project_id)


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    j = ctx_of(request).jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Không tìm thấy job")
    return j


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    if not ctx_of(request).jobs.cancel(job_id):
        raise HTTPException(409, "Job không tồn tại hoặc đã kết thúc")
    return ctx_of(request).jobs.get(job_id)


@router.post("/api/jobs/{job_id}/retry-failed", status_code=202)
def retry_failed(job_id: str, request: Request):
    ctx = ctx_of(request)
    j = ctx.jobs.get(job_id)
    if not j or j["kind"] != "translate":
        raise HTTPException(404, "Không có job dịch này")
    failed = [pid for pid, v in j["pages"].items() if v.get("status") == "failed"]
    if not failed:
        raise HTTPException(409, "Không có trang lỗi để thử lại")
    rt = runtime(request, j["project_id"])
    try:
        return {"job_id": rt.run(rt.start_translate, failed, (j.get("params") or {}).get("keep_manual"))}
    except RuntimeError_ as e:
        raise http(e)


async def progress_ws(ws: WebSocket, project_id: str, token: str = Query(""), since: int = Query(0)):
    """WS /ws/projects/{project_id}/progress?token=...&since=<seq>. Gửi job snapshot + sự kiện page_changed theo thời gian thực."""
    if not ws_token_ok(token):
        await ws.close(code=4401)
        return
    with session_scope() as s:
        if s.get(ProjectRow, project_id) is None:
            await ws.close(code=4404)
            return
    ctx = ws.app.state.ctx
    await ws.accept()
    q = ctx.hub.subscribe(project_id, since)
    await ws.send_json({"type": "hello", "jobs": ctx.jobs.list(project_id, 10)})
    try:
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=25)
                await ws.send_json(ev)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        ctx.hub.unsubscribe(project_id, q)
