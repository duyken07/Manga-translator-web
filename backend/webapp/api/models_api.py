from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..core.model_manager import SPECS
from ..schemas.dto import ModelIdBody
from .deps import ctx_of

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/status")
def status(request: Request):
    return {"models": ctx_of(request).models.status()}


def _check(mid: str) -> None:
    if mid not in SPECS:
        raise HTTPException(404, f"Model không tồn tại: {mid}")


@router.post("/download", status_code=202)
def download(body: ModelIdBody, request: Request):
    _check(body.id)
    mm = ctx_of(request).models
    mm.download(body.id)
    return {"models": mm.status()}


@router.post("/load", status_code=202)
def load(body: ModelIdBody, request: Request):
    _check(body.id)
    ctx = ctx_of(request)
    mm = ctx.models
    if not mm.is_available(body.id):
        raise HTTPException(409, "Model chưa được tải xuống")

    def run(_jid, _jm):
        mm.load(body.id)

    jid = ctx.jobs.submit("model_load", run, params={"model": body.id})
    return {"job_id": jid}
