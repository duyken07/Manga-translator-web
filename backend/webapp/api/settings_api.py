from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from ..core import settings_store as ss
from ..schemas.dto import ApiKeyBody
from .deps import ctx_of

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _view() -> dict:
    from app.config import DEFAULT_GEMINI_MODEL, FALLBACK_GEMINI_MODEL
    return {**ss.get_user_settings().model_dump(),
            "gemini_model": DEFAULT_GEMINI_MODEL, "gemini_fallback_model": FALLBACK_GEMINI_MODEL,   # chỉ đọc (giữ như Desktop)
            "has_api_key": ss.has_api_key(), "api_key_hint": ss.masked_api_key(),                   # KHÔNG trả key
            "fonts": ss.list_fonts()}


@router.get("")
def get_settings():
    return _view()


@router.put("")
def put_settings(patch: Dict[str, Any], request: Request):
    forbidden = {"gemini_model", "has_api_key", "fonts", "api_key"} & set(patch)
    if forbidden:
        raise HTTPException(422, f"Không được đặt: {', '.join(sorted(forbidden))}")
    if patch.get("redraw_font") and not ss.resolve_font(patch["redraw_font"]):
        raise HTTPException(422, "Font không có trên server")
    try:
        ss.update_user_settings(patch)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False, include_input=False))
    ctx_of(request).registry.refresh_all()          # áp ngay (font/màu/koharu/OCR) và xếp lại chữ các trang đã redraw
    return _view()


@router.post("/api-key", status_code=204)
def set_key(body: ApiKeyBody):
    ss.set_api_key(body.api_key)


@router.delete("/api-key", status_code=204)
def del_key():
    ss.delete_api_key()
