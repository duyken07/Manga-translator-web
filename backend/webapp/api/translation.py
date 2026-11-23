from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..adapters import editing
from ..adapters.runtime import RuntimeError_
from ..core import settings_store as ss
from ..schemas.dto import RegionBody, TranslateBody, dialogue_out
from .deps import ctx_of, http, page_runtime, runtime

router = APIRouter(tags=["translation"])


@router.post("/api/projects/{project_id}/translate", status_code=202)
def translate_project(project_id: str, request: Request, body: TranslateBody = TranslateBody()):
    rt = runtime(request, project_id)
    try:
        return {"job_id": rt.run(rt.start_translate, body.page_ids, body.keep_manual)}
    except RuntimeError_ as e:
        raise http(e)


@router.post("/api/pages/{page_id}/translate", status_code=202)
def translate_page(page_id: str, request: Request, body: TranslateBody = TranslateBody()):
    """Dịch (hoặc dịch lại, nếu keep_manual != null) MỘT trang."""
    rt, pid = page_runtime(request, page_id)
    try:
        return {"job_id": rt.run(rt.start_translate, [pid], body.keep_manual)}
    except RuntimeError_ as e:
        raise http(e)


@router.post("/api/pages/{page_id}/translate-region")
async def translate_region(page_id: str, body: RegionBody, request: Request):
    """Dịch MỘT vùng người dùng khoanh [x,y,w,h] (pixel ảnh gốc) -> thoại mới."""
    rt, pid = page_runtime(request, page_id)
    ctx = ctx_of(request)
    x, y, w, h = editing.clean_box(body.box)
    pw, ph = rt.dims.get(pid, (0, 0))
    if pw and ph:                                   # kẹp vào ảnh; vùng nằm hẳn ngoài ảnh -> lỗi
        x2, y2 = min(x + w, pw), min(y + h, ph)
        x, y = max(x, 0), max(y, 0)
        if x2 - x < 2 or y2 - y < 2:
            raise HTTPException(400, "Vùng chọn nằm ngoài ảnh")
        w, h = x2 - x, y2 - y
    st = rt.settings()
    api_key = ss.get_api_key()
    if not api_key:
        raise HTTPException(400, "Chưa có Gemini API key")

    def work():
        for sid in rt._models_needed(st):
            ctx.models.ensure(sid)
        from ..adapters.options import apply_legacy_configuration
        apply_legacy_configuration(ctx.models, st)
        from app.config import DEFAULT_GEMINI_MODEL
        yolo = ctx.models.path("yolo-bubble")
        svc = ctx.service_factory(st.translation_backend, api_key, DEFAULT_GEMINI_MODEL, str(yolo) if yolo else "",
                                  enable_stray_text_removal=st.enable_stray_text_removal)
        img = rt.run(lambda: rt.project.pages[rt.index_of(pid)].image_path)
        return svc.translate_region(img, [x, y, x + w, y + h], st.source_lang, st.target_lang)

    try:
        d = await run_in_threadpool(work)
    except RuntimeError_ as e:
        raise http(e)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Dịch vùng lỗi: {e}")
    if not body.add:
        return dialogue_out(d)

    def add():
        idx = rt.index_of(pid)
        d.box_location = [x, y, w, h] if not d.box_location or len(d.box_location) != 4 else d.box_location
        rt.state.add_dialogue(idx, d)
        return dialogue_out(d)

    return rt.run(add)
