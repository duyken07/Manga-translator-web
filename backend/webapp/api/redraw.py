from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

from ..adapters.runtime import RuntimeError_
from ..core.security import inside
from ..schemas.dto import RedrawBody, RedrawDialogueBody
from .deps import http, page_runtime

router = APIRouter(prefix="/api/pages/{page_id}", tags=["redraw"])


@router.post("/redraw", status_code=202)
def redraw(page_id: str, request: Request, body: RedrawBody = RedrawBody()):
    """Xếp hàng redraw cho trang. Engine tự chỉ làm phần chênh lệch (incremental) nếu không force_full."""
    rt, pid = page_runtime(request, page_id)
    ok = rt.run(lambda: rt.request_redraw(rt.index_of(pid), force_full=body.force_full, relayout=body.relayout))
    if not ok:
        raise HTTPException(409, "Không thể redraw trang này (thiếu ảnh)")
    return {"queued": True}


@router.post("/redraw-dialogue", status_code=202)
def redraw_dialogue(page_id: str, body: RedrawDialogueBody, request: Request):
    """Redraw lại vùng của MỘT câu theo khung hiện tại (bỏ khung neo cũ, xoá lại chữ gốc)."""
    rt, pid = page_runtime(request, page_id)

    def f():
        return rt.state.redraw_dialogue_region(rt.index_of(pid), body.dialogue_id)

    if not rt.run(f):
        raise HTTPException(404, "Không tìm thấy thoại")
    return {"queued": True}


@router.delete("/redraw", status_code=202)
def cancel(page_id: str, request: Request):
    rt, _ = page_runtime(request, page_id)
    rt.run(rt.cancel_redraw)
    return {"cancelled": True}


def _paths(rt, pid):
    def f():
        p = rt.project.pages[rt.index_of(pid)]
        return p.redraw_path, p.layers_path
    return rt.run(f)


@router.get("/redraw-preview")
async def preview(page_id: str, request: Request, kind: str = "translated", hide_text: bool = False, hide_redraw: bool = False):
    """kind: translated | clean | original.
    translated = GHÉP TỪ CÁC LỚP (đúng hàm Desktop dùng để hiển thị / xuất): engine ở "đường tắt" không ghi lại out.png khi chỉ
    sửa chữ, nên phục vụ thẳng out.png sẽ hiển thị chữ cũ. hide_text / hide_redraw: công tắc ẩn toàn trang (như Desktop)."""
    rt, pid = page_runtime(request, page_id)
    if kind == "original":
        name = rt.run(lambda: rt.project.pages[rt.index_of(pid)].page_name)
        return FileResponse(rt.paths.originals / name)
    if kind == "clean":
        out, _ = _paths(rt, pid)
        f = Path(out).parent / "clean.png" if out else None
        if not f or not f.is_file():
            raise HTTPException(404, "Chưa có ảnh clean")
        return FileResponse(f, headers={"Cache-Control": "private, max-age=0, must-revalidate"})
    if kind != "translated":
        raise HTTPException(422, "kind phải là translated | clean | original")

    from app.services.page_export import PageExportData, render_page_image
    data = rt.run(lambda: PageExportData.from_page(rt.project.pages[rt.index_of(pid)]))
    if hide_text:
        for d in data.dialogues:
            d["hide_text"] = True
    if hide_redraw:
        for d in data.dialogues:
            d["hide_redraw"] = True
    lp = Path(data.layers_path) if data.layers_path else None
    sig = json.dumps([lp.stat().st_mtime_ns if lp and lp.is_file() else 0, data.layers_path and lp.parent.name, data.dialogues,
                      data.visible_strokes, [f.get("hidden") for f in data.fills], Path(data.image_path).stat().st_mtime_ns], default=str)
    etag = '"' + hashlib.md5(sig.encode()).hexdigest() + '"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    import cv2
    img = await run_in_threadpool(render_page_image, data)
    if img is None:
        raise HTTPException(404, "Không đọc được ảnh trang")
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png", headers={"ETag": etag, "Cache-Control": "private, no-cache"})


@router.get("/layers")
def layers(page_id: str, request: Request):
    rt, pid = page_runtime(request, page_id)
    _, lp = _paths(rt, pid)
    if not lp or not Path(lp).is_file():
        raise HTTPException(404, "Chưa có dữ liệu lớp")
    return JSONResponse(json.loads(Path(lp).read_text(encoding="utf-8")))


@router.get("/layers/file/{rel:path}")
def layer_file(page_id: str, rel: str, request: Request):
    rt, pid = page_runtime(request, page_id)
    _, lp = _paths(rt, pid)
    if not lp:
        raise HTTPException(404, "Chưa có dữ liệu lớp")
    root = Path(lp).parent
    f = (root / rel)
    if not inside(root, f) or not f.is_file() or f.suffix.lower() not in (".png", ".json"):
        raise HTTPException(404, "Không tìm thấy lớp")
    return FileResponse(f)
