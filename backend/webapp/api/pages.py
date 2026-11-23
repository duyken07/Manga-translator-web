from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..adapters import editing
from ..adapters.runtime import RuntimeError_
from ..database.models import PageRow
from ..database.session import session_scope
from ..schemas.dto import page_detail, page_summary
from ..storage.files import UploadError, make_thumbnail, save_upload
from .deps import http, page_runtime, runtime

router = APIRouter(tags=["pages"])


@router.post("/api/projects/{project_id}/pages/upload", status_code=201)
async def upload(project_id: str, request: Request, files: List[UploadFile] = File(...)):
    rt = runtime(request, project_id)
    if len(files) > 500:
        raise HTTPException(413, "Tối đa 500 ảnh mỗi lần upload")
    existing = set(rt.id_by_name)
    saved, rejected, dup = [], [], []
    for f in files:
        try:
            from ..core.security import safe_filename
            if safe_filename(f.filename or "") in existing:       # kiểm tra TRƯỚC khi ghi để không đè ảnh đang dùng
                dup.append(f.filename)
                continue
            name, w, h = await run_in_threadpool(save_upload, rt.paths, f.filename or "", f.file)
            saved.append((name, w, h))
            existing.add(name)
        except (UploadError, ValueError) as e:
            rejected.append({"file": f.filename, "reason": str(e)})
    try:
        ids, skipped = await run_in_threadpool(rt.run, editing.add_pages, rt, saved) if saved else ([], [])
    except RuntimeError_ as e:
        raise http(e)
    return {"added": len(ids), "page_ids": ids, "skipped_duplicates": dup + skipped, "rejected": rejected}


@router.get("/api/projects/{project_id}/pages")
def list_pages(project_id: str, request: Request):
    rt = runtime(request, project_id)
    return rt.run(lambda: [page_summary(rt, p, rt.id_by_name[p.page_name]) for p in rt.project.pages])


@router.get("/api/pages/{page_id}")
def get_page(page_id: str, request: Request):
    rt, pid = page_runtime(request, page_id)
    return rt.run(lambda: page_detail(rt, rt.project.pages[rt.index_of(pid)], pid))


@router.delete("/api/pages/{page_id}", status_code=204)
def delete_page(page_id: str, request: Request):
    rt, pid = page_runtime(request, page_id)
    try:
        rt.run(editing.remove_page, rt, pid)
    except RuntimeError_ as e:
        raise http(e)


def _page_files(rt, pid):
    def f():
        p = rt.project.pages[rt.index_of(pid)]
        return p.page_name, p.image_path, p.redraw_path
    return rt.run(f)


@router.get("/api/pages/{page_id}/image")
def page_image(page_id: str, request: Request, version: str = "current"):
    """version: current (bản làm việc nếu đã vẽ/fill trực tiếp) | original."""
    rt, pid = page_runtime(request, page_id)
    name, cur, _ = _page_files(rt, pid)
    path = rt.paths.originals / name if version == "original" else cur
    return FileResponse(path, headers={"Cache-Control": "private, max-age=0, must-revalidate"})


@router.get("/api/pages/{page_id}/thumbnail")
def page_thumbnail(page_id: str, request: Request):
    rt, pid = page_runtime(request, page_id)
    name, cur, _ = _page_files(rt, pid)
    from pathlib import Path
    t = make_thumbnail(Path(cur), rt.paths.thumbs / (name + ".jpg"))
    return FileResponse(t, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})
