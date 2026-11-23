from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..core.settings_store import PROJECT_OVERRIDABLE
from ..database.models import ProjectRow
from ..database.session import get_db
from ..schemas.dto import ProjectCreate, ProjectUpdate
from ..storage.files import delete_project_files, project_paths
from .deps import ctx_of, runtime

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _out(r: ProjectRow) -> dict:
    return {"id": r.id, "name": r.name, "source_lang": r.source_lang, "target_lang": r.target_lang, "work_title": r.work_title,
            "page_count": len(r.pages), "glossary": r.glossary or {}, "settings": r.settings or {},
            "created_at": r.created_at.isoformat(), "updated_at": r.updated_at.isoformat()}


@router.get("")
def list_projects(db: Session = Depends(get_db)):
    return [_out(r) for r in db.query(ProjectRow).order_by(ProjectRow.updated_at.desc()).all()]


@router.post("", status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    r = ProjectRow(name=body.name.strip(), source_lang=body.source_lang, target_lang=body.target_lang, work_title=body.work_title)
    db.add(r)
    db.commit()
    project_paths(r.id).ensure()
    return _out(r)


@router.get("/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    r = db.get(ProjectRow, project_id)
    if not r:
        raise HTTPException(404, "Không tìm thấy project")
    return _out(r)


@router.put("/{project_id}")
def update_project(project_id: str, body: ProjectUpdate, request: Request, db: Session = Depends(get_db)):
    r = db.get(ProjectRow, project_id)
    if not r:
        raise HTTPException(404, "Không tìm thấy project")
    rt = runtime(request, project_id)

    def apply() -> None:                       # đổi trên luồng trạng thái rồi persist: tránh bị autosave ghi đè
        p = rt.project
        if body.source_lang is not None:
            p.source_lang = body.source_lang
        if body.target_lang is not None:
            p.target_lang = body.target_lang
        if body.work_title is not None:
            p.work_title = body.work_title
        if body.glossary is not None:
            p.glossary = {str(k): str(v) for k, v in body.glossary.items()}
        rt.persist()

    rt.run(apply)
    db.expire_all()
    r = db.get(ProjectRow, project_id)
    if body.name is not None:
        r.name = body.name.strip()
    if body.settings is not None:
        bad = [k for k in body.settings if k not in PROJECT_OVERRIDABLE]
        if bad:
            raise HTTPException(422, f"Không cho ghi đè theo project: {', '.join(bad)}")
        r.settings = dict(body.settings)
    db.commit()
    return _out(r)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request, db: Session = Depends(get_db)):
    r = db.get(ProjectRow, project_id)
    if not r:
        raise HTTPException(404, "Không tìm thấy project")
    ctx_of(request).registry.drop(project_id)
    db.delete(r)
    db.commit()
    delete_project_files(project_id)
