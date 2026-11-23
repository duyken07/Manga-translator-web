from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..adapters import editing
from ..adapters.runtime import RuntimeError_
from ..schemas.dto import DialogueCreate, DialoguePatch, ReorderBody, dialogue_out
from .deps import http, page_runtime

router = APIRouter(prefix="/api/pages/{page_id}/dialogues", tags=["dialogues"])


def _call(rt, fn, *a):
    try:
        return rt.run(fn, rt, *a)
    except RuntimeError_ as e:
        raise http(e)


@router.get("")
def list_dialogues(page_id: str, request: Request):
    rt, pid = page_runtime(request, page_id)
    return rt.run(lambda: [dialogue_out(d) for d in rt.project.pages[rt.index_of(pid)].dialogues])


@router.post("", status_code=201)
def create_dialogue(page_id: str, body: DialogueCreate, request: Request):
    rt, pid = page_runtime(request, page_id)
    did = _call(rt, editing.add_dialogue, pid, body.model_dump())
    return rt.run(lambda: dialogue_out(next(d for d in rt.project.pages[rt.index_of(pid)].dialogues if d.dialogue_id == did)))


@router.put("/order")
def reorder(page_id: str, body: ReorderBody, request: Request):
    rt, pid = page_runtime(request, page_id)
    _call(rt, editing.reorder, pid, body.order)
    return rt.run(lambda: [dialogue_out(d) for d in rt.project.pages[rt.index_of(pid)].dialogues])


@router.patch("/{dialogue_id}")
def patch(page_id: str, dialogue_id: int, body: DialoguePatch, request: Request):
    rt, pid = page_runtime(request, page_id)
    d = _call(rt, editing.patch_dialogue, pid, dialogue_id, body.model_dump(exclude_unset=True))
    return rt.run(lambda: dialogue_out(d))


@router.delete("/{dialogue_id}", status_code=204)
def delete(page_id: str, dialogue_id: int, request: Request):
    rt, pid = page_runtime(request, page_id)
    _call(rt, editing.delete_dialogue, pid, dialogue_id)
