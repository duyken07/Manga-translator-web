"""Pydantic schemas + serializer. Không bao giờ trả đường dẫn filesystem; ảnh được phục vụ qua URL /api/pages/{id}/image..."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

Box = List[float]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source_lang: str = Field("ja", max_length=16)
    target_lang: str = Field("vi", max_length=16)
    work_title: str = Field("", max_length=300)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    source_lang: Optional[str] = Field(None, max_length=16)
    target_lang: Optional[str] = Field(None, max_length=16)
    work_title: Optional[str] = Field(None, max_length=300)
    glossary: Optional[Dict[str, str]] = None
    settings: Optional[Dict[str, Any]] = None        # ghi đè theo project (whitelist PROJECT_OVERRIDABLE)


class DialogueCreate(BaseModel):
    box_location: Box
    original_text: str = ""
    translated_text: str = ""
    romanization: str = ""
    redraw_only: bool = False
    no_redraw: bool = False


class DialoguePatch(BaseModel):
    model_config = {"extra": "forbid"}
    original_text: Optional[str] = None
    romanization: Optional[str] = None
    translated_text: Optional[str] = None
    box_location: Optional[Box] = None
    text_frame: Optional[Box] = None
    text_rotation: Optional[float] = None
    overrides: Optional[Dict[str, Any]] = None
    hide_text: Optional[bool] = None
    hide_redraw: Optional[bool] = None
    redraw_only: Optional[bool] = None
    no_redraw: Optional[bool] = None


class ReorderBody(BaseModel):
    order: List[int]


class TranslateBody(BaseModel):
    page_ids: Optional[List[str]] = None
    keep_manual: Optional[bool] = None       # != None: "Dịch lại trang" (giữ/bỏ câu sửa tay)


class RegionBody(BaseModel):
    box: Box = Field(description="[x, y, w, h] theo pixel ẢNH GỐC")
    add: bool = True                          # True: thêm thành thoại mới (như Desktop)


class RedrawBody(BaseModel):
    force_full: bool = False
    relayout: bool = False


class RedrawDialogueBody(BaseModel):
    dialogue_id: int


class ModelIdBody(BaseModel):
    id: str


class ApiKeyBody(BaseModel):
    api_key: str = Field(min_length=8, max_length=300)


def dialogue_out(d) -> dict:
    out = asdict(d)
    out["display_original"] = d.display_original()
    return out


def redraw_out(page, has_layers: bool) -> dict:
    return {"status": page.redraw_status, "message": page.redraw_message, "has_image": bool(page.redraw_path),
            "has_layers": has_layers, "layout_boxes": {str(k): v for k, v in page.layout_boxes.items()},
            "layout_info": {str(k): v for k, v in page.layout_info.items()}}


def page_summary(rt, page, page_id: str) -> dict:
    w, h = rt.dims.get(page_id, (0, 0))
    return {"id": page_id, "page_name": page.page_name, "status": page.status, "error_message": page.error_message,
            "width": w, "height": h, "dialogue_count": len(page.dialogues),
            "edited_count": sum(1 for d in page.dialogues if d.edited), "redraw_status": page.redraw_status,
            "direct_edited": page.direct_edited}


def page_detail(rt, page, page_id: str) -> dict:
    out = page_summary(rt, page, page_id)
    out.update({"dialogues": [dialogue_out(d) for d in page.dialogues], "strokes": page.strokes, "fills": page.fills,
                "redraw": redraw_out(page, bool(page.layers_path))})
    return out
