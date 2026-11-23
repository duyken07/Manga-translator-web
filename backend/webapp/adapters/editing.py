"""Thao tác chỉnh sửa trang / thoại. Mọi hàm chạy TRÊN luồng trạng thái (gọi qua rt.run) và dùng lại phương thức AppState của Desktop."""
from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional, Tuple

from app.models import Dialogue
from app.services.redraw import engine as redraw_engine
from app.services.redraw.stage_a import OVERRIDE_KEYS

from ..core import settings_store as ss
from ..database.models import PageRow
from ..database.session import session_scope
from .runtime import NotFound, ProjectRuntime, RuntimeBusy, RuntimeError_, _font_to_abs


def _busy(rt: ProjectRuntime) -> bool:
    return bool(rt._translate_job) or (rt.queue is not None and rt.queue.is_busy()) or not rt.redraw.is_idle()


# ------------------------------------------------------------------------------------------------------------ trang
def add_pages(rt: ProjectRuntime, entries: List[Tuple[str, int, int]]) -> Tuple[List[str], List[str]]:
    """entries: (page_name, width, height) đã lưu file. Trả (page_ids mới, tên bị bỏ qua vì trùng). Chỉ số trang có thể dời nên
    không cho thêm khi đang dịch / redraw (như ghi chú của AppState.add_more_images)."""
    if _busy(rt):
        raise RuntimeBusy("Đang dịch / redraw: hãy đợi xong rồi thêm trang.")
    new, skipped = [], []
    with session_scope() as s:
        for name, w, h in entries:
            if name in rt.id_by_name:
                skipped.append(name)
                continue
            row = PageRow(project_id=rt.id, page_name=name, width=w, height=h, data={})
            s.add(row)
            s.flush()
            rt.id_by_name[name] = row.id
            rt.dims[row.id] = (w, h)
            new.append((name, row.id))
    if new:
        rt.state.add_more_images([str(rt.paths.originals / n) for n, _ in new])
        rt.persist()
        rt.emit("project")
    return [i for _, i in new], skipped


def remove_page(rt: ProjectRuntime, page_id: str) -> None:
    if _busy(rt):
        raise RuntimeBusy("Đang dịch / redraw: hãy đợi xong rồi xóa trang.")
    idx = rt.index_of(page_id)
    removed = rt.state.remove_pages([idx])
    for p in removed:
        redraw_engine.clear_cache(str(rt.paths.cache), p.image_path)
        for f in (rt.paths.originals / p.page_name, rt.paths.working / p.page_name, rt.paths.thumbs / (p.page_name + ".jpg")):
            f.unlink(missing_ok=True)
        rt.id_by_name.pop(p.page_name, None)
    with session_scope() as s:
        row = s.get(PageRow, page_id)
        if row:
            s.delete(row)
    rt.dims.pop(page_id, None)
    rt.persist()
    rt.emit("project")


# ------------------------------------------------------------------------------------------------------------ thoại
def _dlg(rt: ProjectRuntime, page_id: str, dialogue_id: int) -> Tuple[int, Dialogue]:
    idx = rt.index_of(page_id)
    for d in rt.project.pages[idx].dialogues:
        if d.dialogue_id == dialogue_id:
            return idx, d
    raise NotFound("Không tìm thấy thoại")


def clean_box(v: Any, field: str = "box") -> List[int]:
    if not (isinstance(v, (list, tuple)) and len(v) == 4):
        raise RuntimeError_(f"{field} phải là [x, y, w, h]")
    x, y, w, h = (float(n) for n in v)
    if w <= 0 or h <= 0:
        raise RuntimeError_(f"{field}: w/h phải > 0")
    return [round(x), round(y), round(w), round(h)]


def sanitize_overrides(ov: dict) -> dict:
    """Chỉ nhận khoá hợp lệ của engine; font chỉ được chọn theo TÊN trong thư mục font server."""
    out = {}
    for k, v in (ov or {}).items():
        if k not in OVERRIDE_KEYS:
            raise RuntimeError_(f"Khoá override không hợp lệ: {k}")
        out[k] = v
    if out.get("font_path"):
        name = str(out["font_path"])
        if not ss.resolve_font(name.split("/")[-1].split("\\")[-1]):
            raise RuntimeError_("Font không có trên server")
        out = _font_to_abs(out)
    return out


def add_dialogue(rt: ProjectRuntime, page_id: str, data: dict) -> int:
    idx = rt.index_of(page_id)
    d = Dialogue(dialogue_id=0, box_location=clean_box(data["box_location"], "box_location"), original_text=data.get("original_text", ""),
                 translated_text=data.get("translated_text", ""), romanization=data.get("romanization", ""),
                 edited=bool(data.get("edited", True)),            # nhập tay: đánh dấu đã chỉnh sửa như Desktop
                 redraw_only=bool(data.get("redraw_only", False)), no_redraw=bool(data.get("no_redraw", False)))
    rt.state.add_dialogue(idx, d)
    return d.dialogue_id


def patch_dialogue(rt: ProjectRuntime, page_id: str, dialogue_id: int, p: dict) -> Dialogue:
    idx, d = _dlg(rt, page_id, dialogue_id)
    st = rt.state
    redraw_changing = False
    if "box_location" in p:
        st.update_dialogue_box(idx, dialogue_id, clean_box(p["box_location"], "box_location"))
    if "text_frame" in p:
        if p["text_frame"] is None:
            st.reset_text_frame(idx, dialogue_id)
        else:
            st.update_text_frame(idx, dialogue_id, clean_box(p["text_frame"], "text_frame"))
    if "text_rotation" in p:
        st.update_text_rotation(idx, dialogue_id, float(p["text_rotation"]))
    if "overrides" in p:
        st.update_dialogue_overrides(idx, dialogue_id, sanitize_overrides(p["overrides"]))
    if "translated_text" in p:
        st.update_dialogue_translation(idx, dialogue_id, str(p["translated_text"]))
    if "original_text" in p and p["original_text"] != d.original_text:
        d.original_text, d.edited = str(p["original_text"]), True
        redraw_changing = True
    if "romanization" in p and p["romanization"] != d.romanization:
        d.romanization, d.edited = str(p["romanization"]), True
        redraw_changing = True
    vis = {k: bool(p[k]) for k in ("hide_text", "hide_redraw") if k in p}
    if vis:
        st.set_dialogue_visibility(idx, dialogue_id, text=(not vis["hide_text"]) if "hide_text" in vis else None,
                                   redraw=(not vis["hide_redraw"]) if "hide_redraw" in vis else None)
    for flag in ("redraw_only", "no_redraw"):
        if flag in p and bool(p[flag]) != getattr(d, flag):
            setattr(d, flag, bool(p[flag]))
            redraw_changing = True
    if redraw_changing:
        st._set_dirty(True)
        st.dialogue_meta_changed.emit(idx, dialogue_id)
        st.dialogues_changed.emit(idx)
    return d


def delete_dialogue(rt: ProjectRuntime, page_id: str, dialogue_id: int) -> None:
    idx, _ = _dlg(rt, page_id, dialogue_id)
    rt.state.delete_dialogues(idx, [dialogue_id])


def reorder(rt: ProjectRuntime, page_id: str, order: List[int]) -> None:
    idx = rt.index_of(page_id)
    rt.state.reorder_dialogues(idx, order)
    rt.state.dialogues_changed.emit(idx)
