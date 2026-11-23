"""Đọc JSON do Gemini trả về một cách chịu lỗi.

Gemini (nhất là các model "lite") thỉnh thoảng trả JSON hỏng dù đã đặt response_schema: dấu phẩy thừa trước } hoặc ],
bị cắt ngang giữa chừng, bọc trong ```json ...```. Trước đây chỉ cần hỏng một ký tự là cả trang báo lỗi và mất kết quả.
parse_gemini_json thử lần lượt: đọc thẳng -> bỏ rào ``` + dấu phẩy thừa -> CỨU các đối tượng {...} còn nguyên vẹn
(dialogues / glossary_updates). Chỉ ném JSONDecodeError khi không cứu được gì.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger("json_utils")

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _loads_repaired(text: str) -> Any:
    t = _FENCE.sub("", text).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", t))


def _balanced_objects(text: str) -> List[str]:
    """Các đoạn {...} cân bằng ngoặc, ở độ sâu 2 trở lên (đối tượng con trong mảng), bỏ qua ngoặc nằm trong chuỗi."""
    out: List[str] = []
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
            if depth == 2:
                start = i
        elif ch == "}":
            if depth == 2 and start >= 0:
                out.append(text[start:i + 1])
                start = -1
            depth = max(0, depth - 1)
    return out


def _salvage(text: str) -> Optional[Dict[str, Any]]:
    dialogues: List[dict] = []
    glossary: List[dict] = []
    for chunk in _balanced_objects(text):
        try:
            obj = _loads_repaired(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "id" in obj:
            dialogues.append(obj)
        elif "term" in obj and "translation" in obj:
            glossary.append(obj)
    if not dialogues:
        return None
    return {"dialogues": dialogues, "glossary_updates": glossary}


def parse_gemini_json(raw_text: str, expect_ids: Optional[List[int]] = None) -> Any:
    """Trả dữ liệu đã parse. Nếu phải cứu / sửa thì ghi cảnh báo (kèm id còn thiếu nếu biết expect_ids)."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as first:
        try:
            data = _loads_repaired(raw_text)
            log.warning("JSON của Gemini có lỗi nhỏ (%s) - đã tự sửa", first)
            return data
        except json.JSONDecodeError:
            pass
        data = _salvage(raw_text)
        if data is None:
            raise first
        got = {d.get("id") for d in data["dialogues"]}
        missing = sorted(set(expect_ids or []) - got)
        log.warning("JSON của Gemini hỏng (%s) - cứu được %d câu%s", first, len(got),
                    f", THIẾU id={missing} (sẽ để trống, dịch lại trang nếu cần)" if missing else "")
        return data
