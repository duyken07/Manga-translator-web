"""Tìm / thay thế chữ trong các câu thoại của nhiều trang (logic thuần, không phụ thuộc Qt nên test được riêng).

Văn bản được duyệt theo thứ tự trang -> thứ tự câu trong trang. Trường tìm: "translated" (bản dịch, thay thế được) hoặc "original"
(câu gốc, chỉ để tìm). Khớp theo Unicode nên tiếng Việt có dấu / chữ hoa - thường (Á / á) hoạt động bình thường.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

FIELDS = ("translated", "original")


@dataclass(frozen=True)
class Match:
    page_index: int
    row: int            # vị trí của câu trong trang (để giữ đúng thứ tự)
    dialogue_id: int
    page_name: str
    text: str           # toàn bộ văn bản của trường đang tìm
    count: int          # số lần khớp trong câu này
    first: tuple        # (start, end) của lần khớp đầu tiên

    @property
    def key(self) -> tuple:
        return (self.page_index, self.dialogue_id)

    @property
    def order(self) -> tuple:
        return (self.page_index, self.row)

    def snippet(self, radius: int = 28) -> str:
        """Đoạn trích quanh lần khớp đầu, đặt «…» quanh chỗ khớp, xuống dòng thành khoảng trắng."""
        s, e = self.first
        before, hit, after = self.text[:s], self.text[s:e], self.text[e:]
        before = before[-radius:] if len(before) > radius else before
        after = after[:radius] if len(after) > radius else after
        pre = "…" if len(self.text[:s]) > radius else ""
        post = "…" if len(self.text[e:]) > radius else ""
        out = f"{pre}{before}«{hit}»{after}{post}"
        return " ".join(out.split())


def build_pattern(query: str, match_case: bool = False, whole_word: bool = False) -> Optional["re.Pattern"]:
    """Chuỗi tìm là chữ THƯỜNG (không phải regex). Rỗng -> None."""
    if not query:
        return None
    body = re.escape(query)
    if whole_word:
        body = rf"(?<!\w){body}(?!\w)"
    return re.compile(body, 0 if match_case else re.IGNORECASE | re.UNICODE)


def text_of(dialogue, field: str) -> str:
    return str((dialogue.translated_text if field == "translated" else dialogue.original_text) or "")


def find_matches(pages: Sequence, pattern: Optional["re.Pattern"], field: str = "translated",
                 page_indices: Optional[Iterable[int]] = None) -> List[Match]:
    if pattern is None:
        return []
    wanted = None if page_indices is None else set(page_indices)
    out: List[Match] = []
    for pi, page in enumerate(pages):
        if wanted is not None and pi not in wanted:
            continue
        for row, d in enumerate(page.dialogues):
            text = text_of(d, field)
            if not text:
                continue
            found = list(pattern.finditer(text))
            if found:
                out.append(Match(pi, row, int(d.dialogue_id), str(getattr(page, "page_name", "") or f"Trang {pi + 1}"),
                                 text, len(found), (found[0].start(), found[0].end())))
    return out


def replace_text(text: str, pattern: "re.Pattern", replacement: str) -> tuple:
    """(văn bản mới, số lần thay). Chuỗi thay là chữ THƯỜNG: \\1, \\n... không được hiểu đặc biệt."""
    return pattern.subn(lambda _m: replacement, text)


def plan_replacements(matches: Sequence[Match], pattern: "re.Pattern", replacement: str) -> Dict[int, Dict[int, str]]:
    """{page_index: {dialogue_id: văn bản mới}} chỉ gồm những câu thật sự đổi."""
    plan: Dict[int, Dict[int, str]] = {}
    for m in matches:
        new, n = replace_text(m.text, pattern, replacement)
        if n and new != m.text:
            plan.setdefault(m.page_index, {})[m.dialogue_id] = new
    return plan
