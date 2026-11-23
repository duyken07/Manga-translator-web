"""
Kho tri thức "cách dùng app" cho chatbot: lấy thẳng nội dung cửa sổ "Hướng dẫn sử dụng" (help_dialog) rồi đổi HTML -> văn bản,
nên sửa hướng dẫn ở một chỗ thì chatbot cũng cập nhật theo. Mỗi lượt hỏi chỉ chèn vài mục LIÊN QUAN NHẤT (chấm điểm theo từ khoá),
không nhồi cả tài liệu.

Không phụ thuộc Qt ở mức module (help_dialog được nạp lười, lỗi nạp -> kho rỗng, chatbot vẫn chạy nhưng không có căn cứ hướng dẫn).
"""
from __future__ import annotations

import html as _html
import logging
import math
import re
from dataclasses import dataclass
from typing import List, Optional

from app.services.chat_scenarios import fold

log = logging.getLogger("app_guide")

MAX_SECTION_CHARS = 2600
MAX_TOTAL_CHARS = 7000
TOP_K = 3

_STOP = set("la va cua cho toi minh ban the nao nhu sao gi co khong duoc nay do trong tren voi mot cac nhung de khi thi hay "
            "app ung dung phan mem lam".split())


@dataclass
class Section:
    title: str
    text: str
    folded_title: str
    folded_text: str


_cache: Optional[List[Section]] = None


def html_to_text(html: str) -> str:
    """HTML đơn giản của help_dialog -> văn bản thuần, giữ cấu trúc dòng / gạch đầu dòng / số thứ tự."""
    s = html.replace("[[menu]]", "☰ Mục lục")
    s = re.sub(r"\[\[[a-z0-9-]+\]\]", "", s)                       # icon Lucide nội tuyến
    s = re.sub(r"(?is)<(script|style).*?</\1>", "", s)
    s = re.sub(r"(?i)<li[^>]*>", "\n- ", s)
    s = re.sub(r"(?i)</?(p|div|h[1-6]|ul|ol|table|tr|br)[^>]*>", "\n", s)
    s = re.sub(r"(?i)</t[dh]>", " | ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = _html.unescape(s).replace("\xa0", " ")
    lines = [" ".join(ln.split()) for ln in s.splitlines()]
    out: List[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def _collect() -> List[tuple]:
    from app.ui import help_dialog as hd         # nạp lười: kéo theo PySide6
    rows = []
    for group, html in getattr(hd, "GUIDE", []):
        rows.append((group, html))
    for name in ("SETTINGS_ITEMS", "TOOLS_ITEMS", "FEATURE_ITEMS"):
        for title, html in getattr(hd, name, []):
            rows.append((title, html))
    for item in getattr(hd, "EXTRA", []):
        rows.append((item[-2] if len(item) == 3 else item[0], item[-1]))
    return rows


def load_sections(force: bool = False) -> List[Section]:
    global _cache
    if _cache is not None and not force:
        return _cache
    secs: List[Section] = []
    try:
        for title, html in _collect():
            text = html_to_text(html)
            clean_title = re.sub(r"\[\[[a-z0-9-]+\]\]", "", title)
            clean_title = re.sub(r"^[^\wÀ-ỹ]+", "", clean_title).strip() or title
            if text:
                secs.append(Section(clean_title, text, fold(clean_title), fold(text)))
    except Exception:  # noqa: BLE001
        log.exception("Không nạp được nội dung hướng dẫn cho chatbot")
    _cache = secs
    return secs


def _tokens(query: str) -> List[str]:
    return [t for t in fold(query).split() if len(t) >= 2 and t not in _STOP]


def rank(query: str, sections: Optional[List[Section]] = None) -> List[tuple]:
    """[(điểm, Section)] giảm dần. Từ hiếm (ít mục chứa) nặng hơn từ phổ biến (IDF); tiêu đề trùng từ khoá và cụm 2 từ liền nhau
    được cộng thêm. Nhờ đó 'chữ thò ra ngoài bóng thoại' trúng mục sự cố chứ không trôi sang mục nào cũng có chữ 'bóng thoại'."""
    secs = load_sections() if sections is None else sections
    toks = list(dict.fromkeys(_tokens(query)))
    if not toks or not secs:
        return []
    n = len(secs)
    idf = {t: math.log(1 + n / (1 + sum(1 for s in secs if f" {t} " in s.folded_text))) for t in toks}
    bigrams = [(a, b) for a, b in zip(toks, toks[1:])]
    scored = []
    for sec in secs:
        score = 0.0
        title_words = set(sec.folded_title.split())
        for t in toks:
            if t in title_words:
                score += 4 * idf[t]
            score += min(sec.folded_text.count(f" {t} "), 4) * idf[t]
        for a, b in bigrams:
            w = idf[a] + idf[b]
            if f" {a} {b} " in sec.folded_title:
                score += 5 * w
            elif f" {a} {b} " in sec.folded_text:
                score += 2.5 * w
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: -x[0])
    return scored


def retrieve(query: str, k: int = TOP_K, max_chars: int = MAX_TOTAL_CHARS) -> str:
    """Các mục hướng dẫn liên quan nhất tới `query`, nối thành 1 khối văn bản để đưa vào system instruction.
    Không khớp mục nào -> mục 'Bắt đầu nhanh' (tổng quan quy trình)."""
    secs = load_sections()
    if not secs:
        return ""
    ranked = [s for _, s in rank(query, secs)[:k]] or secs[:1]
    parts, used = [], 0
    for sec in ranked:
        body = sec.text if len(sec.text) <= MAX_SECTION_CHARS else sec.text[:MAX_SECTION_CHARS] + "…"
        chunk = f"### {sec.title}\n{body}"
        if used + len(chunk) > max_chars and parts:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts)
