"""
Data Schemas cho toàn bộ ứng dụng.

Cấu trúc khớp với JSON Schema mô tả trong spec:
{
  "page_name": "page_01.png",
  "dialogues": [
    {
      "dialogue_id": 1,
      "box_location": [x, y, w, h],
      "original_text": "...",
      "translated_text": "..."
    }
  ]
}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Natural sort (page_1, page_2, ..., page_10 thay vì page_1, page_10, page_2)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"(\d+)")


def natural_sort_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in _NUM_RE.split(text)]


_NEWLINE_RE = re.compile(r"[ \t]*[\r\n]+[ \t]*")


def flatten_text(text: str) -> str:
    """Thay mọi chỗ xuống dòng trong bản dịch bằng MỘT khoảng trắng ("a\nb" -> "a b"). Dùng khi xuất TXT / copy, nơi
    mỗi câu thoại phải nằm gọn trên một dòng; xuống dòng thủ công chỉ có ý nghĩa khi xếp chữ lên ảnh."""
    return _NEWLINE_RE.sub(" ", text or "").strip()


_CJKISH_RE = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")   # chữ / dấu Nhật - Trung (không dùng dấu cách giữa các từ)


def flatten_source_text(text: str) -> str:
    """Nối các dòng của CÂU GỐC thành một dòng: chỗ ngắt giữa hai ký tự Nhật / Trung thì nối liền (xuống dòng chỉ là cột chữ dọc
    trong bóng), còn lại nối bằng một khoảng trắng."""
    out = ""
    for ln in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if out and not (_CJKISH_RE.match(out[-1]) and _CJKISH_RE.match(ln[0])):
            out += " "
        out += ln
    return out


# ---------------------------------------------------------------------------
# Phiên âm (romaji) tách khỏi câu gốc
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]")   # kana, hán tự
_LATIN_RE = re.compile(r"[A-Za-z]")


def split_legacy_original(text: str) -> tuple:
    """Dữ liệu CŨ lưu câu gốc và phiên âm chung một chuỗi "câu gốc\\nromaji". Tách thành (câu gốc, phiên âm).
    Phiên âm = khối dòng CUỐI không có ký tự Nhật mà có chữ La-tinh (>= 2 chữ cái), đứng sau phần có chữ Nhật.
    Không khớp (không có phiên âm, hoặc không phải tiếng Nhật) -> (text, "") giữ nguyên."""
    if not text or "\n" not in text:
        return text, ""
    lines = text.split("\n")
    k = len(lines)
    while k > 0 and lines[k - 1].strip() and not _CJK_RE.search(lines[k - 1]) and _LATIN_RE.search(lines[k - 1]):
        k -= 1
    if k == 0 or k == len(lines):
        return text, ""
    head, tail = "\n".join(lines[:k]), "\n".join(lines[k:])
    if not _CJK_RE.search(head) or len(_LATIN_RE.findall(tail)) < 2:
        return text, ""
    return head, tail


# Models
# ---------------------------------------------------------------------------
@dataclass
class Dialogue:
    dialogue_id: int
    box_location: List[float]  # [x, y, w, h] tính theo pixel trên ảnh gốc
    original_text: str
    translated_text: str = ""
    edited: bool = False  # đánh dấu người dùng đã chỉnh sửa thủ công
    # Tuỳ chỉnh redraw cho riêng câu này (tự sửa trong file JSON), VD:
    #   {"type": "free", "polarity": "white", "engine": "lama", "font_size": 24, "style": "bold"}
    # Các khoá được hỗ trợ: xem OVERRIDE_KEYS trong app/services/redraw/stage_a.py.
    overrides: Dict[str, Any] = field(default_factory=dict)
    # "Khung neo" của redraw [x, y, w, h]: khung đã dùng để xoá chữ gốc + xếp chữ dịch. Kéo thả / đổi cỡ
    # box_location KHÔNG làm đổi ảnh redraw vì redraw luôn bám khung neo này. None = bám theo box_location.
    redraw_box: Optional[List[int]] = None
    # "Khung chữ" [x, y, w, h] do người dùng kéo thả trong chế độ xem bản dịch (kiểu Canva): chữ dịch được xếp
    # vừa và căn giữa trong khung này. Chỉ ảnh hưởng việc xếp chữ, KHÔNG đụng tới ảnh đã xoá chữ gốc.
    # None = tự động (theo hình bóng thoại).
    text_frame: Optional[List[int]] = None
    # Góc xoay khối chữ (độ, dương = theo chiều kim đồng hồ), xoay quanh tâm khung chữ. 0 = không xoay.
    text_rotation: float = 0.0
    # Số lần người dùng yêu cầu "redraw lại vùng này": đổi số này thì engine xoá lại chữ gốc của câu từ đầu theo khung hiện tại
    redraw_gen: int = 0
    # Ẩn / hiện lẻ từng câu trong ảnh redraw (xem trước và khi xuất ảnh):
    hide_text: bool = False  # True = không hiện chữ dịch của câu này
    hide_redraw: bool = False  # True = không áp vùng xoá chữ gốc của câu này (vùng đó trở lại ảnh gốc)
    # Vùng "chỉ redraw": chỉ xoá chữ gốc trong vùng (không OCR, không bản dịch, không chèn chữ). Dùng để dọn chữ sót / SFX.
    redraw_only: bool = False
    # Khung "chỉ chèn chữ": KHÔNG xoá chữ gốc (không tạo lớp vùng xoá / redraw) nhưng VẪN xếp chữ dịch lên ảnh thành lớp chữ riêng.
    # Bật redraw sau bằng chuột phải trong bảng thoại để xoá chữ gốc. (Tên trường giữ nguyên để project cũ vẫn mở được.)
    no_redraw: bool = False
    # Phiên âm La-tinh (romaji) của câu gốc, TÁCH RIÊNG khỏi original_text (original_text chỉ chứa văn bản gốc thật).
    # Bảng thoại hiển thị ghép "câu gốc\nphiên âm" (display_original) và cho sao chép riêng từng phần.
    romanization: str = ""

    def display_original(self) -> str:
        """Câu gốc + phiên âm (nếu có) để hiển thị: "câu gốc\\nphiên âm"."""
        rom = (self.romanization or "").strip()
        return f"{self.original_text}\n{rom}" if rom else self.original_text

    def flatten_lines(self) -> "Dialogue":
        """Câu gốc, phiên âm và bản dịch mỗi thứ nằm trên MỘT dòng (bỏ các chỗ ngắt hàng do Gemini chèn). Dùng ngay khi nhận kết quả
        từ model; KHÔNG gọi khi nạp project (để giữ xuống dòng người dùng tự gõ bằng Shift+Enter)."""
        self.original_text = flatten_source_text(self.original_text)
        self.romanization = flatten_text(self.romanization)
        self.translated_text = flatten_text(self.translated_text)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        if not (d.get("romanization") or "").strip():
            d.pop("romanization", None)   # không có phiên âm: giữ nguyên schema JSON cũ
        if not d.get("overrides"):
            d.pop("overrides", None)  # giữ nguyên schema JSON cũ khi không có tuỳ chỉnh
        if not d.get("redraw_box") or list(d["redraw_box"]) == list(self.box_location):
            d.pop("redraw_box", None)  # chỉ lưu khi khung neo khác khung hiện tại
        if not d.get("text_frame"):
            d.pop("text_frame", None)
        if not d.get("text_rotation"):
            d.pop("text_rotation", None)
        if not d.get("redraw_gen"):
            d.pop("redraw_gen", None)
        for k in ("hide_text", "hide_redraw", "redraw_only", "no_redraw"):
            if not d.get(k):
                d.pop(k, None)  # chỉ ghi khi bật
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Dialogue":
        original = data.get("original_text", "")
        if "romanization" in data:
            romanization = data.get("romanization") or ""
        else:
            # project cũ (hoặc model trả theo kiểu cũ): phiên âm nằm chung dòng cuối của original_text -> tách ra
            original, romanization = split_legacy_original(original)
        return cls(
            dialogue_id=data["dialogue_id"],
            box_location=list(data.get("box_location", [0, 0, 0, 0])),
            original_text=original,
            romanization=romanization,
            translated_text=data.get("translated_text", ""),
            edited=data.get("edited", False),
            overrides=dict(data.get("overrides") or {}),
            redraw_box=[int(v) for v in data["redraw_box"]] if data.get("redraw_box") else None,
            text_frame=[int(v) for v in data["text_frame"]] if data.get("text_frame") else None,
            text_rotation=float(data.get("text_rotation") or 0.0),
            redraw_gen=int(data.get("redraw_gen") or 0),
            hide_text=bool(data.get("hide_text", False)),
            hide_redraw=bool(data.get("hide_redraw", False)),
            redraw_only=bool(data.get("redraw_only", False)),
            no_redraw=bool(data.get("no_redraw", False)),
        )


@dataclass
class Page:
    page_name: str
    image_path: str  # đường dẫn tuyệt đối trên máy người dùng (không xuất ra JSON chia sẻ)
    dialogues: List[Dialogue] = field(default_factory=list)
    status: str = "pending"  # pending | translating | done | error
    error_message: Optional[str] = None  # chi tiết lỗi khi status == "error", hiển thị trong Workspace
    # Thuật ngữ MỚI (tên riêng, chức danh...) mà Gemini phát hiện khi dịch trang này,
    # dùng để cập nhật vào Project.glossary cho các trang sau - KHÔNG lưu vào JSON
    # cùng Page (chỉ mang tính tạm thời trong bộ nhớ khi dịch tuần tự có ngữ cảnh).
    glossary_updates: Dict[str, str] = field(default_factory=dict)
    # -- Redraw (xoá chữ gốc + chèn bản dịch); chỉ nằm trong bộ nhớ, ảnh kết quả được cache trên đĩa --
    redraw_path: Optional[str] = None  # ảnh redraw hoàn chỉnh của trang (None = chưa có)
    redraw_status: str = "none"  # none | queued | running | done | error
    redraw_message: str = ""  # cảnh báo / lỗi gần nhất của lần redraw
    # Hình chữ nhật bao khối chữ dịch trong ảnh redraw {dialogue_id: [x, y, w, h]} (để hiện khung chữ chỉnh được)
    layout_boxes: Dict[int, List[int]] = field(default_factory=dict)
    # layers.json của lần redraw gần nhất: các lớp riêng từng câu (vùng xoá + chữ) để viewer ẩn/hiện lẻ
    layers_path: Optional[str] = None
    # Số liệu thật đã dùng cho từng khối chữ của lần redraw gần nhất: {dialogue_id: {size, font, line_spacing, bold, color}}
    layout_info: Dict[int, dict] = field(default_factory=dict)
    # Nét vẽ tay (bút / bút dạ / dạ quang) trên trang; lưu trong JSON, in vào ảnh khi xuất. Xem app/services/strokes.py
    strokes: List[dict] = field(default_factory=list)
    # Lịch sử hoàn tác / làm lại nét vẽ (chỉ trong bộ nhớ, không lưu)
    stroke_hist: list = field(default_factory=list)
    stroke_redo: list = field(default_factory=list)
    # Vùng Content-Aware Fill: [{"id", "strokes": [{"pts", "width"}], "engine": auto|lama|texture|opencv, "hidden": bool}]
    # Người dùng bôi vùng cần vá, engine vá theo nền xung quanh; mỗi vùng là 1 lớp riêng ẩn / hiện / xoá được.
    fills: List[dict] = field(default_factory=list)
    # Vẽ / Fill THẲNG lên ảnh: app chỉnh trên một BẢN LÀM VIỆC riêng (image_path trỏ tới bản này), file gốc không bao giờ bị sửa.
    # original_image_path = file gốc của người dùng (chỉ trong bộ nhớ); direct_edited = trang có bản làm việc (lưu trong JSON để nối lại khi mở project).
    original_image_path: str = ""
    direct_edited: bool = False

    def has_redraw_content(self) -> bool:
        """Trang có gì để redraw: câu có bản dịch hoặc vùng 'chỉ redraw'."""
        return bool(self.fills) or any(d.translated_text.strip() or (d.redraw_only and not d.no_redraw) for d in self.dialogues)

    def to_dict(self) -> dict:
        d = {
            "page_name": self.page_name,
            "dialogues": [x.to_dict() for x in self.dialogues],
        }
        if self.strokes:
            d["strokes"] = self.strokes
        if self.fills:
            d["fills"] = self.fills
        if self.direct_edited:
            d["direct_edited"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict, image_path: str = "") -> "Page":
        return cls(
            page_name=data["page_name"],
            image_path=image_path,
            dialogues=[Dialogue.from_dict(d) for d in data.get("dialogues", [])],
            status="done" if data.get("dialogues") else "pending",
            strokes=[dict(s) for s in data.get("strokes", [])],
            fills=[dict(f) for f in data.get("fills", [])],
            direct_edited=bool(data.get("direct_edited", False)),
        )


@dataclass
class Project:
    """Toàn bộ trạng thái của một bộ truyện đang xử lý (Global State)."""

    source_lang: str = "ja"
    target_lang: str = "vi"
    pages: List[Page] = field(default_factory=list)
    json_path: Optional[str] = None  # nơi lưu project_data.json
    # Bảng thuật ngữ tích luỹ qua các trang đã dịch (tên riêng, chức danh...),
    # dùng làm ngữ cảnh khi dịch các trang sau để giữ nhất quán cách dịch.
    # Được lưu/nạp cùng project_data.json để giữ nguyên qua các phiên làm việc.
    glossary: Dict[str, str] = field(default_factory=dict)
    # Cache kết quả "Phân tích ngữ pháp" theo câu gốc (key = câu gốc đã phân
    # tích), để chọn lại đúng câu đó không cần gọi API lần nữa. Lưu/nạp cùng
    # project_data.json nên vẫn còn sau khi đóng và mở lại ứng dụng.
    grammar_cache: Dict[str, dict] = field(default_factory=dict)
    # Tên tác phẩm gốc do người dùng đặt (menu ở đầu hộp chatbot): chatbot dùng để tìm hiểu đúng bộ truyện. Rỗng = chưa đặt.
    work_title: str = ""
    # Tên trang đã bị xoá khỏi project: save_json gộp với file cũ theo page_name nên phải loại các trang này ra,
    # nếu không trang đã xoá sẽ "sống lại" trong file. Không lưu ra JSON.
    removed_names: set = field(default_factory=set)

    def sort_pages(self) -> None:
        self.pages.sort(key=lambda p: natural_sort_key(p.page_name))

    def add_images(self, image_paths: List[str]) -> None:
        existing_names = {p.page_name for p in self.pages}
        for path in image_paths:
            name = Path(path).name
            if name in existing_names:
                continue
            existing_names.add(name)        # hai ảnh cùng tên (khác thư mục) trong cùng một lần nạp: chỉ lấy ảnh đầu
            self.removed_names.discard(name)
            self.pages.append(Page(page_name=name, image_path=path))
        self.sort_pages()

    # -- Export / Persist ---------------------------------------------------
    def to_json_dict(self) -> dict:
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "pages": [p.to_dict() for p in self.pages],
            "glossary": self.glossary,
            "grammar_cache": self.grammar_cache,
            "work_title": self.work_title,
        }

    def cache_grammar_result(self, sentence: str, data: dict) -> None:
        """Lưu kết quả phân tích ngữ pháp của 1 câu vào cache, dùng lại khi gặp đúng câu này."""
        if sentence:
            self.grammar_cache[sentence] = data

    def update_glossary(self, updates: Dict[str, str]) -> None:
        """Gộp thuật ngữ mới phát hiện vào glossary hiện có (không ghi đè thuật ngữ đã có)."""
        for term, translation in updates.items():
            if term and translation and term not in self.glossary:
                self.glossary[term] = translation

    def save_json(self, path: str) -> None:
        """Ghi/Gộp (merge) vào file JSON theo page_name, không nối thô."""
        target = Path(path)
        merged: dict[str, dict] = {}
        existing_glossary: dict = {}
        existing_grammar_cache: dict = {}

        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                for pg in existing.get("pages", []):
                    merged[pg["page_name"]] = pg
                existing_glossary = existing.get("glossary", {})
                existing_grammar_cache = existing.get("grammar_cache", {})
            except (json.JSONDecodeError, OSError):
                pass  # file hỏng/không đọc được -> coi như bắt đầu mới

        for name in self.removed_names:
            merged.pop(name, None)
        for p in self.pages:
            merged[p.page_name] = p.to_dict()

        # Gộp glossary/grammar_cache đã lưu trước đó với bản hiện tại trong bộ
        # nhớ (ưu tiên giá trị trong bộ nhớ nếu trùng key, vì đó là bản mới nhất)
        merged_glossary = {**existing_glossary, **self.glossary}
        merged_grammar_cache = {**existing_grammar_cache, **self.grammar_cache}

        ordered_pages = sorted(merged.values(), key=lambda pg: natural_sort_key(pg["page_name"]))
        output = {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "pages": ordered_pages,
            "glossary": merged_glossary,
            "grammar_cache": merged_grammar_cache,
            "work_title": self.work_title,
        }
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        self.json_path = str(target)
        self.glossary = merged_glossary
        self.grammar_cache = merged_grammar_cache

    @classmethod
    def load_json(cls, path: str, image_dir: Optional[str] = None) -> "Project":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        proj = cls(
            source_lang=data.get("source_lang", "ja"),
            target_lang=data.get("target_lang", "vi"),
            json_path=path,
            glossary=dict(data.get("glossary", {})),
            grammar_cache=dict(data.get("grammar_cache", {})),
            work_title=str(data.get("work_title", "") or ""),
        )
        for pg in data.get("pages", []):
            image_path = ""
            if image_dir:
                candidate = Path(image_dir) / pg["page_name"]
                if candidate.exists():
                    image_path = str(candidate)
            proj.pages.append(Page.from_dict(pg, image_path=image_path))
        proj.sort_pages()
        return proj

    def export_txt(self, path: str) -> None:
        """Xuất file TXT định dạng hội thoại theo trang. Xuống dòng thủ công trong bản dịch được đổi thành khoảng trắng."""
        lines: List[str] = []
        for idx, page in enumerate(self.pages, start=1):
            lines.append(f"Page {idx}")
            for d in page.dialogues:
                lines.append(flatten_text(d.translated_text))  # xuống dòng thủ công -> khoảng trắng
            lines.append("")  # dòng trống ngăn cách giữa các trang
        Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
