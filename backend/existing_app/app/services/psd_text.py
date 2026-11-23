"""psd_text.py - dựng LỚP CHỮ (Type layer) có thể sửa trong Photoshop cho file PSD.

Cách làm: lấy khối "TySh" (Type Tool Object Setting) của một lớp chữ do Photoshop tạo (app/assets/type_layer_template.bin, xem
NOTICE.txt) làm khuôn và chỉ thay những trường cần thiết - nội dung, font, cỡ chữ, giãn dòng, màu, căn giữa, vị trí / xoay.
Phần còn lại giữ nguyên như Photoshop ghi nên cấu trúc luôn hợp lệ (tự dựng từ đầu rất dễ làm Photoshop từ chối cả file).

Lớp vẫn giữ nguyên ẢNH chữ đã vẽ trong app (kể cả viền chữ) làm hình hiển thị; Photoshop chỉ dựng lại chữ bằng bộ máy của nó
khi bạn sửa chữ. Bản dựng lại có thể lệch chút ít so với ảnh trong app (cách ngắt chữ, viền chữ không có).

Mỗi dòng đã ngắt trong app là một đoạn (xuống dòng \\r), căn giữa quanh `cx`, điểm neo nằm trên đường cơ sở dòng đầu - đúng cách
app đặt chữ - nên chữ Photoshop dựng lại trùng vị trí với ảnh chữ.
"""
from __future__ import annotations

import copy
import io
import math
import struct
from functools import lru_cache
from pathlib import Path
from typing import Optional

from psd_tools.psd.engine_data import Float, Integer

_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "type_layer_template.bin"
_TEMPLATE_BOX = Path(__file__).resolve().parent.parent / "assets" / "type_layer_template_box.bin"

# True: lớp chữ xuất ra là chữ KHỐI (paragraph text) có khung bao như khi kéo khung bằng công cụ Type trong Photoshop.
# False: chữ điểm (point text) như bản cũ - đặt False nếu Photoshop của bạn báo lỗi khi mở lớp chữ khối.
PSD_TEXT_BOX = True


# --------------------------------------------------------------------------- tên PostScript của font
@lru_cache(maxsize=256)
def font_info(font_path: str) -> Optional[tuple]:
    """(tên PostScript, là TrueType?) của file font (.ttf / .otf / .ttc), hoặc None nếu không đọc được. Photoshop nhận font theo
    tên PostScript (VD 'ArialMT', 'Anime-Ace-Regular'). Đọc trực tiếp bảng 'name' (nameID 6) nên không cần thêm thư viện."""
    try:
        data = Path(font_path).read_bytes()
        base = 0
        if data[:4] == b"ttcf":                       # bộ sưu tập: lấy font đầu tiên
            base = struct.unpack(">I", data[12:16])[0]
        tag = data[base:base + 4]
        n_tables = struct.unpack(">H", data[base + 4:base + 6])[0]
        truetype = tag in (b"\x00\x01\x00\x00", b"true")
        name_off = None
        for i in range(n_tables):
            rec = data[base + 12 + 16 * i: base + 28 + 16 * i]
            if rec[:4] == b"name":
                name_off = struct.unpack(">I", rec[8:12])[0]
                break
        if name_off is None:
            return None
        count, str_off = struct.unpack(">HH", data[name_off + 2:name_off + 6])
        best = None
        for i in range(count):
            plat, enc, _lang, name_id, length, off = struct.unpack(">HHHHHH", data[name_off + 6 + 12 * i: name_off + 18 + 12 * i])
            if name_id != 6:
                continue
            raw = data[name_off + str_off + off: name_off + str_off + off + length]
            text = raw.decode("utf-16-be", "ignore") if plat in (0, 3) else raw.decode("mac_roman", "ignore")
            text = "".join(c for c in text if 33 <= ord(c) < 127 and c not in "[](){}<>/%")
            if text and (best is None or plat == 3):
                best = text
        return (best, truetype) if best else None
    except (OSError, struct.error, IndexError, ValueError):
        return None


# --------------------------------------------------------------------------- hình học
def type_matrix(cx: float, baseline: float, rotation_deg: float = 0.0, pivot: tuple | None = None) -> tuple:
    """Ma trận (xx, xy, yx, yy, tx, ty) của TySh. Điểm neo chữ căn giữa = (cx, đường cơ sở dòng đầu). rotation_deg > 0 là xoay THEO
    chiều kim đồng hồ quanh `pivot` (tâm ảnh chữ chưa xoay), đúng như app xoay khối chữ."""
    if not rotation_deg:
        return (1.0, 0.0, 0.0, 1.0, float(cx), float(baseline))
    a = math.radians(rotation_deg)
    c, s = math.cos(a), math.sin(a)
    px, py = pivot if pivot is not None else (cx, baseline)
    dx, dy = cx - px, baseline - py
    tx, ty = px + dx * c - dy * s, py + dx * s + dy * c
    return (c, s, -s, c, float(tx), float(ty))


# --------------------------------------------------------------------------- dựng khối TySh
def _set_list(lst, items) -> None:
    while len(lst):
        lst.pop()
    for it in items:
        lst.append(it)


@lru_cache(maxsize=2)
def _template_bytes(box: bool = False) -> bytes:
    return (_TEMPLATE_BOX if box else _TEMPLATE).read_bytes()


def build_type_block(lines: list, font_ps: str, truetype: bool, size: float, leading: float, rgb: tuple, matrix: tuple,
                     box: tuple | None = None, justification: int = 2):
    """Khối TySh cho một khung chữ căn giữa. lines: các dòng (mỗi dòng là một đoạn). rgb: (r, g, b) 0..255.
    box=(rộng, cao): dựng chữ KHỐI có khung bao rộng x cao px (gốc khung = (tx, ty) của `matrix`, tức góc trên trái); None = chữ điểm."""
    from psd_tools.psd.tagged_blocks import TypeToolObjectSetting

    lines = [str(l) for l in lines if str(l) != ""] or [""]
    blk = TypeToolObjectSetting.frombytes(_template_bytes(box is not None))
    text = "\r".join(lines)
    blk.text_data[b"Txt "].value = text + "\x00"          # descriptor: chuỗi Unicode kết thúc bằng NUL, không có \r cuối
    eng = blk.text_data[b"EngineData"].value
    ed = eng["EngineDict"]
    ed["Editor"]["Text"].value = text + "\r"              # EngineData: mỗi đoạn kết thúc bằng \r
    lens = [len(l) + 1 for l in lines]

    # đoạn văn: mỗi dòng 1 đoạn; căn lề theo `justification` (0 trái, 1 phải, 2 giữa)
    para = ed["ParagraphRun"]
    proto = para["RunArray"][0]
    proto["ParagraphSheet"]["Properties"]["Justification"].value = int(justification)   # 0 trái, 1 phải, 2 giữa
    _set_list(para["RunArray"], [copy.deepcopy(proto) for _ in lines])
    _set_list(para["RunLengthArray"], [Integer(n) for n in lens])

    # kiểu chữ: một run phủ toàn bộ văn bản
    style = ed["StyleRun"]
    sd = style["RunArray"][0]["StyleSheet"]["StyleSheetData"]
    sd["Font"].value = 0
    sd["FontSize"].value = float(size)
    sd["FauxBold"].value = False
    sd["FauxItalic"].value = False
    sd["AutoLeading"].value = False
    sd["Leading"].value = float(leading)
    sd["Tracking"].value = 0
    _set_list(sd["FillColor"]["Values"], [Float(1.0)] + [Float(max(0.0, min(1.0, c / 255.0))) for c in rgb[:3]])
    _set_list(style["RunLengthArray"], [Integer(sum(lens))])

    # font: FontSet[0] ở cả ResourceDict và DocumentResources
    for res in (eng["ResourceDict"], eng["DocumentResources"]):
        fs = res["FontSet"][0]
        fs["Name"].value = font_ps
        fs["FontType"].value = 1 if truetype else 0
        try:
            res["StyleSheetSet"][0]["StyleSheetData"]["Font"].value = 0
        except (KeyError, IndexError, TypeError):
            pass
    blk.transform = tuple(float(v) for v in matrix)
    if box is not None:
        bw, bh = float(box[0]), float(box[1])
        cookie = eng["EngineDict"]["Rendered"]["Shapes"]["Children"][0]["Cookie"]["Photoshop"]
        _set_list(cookie["BoxBounds"], [Float(0.0), Float(0.0), Float(bw), Float(bh)])
        for name, v in (("left", 0), ("top", 0), ("right", int(math.ceil(bw))), ("bottom", int(math.ceil(bh)))):
            if hasattr(blk, name):      # hình chữ nhật cuối khối TySh: psd-tools ghi bằng struct "4i" nên PHẢI là số nguyên
                setattr(blk, name, v)
    return blk


def describe_type_block(blk) -> dict:
    """Đọc lại các trường chính của khối TySh (để kiểm tra / thử nghiệm)."""
    eng = blk.text_data[b"EngineData"].value
    ed = eng["EngineDict"]
    sd = ed["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
    buf = io.BytesIO()
    eng.write(buf)
    return dict(
        text=str(blk.text_data[b"Txt "].value).rstrip("\x00"),
        editor_text=str(ed["Editor"]["Text"].value),
        font=str(eng["ResourceDict"]["FontSet"][0]["Name"].value),
        size=float(sd["FontSize"].value), leading=float(sd["Leading"].value),
        auto_leading=bool(sd["AutoLeading"].value),
        color=[float(v.value) for v in sd["FillColor"]["Values"]],
        justification=[int(p["ParagraphSheet"]["Properties"]["Justification"].value) for p in ed["ParagraphRun"]["RunArray"]],
        para_lengths=[int(v.value) for v in ed["ParagraphRun"]["RunLengthArray"]],
        style_lengths=[int(v.value) for v in ed["StyleRun"]["RunLengthArray"]],
        transform=tuple(blk.transform), engine_bytes=len(buf.getvalue()),
        box=_box_of(eng),
    )


def _box_of(eng):
    """(rộng, cao) của khung chữ khối, hoặc None nếu là chữ điểm."""
    try:
        cookie = eng["EngineDict"]["Rendered"]["Shapes"]["Children"][0]["Cookie"]["Photoshop"]
        bb = [float(v.value) for v in cookie["BoxBounds"]]
        return (bb[2] - bb[0], bb[3] - bb[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def attach_type_block(layer, blk) -> None:
    """Gắn khối TySh vào một lớp điểm ảnh: lớp trở thành lớp chữ (vẫn giữ ảnh chữ làm hình hiển thị)."""
    from psd_tools.constants import Tag
    from psd_tools.psd.tagged_blocks import TaggedBlock

    layer.tagged_blocks[Tag.TYPE_TOOL_OBJECT_SETTING] = TaggedBlock(key=Tag.TYPE_TOOL_OBJECT_SETTING, data=blk)
