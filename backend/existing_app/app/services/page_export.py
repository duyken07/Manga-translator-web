"""page_export.py - xuất một trang đã redraw thành ảnh cuối hoặc file PSD nhiều lớp.

PSD gồm (dưới -> trên):
    Anh goc                       ảnh gốc
    [nhóm] Xoa chu (redraw)       mỗi câu một lớp vùng đã xoá chữ gốc
    [nhóm] Fill (vung va)         mỗi vùng Content-Aware Fill một lớp, nếu có
    [nhóm] Net ve                 mỗi nét vẽ tay một lớp trong suốt, nếu có
    [nhóm] Chu dich               mỗi câu một lớp chữ dịch (ảnh trong suốt)
Lớp bị người dùng ẩn trong app (ẩn chữ / tắt vùng redraw) được giữ ở trạng thái ẩn trong PSD. Ảnh gộp trong file (cái mà
các trình xem không hỗ trợ lớp sẽ hiện) chính là ảnh người dùng đang thấy trong app.

Vì sao nhẹ: psd-tools mặc định tự ghép (composite) TOÀN BỘ lớp khi lưu để tạo ảnh gộp - chậm và tốn ~1 GB RAM cho trang 2400x3500.
Ở đây ta đưa sẵn ảnh gộp (app đã có hàm ghép nhanh) và chặn bước đó: ~0.5-2.5 s và ~250 MB RAM mỗi trang (đo trên trang 2400x3500).

Chữ dịch là lớp Type chữ khối (có khung bao, sửa chữ được trong Photoshop; xem psd_text.PSD_TEXT_BOX); PSD tối đa 30000 px mỗi cạnh; tên lớp chỉ dùng ASCII
(tên lớp kiểu cũ của PSD chỉ hỗ trợ MacRoman, tiếng Việt có dấu làm psd-tools lỗi).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from app.qt_compat import QObject, QRunnable, QThreadPool, Signal, Slot

from app.services.redraw import engine
from app.services import psd_text
from app.services.redraw import typeset
from app.services.strokes import render_strokes

log = logging.getLogger("page_export")

PSD_MAX_SIDE = 30000
_WORKER_STACK_BYTES = 32 * 1024 * 1024


class PageExportError(Exception):
    pass


# --------------------------------------------------------------------------- dữ liệu trang (snapshot an toàn đa luồng)
@dataclass
class PageExportData:
    page_name: str
    image_path: str
    layers_path: str = ""
    redraw_path: str = ""
    strokes: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    dialogues: list = field(default_factory=list)  # dict: id, text, hide_text, hide_redraw

    @property
    def visible_strokes(self) -> list:
        return [s for s in self.strokes if not s.get("hidden")]

    @property
    def has_content(self) -> bool:
        return bool((self.redraw_path and Path(self.redraw_path).is_file()) or self.strokes)

    @classmethod
    def from_page(cls, page) -> "PageExportData":
        """Chụp các trường cần thiết của Page (gọi trên luồng UI), để luồng nền xuất mà không đụng vào dữ liệu đang sửa."""
        return cls(
            page_name=page.page_name,
            image_path=page.image_path,
            layers_path=page.layers_path or "",
            redraw_path=page.redraw_path or "",
            strokes=copy.deepcopy(page.strokes or []),
            fills=copy.deepcopy(page.fills or []),
            dialogues=[dict(id=d.dialogue_id, text=d.translated_text or "", hide_text=bool(d.hide_text),
                            hide_redraw=bool(d.hide_redraw)) for d in page.dialogues],
        )


# --------------------------------------------------------------------------- ảnh cuối của trang
def render_page_image(data: PageExportData) -> Optional[np.ndarray]:
    """Ảnh cuối của trang (BGR) theo trạng thái ẩn / hiện hiện tại. Trang chưa redraw và chưa vẽ / chỉnh sửa gì vẫn xuất được:
    khi đó là chính ảnh trang (ảnh gốc, hoặc bản đã vẽ / fill trực tiếp). None chỉ khi không đọc được file ảnh."""
    hide_redraw = {d["id"] for d in data.dialogues if d["hide_redraw"]} | {-f["id"] for f in data.fills if f.get("hidden")}
    hide_text = {d["id"] for d in data.dialogues if d["hide_text"]}
    img = None
    if data.layers_path and Path(data.layers_path).is_file():
        img = engine.compose_from_cache(data.image_path, data.layers_path, hide_redraw=hide_redraw, strokes=data.visible_strokes,
                                        hide_text=hide_text)
    from_layers = img is not None
    if img is None and data.redraw_path and Path(data.redraw_path).is_file():
        img = engine.imread(data.redraw_path)  # chưa có lớp (cache cũ): dùng ảnh redraw đã ghép sẵn
    if img is None and data.image_path:
        img = engine.imread(data.image_path)  # trang chỉ có nét vẽ, hoặc chưa có gì: lấy ảnh trang (nét vẽ vẽ chồng lên bên dưới)
    if img is None:
        return None
    if data.visible_strokes and not from_layers:
        img = render_strokes(img, data.visible_strokes)
    return img


def write_image(img: np.ndarray, target: Path) -> None:
    """Ghi ảnh theo đuôi file (png / jpg / webp); dùng tofile để đường dẫn có dấu / Unicode vẫn ghi được."""
    ext = target.suffix.lower() or ".png"
    params: list = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise OSError(f"Không mã hoá được {target.name}")
    buf.tofile(str(target))


# --------------------------------------------------------------------------- thông tin chữ (để xuất lớp Type)
def load_text_blocks(layers_path: str) -> dict:
    """{id câu: thông tin bố cục chữ} từ layouts.json cạnh bộ lớp, hoặc {} nếu không có / không khớp bộ lớp (cache cũ, hỏng):
    khi đó các lớp chữ được xuất dưới dạng ảnh như trước."""
    if not layers_path:
        return {}
    try:
        lp = Path(layers_path)
        st = json.loads((lp.parent.parent / "layouts.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if st.get("layers") != lp.parent.name:      # bố cục phải thuộc đúng bộ lớp này
        return {}
    out = {}
    for k, v in st.get("blocks", {}).items():
        lay = v.get("lay") or {}
        if lay.get("lines") and lay.get("font_path") and v.get("color"):
            out[int(k)] = dict(lay=lay, color=v["color"], rot=float(v.get("rot") or 0.0))
    return out


def _type_block_for(info: dict, tile_xywh: tuple):
    """Khối TySh cho một khung chữ, hoặc None nếu font không đọc được tên PostScript."""
    lay = info["lay"]
    fi = psd_text.font_info(str(lay["font_path"]))
    if not fi:
        return None
    base = float(lay["top"]) + (float(lay["lh"]) - (float(lay["asc"]) + float(lay["desc"]))) / 2.0 + float(lay["asc"])
    x, y, w, h = tile_xywh
    pivot = (x + w / 2.0, y + h / 2.0)
    rgb = tuple(int(c) for c in info["color"][:3])
    align = lay.get("align") or "center"
    just = {"left": 0, "right": 1}.get(align, 2)
    # bề rộng dòng dài nhất: mép trái / phải của khối (căn lề trái / phải neo ở đó)
    try:
        fss = typeset.get_font(str(lay["font_path"]), int(lay["size"]) * 4)
        mw = max((fss.getlength(l) / 4.0 for l in lay["lines"]), default=0.0)
    except Exception:  # noqa: BLE001
        mw = 0.0
    if not psd_text.PSD_TEXT_BOX:
        ax = float(lay["cx"]) - (mw / 2.0 if align == "left" else (-mw / 2.0 if align == "right" else 0.0))
        matrix = psd_text.type_matrix(ax, base, info["rot"], pivot=pivot)
        return psd_text.build_type_block(list(lay["lines"]), fi[0], fi[1], float(lay["size"]), float(lay["lh"]), rgb, matrix,
                                         justification=just)
    # Chữ khối: khung bao căn giữa quanh cx, rộng hơn ảnh chữ một chút để Photoshop không ngắt thêm dòng (mỗi dòng của app là
    # một đoạn); mép trên khung = mép trên của dòng đầu (đường cơ sở - ascent).
    size, lh = float(lay["size"]), float(lay["lh"])
    cx, pad = float(lay["cx"]), max(6.0, 0.4 * size)
    half = max(cx - x, x + w - cx) + pad
    bx, by = cx - half, base - float(lay["asc"])
    bw, bh = 2.0 * half, len(lay["lines"]) * lh + 0.4 * size
    if align in ("left", "right") and mw > 0:   # khung ôm sát khối chữ, thừa chỗ về phía đối diện để Photoshop không ngắt thêm dòng
        bw = mw + 2.0 * pad
        bx = cx - mw / 2.0 if align == "left" else cx + mw / 2.0 - bw
    matrix = psd_text.type_matrix(bx, by, info["rot"], pivot=pivot)
    return psd_text.build_type_block(list(lay["lines"]), fi[0], fi[1], size, lh, rgb, matrix, box=(bw, bh), justification=just)


# --------------------------------------------------------------------------- PSD
def psd_available() -> tuple[bool, str]:
    """(có thể xuất PSD không, thông báo hướng dẫn nếu chưa)."""
    try:
        import psd_tools  # noqa: F401
    except ImportError:
        return False, "Chưa cài thư viện psd-tools. Chạy: pip install psd-tools"
    return True, ""


def _ascii_name(prefix: str, ident: int, text: str = "", limit: int = 30) -> str:
    """Tên lớp ASCII: 'Chu 03 - Gi chu tien thi...'. Bỏ dấu tiếng Việt (đ -> d) và ký tự lạ."""
    s = unicodedata.normalize("NFKD", (text or "").replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if c.isascii() and (c.isalnum() or c in " ,.!?'-_"))
    s = " ".join(s.split())[:limit].strip()
    head = f"{prefix} {abs(ident):02d}"
    return f"{head} - {s}" if s else head


def _strokes_layer(strokes: list, W: int, H: int):
    """Mọi nét vẽ -> MỘT lớp RGBA trong suốt (cắt theo khung bao của các nét). Trả (rgba, x, y) hoặc None.
    Vẽ cùng các nét lên nền đen và nền trắng rồi suy ra alpha / màu: alpha = 255 - (trắng - đen), màu = đen * 255 / alpha."""
    xs, ys, pad = [], [], 4
    for s in strokes:
        pts = s.get("pts") or []
        if pts:
            pad = max(pad, int(round(float(s.get("width", 5)))) + 4)
            xs += [p[0] for p in pts]
            ys += [p[1] for p in pts]
    if not xs:
        return None
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = min(W, max(xs) + pad + 1), min(H, max(ys) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    shifted = []
    for s in strokes:
        t = dict(s)
        t["pts"] = [[p[0] - x0, p[1] - y0] for p in (s.get("pts") or [])]
        shifted.append(t)
    on_black = render_strokes(np.zeros((y1 - y0, x1 - x0, 3), np.uint8), shifted)
    on_white = render_strokes(np.full((y1 - y0, x1 - x0, 3), 255, np.uint8), shifted)
    alpha = (255 - (on_white[..., 0].astype(np.int16) - on_black[..., 0].astype(np.int16))).clip(0, 255).astype(np.uint16)
    safe = np.maximum(alpha, 1)
    color = ((on_black.astype(np.uint16) * 255 + safe[..., None] // 2) // safe[..., None]).clip(0, 255).astype(np.uint8)
    color[alpha == 0] = 0
    rgba = np.dstack([color[..., 2], color[..., 1], color[..., 0], alpha.astype(np.uint8)])  # BGR -> RGB + A
    return rgba, x0, y0


def _merge_patches(specs: list, read_layer, W: int, H: int):
    """Ghép nhiều lớp vá (mỗi lớp {file, x, y}) thành MỘT ảnh RGBA trong suốt theo thứ tự đưa vào (sau đè trước). Trả (PIL RGBA, x, y) hoặc None."""
    from PIL import Image
    imgs = []
    for sp in specs:
        im = read_layer(sp)
        if im is not None:
            imgs.append((im, int(sp["x"]), int(sp["y"])))
    if not imgs:
        return None
    x0, y0 = max(0, min(x for _i, x, _y in imgs)), max(0, min(y for _i, _x, y in imgs))
    x1, y1 = min(W, max(x + i.width for i, x, _y in imgs)), min(H, max(y + i.height for i, _x, y in imgs))
    if x1 <= x0 or y1 <= y0:
        return None
    canvas = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    for im, x, y in imgs:
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(im, (x - x0, y - y0))
        canvas = Image.alpha_composite(canvas, layer)
    return canvas, x0, y0


def export_page_psd(data: PageExportData, out_path: Path, compression: str = "rle", editable_text: bool = True,
                    separate_layers: bool = True) -> int:
    """Xuất trang thành PSD. Trả số lớp. Ném PageExportError nếu không xuất được. compression: 'rle' (mặc định, tương thích
    nhất), 'zip' (file nhỏ hơn ~30% nhưng chậm hơn), 'raw'. separate_layers=True: mỗi vùng Fill và mỗi nét vẽ là MỘT lớp riêng (ẩn / sửa riêng được); False: gộp mọi Fill thành một lớp và mọi
    nét vẽ đang hiện thành một lớp. editable_text=True: lớp chữ dịch là LỚP CHỮ (Type) sửa được trong
    Photoshop (vẫn giữ ảnh chữ làm hình hiển thị); khung nào không dựng được thì giữ là lớp ảnh."""
    ok, msg = psd_available()
    if not ok:
        raise PageExportError(msg)
    from PIL import Image
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    from psd_tools.constants import Compression

    comp = {"rle": Compression.RLE, "zip": Compression.ZIP, "raw": Compression.RAW}.get(compression, Compression.RLE)
    merged = render_page_image(data)
    if merged is None:
        raise PageExportError(f"Không đọc được ảnh của trang: {data.image_path or '(không có đường dẫn)'}")
    H, W = merged.shape[:2]
    if max(W, H) > PSD_MAX_SIDE:
        raise PageExportError(f"Ảnh {W}×{H} vượt giới hạn PSD ({PSD_MAX_SIDE}px mỗi cạnh).")

    # Tài liệu ở chế độ RGBA (alpha ảnh gộp luôn đục 100%): nếu chỉ RGB, psd-tools lưu độ trong suốt của từng lớp dưới dạng
    # MẶT NẠ LỚP (khó sửa trong Photoshop). Có băng trong suốt thì mỗi lớp chữ / vùng xoá / nét vẽ trong suốt thật.
    psd = PSDImage.frompil(Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGBA)), comp)  # ảnh gộp = ảnh trong app
    del merged
    psd._update_preview = lambda *a, **k: None  # giữ ảnh gộp có sẵn, KHÔNG để psd-tools composite lại toàn bộ khi lưu
    # 72 ppi tường minh: Photoshop đổi cỡ chữ "pt" của lớp Type sang điểm ảnh theo độ phân giải tài liệu; 72 ppi thì 1 pt = 1 px,
    # đúng cỡ chữ (px) mà app đã dùng để vẽ.
    from psd_tools.constants import Resource
    from psd_tools.psd.image_resources import ImageResource, ResoulutionInfo
    psd.image_resources[Resource.RESOLUTION_INFO] = ImageResource(
        key=Resource.RESOLUTION_INFO, data=ResoulutionInfo(72 << 16, 1, 1, 72 << 16, 1, 1))  # (ppi 16.16, đơn vị inch, ...)
    n_layers = 0

    orig = engine.imread(data.image_path)
    if orig is None:
        raise PageExportError(f"Không đọc được ảnh gốc: {data.image_path}")
    psd.append(PixelLayer.frompil(Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)), psd, "Anh goc", 0, 0, comp))
    del orig
    n_layers += 1

    by_id = {d["id"]: d for d in data.dialogues}
    fills_by_id = {-int(f["id"]): f for f in data.fills if "id" in f}
    meta = None
    if data.layers_path and Path(data.layers_path).is_file():
        try:
            meta = json.loads(Path(data.layers_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
    base = Path(data.layers_path).parent if meta else None

    def read_layer(spec) -> Optional[Image.Image]:
        bgra = engine.imread(base / spec["file"], cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            return None
        return Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))

    layers = (meta or {}).get("layers", [])
    if layers:
        patches = [L for L in layers if "patch" in L]
        erase = sorted((L for L in patches if L["id"] > 0), key=lambda L: L["id"])
        fills = sorted((L for L in patches if L["id"] < 0), key=lambda L: -L["id"])   # vùng Content-Aware Fill (id = -fill_id)
        if not separate_layers and fills:
            # Gộp: mọi vùng Fill ĐANG HIỆN ghép thành MỘT lớp (vùng ẩn trong app bị bỏ qua), nằm trên lớp xoá chữ
            merged = _merge_patches([L["patch"] for L in fills if not fills_by_id.get(L["id"], {}).get("hidden")], read_layer, W, H)
            if merged is not None:
                img, mx, my = merged
                fgrp = psd.create_group(name="Fill (vung va)")
                fgrp.append(PixelLayer.frompil(img, psd, "Fill (gop)", my, mx, comp))
                n_layers += 1
            fills = []
        for group_name, items in (("Xoa chu (redraw)", erase), ("Fill (vung va)", fills)):   # xoá chữ ở dưới, mỗi vùng fill một lớp ở trên
            grp = None
            for L in items:
                img = read_layer(L["patch"])
                if img is None:
                    continue
                did = L["id"]
                if did < 0:
                    name, hidden = _ascii_name("Fill", did), bool(fills_by_id.get(did, {}).get("hidden"))
                else:
                    name, hidden = _ascii_name("Xoa", did), bool(by_id.get(did, {}).get("hide_redraw"))
                if grp is None:
                    grp = psd.create_group(name=group_name)
                lay = PixelLayer.frompil(img, psd, name, L["patch"]["y"], L["patch"]["x"], comp)
                lay.visible = not hidden
                grp.append(lay)
                n_layers += 1
    elif data.redraw_path and Path(data.redraw_path).is_file():  # cache cũ chưa tách lớp: một lớp redraw đã ghép
        flat = engine.imread(data.redraw_path)
        if flat is not None:
            psd.append(PixelLayer.frompil(Image.fromarray(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)), psd, "Redraw", 0, 0, comp))
            n_layers += 1

    if data.strokes and not separate_layers:
        visible = sorted(data.visible_strokes, key=lambda t: t.get("id", 0))   # nét ẩn trong app không vào lớp gộp
        sl = _strokes_layer(visible, W, H) if visible else None
        if sl is not None:
            rgba, x0, y0 = sl
            sgrp = psd.create_group(name="Net ve")
            sgrp.append(PixelLayer.frompil(Image.fromarray(rgba, "RGBA"), psd, "Ve (gop)", y0, x0, comp))
            n_layers += 1
    elif data.strokes:
        sgrp = None
        tool_names = {"pen": "but", "marker": "but da", "highlighter": "da quang"}
        for s in sorted(data.strokes, key=lambda t: t.get("id", 0)):   # mỗi nét vẽ một lớp trong suốt riêng (nét ẩn trong app = lớp ẩn)
            sl = _strokes_layer([s], W, H)
            if sl is None:
                continue
            if sgrp is None:
                sgrp = psd.create_group(name="Net ve")
            rgba, x0, y0 = sl
            lay = PixelLayer.frompil(Image.fromarray(rgba, "RGBA"), psd, _ascii_name("Ve", int(s.get("id", 0)), tool_names.get(s.get("tool"), "")),
                                     y0, x0, comp)
            lay.visible = not s.get("hidden")
            sgrp.append(lay)
            n_layers += 1

    texts = sorted((L for L in layers if "text" in L), key=lambda L: L["id"])
    n_type = 0
    if texts:
        text_info = load_text_blocks(data.layers_path) if editable_text else {}
        grp = psd.create_group(name="Chu dich")
        for L in texts:
            img = read_layer(L["text"])
            if img is None:
                continue
            d = by_id.get(L["id"], {})
            lay = PixelLayer.frompil(img, psd, _ascii_name("Chu", L["id"], d.get("text", "")), L["text"]["y"], L["text"]["x"], comp)
            lay.visible = not d.get("hide_text", False)
            info = text_info.get(L["id"])
            if info is not None:
                try:
                    blk = _type_block_for(info, (L["text"]["x"], L["text"]["y"], img.width, img.height))
                    if blk is not None:
                        psd_text.attach_type_block(lay, blk)
                        n_type += 1
                except Exception:  # noqa: BLE001 - không dựng được lớp chữ thì giữ lớp ảnh, không làm hỏng cả file
                    log.exception("Không dựng được lớp chữ cho câu %s; giữ lớp ảnh", L["id"])
            grp.append(lay)
            n_layers += 1
    log.info("PSD %s: %d lớp (%d lớp chữ sửa được)", Path(out_path).name, n_layers, n_type)

    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        psd.save(str(tmp))
        os.replace(tmp, out_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise PageExportError(f"Không ghi được {out_path.name}: {exc}") from exc
    return n_layers


# --------------------------------------------------------------------------- xuất nền (có tiến độ, có huỷ)
class _Signals(QObject):
    progress = Signal(int, int, str)       # đã xong, tổng, tên trang đang làm
    finished = Signal(list, list, bool)    # [(tên trang, đường dẫn, số lớp)], [(tên trang, lỗi)], bị huỷ


class PsdBatchTask(QRunnable):
    """Xuất lần lượt từng trang (tuần tự: RAM chỉ tốn cho 1 trang tại một thời điểm)."""

    def __init__(self, jobs: list, compression: str = "rle", editable_text: bool = True, separate_layers: bool = True):
        super().__init__()
        self.jobs = jobs  # [(PageExportData, Path đích)]
        self.compression = compression
        self.editable_text = editable_text
        self.separate_layers = separate_layers   # False: gộp mọi Fill / nét vẽ thành một lớp mỗi loại
        self.cancel_event = threading.Event()
        self.signals = _Signals()
        self.setAutoDelete(False)  # signals nằm trong task: để Python giữ vòng đời, không để Qt xoá sớm

    @Slot()
    def run(self) -> None:
        done, failed = [], []
        total = len(self.jobs)
        for i, (data, target) in enumerate(self.jobs):
            if self.cancel_event.is_set():
                break
            self.signals.progress.emit(i, total, data.page_name)
            try:
                n = export_page_psd(data, target, self.compression, self.editable_text, self.separate_layers)
                done.append((data.page_name, str(target), n))
            except Exception as exc:  # noqa: BLE001 - một trang lỗi không làm hỏng cả đợt
                log.exception("Xuất PSD thất bại: %s", data.page_name)
                failed.append((data.page_name, str(exc)))
        self.signals.progress.emit(len(done) + len(failed), total, "")
        self.signals.finished.emit(done, failed, self.cancel_event.is_set())


_pool: Optional[QThreadPool] = None


def start_psd_export(task: PsdBatchTask) -> None:
    global _pool
    if _pool is None:
        _pool = QThreadPool()
        _pool.setMaxThreadCount(1)
        _pool.setStackSize(_WORKER_STACK_BYTES)  # như các pool nền khác: stack 1 MB mặc định quá nhỏ cho code native
    _pool.start(task)


def unique_targets(folder: Path, names: list, suffix: str = ".psd") -> list:
    """Tên file đích không trùng nhau trong 1 đợt xuất (hai trang cùng tên gốc -> thêm _2, _3...)."""
    used: dict = {}
    out = []
    for n in names:
        stem = Path(n).stem
        k = used.get(stem, 0) + 1
        used[stem] = k
        out.append(folder / (f"{stem}{suffix}" if k == 1 else f"{stem}_{k}{suffix}"))
    return out


def wait_psd_export(msecs: int = 8000) -> None:
    """Chờ ngắn cho trang đang xuất dở xong (gọi khi đóng app, tránh tắt giữa lúc đang ghi file)."""
    if _pool is not None:
        _pool.waitForDone(msecs)
