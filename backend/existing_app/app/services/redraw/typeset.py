"""
typeset.py - đặt chữ tiếng Việt vào trang đã xoá chữ gốc.

Phần xếp chữ lấy từ manga_cpu_pipeline_v4.py, bổ sung:
  * chia bóng thoại dùng chung: nhiều câu thoại cùng nằm trong 1 bóng thì mỗi câu
    chỉ được dùng phần bóng gần nó nhất (partition_shared_bubbles);
  * xử lý chữ lấn nhau: sau khi xếp xong, nếu 2 khối chữ chồng lên nhau thì
    dịch chuyển khối chữ trong phạm vi cho phép, nếu vẫn lấn thì giảm cỡ chữ (resolve_overlaps);
  * font mặc định có fallback sang font hệ thống hỗ trợ tiếng Việt.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .stage_a import allowed_rect, clamp_bbox, ellipse_kernel, page_gutters, parse_color

log = logging.getLogger("redraw.typeset")

VI_SAMPLE = "ăâđêôơưấềểỗặịựỳỹ"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]          # .../manga_translator

REGULAR_NAMES = ["000 WildWords2 TB Roman.ttf"]
BOLD_NAMES = ["000 ObelixPro TB.ttf"]
FALLBACK_REGULAR = ["segoeui.ttf", "arial.ttf", "Arial.ttf", "tahoma.ttf", "DejaVuSans.ttf",
                    "NotoSans-Regular.ttf", "LiberationSans-Regular.ttf", "Roboto-Regular.ttf"]
FALLBACK_BOLD = ["segoeuib.ttf", "arialbd.ttf", "Arial Bold.ttf", "tahomabd.ttf", "DejaVuSans-Bold.ttf",
                 "NotoSans-Bold.ttf", "LiberationSans-Bold.ttf", "Roboto-Bold.ttf"]


# --------------------------------------------------------------------------- fonts
@lru_cache(maxsize=512)
def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=50000)
def text_len(path: str, size_ss: int, text: str) -> float:
    return get_font(path, size_ss).getlength(text)


def _glyph_sig(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    img = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(img).text((10, 10), ch, font=font, fill=255)
    return img.tobytes()


@lru_cache(maxsize=4096)  # đủ chứa mọi font hệ thống: list_fonts() không được làm các lần kiểm tra sau bị đẩy khỏi cache
def supports_vietnamese(path: str) -> bool:
    try:
        font = ImageFont.truetype(path, 40)
        notdef = _glyph_sig(font, "\uffff")
        blank = bytes(96 * 96)
        for ch in VI_SAMPLE:
            sig = _glyph_sig(font, ch)
            if sig == blank or sig == notdef:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


_FONT_ONLY_DIR = [""]   # thư mục DUY NHẤT được dùng làm nguồn font (rỗng = ./fonts + font hệ thống như mặc định)


def set_font_only_dir(path: str) -> bool:
    """Đặt / bỏ chế độ "chỉ dùng font trong thư mục này". Trả True nếu giá trị đổi (đã xoá cache danh sách font)."""
    p = (path or "").strip()
    if p == _FONT_ONLY_DIR[0]:
        return False
    _FONT_ONLY_DIR[0] = p
    list_fonts.cache_clear()
    _find_font.cache_clear()
    return True


def font_only_dir() -> Path | None:
    p = _FONT_ONLY_DIR[0]
    return Path(p) if p and Path(p).is_dir() else None


def is_in_font_dir(path: str) -> bool:
    only = font_only_dir()
    if only is None:
        return True
    try:
        return Path(path).resolve().is_relative_to(only.resolve())
    except Exception:  # noqa: BLE001
        return False


def font_dirs(free: bool = False) -> list[Path]:
    only = None if free else font_only_dir()
    if only is not None:
        return [only]
    home = Path.home()
    dirs = [
        PROJECT_ROOT / "fonts", SCRIPT_DIR / "fonts", Path.cwd() / "fonts",
        Path(r"C:\Windows\Fonts"), home / "AppData/Local/Microsoft/Windows/Fonts",
        Path("/Library/Fonts"), Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"), home / "Library/Fonts",
        Path("/usr/share/fonts"), home / ".fonts", home / ".local/share/fonts",
    ]
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=32)
def _find_font(names: tuple[str, ...], free: bool = False) -> str | None:
    dirs = font_dirs(free)
    for name in names:
        for d in dirs:
            direct = d / name
            hits = [direct] if direct.is_file() else list(d.rglob(name))
            for hit in hits:
                if supports_vietnamese(str(hit)):
                    return str(hit)
    return None


def find_font(names: list[str]) -> str | None:
    return _find_font(tuple(names))


@lru_cache(maxsize=256)
def font_display_name(path: str) -> str:
    """Tên hiển thị của font (VD 'DejaVu Sans Bold'); rỗng nếu không đọc được."""
    try:
        fam, style = get_font(path, 20).getname()
        return fam if style.lower() in ("regular", "normal", "book") else f"{fam} {style}"
    except Exception:  # noqa: BLE001
        return Path(path).stem if path else ""


_scan_ctx = threading.local()


def _yield_cpu() -> None:
    """Luồng quét font NỀN nhường CPU / GIL sau mỗi font: quét liền tay hàng trăm font bằng Python sẽ chiếm GIL nhiều giây và làm
    giao diện giật liên tục. Gọi trực tiếp (người dùng đang chờ) thì không nhường để xong nhanh nhất."""
    if getattr(_scan_ctx, "background", False):
        time.sleep(0.01)


@lru_cache(maxsize=1)
def list_fonts() -> tuple:
    """((tên hiển thị, đường dẫn), ...) các font hỗ trợ tiếng Việt trong ./fonts và thư mục font hệ thống."""
    seen, out = set(), []
    for d in font_dirs():
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            for f in sorted(d.rglob(ext)):
                key = f.name.lower()
                if key in seen:
                    continue
                seen.add(key)
                _yield_cpu()
                if supports_vietnamese(str(f)):
                    try:
                        fam, style = get_font(str(f), 20).getname()
                        name = fam if style.lower() in ("regular", "normal", "book") else f"{fam} {style}"
                    except Exception:  # noqa: BLE001
                        name = f.stem
                    out.append((name, str(f)))
    return tuple(sorted(out, key=lambda t: t[0].lower()))


def fonts_ready() -> bool:
    """list_fonts() đã quét xong chưa (gọi lúc đó sẽ trả về ngay, không chặn)."""
    return list_fonts.cache_info().currsize > 0


def prewarm_fonts() -> None:
    """Quét font hệ thống ở luồng nền (an toàn gọi nhiều lần). Lần đầu trên Windows phải mở và vẽ thử hàng trăm font; làm ở
    luồng giao diện sẽ làm app 'đứng' vài giây khiến Windows báo Not Responding."""
    if fonts_ready():
        return
    threading.Thread(target=scan_fonts_background, name="font-scan", daemon=True).start()


def scan_fonts_background() -> tuple:
    """list_fonts() nhưng nhường CPU giữa các font (dùng trong luồng nền). Kết quả vào cache chung của list_fonts."""
    _scan_ctx.background = True
    try:
        return list_fonts()
    finally:
        _scan_ctx.background = False


def resolve_fonts(regular: str = "", bold: str = "") -> tuple[str | None, str | None]:
    """(regular, bold). Đường dẫn do người dùng chọn được ưu tiên, sau đó font TB của dự án, rồi font hệ thống."""
    only = font_only_dir()
    reg = regular if regular and Path(regular).is_file() and supports_vietnamese(regular) and is_in_font_dir(regular) else None
    if regular and not reg:
        log.warning("Font '%s' không dùng được (thiếu file, không có dấu tiếng Việt%s), dùng font mặc định.", regular,
                    " hoặc nằm ngoài thư mục font đã chọn" if only else "")
    bld = bold if bold and Path(bold).is_file() and supports_vietnamese(bold) and is_in_font_dir(bold) else None
    reg = reg or find_font(REGULAR_NAMES) or find_font(FALLBACK_REGULAR)
    bld = bld or find_font(BOLD_NAMES) or find_font(FALLBACK_BOLD)
    if only is not None:
        # chế độ chỉ-thư-mục: không có tên font quen thuộc thì lấy font có dấu tiếng Việt đầu tiên trong thư mục (ưu tiên kiểu đậm cho bold)
        listed = list_fonts()
        if reg is None and listed:
            reg = next((p for n, p in listed if "bold" not in n.lower() and "italic" not in n.lower()), listed[0][1])
        if bld is None and listed:
            bld = next((p for n, p in listed if "bold" in n.lower() and "italic" not in n.lower()), reg)
        if reg is None:
            log.warning("Thư mục font '%s' không có font nào hỗ trợ tiếng Việt, tạm dùng font mặc định.", only)
            reg = _find_font(tuple(REGULAR_NAMES), True) or _find_font(tuple(FALLBACK_REGULAR), True)
            bld = bld or _find_font(tuple(BOLD_NAMES), True) or _find_font(tuple(FALLBACK_BOLD), True)
    return reg, bld


# --------------------------------------------------------------------------- line breaking
def row_halfwidths(safe: np.ndarray, cx: int) -> np.ndarray:
    h, w = safe.shape
    half = np.zeros(h, np.int32)
    for y in range(h):
        row = safe[y]
        if not row[cx]:
            continue
        lz = np.flatnonzero(~row[:cx])
        rz = np.flatnonzero(~row[cx + 1:])
        left = lz[-1] + 1 if lz.size else 0
        right = cx + rz[0] if rz.size else w - 1
        half[y] = min(cx - left, right - cx)
    return half


def prepare_shape(b, margin: int, limit: np.ndarray | None = None) -> None:
    """Tính vùng an toàn (safe), tâm (cx, cy) và nửa bề rộng từng hàng của bóng thoại.

    limit: mask (cùng toạ độ ROI với b.interior) giới hạn phần bóng mà câu thoại này được dùng.
    """
    interior = b.interior if limit is None else (b.interior & limit)
    if int(interior.sum()) < 50:
        interior = b.interior
    safe = cv2.erode(interior.astype(np.uint8), ellipse_kernel(margin)) > 0
    if safe.sum() < 50:
        safe = interior.copy()
    b.safe = safe
    dt = cv2.distanceTransform(safe.astype(np.uint8), cv2.DIST_L2, 5)
    mx = float(dt.max())
    b.inr = mx
    if mx <= 0:                                           # bóng quá nhỏ: dùng tâm bbox, half = 0 -> fallback layout
        h, w = safe.shape
        b.cx, b.cy = w // 2, h // 2
        b.half = np.zeros(h, np.int32)
        return
    ys, xs = np.nonzero(dt >= 0.55 * mx)
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    if not safe[cy, cx]:
        cy, cx = np.unravel_index(int(dt.argmax()), dt.shape)
    b.cx, b.cy = int(cx), int(cy)
    b.half = row_halfwidths(safe, b.cx)


GLUE = "\u200b"   # tiền tố của mảnh từ ghép đứng liền sau mảnh trước (không chèn dấu cách khi nối dòng)


def _split_hyphen(word: str) -> list:
    """'Hanna-san' -> ['Hanna-', GLUE+'san']; từ không có gạch nối ở giữa thì giữ nguyên."""
    parts = [p for p in word.split("-")]
    if len(parts) < 2 or any(not p for p in parts):
        return [word]
    return [p + "-" for p in parts[:-1]][:1] + [GLUE + p + ("-" if i < len(parts) - 2 else "") for i, p in enumerate(parts[1:])]


def paragraphs(text: str, split_hyphen: bool = False) -> list:
    """Tách bản dịch thành các đoạn theo ký tự xuống dòng (người dùng gõ Shift+Enter trong bảng thoại); mỗi đoạn là list
    từ. Mỗi đoạn luôn bắt đầu ở một dòng mới; các dòng trống bị bỏ qua. Văn bản không có xuống dòng -> đúng 1 đoạn
    (giống hệt hành vi cũ `text.split()`). split_hyphen=True: từ ghép có gạch nối được tách thành các mảnh (xem GLUE)."""
    norm = str(text).replace("\r\n", "\n").replace("\r", "\n")
    paras = [ln.split() for ln in norm.split("\n")]
    if split_hyphen:
        paras = [[t for w in ws for t in _split_hyphen(w)] for ws in paras]
    return [w for w in paras if w]


def plain(word: str) -> str:
    """Chữ thật của 1 token (bỏ tiền tố GLUE) - dùng khi đo bề rộng / vẽ."""
    return word[1:] if word.startswith(GLUE) else word


def join_line(words: list) -> str:
    out = ""
    for w in words:
        out += plain(w) if (w.startswith(GLUE) or not out) else " " + w
    return out


def flat_words(paras: list) -> list:
    return [w for p in paras for w in p]


def plain_wrap(paras: list, measure, width: float) -> list:
    """Xuống dòng đơn giản theo bề ngang cố định (bố cục dự phòng khi không vừa khung); tôn trọng xuống dòng cứng."""
    lines: list = []
    for words in paras:
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip()
            if cur and measure(cand) > width:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        lines.append(cur)
    return lines


def wrap_paras(paras, wlen, space, spans):
    """Như balanced_wrap nhưng cho nhiều đoạn: mỗi đoạn (xuống dòng cứng) bắt đầu ở dòng mới. None nếu không đủ dòng /
    bề ngang."""
    if len(paras) == 1:
        return balanced_wrap(paras[0], wlen, space, spans)
    mins, off = [], 0
    for para in paras:  # số dòng tối thiểu của từng đoạn khi xếp liên tiếp
        r = greedy_wrap(para, wlen, space, spans[off:], 1.0)
        if r is None:
            return None
        mins.append(len(r))
        off += len(r)
    lines, off = [], 0
    for k, para in enumerate(paras):  # cân đều từng đoạn nhưng chừa đủ dòng cho các đoạn sau
        avail = spans[off: len(spans) - sum(mins[k + 1:])]
        r = balanced_wrap(para, wlen, space, avail) if avail else None
        if r is None:
            lines = []
            break
        lines += r
        off += len(r)
    if lines:
        return lines
    lines, off = [], 0  # cân đều không được (hình bóng thoại lệch) -> xếp tham lam từng đoạn
    for para in paras:
        r = greedy_wrap(para, wlen, space, spans[off:], 1.0)
        if r is None:
            return None
        lines += r
        off += len(r)
    return lines


def greedy_wrap(words, wlen, space, spans, f):
    n = len(spans)
    lines, cur, curw, i = [], [], 0.0, 0
    for w in words:
        ww = wlen[w]
        if cur:
            nw = curw + (0 if w.startswith(GLUE) else space) + ww
            if nw <= spans[i] * f:
                cur.append(w)
                curw = nw
                continue
            lines.append(join_line(cur))
            i += 1
            cur = []
        if i >= n or ww > spans[i]:
            return None
        cur, curw = [w], ww
    lines.append(join_line(cur))
    return lines


def balanced_wrap(words, wlen, space, spans):
    best = greedy_wrap(words, wlen, space, spans, 1.0)
    if best is None:
        return None
    lo, hi = 0.4, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        res = greedy_wrap(words, wlen, space, spans, mid)
        if res is not None:
            best, hi = res, mid
        else:
            lo = mid
    return best


def spans_for(half: np.ndarray, top: int, n: int, lh: int):
    if top < 0 or top + n * lh > len(half):
        return None
    s = [2 * int(half[top + i * lh: top + (i + 1) * lh].min()) for i in range(n)]
    return s if min(s) > 0 else None


LINE_SPACING_RANGE = (0.7, 2.5)


def line_spacing_of(b, a) -> float:
    """Hệ số giãn dòng của khối chữ: override riêng của câu (b.line_spacing) hoặc mặc định chung a.line_spacing."""
    v = getattr(b, "line_spacing", None) or a.line_spacing
    return float(min(max(v, LINE_SPACING_RANGE[0]), LINE_SPACING_RANGE[1]))


def halo_width(b, size: int, a) -> int:
    """Độ dày viền chữ (px). Viền là tuỳ chọn của người dùng, mặc định TẮT (không còn tự thêm viền cho chữ nằm trên tranh)."""
    if not getattr(b, "outline", False):
        return 0
    return max(1, int(getattr(b, "outline_width", 2) or 2))


def try_layout(b, size: int, font_path: str, sw: int, ls: float, ss: int, dy: int = 0):
    """Xếp chữ trong bóng thoại (theo hình dạng). dy: dịch cả khối chữ lên/xuống dy px quanh tâm bóng."""
    paras = paragraphs(b.text, getattr(b, "split_hyphen", False))
    words = flat_words(paras)
    if not words:
        return None
    fss = get_font(font_path, size * ss)
    asc, desc = fss.getmetrics()
    asc, desc = asc / ss, desc / ss
    lh = max(1, int(round((asc + desc) * ls)))
    wlen = {w: text_len(font_path, size * ss, plain(w)) / ss + 2 * sw for w in words}
    space = text_len(font_path, size * ss, " ") / ss
    max_n = max(1, len(b.half) // lh)
    for n in range(1, max_n + 1):
        top = int(round(b.cy + dy - n * lh / 2))
        spans = spans_for(b.half, top, n, lh)
        if spans is None:
            continue
        lines = wrap_paras(paras, wlen, space, spans)
        if lines is None:
            continue
        m = len(lines)
        if m < n:
            top = int(round(b.cy + dy - m * lh / 2))
            sp2 = spans_for(b.half, top, m, lh)
            if sp2 is None or any(text_len(font_path, size * ss, l) / ss + 2 * sw > s for l, s in zip(lines, sp2)):
                continue
        return dict(lines=lines, lh=lh, asc=asc, desc=desc, top=top, size=size,
                    font_path=font_path, sw=sw, fits=True,
                    spans=list(sp2 if m < n else spans))   # bề rộng cho phép của từng dòng (dùng khi căn lề trái / phải)
    return None


def finish_shape_layout(b, lay: dict, a) -> dict:
    lay["cx"] = b.ox + b.cx
    lay["top"] = b.oy + lay["top"]
    lay["hw"] = halo_width(b, lay["size"], a)
    lay["mode"] = "shape"
    return lay


def fallback_layout(b, size: int, font_path: str, sw: int, ls: float, ss: int) -> dict:
    paras = paragraphs(b.text)
    fss = get_font(font_path, size * ss)
    asc, desc = (v / ss for v in fss.getmetrics())
    lh = max(1, int(round((asc + desc) * ls)))
    width = max(2 * int(b.half.max()), int(b.interior.shape[1] * 0.6))
    lines = plain_wrap(paras, lambda t: fss.getlength(t) / ss, width)
    top = int(round(b.cy - len(lines) * lh / 2))
    return dict(lines=lines, lh=lh, asc=asc, desc=desc, top=top, size=size,
                font_path=font_path, sw=sw, fits=False)


def fit_rect(b, size_top: int, font_path: str, fake_sw, a, ss: int, floor: int | None = None) -> dict | None:
    """Xếp chữ trong hình chữ nhật b.allowed (chữ tự do / region do người dùng chỉ định)."""
    paras = paragraphs(b.text, getattr(b, "split_hyphen", False))
    words = flat_words(paras)
    if not words or b.allowed is None:
        return None
    ax1, ay1, ax2, ay2 = b.allowed
    ox1, oy1, ox2, oy2 = b.region if b.region else b.occ
    core_w = max(1, ox2 - ox1)
    max_w, max_h = ax2 - ax1, ay2 - ay1
    for size in range(size_top, (floor if floor is not None else a.min_font) - 1, -1):
        sw, hw = fake_sw(size), halo_width(b, size, a)
        pad = 2 * max(sw, hw)
        fss = get_font(font_path, size * ss)
        asc, desc = (v / ss for v in fss.getmetrics())
        lh = max(1, int(round((asc + desc) * line_spacing_of(b, a))) + (hw if hw else 0) // 2)
        max_n = int(max_h // lh)
        if max_n < 1:
            continue
        wlen = {w: fss.getlength(plain(w)) / ss + pad for w in words}
        space = fss.getlength(" ") / ss
        if max(wlen.values()) > max_w:
            continue
        lo = max(core_w, 4 * size)
        best_pick = None
        for W in np.linspace(min(lo, max_w), max_w, 13):
            for n in range(1, max_n + 1):
                lines = wrap_paras(paras, wlen, space, [float(W)] * n)
                if lines:
                    break
            else:
                continue
            widths = [fss.getlength(l) / ss for l in lines]
            bw = max(widths) + pad
            hh = len(lines) * lh
            # Điểm càng thấp càng tốt: khối chữ gần vuông / hơi ngang (kiểu bóng thoại) và các dòng dài gần bằng nhau.
            # Bản cũ lấy bề rộng hẹp nhất còn vừa -> ra cột chữ dọc 6-10 dòng rất xấu trong khung chữ gốc hẹp (chữ Nhật dọc).
            aspect = math.log(max(bw, 1.0) / max(hh, 1.0) / 1.2)
            ragged = 1.0 - (sum(widths) / len(widths)) / max(max(widths), 1.0)
            score = abs(aspect) + 0.8 * ragged + 0.05 * (len(lines) > 1 and lines[-1] and len(lines[-1]) < 3)
            if best_pick is None or score < best_pick[0] - 1e-6:
                best_pick = (score, lines, bw, hh)
        if best_pick is None:
            continue
        _, lines, bw, hh = best_pick
        cx = (ox1 + ox2) / 2
        cx = min(max(cx, ax1 + bw / 2), ax2 - bw / 2)
        cy = (oy1 + oy2) / 2
        top = min(max(cy - hh / 2, ay1), ay2 - hh)
        return dict(lines=lines, lh=lh, asc=asc, desc=desc, size=size, font_path=font_path,
                    sw=sw, hw=hw, fits=True, cx=cx, top=top, mode="rect")
    return None


OVAL_K = 1.2   # bán trục ellipse = OVAL_K x (nửa rộng, nửa cao) khung: dòng giữa dùng trọn bề rộng khung, dòng đầu / cuối co lại -> khối chữ tròn


def fit_oval(b, size_top: int, font_path: str, fake_sw, a, ss: int, floor: int | None = None) -> dict | None:
    """Khung chữ do người dùng đặt (b.region): xếp chữ thành khối hình OVAL trong khung.

    Bề rộng cho phép của từng dòng = dây cung của ellipse (đồng tâm với khung, bán trục OVAL_K x nửa khung) tại điểm xa tâm nhất của dòng,
    nhưng không vượt bề rộng khung. Cách ngắt dòng CHỈ phụ thuộc kích thước khung + nội dung (không phụ thuộc vị trí), nên dời khung
    đi đâu cũng ra đúng một kiểu trình bày. Cỡ chữ = lớn nhất còn vừa khung. None nếu không vừa ở cỡ nào (rơi về fit_rect).
    """
    paras = paragraphs(b.text, getattr(b, "split_hyphen", False))
    words = flat_words(paras)
    if not words or b.allowed is None:
        return None
    ax1, ay1, ax2, ay2 = b.allowed
    wd, hd = float(ax2 - ax1), float(ay2 - ay1)
    cx, cy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    ra, rb = OVAL_K * wd / 2.0, OVAL_K * hd / 2.0

    def chord(y0: float, y1: float) -> float:
        u = max(abs(y0 - cy), abs(y1 - cy)) / rb          # dải dòng [y0, y1]: lấy chỗ xa tâm nhất (hẹp nhất)
        return 0.0 if u >= 1.0 else 2.0 * ra * math.sqrt(1.0 - u * u)

    def spans_at(top: float, n: int, lh: int) -> list:
        return [min(wd, chord(top + i * lh, top + (i + 1) * lh)) for i in range(n)]

    for size in range(size_top, (floor if floor is not None else a.min_font) - 1, -1):
        sw, hw = fake_sw(size), halo_width(b, size, a)
        pad = 2 * max(sw, hw)
        fss = get_font(font_path, size * ss)
        asc, desc = (v / ss for v in fss.getmetrics())
        lh = max(1, int(round((asc + desc) * line_spacing_of(b, a))) + (hw if hw else 0) // 2)
        max_n = int(hd // lh)
        if max_n < 1:
            continue
        wlen = {w: fss.getlength(plain(w)) / ss + pad for w in words}
        space = fss.getlength(" ") / ss
        if max(wlen.values()) > wd:
            continue
        for n in range(1, max_n + 1):
            top = cy - n * lh / 2.0
            spans = spans_at(top, n, lh)
            if min(spans) <= 0:
                continue
            lines = wrap_paras(paras, wlen, space, spans)
            if lines is None:
                continue
            m = len(lines)
            if m < n:                                           # cân đều ra ít dòng hơn dự kiến: căn lại giữa khung rồi kiểm tra lại
                top = cy - m * lh / 2.0
                spans = spans_at(top, m, lh)
                if min(spans) <= 0 or any(fss.getlength(l) / ss + pad > s for l, s in zip(lines, spans)):
                    continue
            return dict(lines=lines, lh=lh, asc=asc, desc=desc, size=size, font_path=font_path, sw=sw, hw=hw,
                        fits=True, cx=cx, top=top, mode="rect", spans=list(spans))
    return None


# --------------------------------------------------------------------------- render
ALIGNS = ("left", "center", "right")


def line_anchor(lay: dict, i: int, line: str, fss, ss: int, mw: float) -> tuple:
    """(x, anchor PIL) của dòng i theo căn lề của khối. Giữa: neo 'ms' tại cx. Trái / phải: neo 'ls' / 'rs' tại mép trái / phải của khối
    (rộng mw, căn giữa quanh cx); trong bóng thoại (có lay['spans']) mép bị kẹp để dòng không thò ra ngoài hình bóng ở hàng hẹp."""
    al = lay.get("align") or "center"
    cx = lay["cx"]
    if al not in ("left", "right"):
        return cx, "ms"
    w = fss.getlength(line) / ss
    sw = lay.get("sw", 0)
    sp = lay.get("spans")
    half = sp[i] / 2.0 if sp and i < len(sp) else mw / 2.0
    half = max(half, w / 2.0 + sw)
    lo, hi = cx - half + sw, cx + half - sw      # đoạn cho phép của dòng này
    if al == "left":
        return min(max(cx - mw / 2.0, lo), hi - w), "ls"
    return max(min(cx + mw / 2.0, hi), lo + w), "rs"


def _block_masks(lay: dict, ss: int, H: int, W: int, clip: bool = True):
    """(x1, y1, at, ah): toạ độ góc trái-trên của ô ảnh chứa khối chữ, mặt nạ chữ và mặt nạ viền (float 0..1, HxWx1)."""
    fss = get_font(lay["font_path"], lay["size"] * ss)
    lines, hw, sw = lay["lines"], lay["hw"], lay["sw"]
    mw = max(fss.getlength(l) / ss for l in lines)
    reach = int(sw + hw) if hw > 0 else int(sw)   # viền thật sự vươn ra sw+hw px quanh nét chữ (trước đây chỉ tính max(sw,hw) -> cắt viền ở mép ô)
    m = reach + 4
    ink = reach + 2
    # Ô ảnh phải phủ hết MỰC chữ thật chứ không chỉ chiều rộng "advance": font chữ viết / nghiêng / trang trí có nét thò ra ngoài
    # hộp advance, dấu tiếng Việt chồng (Ổ Ễ Ệ...) vượt ascent. Đo hộp mực từng dòng rồi hợp với hộp cũ.
    bx1, bx2 = lay["cx"] - mw / 2 - m, lay["cx"] + mw / 2 + m
    by1, by2 = lay["top"] - m, lay["top"] + len(lines) * lay["lh"] + m
    for i, line in enumerate(lines):
        base_i = lay["top"] + i * lay["lh"] + (lay["lh"] - (lay["asc"] + lay["desc"])) / 2 + lay["asc"]
        ax, anc = line_anchor(lay, i, line, fss, ss, mw)
        try:
            l, t, r, b = fss.getbbox(line, anchor=anc[0] + "s", stroke_width=reach * ss)
        except (ValueError, TypeError, OSError):
            continue
        bx1, bx2 = min(bx1, ax + l / ss - ink), max(bx2, ax + r / ss + ink)
        by1, by2 = min(by1, base_i + t / ss - ink), max(by2, base_i + b / ss + ink)
    if clip:
        x1, x2 = max(0, int(np.floor(bx1))), min(W, int(np.ceil(bx2)) + 1)
        y1, y2 = max(0, int(np.floor(by1))), min(H, int(np.ceil(by2)) + 1)
    else:   # khối chữ sẽ được xoay: chưa cắt theo trang (cắt SAU khi xoay, nếu không phần xoay vào trang sẽ bị mất)
        x1, x2 = int(np.floor(bx1)), int(np.ceil(bx2)) + 1
        y1, y2 = int(np.floor(by1)), int(np.ceil(by2)) + 1
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    txt, halo = Image.new("L", (w * ss, h * ss), 0), Image.new("L", (w * ss, h * ss), 0)
    dt, dh = ImageDraw.Draw(txt), ImageDraw.Draw(halo)
    for i, line in enumerate(lines):
        base = lay["top"] + i * lay["lh"] + (lay["lh"] - (lay["asc"] + lay["desc"])) / 2 + lay["asc"] - y1
        ax, anc = line_anchor(lay, i, line, fss, ss, mw)
        pos = ((ax - x1) * ss, base * ss)
        dt.text(pos, line, font=fss, fill=255, anchor=anc, stroke_width=sw * ss, stroke_fill=255)
        if hw > 0:
            dh.text(pos, line, font=fss, fill=255, anchor=anc, stroke_width=(sw + hw) * ss, stroke_fill=255)
    if hw > 0:
        # Viền "bọc" bằng khoảng cách thật tới nét chữ: bộ vẽ stroke của FreeType có thể bỏ sót viền quanh các nét nhỏ (dấu chấm,
        # dấu thanh) ở một số font; hợp thêm vùng cách chữ <= sw+hw để chắc chắn viền luôn bao kín mọi nét.
        ink_mask = (np.asarray(txt, np.uint8) > 127).astype(np.uint8)
        dist = cv2.distanceTransform(1 - ink_mask, cv2.DIST_L2, 5)
        near = ((dist <= float(hw * ss)) | (ink_mask > 0)).astype(np.uint8) * 255
        halo = Image.fromarray(np.maximum(np.asarray(halo, np.uint8), near))
    at = np.asarray(txt.resize((w, h), Image.LANCZOS), np.float32)[..., None] / 255.0
    ah = np.asarray(halo.resize((w, h), Image.LANCZOS), np.float32)[..., None] / 255.0 if hw > 0 else None
    return x1, y1, at, ah


def render_block(out: np.ndarray, lay: dict, color: tuple, halo_color: tuple, ss: int) -> None:
    H, W = out.shape[:2]
    mk = _block_masks(lay, ss, H, W)
    if mk is None:
        return
    x1, y1, at, ah = mk
    h, w = at.shape[:2]
    roi = out[y1:y1 + h, x1:x1 + w].astype(np.float32)
    if ah is not None:
        roi = roi * (1 - ah) + np.array(halo_color, np.float32).reshape(1, 1, 3) * ah
    roi = roi * (1 - at) + np.array(color, np.float32).reshape(1, 1, 3) * at
    out[y1:y1 + h, x1:x1 + w] = np.clip(roi, 0, 255).astype(np.uint8)


def rotate_bgra(x: int, y: int, bgra: np.ndarray, angle: float):
    """Xoay ảnh BGRA trong suốt quanh tâm của nó (angle dương = theo chiều kim đồng hồ). Trả về (x, y, bgra) mới.
    Xoay trên ảnh nhân sẵn alpha để mép chữ không bị viền tối."""
    h, w = bgra.shape[:2]
    a = np.deg2rad(angle)
    nw = int(np.ceil(abs(w * np.cos(a)) + abs(h * np.sin(a)))) + 2
    nh = int(np.ceil(abs(w * np.sin(a)) + abs(h * np.cos(a)))) + 2
    f = bgra.astype(np.float32)
    al = f[..., 3:4] / 255.0
    prem = np.concatenate([f[..., :3] * al, f[..., 3:4]], axis=2)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)   # OpenCV: dương = ngược chiều kim đồng hồ
    M[0, 2] += (nw - w) / 2.0
    M[1, 2] += (nh - h) / 2.0
    out = cv2.warpAffine(prem, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = np.clip(out[..., 3:4], 0, 255)
    bgr = np.where(alpha > 0.5, out[..., :3] / np.maximum(alpha / 255.0, 1e-4), 0.0)
    res = np.concatenate([np.clip(bgr, 0, 255), alpha], axis=2).astype(np.uint8)
    return int(round(x - (nw - w) / 2.0)), int(round(y - (nh - h) / 2.0)), res


def clip_tile(x: int, y: int, bgra: np.ndarray, W: int, H: int):
    """Cắt ảnh BGRA đặt tại (x, y) theo trang W x H; None nếu nằm hẳn ngoài trang."""
    h, w = bgra.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, bgra[y1 - y: y2 - y, x1 - x: x2 - x]


def render_block_bgra(lay: dict, color: tuple, halo_color: tuple, ss: int, H: int, W: int, clip: bool = True):
    """Khối chữ dưới dạng ảnh BGRA trong suốt (cùng kết quả khi dán lên nền như render_block).
    Trả về (x, y, bgra) hoặc None."""
    mk = _block_masks(lay, ss, H, W, clip)
    if mk is None:
        return None
    x1, y1, at, ah = mk
    col = np.array(color, np.float32).reshape(1, 1, 3)
    if ah is None:
        alpha = at
        prem = col * at
    else:
        hal = np.array(halo_color, np.float32).reshape(1, 1, 3)
        alpha = at + ah * (1 - at)
        prem = col * at + hal * ah * (1 - at)
    bgr = np.where(alpha > 1e-4, prem / np.maximum(alpha, 1e-4), 0.0)
    bgra = np.concatenate([np.clip(bgr, 0, 255), np.clip(alpha * 255.0, 0, 255)], axis=2).astype(np.uint8)
    return x1, y1, bgra


# --------------------------------------------------------------------------- shared bubbles
def partition_shared_bubbles(items: list, min_share: float = 0.5) -> int:
    """Nhiều câu thoại cùng nằm trong 1 bóng (interior chồng nhau) -> chia bóng theo khoảng cách
    tới bbox từng câu, mỗi câu chỉ được dùng phần của mình (gán b.limit). Trả về số nhóm đã chia."""
    cand = [b for b in items if b.shaped and b.interior is not None and b.interior.any() and b.text.strip()]
    n = len(cand)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def shared(p, q) -> bool:
        ax1, ay1 = p.ox, p.oy
        ax2, ay2 = ax1 + p.interior.shape[1], ay1 + p.interior.shape[0]
        bx1, by1 = q.ox, q.oy
        bx2, by2 = bx1 + q.interior.shape[1], by1 + q.interior.shape[0]
        x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        if x2 <= x1 or y2 <= y1:
            return False
        wp = p.interior[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1]
        wq = q.interior[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1]
        inter = int((wp & wq).sum())
        # cùng 1 bóng khi phần bóng mà cả 2 ROI cùng nhìn thấy là MỘT vùng trắng chung
        # (2 bóng sát nhau thì trong cửa sổ chung, vùng trắng của mỗi bên là khác nhau -> inter ~ 0)
        return inter >= max(30, min_share * min(int(wp.sum()), int(wq.sum())))

    for i in range(n):
        for j in range(i + 1, n):
            if shared(cand[i], cand[j]):
                parent[find(j)] = find(i)
    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(cand[i])

    done = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        ux1 = min(m.ox for m in members)
        uy1 = min(m.oy for m in members)
        ux2 = max(m.ox + m.interior.shape[1] for m in members)
        uy2 = max(m.oy + m.interior.shape[0] for m in members)
        ys, xs = np.mgrid[uy1:uy2, ux1:ux2]
        dists = []
        for m in members:
            x1, y1, x2, y2 = m.bbox
            dx = np.maximum(np.maximum(x1 - xs, xs - (x2 - 1)), 0)
            dy = np.maximum(np.maximum(y1 - ys, ys - (y2 - 1)), 0)
            dists.append(np.hypot(dx, dy))
        owner = np.argmin(np.stack(dists), axis=0)
        for k, m in enumerate(members):
            h, w = m.interior.shape
            m.limit = (owner == k)[m.oy - uy1:m.oy - uy1 + h, m.ox - ux1:m.ox - ux1 + w]
        done += 1
    return done


# --------------------------------------------------------------------------- overlaps
def _rects(lay: dict, ss: int) -> list[tuple]:
    """Mỗi dòng chữ là 1 hình chữ nhật (căn giữa quanh lay['cx'])."""
    fss = get_font(lay["font_path"], lay["size"] * ss)
    m = max(lay["sw"], lay.get("hw", 0)) + 1
    out = []
    mw = max((fss.getlength(l) / ss for l in lay["lines"]), default=0.0)
    for i, line in enumerate(lay["lines"]):
        w = fss.getlength(line) / ss + 2 * m
        y1 = lay["top"] + i * lay["lh"]
        ax, anc = line_anchor(lay, i, line, fss, ss, mw)
        if anc == "ls":
            out.append((ax - m, y1, ax - m + w, y1 + lay["lh"]))
        elif anc == "rs":
            out.append((ax + m - w, y1, ax + m, y1 + lay["lh"]))
        else:
            out.append((lay["cx"] - w / 2, y1, lay["cx"] + w / 2, y1 + lay["lh"]))
    return out


def _overlap(ra: list, rb: list, gap: float = 0.0) -> float:
    tot = 0.0
    for p in ra:
        for q in rb:
            w = min(p[2], q[2]) - max(p[0], q[0]) + gap
            h = min(p[3], q[3]) - max(p[1], q[1]) + gap
            if w > 0 and h > 0:
                tot += w * h
    return tot


def resolve_overlaps(active: list, relayout, a, ss: int, max_iter: int = 80) -> dict:
    """Làm cho các khối chữ không còn chồng lên nhau.

    Mỗi vòng lấy cặp chồng nhiều nhất rồi thử lần lượt:
      1) dịch 1 trong 2 khối (shaped: lên/xuống trong bóng, rect: 4 hướng trong vùng allowed)
         tới vị trí gần nhất mà không chạm khối nào khác (chừa 3 px);
      2) giảm cỡ chữ của khối to hơn 1 px (không thấp hơn a.overlap_floor, không đụng font_size do người dùng ghim);
      3) bỏ cặp đó và ghi nhận là chưa xử lý được.
    relayout(b, size, floor) -> layout dict | None.
    """
    stats = dict(nudged=0, shrunk=0, unresolved=0)
    rects = {id(b): _rects(b.layout, ss) for b in active}
    ignore: set = set()

    def others_ok(b, lay_try) -> bool:
        rb = _rects(lay_try, ss)
        return all(_overlap(rb, rects[id(o)], 3.0) == 0 for o in active if o is not b)

    def nudge(b):
        lay = b.layout
        best = None                                              # (|shift|, new_layout)
        if lay.get("mode") == "shape" and b.shaped:
            limit = min(len(b.half), int(2 * len(lay["lines"]) * lay["lh"]) + 6)
            step = 3
            while step < limit and (best is None or step < best[0]):
                for sign in (1, -1):
                    cand = try_layout(b, lay["size"], lay["font_path"], lay["sw"], line_spacing_of(b, a), ss, dy=sign * step)
                    if cand is None:
                        continue
                    cand = finish_shape_layout(b, cand, a)
                    if others_ok(b, cand):
                        best = (step, cand)
                        break
                step += 3
        elif lay.get("mode") == "rect" and b.allowed is not None:
            ax1, ay1, ax2, ay2 = b.allowed
            rs = rects[id(b)]
            lo_x, hi_x = ax1 - min(r[0] for r in rs), ax2 - max(r[2] for r in rs)
            lo_y, hi_y = ay1 - rs[0][1], ay2 - rs[-1][3]
            step, limit = 3, int(min(max(ax2 - ax1, ay2 - ay1), 3 * lay["lh"] * len(lay["lines"]) + 6))
            while step < limit and best is None:
                for ddx, ddy in ((0, -step), (0, step), (-step, 0), (step, 0)):
                    if not (lo_x <= ddx <= hi_x and lo_y <= ddy <= hi_y):
                        continue
                    cand = dict(lay, cx=lay["cx"] + ddx, top=lay["top"] + ddy)
                    if others_ok(b, cand):
                        best = (step, cand)
                        break
                step += 3
        return best

    def set_layout(b, lay):
        b.layout = lay
        rects[id(b)] = _rects(lay, ss)

    for _ in range(max_iter):
        worst, pair = 0.0, None
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if (i, j) in ignore:
                    continue
                ov = _overlap(rects[id(active[i])], rects[id(active[j])])
                if ov > worst:
                    worst, pair = ov, (i, j)
        if pair is None:
            break
        bi, bj = active[pair[0]], active[pair[1]]
        moves = [m for m in ((nudge(bi), bi), (nudge(bj), bj)) if m[0] is not None]
        if moves:
            (shift, lay), blk = min(moves, key=lambda m: m[0][0])
            set_layout(blk, lay)
            stats["nudged"] += 1
            continue
        shrinkable = [b for b in (bi, bj) if not b.font_size and b.layout["size"] > a.overlap_floor]
        shrinkable.sort(key=lambda b: (-b.layout["size"], -sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects[id(b)])))
        shrunk = False
        for b in shrinkable:
            new = relayout(b, b.layout["size"] - 1, a.overlap_floor)
            if new is not None and new["size"] < b.layout["size"]:
                set_layout(b, new)
                stats["shrunk"] += 1
                shrunk = True
                break
        if not shrunk:
            ignore.add(pair)
            stats["unresolved"] += 1
    return stats


# --------------------------------------------------------------------------- page
def typeset_page(out: np.ndarray, items: list, gray: np.ndarray, a, regular: str, bold: str | None,
                 draw: bool = True, frozen: dict | None = None, dirty: set | None = None,
                 frozen_best: dict | None = None, skip_tiles: set | None = None) -> dict:
    """Xếp chữ của mọi item có bản dịch. draw=True: vẽ thẳng lên `out`; luôn trả về `tiles` (mỗi khối chữ là 1 ảnh BGRA
    trong suốt riêng, để bật/tắt từng câu) cùng thống kê / báo cáo.

    Hai chế độ:
      * ĐẦY ĐỦ (frozen=None) - lần xếp đầu của trang / redraw hàng loạt: cỡ chữ được chuẩn hoá theo trung vị của cả trang và
        các khối chồng nhau được tự dịch / thu nhỏ cho khỏi đè (resolve_overlaps).
      * TĂNG DẦN (frozen = {id câu: bố cục đã xếp}, frozen_best = {id: cỡ tối đa vừa khung lúc đó}) - chỉnh một vài câu: khối
        nào không nằm trong `dirty` giữ NGUYÊN bố cục cũ, chỉ khối trong `dirty` được xếp lại, và KHÔNG tự dịch / thu nhỏ khối
        xung quanh nữa - nên sửa 1 khung không làm các khung khác nhúc nhích. Ngưỡng cỡ chữ (trung vị) của khối được xếp lại
        tính trên MỌI khối (khối đóng băng góp đúng cỡ cũ của chúng) như khi xếp đầy đủ.
    """
    H, W = out.shape[:2]
    incremental = frozen is not None
    dirty = dirty or set()

    def key_of(b):
        return int(b.ids[0]) if b.ids else None

    def is_frozen(b) -> bool:
        k = key_of(b)
        return incremental and k is not None and k in frozen and k not in dirty
    ss = max(1, a.supersample)
    dark_text = parse_color(a.text_color) or (20, 20, 20)

    active = [b for b in items if b.text.strip() and b.kind != "sfx"]
    shared_groups = partition_shared_bubbles(items)
    gutters = page_gutters(gray)
    for b in items:
        if b.kind == "sfx" or is_frozen(b):  # khối đóng băng không cần dựng lại hình dạng bóng thoại / vùng cho phép
            continue
        if b.region is not None:
            rx1, ry1, rx2, ry2 = b.region                          # khung chữ người dùng đặt: KHÔNG nhốt trong trang (có thể thò ra ngoài)
            rx1, rx2 = sorted((int(round(rx1)), int(round(rx2))))
            ry1, ry2 = sorted((int(round(ry1)), int(round(ry2))))
            b.allowed = (rx1, ry1, max(rx1 + 1, rx2), max(ry1 + 1, ry2))
        elif b.kind == "free":
            b.allowed = allowed_rect(b, items, gray, gutters, a)
        if b.shaped and b.interior is not None:
            prepare_shape(b, a.margin, getattr(b, "limit", None))

    custom_ok: dict[str, bool] = {}
    warned: list[str] = []

    def custom_font(b) -> str | None:
        """Font do người dùng chọn riêng cho câu này (override font_path); None nếu không có / không dùng được."""
        p = b.font_path
        if not p:
            return None
        if p not in custom_ok:
            custom_ok[p] = Path(p).is_file() and supports_vietnamese(p)
            if not custom_ok[p]:
                warned.append(p)
                log.warning("Font riêng '%s' không dùng được (thiếu file hoặc không có dấu tiếng Việt) -> dùng font mặc định.", p)
        return p if custom_ok[p] else None

    def font_for(b) -> str:
        c = custom_font(b)
        if c:
            return c
        if b.bold and bold:
            return bold
        return regular

    def fake_sw(b):
        # Chữ đậm giả (stroke) khi không có font đậm thật: font riêng + style "bold" cũng dùng cách này
        faux = b.bold and (not bold or custom_font(b) is not None)
        return lambda size: max(1, round(size * 0.035)) if faux else 0

    def align_of(b) -> str:
        return b.align if getattr(b, "align", None) in ALIGNS else "center"

    def layout_once(b, top: int, floor: int) -> dict | None:
        path = font_for(b)
        for size in range(top, floor - 1, -1):
            if b.shaped:
                lay = try_layout(b, size, path, fake_sw(b)(size), line_spacing_of(b, a), ss)
                if lay:
                    lay = finish_shape_layout(b, lay, a)
                    lay["align"] = align_of(b)
                    return lay
            else:
                lay = fit_oval(b, size, path, fake_sw(b), a, ss, floor) if b.region is not None else None   # khung người dùng đặt: chữ hình oval
                if lay is None:
                    lay = fit_rect(b, size, path, fake_sw(b), a, ss, floor)
                if lay:
                    lay["align"] = align_of(b)
                    return lay
                break
        return None

    def layout(b, top: int, floor: int) -> dict | None:
        lay = layout_once(b, top, floor)
        if lay is None and not getattr(b, "split_hyphen", False) and any("-" in w.strip("-") for w in flat_words(paragraphs(b.text))):
            b.split_hyphen = True        # từ ghép dài không vừa khung ở cỡ nào -> thử lại cho phép ngắt tại dấu gạch nối
            lay = layout_once(b, top, floor)
            if lay is None:
                b.split_hyphen = False
        return lay

    todo = [b for b in active if not is_frozen(b)]  # khối cần xếp lại (chế độ đầy đủ: tất cả)
    for b in active:
        if is_frozen(b):
            b.layout = dict(frozen[key_of(b)])
            b.layout["align"] = align_of(b)
            b.best = int((frozen_best or {}).get(key_of(b), b.layout["size"]))
    for b in todo:
        lay = layout(b, b.font_size or a.max_font, a.min_font)
        b.best = lay["size"] if lay else a.min_font

    def med_of(group):
        s = [b.best for b in group if not b.font_size]
        return int(np.median(s)) if s else a.min_font

    # Ngưỡng cỡ chữ theo trung vị trên MỌI khối (kể cả khối đóng băng, góp đúng cỡ cũ). Chỉ khối được xếp lại (`todo`) bị ngưỡng
    # này chi phối nên các khối khác không đổi. (Không dùng ngưỡng lưu từ lần xếp đầu: nhóm "khung tự do" lúc đó có thể rỗng
    # -> ngưỡng = cỡ tối thiểu, làm khung vừa được kéo thả bị thu nhỏ hẳn.)
    shaped = [b for b in active if b.shaped]
    rect = [b for b in active if not b.shaped]
    cap_s = max(a.min_font, int(round(med_of(shaped) * a.uniform_cap)))
    cap_r = max(a.min_font, int(round(med_of(rect) * a.free_cap)))

    for b in todo:
        if b.shaped:
            cap_b = max(cap_s, int(round(getattr(a, "geo_k", 0.0) * b.inr)))
        else:
            ax1_, ay1_, ax2_, ay2_ = b.allowed or b.bbox
            cap_b = max(cap_r, int(round(getattr(a, "geo_k_free", 0.0) * math.sqrt(max(1, (ax2_ - ax1_) * (ay2_ - ay1_))))))
        target = b.font_size or min(b.best, cap_b)
        lay = layout(b, target, a.min_font)
        if lay is None:
            log.warning("Câu %s không vừa khung ở cỡ %dpx -> dùng bố cục tràn.", b.ids, a.min_font)
            if b.shaped:
                lay = finish_shape_layout(b, fallback_layout(b, a.min_font, font_for(b), fake_sw(b)(a.min_font),
                                                             line_spacing_of(b, a), ss), a)
                lay["mode"] = "shape"
            else:
                x1, y1, x2, y2 = b.allowed or b.bbox
                fss = get_font(font_for(b), a.min_font * ss)
                asc, desc = (v / ss for v in fss.getmetrics())
                lines = plain_wrap(paragraphs(b.text), lambda t: fss.getlength(t) / ss, x2 - x1)
                lh = int(round((asc + desc) * line_spacing_of(b, a)))
                lay = dict(lines=lines, lh=lh, asc=asc, desc=desc, size=a.min_font, font_path=font_for(b),
                           sw=fake_sw(b)(a.min_font), hw=halo_width(b, a.min_font, a), fits=False,
                           cx=(x1 + x2) / 2, top=(y1 + y2) / 2 - len(lines) * lh / 2, mode="rect")
            lay["fits"] = False
        lay["align"] = align_of(b)
        b.layout = lay

    overflow = sum(1 for b in active if not b.layout.get("fits", True))
    if incremental:
        ov = dict(nudged=0, shrunk=0, unresolved=0)  # chỉnh tay: không tự đẩy / thu nhỏ khối xung quanh cho khỏi đè
    else:
        ov = resolve_overlaps(active, layout, a, ss)

    tiles = []
    H_, W_ = out.shape[:2]
    for b in active:
        lay = b.layout
        lay["align"] = getattr(b, "align", None) if getattr(b, "align", None) in ALIGNS else "center"
        if b.text_color:
            color = b.text_color
        elif b.polarity == "white":
            color = (255, 255, 255)
        else:
            color = dark_text
        halo_color = b.outline_color or ((0, 0, 0) if b.polarity == "white" else (255, 255, 255))
        b.final_color = (int(color[2]), int(color[1]), int(color[0]))  # RGB để hiển thị
        if skip_tiles and key_of(b) in skip_tiles:
            continue  # ảnh lớp chữ của khối này đã có sẵn (tái dùng từ lần ghi trước): khỏi vẽ lại
        if draw:
            render_block(out, lay, color, halo_color, ss)
        rotated = abs(getattr(b, "rotation", 0.0)) >= 0.2
        t = render_block_bgra(lay, color, halo_color, ss, H_, W_, clip=not rotated)
        if t is not None and rotated:
            t = rotate_bgra(t[0], t[1], t[2], b.rotation)          # xoay quanh tâm khối chữ
            t = clip_tile(t[0], t[1], t[2], W_, H_)                # rồi mới cắt theo trang
        if t is not None and b.ids:
            tiles.append(dict(id=int(b.ids[0]), x=t[0], y=t[1], bgra=t[2]))

    return dict(
        blocks=[dict(ids=b.ids, kind=b.kind, size=b.layout["size"], lines=len(b.layout["lines"]),
                     fits=bool(b.layout.get("fits", True)), font=b.layout["font_path"],
                     line_spacing=round(line_spacing_of(b, a), 2), bold=bool(b.bold),
                     color=list(getattr(b, "final_color", (20, 20, 20))),
                     outline=bool(b.outline), outline_width=int(b.outline_width), align=b.layout.get("align", "center"),
                     outline_color=[int(b.outline_color[2]), int(b.outline_color[1]), int(b.outline_color[0])]
                     if b.outline_color else ([0, 0, 0] if b.polarity == "white" else [255, 255, 255]),
                     rects=[[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in _rects(b.layout, ss)])
                for b in active],
        shared_bubble_groups=shared_groups, overflow=overflow, **{f"overlap_{k}": v for k, v in ov.items()},
        caps=dict(shaped=cap_s, free=cap_r), bad_fonts=warned, tiles=tiles,
        layouts={key_of(b): b.layout for b in active if key_of(b) is not None},
        bests={key_of(b): int(b.best) for b in active if key_of(b) is not None}, incremental=incremental,
        relaid=sorted(key_of(b) for b in todo if key_of(b) is not None),
    )
