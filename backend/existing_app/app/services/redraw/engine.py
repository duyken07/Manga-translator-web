"""
engine.py - redraw 1 trang: xoá chữ gốc (OpenCV / lama-manga) + đặt bản dịch tiếng Việt.

Điểm khác với chạy lẻ manga_redraw_v8.py / manga_cpu_pipeline_v4.py: kết quả được CACHE trên đĩa và
mỗi lần gọi chỉ làm phần chênh lệch (sync_page):

    <cache>/<key>/clean.png    trang đã xoá chữ gốc (chưa có chữ dịch)
    <cache>/<key>/touched.png  vùng đã bị sửa (để khôi phục ảnh gốc khi xoá 1 khung thoại)
    <cache>/<key>/state.json   danh sách khung thoại đã được xoá chữ gốc
    <cache>/<key>/out.png      trang hoàn chỉnh = clean + chữ dịch

    khung thoại MỚI (có bản dịch)  -> chỉ xoá chữ gốc ở vị trí đó (LaMa/OpenCV), giữ nguyên phần còn lại
    khung thoại bị XOÁ / bản dịch rỗng -> trả lại điểm ảnh gốc ở vị trí đó
    sửa chữ dịch / thứ tự            -> không xoá gì, chỉ xếp chữ lại (vài trăm ms)

Toàn bộ hàm ở đây không phụ thuộc Qt, gọi được từ thread nền.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import stage_a as sa
from . import typeset as ts
from app.services.strokes import render_strokes

log = logging.getLogger("redraw.engine")

STATE_VERSION = 3   # đổi số này => mọi trang đã redraw bị xoá chữ lại từ đầu (không đổi thì khung đã xoá bằng bản cũ, kể cả kết quả còn sót
#                    bóng chữ, được dùng lại nguyên từ cache).  3: thuật toán xoá đổi (nới khung theo nét chữ, ngưỡng nét thích nghi, quét viền)
#                    2: nhận diện bóng thoại theo khung (box_bubble)
TOUCH_RADIUS = 10                  # px: vùng quanh nét chữ có thể bị đổi màu khi xoá (LaMa blend, snap trắng)
ERASE_KEYS = ("type", "fill", "keep", "polarity", "cleanup", "engine", "kz")   # override làm đổi kết quả XOÁ
_LOCK = threading.RLock()          # stage_a dùng biến toàn cục _LAMA_QUEUE -> chỉ cho 1 trang chạy tại 1 thời điểm


class RedrawError(RuntimeError):
    pass


@dataclass
class RedrawOptions:
    lama_path: str = ""            # .safetensors (lama-manga) hoặc .onnx; rỗng = chỉ dùng OpenCV
    device: str = "cpu"
    threads: int = 0               # 0 = để torch tự chọn
    font_path: str = ""
    text_color: str = "20,20,20"   # R,G,B chữ tối
    font_scale: float = 0.0        # 0 = tự theo chiều cao trang (chuẩn 1600 px)
    auto_bold: bool = False        # True: tự đoán chữ đậm theo độ dày nét gốc (mặc định tắt: chữ thường)
    thick_stroke: float = 0.0      # 0 = tắt (hành vi gốc v8). >0: bật nhận diện chữ cỡ lớn theo độ dày nét (gợi ý 7)
    line_spacing: float = 1.06     # hệ số giãn dòng mặc định (0.7 - 2.5); từng câu có thể ghi đè bằng override "line_spacing"
    box_bubble: bool = True        # True: khung thoại = khung bao bóng thoại -> nhận lòng bóng theo khung, không xoá / xếp chữ ra ngoài khung
    # -- koharu-yolo26s: bóng thoại + vùng chữ có mask để xác định chính xác vùng cần xoá (xem app/services/koharu_detector.py)
    koharu: bool = False
    koharu_model_dir: str = ""     # trống = tải mayocream/koharu-yolo26s từ HuggingFace
    koharu_imgsz: int = 0          # 0 = theo config.json (1280); CPU chậm thì 960
    koharu_gate: bool = True       # chỉ xoá nét nằm trong vùng chữ koharu (có lưới an toàn khi koharu sót chữ)

    def koharu_cfg(self, cache_root=None):
        from app.services import koharu_detector as kd
        return kd.KoharuConfig(enabled=self.koharu, model_dir=self.koharu_model_dir, imgsz=self.koharu_imgsz,
                               device=self.device or "cpu", threads=int(self.threads or 0),
                               cache_dir=str(Path(cache_root) / "koharu") if cache_root else "")

    def lama_key(self) -> tuple:
        return (self.lama_path, self.device, int(self.threads))


@dataclass
class RedrawResult:
    out_path: str
    clean_path: str
    erased: int = 0
    restored: int = 0
    seconds: float = 0.0
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    layout: dict = field(default_factory=dict)   # {dialogue_id: [x, y, w, h]} hình chữ nhật bao khối chữ dịch
    info: dict = field(default_factory=dict)     # {dialogue_id: {size, font, line_spacing, bold, color:[r,g,b]}} số liệu thật đã dùng
    layers_path: str = ""                        # layers.json: các lớp riêng từng câu (vùng xoá + chữ) cho viewer


# --------------------------------------------------------------------------- image io (đường dẫn có dấu / Unicode trên Windows)
def imread(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_atomic(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RedrawError(f"Không ghi được {path.name}")
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- cache
def cache_dir_for(root, image_path: str) -> Path:
    p = Path(image_path)
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:16]
    d = Path(root) / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_path_for(root, image_path: str) -> Path:
    return cache_dir_for(root, image_path) / "out.png"


def _wipe_dir(cdir: Path) -> None:
    """Xoá toàn bộ nội dung cache của 1 trang (file lẫn thư mục layers_*). Trên Windows, unlink() một thư mục báo
    'Access is denied' và 1 file đang bị khoá cũng không được làm hỏng cả lần redraw -> xoá từng mục, bỏ qua lỗi."""
    for f in list(cdir.iterdir()):
        try:
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Không xoá được %s: %s", f.name, exc)


def clear_cache(root, image_path: str) -> None:
    d = cache_dir_for(root, image_path)
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- LaMa
_LAMA: dict = {"key": None, "obj": None, "error": ""}


def get_lama(opts: RedrawOptions):
    """(model | None, thông báo lỗi). Nạp 1 lần, dùng lại cho mọi trang; lỗi nạp cũng được nhớ để khỏi thử lại mỗi trang."""
    if not opts.lama_path:
        return None, ""
    key = opts.lama_key()
    with _LOCK:
        if _LAMA["key"] == key:
            return _LAMA["obj"], _LAMA["error"]
        obj, err = None, ""
        try:
            if not Path(opts.lama_path).is_file():
                raise FileNotFoundError(opts.lama_path)
            if opts.lama_path.lower().endswith(".safetensors"):
                from .lama_manga import LamaManga
                obj = LamaManga(opts.lama_path, device=opts.device or "cpu", threads=opts.threads or None)
            else:
                obj = sa.Lama(opts.lama_path)
            log.info("Đã nạp LaMa: %s (%s)", opts.lama_path, opts.device)
        except Exception as exc:  # noqa: BLE001
            err = f"Không nạp được LaMa ({exc}); đang dùng OpenCV cho mọi vùng."
            log.warning(err)
        _LAMA.update(key=key, obj=obj, error=err)
        return obj, err


def make_args(opts: RedrawOptions, height: int, lama_ok: bool):
    """Tuỳ chọn stage A (mặc định của manga_redraw_v8.py) + tuỳ chọn xếp chữ (mặc định của pipeline v4)."""
    a = sa.build_parser().parse_args(["--json", "-", "--image", "-"])
    a.lama_ok = lama_ok
    a.thick_stroke = float(opts.thick_stroke or 0.0)
    a.box_bubble = int(bool(opts.box_bubble))
    a.koharu = int(bool(opts.koharu))
    a.k_gate = int(bool(opts.koharu_gate))
    s = opts.font_scale if opts.font_scale > 0 else min(2.5, max(0.6, height / 1600.0))
    a.min_font = max(8, int(round(12 * s)))
    a.max_font = max(a.min_font + 4, int(round(38 * s)))
    a.overlap_floor = max(7, int(round(9 * s)))
    a.uniform_cap, a.free_cap, a.margin = 1.18, 1.5, 5
    # Cỡ chữ theo độ to của khung (không chỉ theo trung vị cả trang): trần của bóng = max(trung vị x cap, geo_k x bán kính nội tiếp).
    # Trung vị bị các bóng hẹp (chỉ chứa được chữ rất nhỏ) kéo xuống làm bóng to bị khoá chữ nhỏ; geo_k nâng trần đó lên theo kích thước thật.
    a.geo_k, a.geo_k_free = 0.34, 0.18
    a.grow_ink, a.grow_gap_mult = 1, 1.6   # nới khung chữ tự do theo nét chữ nối tiếp khi khung dò bị cắt (1 = bật)
    a.free_adaptive, a.free_ink_delta = 1, 80   # ngưỡng ink của chữ tự do theo mức giấy trắng phẳng (bắt nét mảnh / scan mềm); 0 = ngưỡng cố định
    a.halo_sweep, a.snap_thr_low = 1, 150       # sau khi tô phẳng: ép viền mờ sát nét (>= ngưỡng này) về màu nền; halo_sweep = 0 để tắt
    a.flat_frac = 0.15                     # nới khung chữ dịch chỉ qua chỗ là nền: cột/hàng có > 15% điểm khác nền là rào cản (0 = tắt)
    a.line_spacing = min(max(opts.line_spacing or 1.06, 0.7), 2.5)
    a.edge_black, a.edge_white, a.supersample = 0.13, 0.06, 3
    a.text_color = opts.text_color or "20,20,20"
    return a


# --------------------------------------------------------------------------- helpers
def _raw_items(dialogues: list[dict], W: int, H: int) -> list[dict]:
    raw = []
    for i, d in enumerate(dialogues, 1):
        box = d.get("box_location") or []
        if len(box) != 4:
            continue
        x, y, w, h = box
        if w <= 1 or h <= 1 or x >= W or y >= H:
            continue
        it = dict(id=d.get("dialogue_id", i), bbox=[x, y, x + w, y + h],
                  source_text=d.get("original_text", ""), translated_text=d.get("translated_text", ""))
        for k, v in (d.get("overrides") or {}).items():
            if k in sa.OVERRIDE_KEYS:
                it[k] = v
        it["hide_text"] = bool(d.get("hide_text"))        # chỉ ảnh hưởng ảnh ghép out.png, không ảnh hưởng xoá / xếp chữ
        it["hide_redraw"] = bool(d.get("hide_redraw"))
        it["redraw_only"] = bool(d.get("redraw_only"))   # vùng chỉ xoá chữ gốc: không cần bản dịch
        it["no_erase"] = bool(d.get("no_erase"))         # khung "chỉ chèn chữ": KHÔNG xoá chữ gốc (không có lớp vùng xoá), vẫn xếp chữ dịch
        it["rotation"] = float(d.get("text_rotation") or 0.0)  # góc xoay khối chữ (độ, chiều kim đồng hồ)
        it["redraw_gen"] = int(d.get("redraw_gen") or 0)       # đổi số này = xoá lại chữ gốc của câu từ đầu
        raw.append(it)
    return raw


def _erase_key(b, raw_item: dict) -> str:
    ov = {k: raw_item[k] for k in ERASE_KEYS if k in raw_item}
    if raw_item.get("redraw_gen"):
        ov["gen"] = raw_item["redraw_gen"]
    return f"{b.bbox[0]},{b.bbox[1]},{b.bbox[2]},{b.bbox[3]}|{json.dumps(ov, sort_keys=True, default=str)}"


LAYOUT_VERSION = 4   # 4: cỡ chữ theo độ to của bóng, khung chữ tự do, ngắt gạch nối (bản cũ dùng lại bố cục cache nên không thấy cải thiện)
#                      3: ô ảnh chữ phủ hết mực chữ (trước đó có thể cắt nét thò ra của font nghiêng / chữ viết)
# raw item: các trường KHÔNG ảnh hưởng bố cục chữ (chỉ ảnh hưởng ảnh ghép / việc xoá chữ gốc) -> đổi chúng không làm khối "bẩn"
# (rotation: chỉ xoay ảnh khối chữ sau khi đã xếp, không đổi cỡ chữ / ngắt dòng)
_LAYOUT_IGNORE = ("hide_text", "hide_redraw", "redraw_only", "no_erase", "redraw_gen", "rotation")


# Khung chữ tự động hiển thị ở workspace được đệm AUTO_FRAME_PAD px mỗi cạnh (workspace_view._refresh_text_frames, hằng số pad). Kéo khung tự động
# lần đầu thì khung người dùng (region) lớn hơn bbox khối chữ cũ 2 * AUTO_FRAME_PAD: vẫn là "chỉ dời khung", không phải đổi kích thước.
AUTO_FRAME_PAD = 4


def _block_sig(raw_item: dict, ignore_region: bool = False) -> str:
    """Chữ ký đầu vào bố cục của MỘT khung: bản dịch, vị trí / kích thước, mọi override định dạng. Khung có chữ ký không đổi
    thì giữ nguyên bố cục đã xếp, dù khung bên cạnh vừa đổi. ignore_region=True: bỏ qua khung chữ người dùng kéo (để nhận ra
    trường hợp CHỈ dời khung)."""
    skip = _LAYOUT_IGNORE + (("region",) if ignore_region else ())
    d = {k: v for k, v in raw_item.items() if k not in skip}
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _shift_sig(sig: str, x: int, y: int) -> str:
    """Chữ ký của lớp cũ sau khi dời vị trí (điểm ảnh y nguyên)."""
    return hashlib.md5(f"{sig}@{x},{y}".encode()).hexdigest()[:20]


def _rects_bbox(rects) -> list | None:
    if not rects:
        return None
    return [int(min(r[0] for r in rects)), int(min(r[1] for r in rects)), int(max(r[2] for r in rects)), int(max(r[3] for r in rects))]


def _translated_layout(old: dict, new_region, dx: int, dy: int) -> dict:
    """Bố cục cũ dời nguyên khối (dx, dy): giữ cỡ chữ và cách ngắt dòng; chỉ vị trí đổi."""
    lay = dict(old["lay"])
    lay["cx"] = float(lay["cx"]) + dx
    lay["top"] = float(lay["top"]) + dy
    return lay


def _opts_sig(opts, regular: str, bold, W: int, H: int) -> str:
    """Chữ ký các cài đặt toàn cục ảnh hưởng bố cục (font, cỡ, giãn dòng, kích thước ảnh...). Đổi -> xếp lại cả trang."""
    from dataclasses import asdict

    d = dict(opts=asdict(opts), regular=regular, bold=bold, size=[W, H])
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):  # numpy scalar
        return o.item()
    return o


def _load_layouts(cdir: Path):
    try:
        st = json.loads((cdir / "layouts.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return st if st.get("version") == LAYOUT_VERSION else None


def _load_state(cdir: Path, image_path: str, W: int, H: int):
    p = cdir / "state.json"
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sig = Path(image_path).stat()
    if st.get("version") != STATE_VERSION or st.get("size") != [W, H] or st.get("image") != [sig.st_size, sig.st_mtime_ns]:
        return None
    return st


def _restore(clean, touched, image, removed_bboxes, kept_items, a) -> None:
    """Trả lại điểm ảnh gốc quanh các khung thoại đã bị xoá, trừ vùng của những khung còn lại."""
    H, W = touched.shape
    pad = 40 + a.lama_grow + 6          # >= vùng phân tích mở rộng của chữ lớn (stage_a.analyze)
    prot = np.zeros((H, W), np.uint8)
    for b in kept_items:
        h, w = b.strokes.shape
        prot[b.oy:b.oy + h, b.ox:b.ox + w] |= b.strokes.astype(np.uint8)
    if prot.any():
        prot = cv2.dilate(prot, sa.ellipse_kernel(TOUCH_RADIUS))
    for x1, y1, x2, y2 in removed_bboxes:
        rx1, ry1, rx2, ry2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
        reg = (touched[ry1:ry2, rx1:rx2] > 0) & (prot[ry1:ry2, rx1:rx2] == 0)
        clean[ry1:ry2, rx1:rx2][reg] = image[ry1:ry2, rx1:rx2][reg]
        touched[ry1:ry2, rx1:rx2][reg] = 0


def _erase(clean, image, items_to_erase, all_items, a, lama) -> np.ndarray:
    """Xoá chữ gốc của `items_to_erase` ngay trên `clean` (cùng thuật toán main() của manga_redraw_v8.py). Trả về mask."""
    H, W = clean.shape[:2]
    mask_full = np.zeros((H, W), np.uint8)
    tex_mask = np.zeros((H, W), np.uint8)
    keep_map = np.zeros((H, W), np.uint8)
    for b in all_items:
        for kx1, ky1, kx2, ky2 in b.keep:
            keep_map[max(0, ky1):ky2, max(0, kx1):kx2] = 1
        if b.kind == "sfx":
            x1, y1, x2, y2 = b.bbox
            keep_map[y1:y2, x1:x2] = 1
    sa._LAMA_QUEUE = np.zeros((H, W), np.uint8) if lama is not None else None
    try:
        for b in items_to_erase:
            m = sa.erase_item(clean, b, a, lama)
            mask_full |= m
            if b.fill_mode == "texture":
                tex_mask |= m
        sa.flush_lama(clean, a, lama)          # 1 lượt LaMa cho mọi khung mới của trang
    finally:
        sa._LAMA_QUEUE = None
    if tex_mask.any():
        if a.tex_grow > 0:
            tex_mask = cv2.dilate(tex_mask, sa.ellipse_kernel(a.tex_grow))
        thin = tex_mask & keep_map
        tex_mask[keep_map > 0] = 0
        mask_full |= tex_mask | thin
        forbid = cv2.dilate(mask_full, sa.ellipse_kernel(3)) | keep_map
        sa.texture_fill(clean, tex_mask, forbid, a, report=[])
        if thin.any():
            res = cv2.inpaint(clean, thin * 255, 3, cv2.INPAINT_TELEA)
            clean[thin > 0] = res[thin > 0]
    if a.retry > 0:
        sa.residual_pass(clean, items_to_erase, a, lama, mask_full)
    return mask_full


# --------------------------------------------------------------------------- Content-Aware Fill (bôi vùng rồi vá)
FILL_ENGINES = ("auto", "lama", "texture", "opencv")
FILL_LAMA_MAX_PIXELS = 1_600_000   # cửa sổ ngữ cảnh tối đa cho LaMa (≈ 1265 x 1265 px)
FILL_LAMA_MAX_MASK = 400_000       # số điểm ảnh bôi tối đa cho LaMa (≈ 630 x 630 px); lớn hơn thì dùng patch-match / OpenCV


def _fill_mask(strokes: list, H: int, W: int, grow: int):
    """Vẽ các nét bôi thành mặt nạ. Trả về ((x1, y1, x2, y2), mask uint8 trong khung đó) hoặc None."""
    pts_all = [p for s in strokes for p in s.get("pts", [])]
    if not pts_all:
        return None
    wmax = max(float(s.get("width", 20)) for s in strokes)
    pad = int(wmax / 2) + grow + 3
    xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
    x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x2, y2 = min(W, int(max(xs)) + pad + 1), min(H, int(max(ys)) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    m = np.zeros((y2 - y1, x2 - x1), np.uint8)
    for s in strokes:
        pts = np.asarray(s.get("pts", []), np.int32)
        if pts.size == 0:
            continue
        th = max(1, int(round(float(s.get("width", 20)))))
        local = (pts - np.array([x1, y1], np.int32)).reshape(-1, 1, 2)
        if len(local) > 1:
            cv2.polylines(m, [local], False, 255, th)
        for p in local.reshape(-1, 2)[:: max(1, len(local) // 400)]:
            cv2.circle(m, (int(p[0]), int(p[1])), max(1, th // 2), 255, -1)
    return (x1, y1, x2, y2), (m > 0).astype(np.uint8)


def _compute_fill(base: np.ndarray, strokes: list, engine_name: str, a, lama):
    """Vá vùng bôi bằng nền xung quanh. Trả về dict(x, y, bgra) (alpha mềm ở mép) hoặc None nếu mặt nạ rỗng."""
    H, W = base.shape[:2]
    got = _fill_mask(strokes, H, W, a.lama_grow)
    if got is None:
        return None
    (x1, y1, x2, y2), m_local = got
    w, h = x2 - x1, y2 - y1
    ctx = max(48, int(0.6 * max(w, h)))                       # vùng nền xung quanh làm ngữ cảnh
    cx1, cy1, cx2, cy2 = max(0, x1 - ctx), max(0, y1 - ctx), min(W, x2 + ctx), min(H, y2 + ctx)
    crop = np.ascontiguousarray(base[cy1:cy2, cx1:cx2]).copy()
    m = np.zeros(crop.shape[:2], np.uint8)
    m[y1 - cy1: y2 - cy1, x1 - cx1: x2 - cx1] = m_local
    m = cv2.dilate(m, sa.ellipse_kernel(2))                    # lấn ra một chút để không sót viền nét bôi
    mode = engine_name if engine_name in FILL_ENGINES else "auto"
    if mode == "lama" and lama is None:
        mode = "auto"
    if mode == "auto":
        if lama is not None:
            mode = "lama"
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ring = (cv2.dilate(m, sa.ellipse_kernel(15)) > 0) & (m == 0)
            mode = "texture" if ring.any() and float(gray[ring].std()) > 25 else "opencv"
    note = ""
    if mode == "lama" and (crop.shape[0] * crop.shape[1] > FILL_LAMA_MAX_PIXELS or int(m.sum()) > FILL_LAMA_MAX_MASK):
        # Vùng quá lớn cho LaMa trên CPU (nguy cơ hết bộ nhớ / treo máy): dùng patch-match / OpenCV thay thế
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ring = (cv2.dilate(m, sa.ellipse_kernel(15)) > 0) & (m == 0)
        mode = "texture" if ring.any() and float(gray[ring].std()) > 25 else "opencv"
        note = "Vùng fill quá lớn cho LaMa nên đã dùng " + ("patch-match" if mode == "texture" else "OpenCV") + \
               ". Hãy bôi từng vùng nhỏ hơn nếu muốn dùng LaMa."
    ok = False
    try:
        if mode == "lama":
            ok = sa._lama_apply(crop, m, a, lama)
        elif mode == "texture":
            sa.texture_fill(crop, m, np.zeros_like(m), a, report=[])
            ok = True
    except (MemoryError, Exception) as exc:  # noqa: BLE001 - mọi lỗi Python của bước vá đều rơi về OpenCV
        log.warning("Fill (%s) lỗi, dùng OpenCV: %s", mode, exc)
        ok, mode = False, "opencv"
    if not ok and mode != "texture":
        res = cv2.inpaint(crop, m * 255, a.inpaint_radius, cv2.INPAINT_TELEA)
        crop[m > 0] = res[m > 0]
    soft = cv2.GaussianBlur(cv2.dilate(m, sa.ellipse_kernel(3)).astype(np.float32), (0, 0), 1.3)
    alpha = np.clip(soft * 1.6, 0, 1)
    ys, xs = np.nonzero(alpha > 0.01)
    if xs.size == 0:
        return None
    bx1, bx2, by1, by2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    bgra = np.dstack([crop[by1:by2, bx1:bx2], (alpha[by1:by2, bx1:bx2] * 255).astype(np.uint8)])
    return dict(x=cx1 + bx1, y=cy1 + by1, bgra=bgra, note=note)


def _apply_fills(clean: np.ndarray, fills: list, a, lama, cdir: Path, warnings: list) -> list:
    """Các vùng fill của trang -> danh sách lớp vá (id âm = -id vùng fill). Kết quả được cache theo nét bôi + nền bên dưới,
    nên thêm 1 vùng mới chỉ tính vùng đó. Không sửa `clean`: mỗi fill là 1 lớp riêng nằm trên ảnh đã xoá chữ."""
    fdir = cdir / "fills"
    keep: set = set()
    out: list = []
    H, W = clean.shape[:2]
    for f in fills:
        try:
            strokes = f.get("strokes") or []
            got = _fill_mask(strokes, H, W, a.lama_grow)
            if got is None:
                continue
            (x1, y1, x2, y2), _m = got
            ctx = max(48, int(0.6 * max(x2 - x1, y2 - y1)))
            under = clean[max(0, y1 - ctx): y2 + ctx, max(0, x1 - ctx): x2 + ctx]
            key = hashlib.md5(json.dumps([strokes, f.get("engine", "auto"), lama is not None, a.lama_grow],
                                         sort_keys=True).encode() + under.tobytes()).hexdigest()[:20]
            keep.add(key)
            hit = next(iter(fdir.glob(f"{key}.*.*.png")), None) if fdir.is_dir() else None
            if hit is not None:
                _, px, py, _ = hit.name.split(".")
                bgra = imread(hit, cv2.IMREAD_UNCHANGED)
                res = dict(x=int(px), y=int(py), bgra=bgra) if bgra is not None else None
            else:
                res = None
            if res is None:
                res = _compute_fill(clean, strokes, f.get("engine", "auto"), a, lama)
                if res is None:
                    continue
                fdir.mkdir(parents=True, exist_ok=True)
                imwrite_atomic(fdir / f"{key}.{res['x']}.{res['y']}.png", res["bgra"])
            if res.get("note"):
                warnings.append(res["note"])
            out.append(dict(id=-int(f["id"]), x=res["x"], y=res["y"], bgra=res["bgra"]))
        except Exception as exc:  # noqa: BLE001
            log.exception("Fill vùng %s lỗi", f.get("id"))
            warnings.append(f"Fill vùng #{f.get('id')} lỗi: {exc}")
    if fdir.is_dir():
        for p in fdir.glob("*.png"):
            if p.name.split(".")[0] not in keep:
                p.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------- lớp riêng từng câu thoại
def _paste_bgra(out: np.ndarray, bgra: np.ndarray, x: int, y: int) -> None:
    h, w = bgra.shape[:2]
    H, W = out.shape[:2]
    x1, y1 = max(0, x), max(0, y)                       # lớp có thể thò ra ngoài trang (chữ kéo ra ngoài): cắt cả hai phía
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    src = bgra[y1 - y: y2 - y, x1 - x: x2 - x].astype(np.float32)
    a = src[..., 3:4] / 255.0
    roi = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = np.clip(roi * (1 - a) + src[..., :3] * a, 0, 255).astype(np.uint8)


def compose_layers(image: np.ndarray, patches: list, tiles: list, hide_redraw=(), hide_text=(),
                   strokes: list | None = None) -> np.ndarray:
    """Ghép ảnh cuối theo thứ tự lớp (dưới -> trên):
        ảnh gốc -> vùng xoá chữ của từng câu và vùng fill (trừ vùng bị tắt) -> nét vẽ tay -> CHỮ DỊCH (trừ câu bị ẩn).
    Chữ dịch luôn nằm trên cùng, không bị nét vẽ hay fill đè lên."""
    out = image.copy()
    for p in patches:
        if p["id"] not in hide_redraw:
            _paste_bgra(out, p["bgra"], p["x"], p["y"])
    if strokes:
        out = render_strokes(out, strokes)
    for t in tiles:
        if t["id"] not in hide_text:
            _paste_bgra(out, t["bgra"], t["x"], t["y"])
    return out


def _solidify_mask(mask: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Làm đặc mặt nạ lớp vá: lấp lỗ li ti (điểm ảnh mà bước xoá chữ trả về đúng giá trị gốc nên `clean == image` -> alpha 0)
    rồi nới 1px cho phủ kín viền anti-alias của chữ gốc. Nếu không, viewer (Qt nội suy alpha khi zoom) để lộ lại viền chữ gốc
    dưới lớp vá. Chỉ lấn vào điểm ảnh chưa có chủ (`allowed`); điểm ảnh lấn thêm có giá trị clean == ảnh gốc nên bản xuất không đổi."""
    m = mask.astype(np.uint8)
    if not m.any():
        return mask
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, sa.ellipse_kernel(3))
    m = cv2.dilate(m, sa.ellipse_kernel(1))
    return (m > 0) & allowed


def _build_patches(image, clean, items, a) -> list:
    """Chia các điểm ảnh đã bị đổi (clean != image) cho từng câu thoại đã xoá chữ -> mỗi câu 1 ảnh BGRA (alpha 255 ở
    vùng nó sở hữu). Tắt 1 câu = bỏ lớp này, vùng đó trở lại ảnh gốc."""
    H, W = image.shape[:2]
    diff = cv2.absdiff(clean, image).max(axis=2) > 0
    owner = np.full((H, W), -1, np.int16)
    k_el = sa.ellipse_kernel(TOUCH_RADIUS)
    for k, b in enumerate(items):
        h, w = b.strokes.shape
        rx1, ry1 = max(0, b.ox - TOUCH_RADIUS), max(0, b.oy - TOUCH_RADIUS)
        rx2, ry2 = min(W, b.ox + w + TOUCH_RADIUS), min(H, b.oy + h + TOUCH_RADIUS)
        canvas = np.zeros((ry2 - ry1, rx2 - rx1), np.uint8)
        canvas[b.oy - ry1: b.oy - ry1 + h, b.ox - rx1: b.ox - rx1 + w] = b.strokes.astype(np.uint8)
        near = cv2.dilate(canvas, k_el) > 0
        sub = near & diff[ry1:ry2, rx1:rx2] & (owner[ry1:ry2, rx1:rx2] == -1)
        owner[ry1:ry2, rx1:rx2][sub] = k
    for k, b in enumerate(items):                      # điểm ảnh đổi nhưng không nằm gần nét nào: giao cho khung chứa nó
        x1, y1, x2, y2 = b.bbox
        rx1, ry1, rx2, ry2 = max(0, x1 - 60), max(0, y1 - 60), min(W, x2 + 60), min(H, y2 + 60)
        sub = diff[ry1:ry2, rx1:rx2] & (owner[ry1:ry2, rx1:rx2] == -1)
        owner[ry1:ry2, rx1:rx2][sub] = k
    patches = []
    for k, b in enumerate(items):
        x1, y1, x2, y2 = b.bbox
        rx1, ry1, rx2, ry2 = max(0, x1 - 70), max(0, y1 - 70), min(W, x2 + 70), min(H, y2 + 70)
        own = owner[ry1:ry2, rx1:rx2]
        mine = _solidify_mask(own == k, (own == k) | (own == -1))
        ys, xs = np.nonzero(mine)
        if xs.size == 0:
            continue
        cx1, cx2, cy1, cy2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = mine[cy1:cy2, cx1:cx2]
        bgr = clean[ry1 + cy1: ry1 + cy2, rx1 + cx1: rx1 + cx2]
        bgra = np.dstack([bgr, (crop * 255).astype(np.uint8)])
        patches.append(dict(id=int(b.ids[0]), x=int(rx1 + cx1), y=int(ry1 + cy1), bgra=bgra))
    return patches


def _layer_sig(item: dict) -> str:
    """Chữ ký nội dung một lớp (vị trí + điểm ảnh). Lớp có cùng chữ ký ở lần ghi trước thì khỏi mã hoá PNG lại, và viewer
    khỏi giải mã lại."""
    h = hashlib.md5()
    bgra = np.ascontiguousarray(item["bgra"])
    h.update(f"{item['x']},{item['y']},{bgra.shape}".encode())
    h.update(bgra)
    return h.hexdigest()[:20]


def _previous_layers(cdir: Path) -> tuple[Path | None, dict]:
    """(thư mục, {(kind, id): (sig, file)}) của bản lớp gần nhất đã ghi, để tái dùng file PNG của lớp không đổi."""
    for old in sorted(cdir.glob("layers_*"), reverse=True):
        try:
            meta = json.loads((old / "layers.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prev = {}
        for L in meta.get("layers", []):
            for kind in ("patch", "text"):
                if kind in L and L[kind].get("sig"):
                    prev[(kind, L["id"])] = (L[kind]["sig"], L[kind]["file"])
        return old, prev
    return None, {}


def _prev_layer_specs(cdir: Path, W: int, H: int):
    """(thư mục, {(kind, id): spec}) của bộ lớp gần nhất đã ghi (chỉ nhận lớp có chữ ký và file còn đủ); thư mục None nếu không dùng
    được (thiếu / hỏng / khác kích thước ảnh)."""
    for old in sorted(cdir.glob("layers_*"), reverse=True):
        try:
            meta = json.loads((old / "layers.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if list(meta.get("size", [])) != [W, H]:
            return None, {}
        specs = {}
        for L in meta.get("layers", []):
            for kind in ("patch", "text"):
                spec = L.get(kind)
                if spec and spec.get("sig") and (old / spec["file"]).is_file():
                    specs[(kind, int(L["id"]))] = dict(spec)
        return old, specs
    return None, {}


def _write_layers(cdir: Path, W: int, H: int, patches: list, tiles: list, reuse: list | tuple = ()) -> str:
    """Ghi bộ lớp mới. `reuse`: các lớp ĐÃ CÓ ẢNH từ lần ghi trước, chỉ chép file (dict: kind, id, dir, file, x, y, sig) - khỏi tính
    lại điểm ảnh, chữ ký và mã hoá PNG."""
    prev_dir, prev = _previous_layers(cdir)
    d = cdir / f"layers_{int(time.time() * 1000)}"
    d.mkdir(parents=True, exist_ok=True)
    layers: dict = {}
    for r in reuse:
        name = f"{r['kind'][0]}{r['id']}.png"
        shutil.copyfile(Path(r["dir"]) / r["file"], d / name)
        layers.setdefault(r["id"], {"id": r["id"]})[r["kind"]] = dict(file=name, x=r["x"], y=r["y"], sig=r["sig"])
    for kind, seq in (("patch", patches), ("text", tiles)):
        for item in seq:
            name = f"{kind[0]}{item['id']}.png"
            sig = _layer_sig(item)
            old = prev.get((kind, item["id"]))
            if prev_dir is not None and old and old[0] == sig and (prev_dir / old[1]).is_file():
                try:
                    shutil.copyfile(prev_dir / old[1], d / name)  # lớp không đổi: chép file cũ, khỏi mã hoá lại
                    ok = True
                except OSError:
                    ok = False
            else:
                ok = False
            if not ok:
                enc_ok, buf = cv2.imencode(".png", item["bgra"], [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not enc_ok:
                    continue
                buf.tofile(str(d / name))
            layers.setdefault(item["id"], {"id": item["id"]})[kind] = dict(file=name, x=item["x"], y=item["y"], sig=sig)
    write_json_atomic(d / "layers.json", dict(version=1, size=[W, H], layers=list(layers.values())))
    for old in sorted(cdir.glob("layers_*"))[:-3]:        # giữ 3 bản gần nhất (viewer có thể đang đọc bản trước)
        shutil.rmtree(old, ignore_errors=True)
    return str(d / "layers.json")


def compose_from_cache(image_path: str, layers_path: str, hide_redraw=(), hide_text=(),
                       strokes: list | None = None) -> np.ndarray | None:
    """Ghép ảnh cuối từ các lớp đã lưu (dùng khi xuất ảnh): nhanh, không chạy lại redraw. strokes: nét vẽ tay của trang
    (đặt dưới chữ dịch)."""
    image = imread(image_path)
    try:
        meta = json.loads(Path(layers_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if image is None or list(meta.get("size", [])) != [image.shape[1], image.shape[0]]:
        return None
    base = Path(layers_path).parent
    patches, tiles = [], []
    for L in meta.get("layers", []):
        for kind, seq in (("patch", patches), ("text", tiles)):
            if kind in L:
                img = imread(base / L[kind]["file"], cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 4:
                    seq.append(dict(id=L["id"], x=L[kind]["x"], y=L[kind]["y"], bgra=img))
    return compose_layers(image, patches, tiles, set(hide_redraw), set(hide_text), strokes)


# --------------------------------------------------------------------------- main API
def sync_page(image_path: str, dialogues: list[dict], cache_root, opts: RedrawOptions,
              force_full: bool = False, fills: list | None = None, relayout: bool = False,
              fast: bool = True) -> RedrawResult:
    """Đưa ảnh redraw của trang về khớp với `dialogues`.

    relayout=False (mặc định, khi sửa 1 khung): khung nào không đổi giữ nguyên bố cục đã xếp, chỉ xếp lại khung vừa sửa, không
    tự dịch / thu nhỏ các khung xung quanh. relayout=True (dịch / redraw hàng loạt) hoặc trang chưa có bố cục: xếp lại cả
    trang và tự chỉnh cho các khung khỏi đè nhau.

    fast=True: nếu việc xoá chữ gốc không đổi (chỉ sửa chữ / định dạng / vị trí) thì đi ĐƯỜNG TẮT: không nạp lại ảnh đã xoá, không dựng
    lại các vùng xoá, không vẽ lại lớp chữ của khung không đổi, không ghép lại out.png - việc còn lại không phụ thuộc kích thước
    ảnh nhiều như đường đầy đủ. Kết quả (các lớp) giống hệt đường đầy đủ; tự quay về đường đầy đủ khi có bất kỳ điều kiện nào lệch.

    dialogues: list dict {dialogue_id, box_location=[x,y,w,h], original_text, translated_text, overrides?}
    """
    with _LOCK:
        return _sync_page(image_path, dialogues, cache_root, opts, force_full, fills or [], relayout, fast)


def _sync_page(image_path, dialogues, cache_root, opts, force_full, fills, relayout=False, fast=True) -> RedrawResult:
    t0 = time.perf_counter()
    image = imread(image_path)
    if image is None:
        raise RedrawError(f"Không đọc được ảnh: {image_path}")
    H, W = image.shape[:2]
    cdir = cache_dir_for(cache_root, image_path)
    if force_full:
        _wipe_dir(cdir)

    warnings: list[str] = []
    regular, bold = ts.resolve_fonts(opts.font_path)
    if not regular:
        raise RedrawError("Không tìm thấy font hỗ trợ tiếng Việt. Chọn font trong 🎨 Redraw hoặc đặt file .ttf vào thư mục ./fonts")

    lama, lama_err = get_lama(opts)
    if lama_err:
        warnings.append(lama_err)
    a = make_args(opts, H, lama is not None)

    # ---- trạng thái cũ
    osig = _opts_sig(opts, regular, bold, W, H)
    state = _load_state(cdir, image_path, W, H)
    raw = _raw_items(dialogues, W, H)
    items = sa.build_items(raw, W, H, a.merge_thr, False, [])     # không gộp: mỗi câu thoại là 1 khối riêng
    if opts.koharu and items:
        # koharu: gán bóng thoại + vùng chữ cho từng khung. Chữ ký hình học vào khoá xoá (r["kz"]) nên khi koharu cho kết quả khác
        # (đổi model / bật tắt) chữ gốc của khung đó được xoá lại; không bật koharu thì không thêm khoá nào -> cache cũ nguyên vẹn.
        try:
            from app.services import koharu_detector as kd
            layout = kd.get_layout(image_path, opts.koharu_cfg(cache_root))
            for b, r in zip(items, raw):
                b.kz = kd.match_item(layout, b.bbox)
                if b.kz is not None:
                    r["kz"] = b.kz.sig
        except Exception as exc:  # noqa: BLE001  (thiếu torch / model / lỗi suy luận: redraw vẫn chạy bằng cách cũ)
            msg = f"koharu không dùng được ({exc}); đang xác định vùng chữ bằng cách cũ."
            log.warning(msg)
            warnings.append(msg)
    keys: dict[int, str] = {b.gid: _erase_key(b, r) for b, r in zip(items, raw)}

    # ---- ĐƯỜNG TẮT: việc xoá chữ gốc không đổi so với lần ghi trước (cùng ảnh, cùng cài đặt, mỗi khung cùng khoá xoá và cùng
    # "có / không có bản dịch") thì clean.png, vùng xoá của từng câu và out.png đều không đổi -> dùng lại, chỉ xếp lại chữ.
    all_now = {str(int(b.ids[0])): [keys[b.gid], bool(b.text.strip() or r.get("redraw_only")), bool(r.get("no_erase"))]
               for b, r in zip(items, raw) if b.ids}
    prev_layout = None if force_full else _load_layouts(cdir)
    prev_dir, prev_specs = _prev_layer_specs(cdir, W, H)
    # chữ ký các vùng Fill (bỏ cờ ẩn: ẩn/hiện chỉ đổi lúc ghép, không đổi ảnh lớp). Trùng với lần trước => lớp Fill cũ dùng lại nguyên,
    # không cần chạy lại cả trang chỉ vì trang có vùng Fill.
    fsig = json.dumps([[int(f["id"]), f.get("strokes"), f.get("engine")] for f in fills], sort_keys=True, default=str) if fills else ""
    prev_fsig = prev_layout.get("fills") if prev_layout else None
    if prev_fsig is None and prev_layout:   # layouts.json cũ chưa ghi chữ ký: chỉ coi là "không Fill" nếu bộ lớp cũ không có lớp Fill
        prev_fsig = "" if not any(k[0] == "patch" and k[1] < 0 for k in prev_specs) else None
    use_fast = bool(
        fast and state and prev_layout is not None and prev_fsig == fsig and prev_layout and prev_layout.get("opts") == osig
        and prev_layout.get("all") == all_now and prev_dir is not None and prev_layout.get("layers") == prev_dir.name
        and all((cdir / n).is_file() for n in ("out.png", "clean.png", "touched.png"))
    )
    clean = touched = None
    if state and not use_fast:
        clean = imread(cdir / "clean.png")
        touched = imread(cdir / "touched.png", cv2.IMREAD_GRAYSCALE)
        if clean is None or touched is None or clean.shape != image.shape or touched.shape != (H, W):
            state, clean, touched = None, None, None
    if state is None:
        clean, touched = image.copy(), np.zeros((H, W), np.uint8)
        old = {}
    else:
        old = {e["key"]: e["bbox"] for e in state.get("erased", [])}

    # ---- phân tích mọi khung thoại trên ẢNH GỐC
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for b, r in zip(items, raw):
        b.rotation = float(r.get("rotation") or 0.0)
        sa.analyze(gray, b, a, others=[o.bbox for o in items if o is not b])
        if b.kind == "sfx":
            b.cleanup = False
        if not use_fast:                       # quyết định cách lấp chỗ chữ gốc: chỉ cần khi có thể phải xoá chữ
            sa.decide_fill(image, gray, b, a)
            log.info("Khung %s: loại=%s (%s) · lấp=%s · engine=%s · %s", b.ids, b.kind, b.method, b.fill_mode, b.engine, b.why)
    th = [b.thickness for b in items if b.thickness > 0 and b.kind != "sfx"]
    med_th = float(np.median(th)) if th else 0.0
    for b in items:
        # Chữ luôn THƯỜNG: tuỳ chọn in đậm của người dùng đã bỏ (override style = "bold" của project cũ bị bỏ qua).
        # Chỉ khi bật auto_bold (đoán đậm theo độ dày nét gốc, cách cũ của pipeline v4; mặc định tắt) mới có chữ đậm.
        if opts.auto_bold and med_th > 0 and b.kind in ("bubble", "free"):
            b.bold = b.thickness > a.bold_ratio * med_th
        else:
            b.bold = False

    # ---- khung nào cần xoá chữ gốc: có bản dịch + cho phép cleanup + không phải SFX
    only = {b.gid for b, r in zip(items, raw) if r.get("redraw_only")}
    want = {keys[b.gid]: b for b, r in zip(items, raw)
            if (b.text.strip() or b.gid in only) and b.cleanup and b.kind != "sfx" and not r.get("no_erase")}
    to_restore = [k for k in old if k not in want] if not use_fast else []
    to_erase = [k for k in want if k not in old] if not use_fast else []

    if to_restore:
        kept = [want[k] for k in want if k in old]
        _restore(clean, touched, image, [tuple(old[k]) for k in to_restore], kept, a)
        for k in to_restore:
            old.pop(k)
    erased_now = 0
    if to_erase:
        new_items = [want[k] for k in to_erase]
        mask = _erase(clean, image, new_items, items, a, lama)
        touched |= cv2.dilate(mask, sa.ellipse_kernel(TOUCH_RADIUS))
        for k in to_erase:
            old[k] = list(want[k].bbox)
        erased_now = len(to_erase)

    # ---- lưu trước khi xếp chữ để công xoá chữ không mất nếu bước sau lỗi
    if to_restore or to_erase or state is None:
        imwrite_atomic(cdir / "clean.png", clean)
        imwrite_atomic(cdir / "touched.png", touched)
        sig = Path(image_path).stat()
        write_json_atomic(cdir / "state.json", dict(
            version=STATE_VERSION, size=[W, H], image=[sig.st_size, sig.st_mtime_ns],
            erased=[dict(key=k, bbox=v) for k, v in old.items()], lama=lama is not None))

    # ---- xếp chữ dịch (mỗi câu 1 lớp riêng) + vùng xoá của từng câu -> ghi lớp, ghép ảnh cuối theo cờ ẩn/hiện
    # bố cục đã xếp của từng khung (lần xếp đầu / redraw hàng loạt) -> sửa 1 khung thì các khung còn lại giữ nguyên
    raw_by_id = {int(b.ids[0]): r for b, r in zip(items, raw) if b.ids}
    block_sigs = {k: _block_sig(r) for k, r in raw_by_id.items()}
    prev = None if (force_full or relayout) else prev_layout
    frozen = dirty = frozen_best = None
    shifted: dict = {}                                   # id -> (dx, dy): khối chỉ bị dời khung
    reuse_ok: set = set()                                # id khối đóng băng có thể dùng lại nguyên ảnh lớp chữ cũ
    if prev and prev.get("opts") == osig:
        old_blocks = {int(k): v for k, v in prev.get("blocks", {}).items()}
        keep, keep_best = {}, {}
        for k, r in raw_by_id.items():
            ob = old_blocks.get(k)
            if ob is None:
                continue
            rot_same = abs(float(ob.get("rot", 0.0)) - float(r.get("rotation") or 0.0)) < 1e-6
            if ob.get("sig") == block_sigs[k]:
                keep[k], keep_best[k] = ob["lay"], ob.get("best", ob["lay"]["size"])
                if rot_same:
                    reuse_ok.add(k)
                continue
            # CHỈ dời khung chữ (cùng kích thước, mọi thứ khác không đổi) -> dời nguyên khối chữ, khỏi xếp lại: cỡ chữ và cách
            # ngắt dòng giữ nguyên (xếp lại ở chỗ mới có thể ra cỡ khác vì khung / bóng thoại ở đó khác)
            reg = r.get("region")
            old_frame = ob.get("region") or ob.get("bbox")
            pad2 = 0 if ob.get("region") else 2 * AUTO_FRAME_PAD     # khung cũ là bbox khối chữ (chưa đệm) hay khung người dùng đặt (đã đệm sẵn)
            if (reg and old_frame and ob.get("sig_nr") == _block_sig(r, ignore_region=True)
                    and abs((reg[2] - reg[0]) - (old_frame[2] - old_frame[0]) - pad2) <= 1
                    and abs((reg[3] - reg[1]) - (old_frame[3] - old_frame[1]) - pad2) <= 1):
                # dời theo TÂM khung (đệm đối xứng nên tâm không đổi, còn góc trái-trên thì lệch pad)
                dx = int(round(((reg[0] + reg[2]) - (old_frame[0] + old_frame[2])) / 2.0))
                dy = int(round(((reg[1] + reg[3]) - (old_frame[1] + old_frame[3])) / 2.0))
                keep[k] = _translated_layout(ob, reg, dx, dy)
                keep_best[k] = ob.get("best", ob["lay"]["size"])
                shifted[k] = (dx, dy)
                if rot_same:
                    reuse_ok.add(k)
        dirty = {k for k in block_sigs if k not in keep}
        if len(dirty) <= max(1, len(block_sigs) // 2):   # đổi nhiều khung cùng lúc (dịch lại cả trang...) -> xếp lại cả trang
            frozen, frozen_best = keep, keep_best
        else:
            dirty, shifted, reuse_ok = None, {}, set()

    # ảnh lớp chữ dùng lại (khối không đổi, hoặc chỉ bị dời): chép file cũ thay vì vẽ + mã hoá lại
    reuse: list = []
    if use_fast and frozen is not None:
        for k in sorted(reuse_ok):
            spec = prev_specs.get(("text", k))
            if spec is None:
                reuse_ok.discard(k)
                continue
            dx, dy = shifted.get(k, (0, 0))
            x, y = spec["x"] + dx, spec["y"] + dy
            reuse.append(dict(kind="text", id=k, dir=prev_dir, file=spec["file"], x=x, y=y,
                              sig=spec["sig"] if (dx, dy) == (0, 0) else _shift_sig(spec["sig"], x, y)))
    else:
        reuse_ok = set()
    stats = ts.typeset_page(image if use_fast else clean, items, gray, a, regular, bold, draw=False, frozen=frozen, dirty=dirty,
                            frozen_best=frozen_best, skip_tiles=reuse_ok)
    stats["shifted"] = sorted(shifted)
    stats["fast"] = use_fast
    tiles = stats.pop("tiles")
    hide_text = {b.ids[0] for b, r in zip(items, raw) if r.get("hide_text") and b.ids}
    hide_redraw = {b.ids[0] for b, r in zip(items, raw) if r.get("hide_redraw") and b.ids}
    if use_fast:
        # vùng xoá của từng câu không đổi: dùng lại nguyên bộ lớp vá của lần trước
        patches = []
        reuse += [dict(kind="patch", id=k[1], dir=prev_dir, file=spec["file"], x=spec["x"], y=spec["y"], sig=spec["sig"])
                  for k, spec in prev_specs.items() if k[0] == "patch"]
    else:
        patches = _build_patches(image, clean, [b for b in items if keys[b.gid] in old and b.ids], a)
        patches += _apply_fills(clean, fills, a, lama, cdir, warnings)     # Content-Aware Fill: lớp riêng, nằm trên cùng
    hide_redraw |= {-int(f["id"]) for f in fills if f.get("hidden")}
    layers_path = _write_layers(cdir, W, H, patches, tiles, reuse)
    try:
        lays, bests = stats.pop("layouts", {}), stats.pop("bests", {})
        bboxes = {blk["ids"][0]: _rects_bbox(blk["rects"]) for blk in stats.get("blocks", []) if blk["ids"]}
        colors = {blk["ids"][0]: blk.get("color") for blk in stats.get("blocks", []) if blk["ids"]}   # RGB; dùng khi xuất lớp chữ PSD
        engines_prev = (prev_layout or {}).get("engines", {}) if use_fast else None
        if use_fast:
            engines = engines_prev
        else:
            engines = {}
            for b in want.values():
                engines[b.engine] = engines.get(b.engine, 0) + 1
        # ghi SAU khi bộ lớp đã ghi xong và ghi tên thư mục lớp: lần sau chỉ tin bố cục này nếu khớp đúng bộ lớp ấy
        write_json_atomic(cdir / "layouts.json", dict(
            version=LAYOUT_VERSION, opts=osig, layers=Path(layers_path).parent.name, all=all_now, engines=engines, fills=fsig,
            blocks={str(k): dict(sig=block_sigs.get(k), sig_nr=_block_sig(raw_by_id[k], ignore_region=True) if k in raw_by_id else None,
                                 region=raw_by_id.get(k, {}).get("region"), bbox=bboxes.get(k), best=bests.get(k),
                                 rot=float(raw_by_id[k].get("rotation") or 0.0) if k in raw_by_id else 0.0,
                                 color=colors.get(k),
                                 lay=_jsonable(v)) for k, v in lays.items()}))
    except OSError as exc:  # không lưu được thì lần sau chỉ xếp lại cả trang, không làm hỏng redraw
        log.warning("Không lưu được bố cục từng khung: %s", exc)
        engines = {}
    out_path = cdir / "out.png"
    if not use_fast:      # đường tắt: ảnh ghép toàn trang không đổi và không ai đọc (hiển thị / xuất đều ghép từ các lớp)
        page = compose_layers(image, patches, tiles, hide_redraw, hide_text)
        imwrite_atomic(out_path, page)
    if stats.get("overlap_unresolved"):
        warnings.append(f"{stats['overlap_unresolved']} cặp câu thoại vẫn lấn nhau ở cỡ chữ nhỏ nhất; hãy thu nhỏ/sửa bản dịch.")
    if stats.get("overflow"):
        warnings.append(f"{stats['overflow']} câu thoại quá dài, không vừa khung ở cỡ chữ nhỏ nhất.")
    if stats.get("bad_fonts"):
        warnings.append("Font riêng không dùng được (thiếu file hoặc không có dấu tiếng Việt), đã dùng font mặc định: "
                        + ", ".join(Path(p).name for p in stats["bad_fonts"]))
    stats["engines"] = engines
    dt = time.perf_counter() - t0
    log.info("Redraw %s: xoá %d, khôi phục %d, %.2fs, engine=%s, bố cục=%s%s", Path(image_path).name, erased_now, len(to_restore), dt, engines,
             f"tăng dần ({len(stats.get('relaid', []))} khung xếp lại)" if stats.get("incremental") else "đầy đủ",
             ", đường tắt" if use_fast else "")
    layout, info = {}, {}
    for blk in stats.get("blocks", []):
        if blk["ids"]:
            info[blk["ids"][0]] = dict(size=blk["size"], font=blk["font"], line_spacing=blk["line_spacing"],
                                       bold=blk["bold"], color=blk["color"], outline=blk["outline"],
                                       outline_width=blk["outline_width"], outline_color=blk["outline_color"], align=blk.get("align", "center"))
    for blk in stats.get("blocks", []):
        if blk["rects"] and blk["ids"]:
            x1 = min(r[0] for r in blk["rects"]); y1 = min(r[1] for r in blk["rects"])
            x2 = max(r[2] for r in blk["rects"]); y2 = max(r[3] for r in blk["rects"])
            layout[blk["ids"][0]] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    return RedrawResult(out_path=str(out_path), clean_path=str(cdir / "clean.png"), erased=erased_now,
                        restored=len(to_restore), seconds=dt, warnings=warnings, stats=stats, layout=layout,
                        layers_path=layers_path, info=info)
