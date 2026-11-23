"""
Bộ phát hiện bố cục trang truyện bằng mayocream/koharu-yolo26s (YOLO26s-seg, 4 lớp, có mask đa giác).

Lớp: 0 frame (khung tranh) | 1 dialogue_text (vùng chữ thoại) | 2 balloon (bóng thoại) | 3 onomatopoeia_text (SFX).

Dùng ở HAI nơi, chạy model đúng MỘT lần cho mỗi trang (kết quả cache RAM + đĩa):
  1. Dịch (HybridOcr / YoloGeminiVision): `detection_units()` -> danh sách "đơn vị cần dịch" (bóng thoại có chữ + chữ tự do
     ngoài bóng), kèm vùng cắt cho OCR (xoá trắng phần ngoài bóng) thay cho watershed của extract_bubble_mask.
  2. Redraw (engine.sync_page): `match_item()` gán cho từng khung thoại: bóng thoại nào + vùng chữ nào của nó. stage_a.analyze()
     dùng để (a) lấy lòng bóng chính xác kể cả bóng hở viền / bóng gai, (b) chỉ xoá nét NẰM TRONG vùng chữ.

Nguyên tắc: koharu cho biết "Ở ĐÂU" (đa giác khá thô, sai số vài px), ngưỡng sáng/tối trong stage_a quyết định "ĐIỂM ẢNH NÀO".
Không dùng đa giác làm mask xoá trực tiếp (trừ SFX theo yêu cầu).

ultralytics / torch / safetensors / huggingface_hub chỉ import LÚC GỌI model (app không bật koharu vẫn chạy bình thường).
Phần so khớp + dựng mask chỉ cần numpy + opencv nên test được không cần torch.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("koharu_detector")

KOHARU_REPO = "mayocream/koharu-yolo26s"
C_FRAME, C_TEXT, C_BALLOON, C_SFX = 0, 1, 2, 3
CLASS_NAMES = {C_FRAME: "frame", C_TEXT: "dialogue_text", C_BALLOON: "balloon", C_SFX: "onomatopoeia_text"}
# Ngưỡng conf dùng khi DÙNG kết quả (model luôn chạy với conf thấp rồi lọc ở đây, nên đổi ngưỡng không phải chạy lại model)
MIN_CONF = {C_FRAME: 0.50, C_TEXT: 0.35, C_BALLOON: 0.50, C_SFX: 0.40}
LAYOUT_SCHEMA = 1


# --------------------------------------------------------------------------- cấu hình
@dataclass
class KoharuConfig:
    enabled: bool = False
    model_dir: str = ""          # thư mục chứa model.safetensors + yolo26s-seg.yaml + config.json; trống = tải từ HuggingFace
    repo: str = KOHARU_REPO
    infer_conf: float = 0.25     # conf chạy model (thấp); lọc theo MIN_CONF khi dùng
    imgsz: int = 0               # 0 = lấy từ config.json (1280); CPU chậm thì đặt 960
    device: str = "cpu"
    threads: int = 0
    erase_sfx: bool = False      # thêm vùng "chỉ redraw" cho SFX khi dịch (mặc định tắt: SFX giữ nguyên)
    use_yolo: bool = True        # có file YOLO bóng thoại thì kết hợp (hybrid); False = chỉ koharu (1 lượt model / trang, nhanh hơn)
    cache_dir: str = ""          # trống = <AppData>/redraw_cache/koharu (cùng thư mục với cache redraw)

    def model_key(self) -> str:
        return f"{self.model_dir or self.repo}|{self.imgsz}|{self.infer_conf}"


_CFG = KoharuConfig()


def configure(cfg: KoharuConfig) -> None:
    """Đặt cấu hình dùng chung (MainWindow gọi khi mở app và sau khi lưu cài đặt); các service dịch đọc qua current()."""
    global _CFG
    _CFG = cfg


def current() -> KoharuConfig:
    return _CFG


def default_cache_dir() -> str:
    try:
        from app.qt_compat import QStandardPaths
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        root = Path(base) if base else Path.home() / ".manga_translator"
    except Exception:  # noqa: BLE001  (không có Qt: dùng thư mục người dùng)
        root = Path.home() / ".manga_translator"
    return str(root / "redraw_cache" / "koharu")


# --------------------------------------------------------------------------- dữ liệu
@dataclass
class Det:
    cls: int
    conf: float
    box: Tuple[float, float, float, float]          # x1, y1, x2, y2 (toạ độ ảnh gốc)
    poly: Optional[np.ndarray] = None                # (N, 2) float32, toạ độ ảnh gốc; None nếu model không trả mask

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    def xywh(self) -> List[int]:
        x1, y1, x2, y2 = self.box
        return [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]


@dataclass
class PageLayout:
    size: Tuple[int, int]                            # (W, H)
    dets: List[Det] = field(default_factory=list)

    def of(self, cls: int) -> List[Det]:
        return [d for d in self.dets if d.cls == cls]

    @property
    def balloons(self) -> List[Det]:
        return self.of(C_BALLOON)

    @property
    def texts(self) -> List[Det]:
        return self.of(C_TEXT)

    @property
    def sfx(self) -> List[Det]:
        return self.of(C_SFX)

    def to_json(self) -> dict:
        return dict(schema=LAYOUT_SCHEMA, size=list(self.size), dets=[
            dict(cls=d.cls, conf=round(float(d.conf), 3), box=[round(float(v), 1) for v in d.box],
                 poly=[] if d.poly is None else np.round(d.poly, 1).tolist()) for d in self.dets])

    @classmethod
    def from_json(cls, o: dict) -> "PageLayout":
        dets = []
        for e in o.get("dets", []):
            p = np.asarray(e.get("poly") or [], np.float32).reshape(-1, 2)
            dets.append(Det(int(e["cls"]), float(e["conf"]), tuple(float(v) for v in e["box"]), p if len(p) >= 3 else None))
        return cls(size=tuple(o["size"]), dets=dets)


# --------------------------------------------------------------------------- hình học
def _inter(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def iou(a, b) -> float:
    i = _inter(a, b)
    return i / max(1e-6, _area(a) + _area(b) - i)


def _poly_or_box(d: Det) -> np.ndarray:
    if d.poly is not None and len(d.poly) >= 3:
        return d.poly.astype(np.float32)
    x1, y1, x2, y2 = d.box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def inside_ratio(inner: Det, outer: Det) -> float:
    """Tỉ lệ 'inner' nằm trong đa giác của 'outer', ước lượng bằng lưới 5x5 điểm trong hộp của inner (rẻ, đủ cho việc gán)."""
    poly = _poly_or_box(outer).reshape(-1, 1, 2)
    x1, y1, x2, y2 = inner.box
    n = hit = 0
    for fx in (0.1, 0.3, 0.5, 0.7, 0.9):
        for fy in (0.1, 0.3, 0.5, 0.7, 0.9):
            n += 1
            if cv2.pointPolygonTest(poly, (float(x1 + fx * (x2 - x1)), float(y1 + fy * (y2 - y1))), False) >= 0:
                hit += 1
    return hit / n


def poly_mask(poly: np.ndarray, shape: Tuple[int, int], ox: int = 0, oy: int = 0) -> np.ndarray:
    """Đa giác (toạ độ ảnh gốc) -> mask bool trong ROI có gốc (ox, oy)."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(poly - np.array([ox, oy], np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m > 0


def det_mask(d: Det, shape: Tuple[int, int], ox: int = 0, oy: int = 0) -> np.ndarray:
    return poly_mask(_poly_or_box(d), shape, ox, oy)


def dedupe(dets: List[Det], contain: float = 0.7) -> List[Det]:
    """Bỏ hộp cùng lớp chồng gần hết lên nhau (SFX hay bị lặp), giữ conf cao hơn."""
    out: List[Det] = []
    for d in sorted(dets, key=lambda e: -e.conf):
        small = d.area
        if any(o.cls == d.cls and _inter(o.box, d.box) >= contain * max(1.0, min(small, o.area)) for o in out):
            continue
        out.append(d)
    return out


def filter_layout(layout: PageLayout, min_conf: Optional[Dict[int, float]] = None) -> PageLayout:
    mc = dict(MIN_CONF, **(min_conf or {}))
    kept = [d for d in layout.dets if d.conf >= mc.get(d.cls, 0.5) and d.area >= 16]
    return PageLayout(layout.size, dedupe(kept))


# --------------------------------------------------------------------------- nạp model + chạy
_LOAD_LOCK = threading.RLock()
_MODELS: Dict[str, Any] = {}
_FAILED: Dict[str, str] = {}
_MEM: "OrderedDict[str, PageLayout]" = OrderedDict()
_MEM_MAX = 24


def _resolve_dir(cfg: KoharuConfig) -> Path:
    if cfg.model_dir:
        d = Path(cfg.model_dir)
        if not (d / "model.safetensors").is_file():
            raise FileNotFoundError(f"Thư mục model koharu thiếu model.safetensors: {d}")
        return d
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(cfg.repo, allow_patterns=["*.safetensors", "*.yaml", "*.json"]))


def _load_model(cfg: KoharuConfig):
    key = cfg.model_key()
    got = _MODELS.get(key)
    if got is not None:
        return got
    if key in _FAILED:                                         # lỗi nạp được nhớ: không thử lại (và không chờ tải lại) ở mỗi trang
        raise RuntimeError(_FAILED[key])
    with _LOAD_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        try:
            return _load_model_locked(cfg, key)
        except Exception as exc:  # noqa: BLE001
            _FAILED[key] = f"Không nạp được koharu: {exc}"
            raise


def _load_model_locked(cfg: KoharuConfig, key: str):
    import torch
    from safetensors.torch import load_file
    from ultralytics import YOLO
    from app.services.manga_ocr_pipeline import _SerializedModel     # cùng khoá suy luận với YOLO / MangaOcr của app

    d = _resolve_dir(cfg)
    info = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").is_file() else {}
    yaml = next(iter(sorted(d.glob("*.yaml"))), None)
    if yaml is None:
        raise FileNotFoundError(f"Thiếu file kiến trúc *.yaml trong {d}")
    log.info("Đang nạp koharu-yolo26s từ %s ...", d)
    model = YOLO(str(yaml))
    state = load_file(str(d / "model.safetensors"))
    try:
        model.model.load_state_dict(state, strict=True)
    except RuntimeError:                                   # tiền tố "model." thừa / thiếu
        state = {k.removeprefix("model."): v for k, v in state.items()}
        model.model.load_state_dict(state, strict=True)
    names = info.get("names") or CLASS_NAMES
    model.model.names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    from app.threading_utils import apply_torch_threads
    apply_torch_threads(cfg.threads)   # một lần, dưới khoá chung (set_num_threads song song với suy luận có thể abort)
    model.to(cfg.device or "cpu")
    imgsz = int(cfg.imgsz or info.get("imgsz", 1280))
    _MODELS[key] = (_SerializedModel(model), imgsz)
    log.info("Đã nạp koharu (imgsz=%d, device=%s)", imgsz, cfg.device)
    return _MODELS[key]


def _imread(path: str):
    data = np.fromfile(str(path), dtype=np.uint8)           # đọc được đường dẫn Unicode trên Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _run_model(image_path: str, cfg: KoharuConfig) -> PageLayout:
    model, imgsz = _load_model(cfg)
    img = _imread(image_path)
    if img is None:
        raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    H, W = img.shape[:2]
    res = model.predict(img, imgsz=imgsz, conf=cfg.infer_conf, device=cfg.device or "cpu", verbose=False)[0]
    dets: List[Det] = []
    if res.boxes is not None:
        for i, b in enumerate(res.boxes):
            poly = None
            if res.masks is not None and i < len(res.masks.xy):
                p = np.asarray(res.masks.xy[i], np.float32).reshape(-1, 2)
                if len(p) >= 3:
                    poly = cv2.approxPolyDP(p.reshape(-1, 1, 2), 0.8, True).reshape(-1, 2).astype(np.float32)
                    if len(poly) < 3:
                        poly = p
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            dets.append(Det(int(b.cls), float(b.conf), (max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)), poly))
    return PageLayout((W, H), dets)


def get_layout(image_path: str, cfg: Optional[KoharuConfig] = None, run: bool = True) -> Optional[PageLayout]:
    """Bố cục thô của trang (đã lọc conf + bỏ trùng). Cache RAM + đĩa theo (đường dẫn, kích thước, mtime, model, conf, imgsz).
    run=False: chỉ đọc cache, không chạy model (trả None nếu chưa có)."""
    cfg = cfg or _CFG
    p = Path(image_path)
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{cfg.model_key()}|{LAYOUT_SCHEMA}".encode()).hexdigest()[:20]
    with _LOAD_LOCK:
        if key in _MEM:
            _MEM.move_to_end(key)
            return filter_layout(_MEM[key])
    cdir = Path(cfg.cache_dir or default_cache_dir())
    f = cdir / f"{key}.json"
    raw: Optional[PageLayout] = None
    if f.is_file():
        try:
            raw = PageLayout.from_json(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache koharu hỏng (%s), chạy lại model: %s", f.name, exc)
    if raw is None:
        if not run:
            return None
        raw = _run_model(image_path, cfg)
        try:
            cdir.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_text(json.dumps(raw.to_json(), ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, f)
        except OSError as exc:
            log.warning("Không ghi được cache koharu: %s", exc)
    with _LOAD_LOCK:
        _MEM[key] = raw
        while len(_MEM) > _MEM_MAX:
            _MEM.popitem(last=False)
    return filter_layout(raw)


# --------------------------------------------------------------------------- gán cho khung thoại (dùng trong redraw)
@dataclass
class ItemMatch:
    balloon: Optional[Det] = None
    texts: List[Det] = field(default_factory=list)       # vùng chữ thuộc khung này
    sfx: List[Det] = field(default_factory=list)         # SFX nằm trong khung (dùng cho type="mask")
    sig: str = "0"                                       # chữ ký hình học: đổi => xoá lại chữ gốc của khung

    @property
    def usable(self) -> bool:
        return self.balloon is not None or bool(self.texts) or bool(self.sfx)


def _sig(parts: Sequence[Det]) -> str:
    s = "|".join(f"{d.cls}:{','.join(str(int(round(v))) for v in d.box)}" for d in parts)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def match_item(layout: Optional[PageLayout], bbox: Sequence[float]) -> Optional[ItemMatch]:
    """Gán bóng thoại + vùng chữ + SFX cho khung `bbox` (x1, y1, x2, y2). None nếu không khớp gì."""
    if layout is None:
        return None
    x1, y1, x2, y2 = bbox
    ib = (float(x1), float(y1), float(x2), float(y2))
    me = Det(-1, 1.0, ib)
    best, best_s = None, 0.0
    for d in layout.balloons:
        s = max(iou(ib, d.box), 0.9 * inside_ratio(me, d) if _area(ib) > 0 else 0.0,
                0.9 * _inter(ib, d.box) / max(1.0, d.area))     # khung ôm cả bóng / khung nằm gọn trong bóng / hai khung gần trùng
        if s > best_s:
            best, best_s = d, s
    if best is not None and best_s < 0.45:
        best = None

    texts: List[Det] = []
    if best is not None:
        texts = [t for t in layout.texts if inside_ratio(t, best) >= 0.6]
        if len(texts) >= 2:   # bóng koharu dính nhiều bóng: chỉ lấy chữ giao với khung này (không xoá nhầm chữ của câu bên cạnh)
            mine = [t for t in texts if _inter(ib, t.box) >= 0.3 * max(1.0, t.area)]
            if mine:
                texts = mine
    else:
        others = layout.balloons
        for t in layout.texts:
            if _inter(ib, t.box) < 0.5 * max(1.0, t.area):
                continue
            if any(inside_ratio(t, o) >= 0.6 for o in others):   # chữ của một bóng thoại khác: không phải chữ tự do của khung này
                continue
            texts.append(t)
    sfx = [s for s in layout.sfx if _inter(ib, s.box) >= 0.5 * max(1.0, s.area)
           and not (best is not None and inside_ratio(s, best) >= 0.6)]
    if best is None and not texts and not sfx:
        return None
    return ItemMatch(best, texts, sfx, _sig(([best] if best else []) + texts + sfx))


@dataclass
class Zone:
    interior: Optional[np.ndarray]    # bool ROI-local: lòng bóng theo koharu (CHƯA co / chưa bỏ viền); None nếu không có bóng
    text: Optional[np.ndarray]        # bool ROI-local: vùng chữ đã nở thêm `pad`; None nếu koharu không thấy chữ nào
    sfx: Optional[np.ndarray]         # bool ROI-local: đa giác SFX trong khung; None nếu không có


def rasterize(m: ItemMatch, shape: Tuple[int, int], ox: int, oy: int, zone_pad: int = 6, box_pad: int = 4) -> Zone:
    """Dựng mask trong ROI (gốc ox, oy). Vùng chữ = hợp của (đa giác nở zone_pad) và (hộp chữ nở box_pad): đa giác của
    koharu khá thô nên lấy hợp với hộp để không cắt cụt đầu / đuôi cột chữ."""
    h, w = shape
    interior = det_mask(m.balloon, shape, ox, oy) if m.balloon is not None else None
    text = None
    if m.texts:
        text = np.zeros(shape, bool)
        for t in m.texts:
            pm = det_mask(t, shape, ox, oy)
            if zone_pad > 0:
                pm = cv2.dilate(pm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * zone_pad + 1,) * 2)) > 0
            text |= pm
            bx1, by1, bx2, by2 = (int(round(v)) for v in t.box)
            text[max(0, by1 - oy - box_pad):max(0, by2 - oy + box_pad), max(0, bx1 - ox - box_pad):max(0, bx2 - ox + box_pad)] = True
    sfx = None
    if m.sfx:
        sfx = np.zeros(shape, bool)
        for s in m.sfx:
            sfx |= det_mask(s, shape, ox, oy)
    return Zone(interior, text, sfx)


# --------------------------------------------------------------------------- đơn vị cần dịch (dùng ở bước OCR / dịch)
@dataclass
class Unit:
    kind: str                         # "balloon" | "free"
    box: List[int]                    # [x, y, w, h] -> Dialogue.box_location (bóng: hộp cả bóng; tự do: hộp chữ)
    crop: Tuple[int, int, int, int]   # x1, y1, x2, y2 vùng cắt đưa cho OCR
    balloon: Optional[Det] = None
    texts: List[Det] = field(default_factory=list)
    others: List[Det] = field(default_factory=list)   # chữ của bóng KHÁC lấn vào vùng cắt: tô trắng trước khi đưa cho OCR
    frame: int = -1                   # thứ hạng khung tranh (panel) theo thứ tự đọc; -1 = không có khung nào


def _reading_key(box: Sequence[int], band: int = 100):
    return (box[1] // band, -box[0])


# --------------------------------------------------------------------------- thứ tự đọc theo khung tranh (lớp frame của koharu)
def _shrunk(b: Sequence[float], f: float = 0.03) -> Tuple[float, float, float, float]:
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] + f * w, b[1] + f * h, b[2] - f * w, b[3] - f * h)


def _split_axis(boxes: List[Tuple[float, float, float, float]], idx: List[int], axis: int) -> List[List[int]]:
    """Chia theo trục (0 = x, 1 = y): khe trống không khung nào bắt qua -> nhóm mới."""
    order = sorted(idx, key=lambda i: boxes[i][axis])
    groups = [[order[0]]]
    cur_max = boxes[order[0]][axis + 2]
    for i in order[1:]:
        if boxes[i][axis] > cur_max:
            groups.append([i])
            cur_max = boxes[i][axis + 2]
        else:
            groups[-1].append(i)
            cur_max = max(cur_max, boxes[i][axis + 2])
    return groups


def order_frames(frames: Sequence[Sequence[float]]) -> List[int]:
    """Thứ tự đọc manga (hàng trên -> dưới, trong hàng phải -> trái) cho các khung [x1, y1, x2, y2] bằng XY-cut đệ quy.
    Khung cao cắt ngang nhiều hàng (cột bên phải cao bằng hai khung bên trái) vẫn đúng vì thử cắt ngang trước, không được thì cắt dọc.
    Không cắt được (khung chồng / xoay vòng) thì lùi về (y trên, x phải)."""
    boxes = [_shrunk(b) for b in frames]

    def cut(idx: List[int]) -> List[int]:
        if len(idx) <= 1:
            return list(idx)
        for axis in (1, 0):
            groups = _split_axis(boxes, idx, axis)
            if len(groups) > 1:
                if axis == 0:
                    groups = groups[::-1]            # phải -> trái
                out: List[int] = []
                for g in groups:
                    out += cut(g)
                return out
        return sorted(idx, key=lambda i: (boxes[i][1], -boxes[i][2]))

    return cut(list(range(len(boxes))))


def _assign_frame(box_xyxy: Sequence[float], frames: Sequence[Sequence[float]]) -> int:
    best, best_a = -1, 0.0
    for i, f in enumerate(frames):
        a = _inter(box_xyxy, f)
        if a > best_a:
            best, best_a = i, a
    if best >= 0:
        return best
    cx, cy = (box_xyxy[0] + box_xyxy[2]) / 2, (box_xyxy[1] + box_xyxy[3]) / 2      # chữ nằm ngoài mọi khung (trên rãnh giữa các khung)
    def dist(f):
        dx = max(f[0] - cx, 0.0, cx - f[2])
        dy = max(f[1] - cy, 0.0, cy - f[3])
        return dx * dx + dy * dy
    return min(range(len(frames)), key=lambda i: dist(frames[i]))


def order_units(layout: PageLayout, units: List[Unit]) -> List[Unit]:
    """Sắp các đơn vị theo thứ tự đọc: khung tranh (nếu koharu thấy khung) -> trong khung: hàng trên trước, trong hàng phải -> trái.
    Không có khung nào: dùng cách cũ (dải 100 px). Gán u.frame = thứ hạng khung (cho prompt Gemini: câu cùng khung = cùng cảnh)."""
    frames = [d.box for d in layout.of(C_FRAME)]
    if not frames or not units:
        for u in units:
            u.frame = -1
        return sorted(units, key=lambda u: _reading_key(u.box))
    rank = {fi: r for r, fi in enumerate(order_frames(frames))}
    keyed = []
    for u in units:
        x, y, w, h = u.box
        fi = _assign_frame((x, y, x + w, y + h), frames)
        u.frame = rank[fi]
        f = frames[fi]
        band = min(200.0, max(60.0, 0.15 * (f[3] - f[1])))
        keyed.append(((u.frame, int((y - f[1]) // band), -(x + w / 2.0)), u))
    keyed.sort(key=lambda t: t[0])
    return [u for _, u in keyed]


def _balloon_has_ink(gray: np.ndarray, b: Det, min_frac: float = 0.008) -> bool:
    x1, y1, x2, y2 = (int(round(v)) for v in b.box)
    sub = gray[max(0, y1):y2, max(0, x1):x2]
    if sub.size == 0:
        return False
    m = poly_mask(_poly_or_box(b), sub.shape, max(0, x1), max(0, y1))
    m = cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0   # bỏ viền bóng
    return bool(m.any()) and float((sub[m] < 120).mean()) >= min_frac


# Tách bóng thoại dính nhau: model hay trả MỘT đa giác bóng cho hai bóng liền nhau (không có viền ngăn) -> cùng một Unit -> một cụm text.
SPLIT_FUSED_BALLOONS = True   # tắt = hành vi cũ (mỗi bóng của model là một đơn vị)
_CORE_FRAC = 0.50             # lõi bóng = vùng có khoảng cách tới viền >= 50% bán kính lớn nhất
_NECK_RATIO = 0.75            # chỉ tách khi "cổ" nối hẹp hơn 75% bán kính của bóng nhỏ hơn trong hai bóng


def _lobe_labels(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[int, float]]:
    """Chia mask một bóng thành các 'thùy' (bóng con) theo cổ thắt. Trả (nhãn int32, {nhãn: bán kính}) hoặc (None, {}) nếu chỉ có 1 thùy.
    Lõi = vùng cách viền >= _CORE_FRAC * bán kính lớn nhất; mỗi điểm trong bóng thuộc lõi gần nhất (Voronoi); hai thùy chỉ được
    giữ riêng nếu 'cổ' giữa chúng hẹp hơn _NECK_RATIO * bán kính thùy nhỏ hơn."""
    m8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax < 8:
        return None, {}
    n, cores = cv2.connectedComponents((dist >= _CORE_FRAC * dmax).astype(np.uint8), connectivity=8)
    if n <= 2:
        return None, {}
    # nhãn lõi gần nhất cho từng điểm: distanceTransformWithLabels cho nhãn điểm 0 gần nhất -> dùng ảnh mà lõi là điểm 0
    src = (cores == 0).astype(np.uint8)
    _, near = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    pix_label = np.zeros(int(near.max()) + 1, np.int32)
    pix_label[near[cores > 0]] = cores[cores > 0]
    lab = pix_label[near]
    lab[~mask] = 0
    ids = [i for i in range(1, n) if (lab == i).any()]
    peaks = {i: float(dist[lab == i].max()) for i in ids}
    # độ cao 'cổ' giữa từng cặp thùy kề nhau = điểm cao nhất của dist dọc biên giữa hai thùy
    saddle: Dict[Tuple[int, int], float] = {}
    H, W = lab.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ya, yb = (slice(0, H - dy), slice(dy, H))
        xa, xb = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
        la, lb_ = lab[ya, xa], lab[yb, xb]
        d = np.minimum(dist[ya, xa], dist[yb, xb])
        sel = (la > 0) & (lb_ > 0) & (la != lb_)
        for u, v, dv in zip(la[sel], lb_[sel], d[sel]):
            k = (int(min(u, v)), int(max(u, v)))
            saddle[k] = max(saddle.get(k, 0.0), float(dv))
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (u, v), sd in saddle.items():
        if sd > _NECK_RATIO * min(peaks[u], peaks[v]):       # không có cổ thắt rõ: cùng một bóng
            parent[find(u)] = find(v)
    roots = sorted({find(i) for i in ids})
    if len(roots) < 2:
        return None, {}
    rmap = {r: k + 1 for k, r in enumerate(roots)}
    out = np.zeros_like(lab)
    for i in ids:
        out[lab == i] = rmap[find(i)]
    rad = {rmap[r]: max(peaks[i] for i in ids if find(i) == r) for r in roots}
    return out, rad


def _split_balloon(b: Det, ts: List[Det]) -> Optional[List[Tuple[Det, List[Det]]]]:
    """Nếu bóng `b` thực ra là >= 2 bóng dính nhau và chữ nằm ở >= 2 thùy khác nhau thì trả [(bóng con, [chữ của nó]), ...]; ngược lại None."""
    if not ts:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in b.box)
    x1, y1 = max(0, x1), max(0, y1)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    mask = poly_mask(_poly_or_box(b), (y2 - y1, x2 - x1), x1, y1)
    # chữ (mực) làm thủng mask bóng -> lấp lỗ để khoảng cách tới viền không bị chữ làm nhiễu
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    lab, rad = _lobe_labels(filled > 0)
    if lab is None:
        return None
    H, W = lab.shape
    groups: Dict[int, List[Det]] = {}
    for t in ts:
        tx1, ty1, tx2, ty2 = (int(round(v)) for v in t.box)
        sub = lab[max(0, ty1 - y1):max(0, ty2 - y1), max(0, tx1 - x1):max(0, tx2 - x1)]
        cnt = np.bincount(sub[sub > 0].ravel(), minlength=len(rad) + 1)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        order = np.argsort(-cnt)
        main, second = int(order[0]), int(order[1]) if len(order) > 1 else 0
        if second > 0 and cnt[second] >= 0.25 * tot and len(ts) == 1:
            # MỘT vùng chữ trải qua hai thùy -> cắt vùng chữ theo thùy
            for lb in (main, second):
                ys, xs = np.nonzero(lab[:H, :W] == lb)
                ox1, oy1 = max(tx1, x1 + int(xs.min())), max(ty1, y1 + int(ys.min()))
                ox2, oy2 = min(tx2, x1 + int(xs.max()) + 1), min(ty2, y1 + int(ys.max()) + 1)
                if ox2 - ox1 >= 8 and oy2 - oy1 >= 8:
                    groups.setdefault(lb, []).append(Det(t.cls, t.conf, (float(ox1), float(oy1), float(ox2), float(oy2))))
        else:
            groups.setdefault(main, []).append(t)
    if len(groups) < 2:
        return None
    out = []
    for lb, tl in groups.items():
        ys, xs = np.nonzero(lab == lb)
        piece = (float(x1 + xs.min()), float(y1 + ys.min()), float(x1 + xs.max() + 1), float(y1 + ys.max() + 1))
        pm = (lab == lb).astype(np.uint8)
        pc, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = max(pc, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) + np.array([x1, y1], np.float32)
        out.append((Det(C_BALLOON, b.conf, piece, big), tl))
    return out



# Tách bóng dính nhau theo VÙNG TRẮNG THẬT trong ảnh: hai bóng liền nhau luôn có viền đen ngăn cách nên là hai vùng trắng rời nhau,
# dù model trả MỘT đa giác bóng / MỘT hộp chữ trùm cả hai (đa giác của model khá thô nên _split_balloon theo "cổ thắt" hay bỏ sót).
SPLIT_BY_IMAGE = True         # tắt = chỉ dùng _split_balloon + gom theo khoảng hở như cũ
_REGION_MIN_FRAC = 0.12       # mỗi vùng trắng phải >= 12% diện tích đa giác bóng mới tính là một bóng riêng
_REGION_MIN_INK = 0.20        # một hộp chữ chỉ bị cắt đôi nếu >= 20% mực của nó nằm ở vùng thứ hai


def _split_by_regions(gray: Optional[np.ndarray], b: Det, ts: List[Det]) -> Optional[List[Tuple[Det, List[Det]]]]:
    """Bóng `b` có >= 2 vùng trắng lớn rời nhau (và chữ nằm ở >= 2 vùng) -> [(bóng con, [chữ của nó]), ...]; ngược lại None.
    Hộp chữ trải qua hai vùng được cắt theo mực của từng vùng. Bóng nền xám / đen (không đủ trắng) -> None."""
    if gray is None or not ts:
        return None
    Hh, Ww = gray.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in b.box)
    x1, y1, x2, y2 = max(0, x1 - 4), max(0, y1 - 4), min(Ww, x2 + 4), min(Hh, y2 + 4)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    sub = gray[y1:y2, x1:x2]
    poly = poly_mask(_poly_or_box(b), sub.shape, x1, y1)
    pa = int(poly.sum())
    if pa < 400:
        return None
    p90 = float(np.percentile(sub[poly], 90))
    if p90 < 190:                                   # không phải bóng trắng (nền xám / đen / tông) -> không phán đoán
        return None
    thr = min(215.0, p90 - 35.0)
    area = cv2.dilate(poly.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    white = cv2.erode(((sub > thr) & area).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= _REGION_MIN_FRAC * pa]
    if len(comps) < 2:
        return None
    # nhãn vùng cho MỌI điểm (kể cả mực / viền): lấy nhãn của điểm trắng gần nhất
    seeds = np.isin(lab, comps)
    _, near = cv2.distanceTransformWithLabels(np.where(seeds, 0, 255).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.nonzero(seeds)                      # nhãn PIXEL k (k >= 1) = điểm không thứ k-1 theo thứ tự quét dòng
    comp_of = lab[zy, zx]
    region = np.where(near > 0, comp_of[np.clip(near - 1, 0, len(comp_of) - 1)], 0)
    region[~area] = 0
    ink_all = (sub < thr) & area
    # bỏ viền bóng (và viền chung giữa hai bóng) khỏi "mực chữ": các cụm mực chạm sát mép vùng bóng chính là đường viền
    edge = area & ~(cv2.erode(area.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    ncc, icc = cv2.connectedComponents(ink_all.astype(np.uint8), connectivity=8)
    touch = np.unique(icc[edge & ink_all])
    ink_all &= ~np.isin(icc, touch[touch > 0])

    groups: Dict[int, List[Det]] = {}
    for t in ts:
        tx1, ty1 = max(0, int(round(t.box[0])) - x1), max(0, int(round(t.box[1])) - y1)
        tx2, ty2 = min(sub.shape[1], int(round(t.box[2])) - x1), min(sub.shape[0], int(round(t.box[3])) - y1)
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        rl = region[ty1:ty2, tx1:tx2]
        ink = ink_all[ty1:ty2, tx1:tx2] & (rl > 0)
        cnt = {int(c): int((ink & (rl == c)).sum()) for c in comps}
        tot = sum(cnt.values())
        if tot < 20:                                # gần như không có mực: theo vùng ở tâm hộp
            c = int(region[(ty1 + ty2) // 2, (tx1 + tx2) // 2])
            if c:
                groups.setdefault(c, []).append(t)
            continue
        order = sorted(cnt, key=lambda c: -cnt[c])
        main, second = order[0], order[1]
        if cnt[second] >= _REGION_MIN_INK * tot and cnt[second] >= 30:
            for c in (main, second):                # hộp chữ vắt qua hai bóng: cắt theo mực của từng vùng
                ys, xs = np.nonzero(ink & (rl == c))
                if len(xs) == 0:
                    continue
                bx1, by1 = x1 + tx1 + int(xs.min()) - 2, y1 + ty1 + int(ys.min()) - 2
                bx2, by2 = x1 + tx1 + int(xs.max()) + 3, y1 + ty1 + int(ys.max()) + 3
                if bx2 - bx1 >= 8 and by2 - by1 >= 8:
                    groups.setdefault(c, []).append(Det(t.cls, t.conf, (float(bx1), float(by1), float(bx2), float(by2))))
        else:
            groups.setdefault(main, []).append(t)
    if len(groups) < 2:
        return None
    out: List[Tuple[Det, List[Det]]] = []
    for c, tl in groups.items():
        m = (region == c).astype(np.uint8)
        ys, xs = np.nonzero(m)
        piece = (float(x1 + xs.min()), float(y1 + ys.min()), float(x1 + xs.max() + 1), float(y1 + ys.max() + 1))
        pc, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = max(pc, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) + np.array([x1, y1], np.float32)
        out.append((Det(C_BALLOON, b.conf, piece, big), tl))
    return out


SPLIT_TEXT_GAP = 0.5          # bóng dính liền không có cổ thắt: tách các vùng chữ cách nhau > 50% cạnh ngắn trung bình của hai hộp chữ (~ nửa bề rộng một cột); 0 = tắt


def _box_gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def _cluster_texts(ts: List[Det], frac: float) -> List[List[Det]]:
    """Gom các vùng chữ cùng một bóng: hai vùng thuộc cùng cụm nếu khoảng hở giữa hộp <= frac * cạnh ngắn trung bình của hai hộp."""
    parent = list(range(len(ts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def short(d):
        return max(1.0, min(d.box[2] - d.box[0], d.box[3] - d.box[1]))

    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if _box_gap(ts[i].box, ts[j].box) <= frac * 0.5 * (short(ts[i]) + short(ts[j])):
                parent[find(i)] = find(j)
    groups: Dict[int, List[Det]] = {}
    for i, t in enumerate(ts):
        groups.setdefault(find(i), []).append(t)
    return list(groups.values())



def detection_units(layout: PageLayout, crop_pad: int = 6, gray: Optional[np.ndarray] = None) -> List[Unit]:
    """Bóng thoại CÓ chữ (+ mỗi vùng chữ tự do ngoài bóng) theo thứ tự đọc manga. SFX và khung tranh bị bỏ.
    gray (tuỳ chọn): bóng không được gán vùng chữ nào (model sót / vùng chữ dính hai bóng) vẫn được giữ nếu bên trong có mực,
    để không mất câu thoại; bóng rỗng thật thì bỏ."""
    W, H = layout.size
    units: List[Unit] = []
    used = set()
    texts = layout.texts
    for bi, b in enumerate(layout.balloons):
        inside = [i for i, t in enumerate(texts) if inside_ratio(t, b) >= 0.6]
        if not inside:
            if gray is None or not _balloon_has_ink(gray, b):
                continue
            ts = []
            cx1, cy1, cx2, cy2 = b.box                       # không có vùng chữ: cắt theo cả bóng
        else:
            used.update(inside)
            ts = [texts[i] for i in inside]
            cx1 = min(t.box[0] for t in ts); cy1 = min(t.box[1] for t in ts)
            cx2 = max(t.box[2] for t in ts); cy2 = max(t.box[3] for t in ts)
        parts = _split_by_regions(gray, b, ts) if (SPLIT_BY_IMAGE and SPLIT_FUSED_BALLOONS and ts) else None   # ưu tiên: viền bóng thật trong ảnh
        if not parts:
            parts = _split_balloon(b, ts) if (SPLIT_FUSED_BALLOONS and ts) else None                          # dự phòng: cổ thắt của đa giác
        if parts:   # nhiều bóng dính nhau -> mỗi bóng một đơn vị, OCR/dịch/redraw riêng
            for pb, pts in parts:
                px1 = min(t.box[0] for t in pts); py1 = min(t.box[1] for t in pts)
                px2 = max(t.box[2] for t in pts); py2 = max(t.box[3] for t in pts)
                pcrop = (max(0, int(px1) - crop_pad), max(0, int(py1) - crop_pad), min(W, int(px2) + crop_pad), min(H, int(py2) + crop_pad))
                units.append(Unit("balloon", pb.xywh(), pcrop, pb, pts))
            continue
        if SPLIT_FUSED_BALLOONS and SPLIT_TEXT_GAP > 0 and len(ts) >= 2:
            clusters = _cluster_texts(ts, SPLIT_TEXT_GAP)
            if len(clusters) >= 2:     # nhiều khối chữ tách rời trong một đa giác bóng -> mỗi khối một đơn vị
                bx1, by1, bx2, by2 = b.box
                for cl in clusters:
                    px1 = min(t.box[0] for t in cl); py1 = min(t.box[1] for t in cl)
                    px2 = max(t.box[2] for t in cl); py2 = max(t.box[3] for t in cl)
                    pcrop = (max(0, int(px1) - crop_pad), max(0, int(py1) - crop_pad), min(W, int(px2) + crop_pad), min(H, int(py2) + crop_pad))
                    pad = 8
                    fbox = [max(int(bx1), int(px1) - pad), max(int(by1), int(py1) - pad),
                            min(int(bx2), int(px2) + pad), min(int(by2), int(py2) + pad)]
                    units.append(Unit("balloon", [fbox[0], fbox[1], fbox[2] - fbox[0], fbox[3] - fbox[1]], pcrop, b, cl))
                continue
        crop = (max(0, int(cx1) - crop_pad), max(0, int(cy1) - crop_pad), min(W, int(cx2) + crop_pad), min(H, int(cy2) + crop_pad))
        units.append(Unit("balloon", b.xywh(), crop, b, ts))
    for i, t in enumerate(texts):
        if i in used or any(inside_ratio(t, b) >= 0.6 for b in layout.balloons):
            continue
        x1, y1, x2, y2 = t.box
        crop = (max(0, int(x1) - crop_pad), max(0, int(y1) - crop_pad), min(W, int(x2) + crop_pad), min(H, int(y2) + crop_pad))
        units.append(Unit("free", t.xywh(), crop, None, [t]))
    return order_units(layout, units)


HYBRID_WITH_YOLO = True   # có model YOLO bóng thoại + koharu bật: YOLO quyết định các khung thoại, koharu bổ sung / làm sạch vùng cắt / SFX


def hybrid_units(layout: PageLayout, yolo_boxes: Sequence[Sequence[int]], gray: Optional[np.ndarray] = None,
                 crop_pad: int = 6) -> List[Unit]:
    """Kết hợp: mỗi hộp YOLO [x, y, w, h] là MỘT đơn vị dịch (YOLO tách bóng thoại sát nhau tốt hơn koharu).
    koharu dùng để: (a) thu vùng cắt OCR về đúng khối chữ trong hộp; (b) tô trắng chữ của bóng khác lấn vào; (c) lấy đa giác bóng
    để xoá nền ngoài bóng; (d) THÊM các vùng chữ mà YOLO bỏ sót (chữ tự do / bóng YOLO không thấy)."""
    W, H = layout.size
    texts, balloons = layout.texts, layout.balloons
    units: List[Unit] = []
    yb_all = [(float(x), float(y), float(x + w), float(y + h)) for (x, y, w, h) in yolo_boxes]
    # chữ coi là đã được YOLO phủ nếu tổng phần giao với mọi hộp YOLO >= 50% (vùng chữ gộp hai câu vắt qua hai hộp vẫn tính là phủ)
    covered = {i for i, t in enumerate(texts) if sum(_inter(t.box, b) for b in yb_all) >= 0.5 * max(1.0, t.area)}
    for (x, y, w, h) in yolo_boxes:
        yb = (float(x), float(y), float(x + w), float(y + h))
        ya = max(1.0, _area(yb))
        mine = []
        for i, t in enumerate(texts):
            cx, cy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            if (yb[0] <= cx <= yb[2] and yb[1] <= cy <= yb[3]) or _inter(t.box, yb) >= 0.25 * max(1.0, t.area):
                mine.append(i)
        ts = [texts[i] for i in mine]
        if ts:
            cx1 = min(t.box[0] for t in ts); cy1 = min(t.box[1] for t in ts)
            cx2 = max(t.box[2] for t in ts); cy2 = max(t.box[3] for t in ts)
        else:
            cx1, cy1, cx2, cy2 = yb
        # vùng cắt không vượt khỏi hộp YOLO (nở 8 px): koharu gộp hai câu thành MỘT vùng chữ thì mỗi hộp YOLO chỉ lấy phần của mình
        lim = 8
        cx1, cy1 = max(cx1, yb[0] - lim), max(cy1, yb[1] - lim)
        cx2, cy2 = min(cx2, yb[2] + lim), min(cy2, yb[3] + lim)
        crop = (max(0, int(cx1) - crop_pad), max(0, int(cy1) - crop_pad), min(W, int(cx2) + crop_pad), min(H, int(cy2) + crop_pad))
        # bóng koharu phủ gần hết hộp YOLO; nếu nhiều bóng thì lấy bóng nhỏ nhất (cụ thể nhất)
        cands = [b for b in balloons if _inter(b.box, yb) / ya >= 0.6]
        best = min(cands, key=lambda b: b.area) if cands else None
        others = [t for i, t in enumerate(texts) if i not in mine and _inter(t.box, (crop[0], crop[1], crop[2], crop[3])) > 0]
        units.append(Unit("balloon", [int(x), int(y), int(w), int(h)], crop, best, ts, others))
    # chữ koharu mà không hộp YOLO nào phủ: thêm như đơn vị bổ sung (bóng YOLO bỏ sót / chữ tự do)
    rest = [t for i, t in enumerate(texts) if i not in covered]
    if rest:
        keep_b = [b for b in balloons if any(inside_ratio(t, b) >= 0.6 for t in rest)]
        extra = detection_units(PageLayout(layout.size, keep_b + rest), crop_pad, gray=None)
        units.extend(extra)
    return order_units(layout, units)


def _own_text_keep(u: Unit, shape: Tuple[int, int], ox: int, oy: int, pad: int) -> np.ndarray:
    """Vùng chữ của chính đơn vị này (đa giác nở + hộp nở): KHÔNG bao giờ tô trắng, vì đa giác koharu khá thô."""
    keep = np.zeros(shape, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1,) * 2)
    for t in u.texts:
        keep |= cv2.dilate(det_mask(t, shape, ox, oy).astype(np.uint8), k) > 0
        bx1, by1, bx2, by2 = (int(round(v)) for v in t.box)
        keep[max(0, by1 - oy - pad):max(0, by2 - oy + pad), max(0, bx1 - ox - pad):max(0, bx2 - ox + pad)] = True
    return keep


def crop_for_ocr(cv_img: np.ndarray, u: Unit, outside_white: bool = True, variant: str = "default") -> np.ndarray:
    """Cắt vùng chữ của đơn vị để đưa cho OCR. Ba biến thể (OCR lần đầu dùng "default"; câu đọc chưa chắc thử tiếp hai biến thể còn lại):
      default   : tô trắng mọi điểm ngoài (bóng nở 9 px  HỢP chữ của chính đơn vị) - bỏ đuôi bóng, nét tranh, chữ bóng bên cạnh;
                  hợp thêm vùng chữ để đa giác bóng thô không cắt cụt chữ sát viền (bản cũ chỉ nở 5 px nên có thể cắt).
      text_only : chỉ giữ vùng chữ của đơn vị (nở 4 px) - sạch nhất, dùng khi đuôi bóng / nét tranh làm OCR đọc nhầm.
      wide      : nở thêm 12 px, không tô gì ngoài chữ bóng khác - dùng khi koharu cắt sát làm mất nét chữ; cũng là ảnh gửi Gemini đối chiếu.
    Luôn tô trắng chữ của bóng KHÁC lấn vào vùng cắt (u.others)."""
    H, W = cv_img.shape[:2]
    x1, y1, x2, y2 = u.crop
    if variant == "wide":
        x1, y1, x2, y2 = max(0, x1 - 12), max(0, y1 - 12), min(W, x2 + 12), min(H, y2 + 12)
    roi = cv_img[y1:y2, x1:x2].copy()
    if not roi.size:
        return roi
    shape = roi.shape[:2]
    keep: Optional[np.ndarray] = None
    if variant == "text_only" and u.texts:
        keep = _own_text_keep(u, shape, x1, y1, 4)
    elif variant == "default" and outside_white and u.balloon is not None:
        inside = poly_mask(_poly_or_box(u.balloon), shape, x1, y1)
        keep = cv2.dilate(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        keep |= _own_text_keep(u, shape, x1, y1, 3)
    if keep is not None:
        roi[~keep] = 255
    for o in u.others:
        ox1, oy1, ox2, oy2 = (int(round(v)) for v in o.box)
        roi[max(0, oy1 - 2 - y1):max(0, oy2 + 2 - y1), max(0, ox1 - 2 - x1):max(0, ox2 + 2 - x1)] = 255
    return roi


def sfx_units(layout: PageLayout, min_side: int = 14, exclude_boxes: Optional[Sequence[Sequence[int]]] = None) -> List[List[int]]:
    """SFX ngoài bóng thoại, hộp [x, y, w, h] - cho tuỳ chọn 'xoá SFX' (redraw_only + type=mask).
    exclude_boxes: các hộp [x, y, w, h] của khung thoại (VD hộp YOLO); SFX nằm trong đó không phải SFX ngoài bóng."""
    out = []
    ex = [(float(x), float(y), float(x + w), float(y + h)) for x, y, w, h in (exclude_boxes or [])]
    for s in layout.sfx:
        if any(inside_ratio(s, b) >= 0.6 for b in layout.balloons):
            continue
        if any(_inter(s.box, e) >= 0.6 * max(1.0, s.area) for e in ex):
            continue
        b = s.xywh()
        if min(b[2], b[3]) >= min_side:
            out.append(b)
    return sorted(out, key=_reading_key)


def units_for_page(image_path: str, yolo_boxes: Optional[Sequence[Sequence[int]]] = None) -> Optional[Tuple[PageLayout, List[Unit]]]:
    """(layout, đơn vị cần dịch) nếu koharu đang bật và chạy được; None để nơi gọi dùng YOLO cũ (lỗi chỉ cảnh báo, không làm hỏng dịch)."""
    cfg = current()
    if not cfg.enabled:
        return None
    try:
        layout = get_layout(image_path, cfg)
        img = _imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
        if yolo_boxes is not None and HYBRID_WITH_YOLO:
            return layout, hybrid_units(layout, yolo_boxes, gray=gray)
        return layout, detection_units(layout, gray=gray)
    except Exception as exc:  # noqa: BLE001
        log.warning("koharu lỗi, dùng YOLO cũ cho trang %s: %s", Path(image_path).name, exc)
        return None
