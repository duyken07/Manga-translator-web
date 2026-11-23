#!/usr/bin/env python3
"""
manga_redraw.py (v8, stage A) - erase the original text and export a layout for stage B.

  v7: two erase engines, chosen per text block (see decide_fill):
      OpenCV  : plain white bubble / solid black / flat paper  -> flat colour fill (+ snap to white)
      LaMa    : anything with a complex background (screentone, gradient, line art behind the
                text, glyphs touching the bubble outline)      -> lama-manga inpainting
      Without --lama everything falls back to OpenCV (flat / cv2.inpaint / texture fill).
      v8: every LaMa block of a page is queued and run in ONE pass over the union mask (fewer, smaller
      windows with the adaptive lama_manga.py); residual retry skips hatching (--retry-max-busy).
      Per-block override in the JSON: "engine": "opencv" | "lama"; global: --force-engine.

  input : page image + data.json (same formats as v4)
  output (in --outdir, prefix = image stem):
      <stem>_clean.png     page with the text erased (no Vietnamese text on it)
      <stem>_mask.png      pixels that were rebuilt
      <stem>_qa.png        review image: green = OpenCV, red = LaMa, blue = texture/other,
                           orange boxes = texture tiles that
                           fell back to inpaint, magenta boxes = tiles with a weak match
      <stem>_layout.json   what stage B needs to typeset (regenerated on every run,
                           never edit it by hand: use <stem>_overrides.json instead)

Stage B is manga_typeset.py. Stage A never reads <stem>_overrides.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("redraw.stage_a")

SCHEMA = 1


# --------------------------------------------------------------------------- geometry / data
def clamp_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def overlap_ratio(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / max(1, min(area(a), area(b)))


def rects_intersect(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def ellipse_kernel(radius: int) -> np.ndarray:
    k = 2 * max(0, radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def parse_rects(v: Any) -> list[tuple[int, int, int, int]]:
    if not v:
        return []
    if isinstance(v[0], (int, float)):
        v = [v]
    return [tuple(int(x) for x in r) for r in v]


def parse_color(s: Any) -> tuple[int, int, int] | None:
    """'r,g,b' (or [r,g,b]) -> BGR tuple."""
    if s is None:
        return None
    vals = [int(v) for v in (s.split(",") if isinstance(s, str) else s)]
    r, g, b = vals
    return (b, g, r)


@dataclass
class Item:
    gid: int
    ids: list
    bbox: tuple
    text: str
    # manual overrides
    type: str | None = None
    fill: str | None = None
    region: tuple | None = None
    polarity: str | None = None
    style: str | None = None
    font_size: int | None = None
    line_spacing: float | None = None      # hệ số giãn dòng riêng của câu (None = mặc định chung)
    align: str | None = None               # căn lề các dòng trong khối chữ: "left" | "center" | "right" (None = giữa)
    outline: bool = False                  # viền chữ (mặc định TẮT)
    outline_width: int = 2                 # độ dày viền (px ảnh)
    outline_color: tuple | None = None     # màu viền BGR (None = trắng, hoặc đen khi chữ trắng)
    font_path: str | None = None
    text_color: tuple | None = None
    cleanup: bool = True
    engine_force: str | None = None
    keep: list = field(default_factory=list)
    # analysis (ROI-local arrays, ROI origin = (ox, oy))
    kind: str = "free"
    method: str = ""
    ox: int = 0
    oy: int = 0
    strokes: np.ndarray | None = None      # bool: pixels to erase
    interior: np.ndarray | None = None     # bool: bubble / dark-blob shape
    thickness: float = 0.0
    fill_mode: str = "inpaint"
    flat_color: tuple = (255, 255, 255)
    occ: tuple = (0, 0, 0, 0)              # global rect occupied by the original text
    bold: bool = False
    guard: np.ndarray | None = None        # bubble: pixels far enough from the outline to be edited
    strokes_w: np.ndarray | None = None    # white glyphs found OUTSIDE a bubble (e.g. on black hair)
    busy: float = 0.0
    large: bool = False                    # chữ cỡ lớn (xem analyze)
    engine: str = "opencv"                # opencv | lama | opencv+lama (decided in decide_fill)
    why: str = ""                          # human-readable reason for the engine choice
    # layout
    safe: np.ndarray | None = None
    half: np.ndarray | None = None
    cx: int = 0
    cy: int = 0
    split_hyphen: bool = False             # từ ghép (Hanna-san) không vừa ở bất kỳ cỡ nào -> cho phép ngắt dòng tại dấu gạch nối
    inr: float = 0.0                       # bán kính vòng tròn nội tiếp của vùng an toàn (px) - đo độ "to" thật của bóng thoại
    best: int = 0
    allowed: tuple | None = None
    layout: dict | None = field(default=None)
    kz: Any = None                         # koharu_detector.ItemMatch: bóng thoại + vùng chữ + SFX koharu gán cho khung này

    @property
    def shaped(self) -> bool:
        return self.kind in ("bubble", "dark") and self.region is None


OVERRIDE_KEYS = ("type", "fill", "region", "keep", "polarity", "style", "font_size", "font_path",
                 "text_color", "cleanup", "engine", "line_spacing", "outline", "outline_width", "outline_color", "align")


def load_items(data: dict, page: str | None, image_name: str) -> tuple[list[dict], list[dict], str]:
    """Return (items, manual_groups, page_name) normalised to {id,bbox,source_text,translated_text,...}."""
    if isinstance(data.get("pages"), list):
        pages = data["pages"]
        want = page or image_name
        pg = next((p for p in pages if p.get("page_name") == want), None)
        if pg is None and want:
            pg = next((p for p in pages if Path(p.get("page_name", "")).stem == Path(want).stem), None)
        if pg is None and len(pages) == 1:
            pg = pages[0]
        if pg is None:
            names = ", ".join(p.get("page_name", "?") for p in pages)
            sys.exit(f"Page '{want}' not found in JSON. Use --page. Available: {names}")
        items = []
        for i, d in enumerate(pg.get("dialogues", []), 1):
            x, y, w, h = d["box_location"]
            it = dict(id=d.get("dialogue_id", i), bbox=[x, y, x + w, y + h],
                      source_text=d.get("original_text", ""),
                      translated_text=d.get("translated_text", ""))
            for k in OVERRIDE_KEYS:
                if k in d:
                    it[k] = d[k]
            items.append(it)
        return items, pg.get("groups", data.get("groups", [])), pg.get("page_name", "")
    if isinstance(data.get("bubbles"), list):
        return data["bubbles"], data.get("groups", []), data.get("image", "")
    sys.exit('JSON must contain "pages" (with "dialogues") or "bubbles".')


def norm_src(item: dict) -> str:
    first = str(item.get("source_text", "")).split("\n")[0]
    return re.sub(r"[\s、。，,.!?！？…・「」『』\"'~〜ー-]+", "", first)


def pick_text(members: list[dict]) -> str:
    if len(members) == 1:
        return str(members[0].get("translated_text", "")).strip()
    srcs = [norm_src(m) for m in members]
    keep = []
    for i, m in enumerate(members):
        subsumed = False
        for j in range(len(members)):
            if i == j or not srcs[i]:
                continue
            if srcs[i] in srcs[j] and (len(srcs[j]) > len(srcs[i]) or j < i):
                subsumed = True
                break
        if not subsumed:
            keep.append(m)
    keep = keep or members
    keep.sort(key=lambda m: (-(m["bbox"][0] + m["bbox"][2]) / 2, m["bbox"][1]))
    return " ".join(str(m.get("translated_text", "")).strip() for m in keep).strip()


def build_items(raw: list[dict], W: int, H: int, merge_thr: float, do_merge: bool,
                manual: list[dict]) -> list[Item]:
    n = len(raw)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    boxes = [clamp_bbox(it["bbox"], W, H) for it in raw]
    for it, bx in zip(raw, boxes):
        it["bbox"] = list(bx)
    id_index = {it.get("id", i + 1): i for i, it in enumerate(raw)}
    if do_merge:
        for i in range(n):
            for j in range(i + 1, n):
                if overlap_ratio(boxes[i], boxes[j]) >= merge_thr:
                    union(i, j)
    manual_idx = []
    for g in manual:
        idx = [id_index[x] for x in g.get("ids", []) if x in id_index]
        for k in idx[1:]:
            union(idx[0], k)
        if idx:
            manual_idx.append((idx[0], g))

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    items: list[Item] = []
    for gid, (root, idxs) in enumerate(sorted(clusters.items(), key=lambda kv: min(kv[1])), 1):
        members = [raw[i] for i in idxs]
        first = lambda key: next((m[key] for m in members if m.get(key) not in (None, "")), None)
        region = first("region")
        it = Item(
            gid=gid, ids=[raw[i].get("id", i + 1) for i in idxs],
            bbox=(min(boxes[i][0] for i in idxs), min(boxes[i][1] for i in idxs),
                  max(boxes[i][2] for i in idxs), max(boxes[i][3] for i in idxs)),
            text=pick_text(members),
            type=first("type"), fill=first("fill"), engine_force=first("engine"),
            region=tuple(int(v) for v in region) if region else None,
            polarity=first("polarity"), style=first("style"),
            font_size=int(first("font_size")) if first("font_size") else None,
            line_spacing=float(first("line_spacing")) if first("line_spacing") else None,
            align=(first("align") if first("align") in ("left", "center", "right") else None),
            outline=bool(first("outline")),
            outline_width=max(1, int(first("outline_width") or 2)),
            outline_color=parse_color(first("outline_color")),
            font_path=first("font_path"), text_color=parse_color(first("text_color")),
            cleanup=any(m.get("cleanup", True) is not False for m in members),
            keep=[r for m in members for r in parse_rects(m.get("keep"))],
        )
        for first_idx, spec in manual_idx:
            if find(first_idx) == root:
                if "translated_text" in spec:
                    it.text = str(spec["translated_text"]).strip()
                if spec.get("style"):
                    it.style = spec["style"]
        items.append(it)
    return items


# --------------------------------------------------------------------------- analysis
def pick_components(cand: np.ndarray, gray: np.ndarray, ring_ok, ring_min: float, max_side: int,
                    keep_box: tuple, min_area: int = 4) -> np.ndarray:
    """Keep connected components that look like glyph strokes.

    ring_ok(values)->bool array tells whether a ring pixel has the right polarity
    (bright ring for black strokes, dark ring for white strokes). Hatching / outlines of the
    artwork fail either the size limit or the ring test.
    """
    H, W = gray.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros((H, W), bool)
    k3 = np.ones((3, 3), np.uint8)
    bx1, by1, bx2, by2 = keep_box
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < min_area or max(w, h) > max_side:
            continue
        cx, cy = x + w / 2, y + h / 2
        if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
            continue
        x0, y0, x1, y1 = max(0, x - 5), max(0, y - 5), min(W, x + w + 5), min(H, y + h + 5)
        sub = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(sub, k3, iterations=3) > 0) & (cv2.dilate(sub, k3, iterations=1) == 0)
        vals = gray[y0:y1, x0:x1][ring]
        if vals.size < 6 or ring_ok(vals).mean() < ring_min:
            continue
        keep[y0:y1, x0:x1] |= sub > 0
    return keep


def fill_holes(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=-1)
    return out > 0


def detect_dark(roi: np.ndarray, box: tuple, a) -> np.ndarray | None:
    """Large dark blob around the box centre (white text on black). Returns filled blob mask."""
    h, w = roi.shape
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    dark = (roi < a.dark_thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(dark, connectivity=8)
    core = lab[by1 + bh // 4: max(by1 + bh // 4 + 1, by2 - bh // 4),
               bx1 + bw // 4: max(bx1 + bw // 4 + 1, bx2 - bw // 4)]
    ids, cnt = np.unique(core[core > 0], return_counts=True)
    if not len(ids) or cnt.max() < 0.12 * core.size:
        return None
    comp = lab == ids[np.argmax(cnt)]
    filled = fill_holes(comp)
    if filled.sum() < 0.18 * bw * bh:
        return None
    inside = roi[filled]
    if (inside < a.dark_thr).mean() < a.dark_frac or (inside > 140).mean() < 0.01:
        return None
    return filled


def detect_bubble(roi: np.ndarray, box: tuple, a, force: bool) -> tuple[np.ndarray, str] | None:
    """White closed component (v2 idea) + extra checks so halftone is not taken for a bubble."""
    h, w = roi.shape
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    white = (roi > a.white_thr).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(white, connectivity=4)
    core = lab[by1 + bh // 4: max(by1 + bh // 4 + 1, by2 - bh // 4),
               bx1 + bw // 4: max(bx1 + bw // 4 + 1, bx2 - bw // 4)]
    ids, cnt = np.unique(core[core > 0], return_counts=True)
    if len(ids):
        comp = (lab == ids[np.argmax(cnt)]).astype(np.uint8)
        filled = fill_holes(comp)
        area = int(filled.sum())
        if 0.12 * bw * bh <= area <= 1.6 * bw * bh and area < 0.92 * h * w:
            ring = (cv2.dilate(filled.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0) & ~filled
            outline = float((roi[ring] < 120).mean()) if ring.any() else 0.0
            bg = roi[filled & (roi > a.white_thr)]
            clean = bg.size > 0 and float(np.median(bg)) >= 225 and float(bg.std()) < 16
            # a real bubble covers (almost) the whole box; a sliver of white next to the
            # text (e.g. a gap in the artwork) does not
            cover = float(filled[by1:by2, bx1:bx2].sum()) / max(1, bw * bh)
            if force or (outline >= a.outline_frac and clean and cover >= a.bubble_cover):
                return cv2.erode(filled.astype(np.uint8), ellipse_kernel(a.erode)) > 0, "shape"
    if not force:
        return None
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, ((bx1 + bx2) // 2, (by1 + by2) // 2),
                (max(1, int(bw / 2 * 0.88)), max(1, int(bh / 2 * 0.88))), 0, 0, 360, 1, -1)
    return m > 0, "ellipse"


def detect_bubble_boxed(roi: np.ndarray, box: tuple, a) -> tuple[np.ndarray, str] | None:
    """Nhận diện bóng thoại khi KHUNG CHÍNH LÀ khung bao của bóng (toạ độ bóng thoại đã chính xác).

    detect_bubble() chỉ nhìn "vùng trắng khép kín" nên hỏng trong các trường hợp rất hay gặp ở truyện:
      * viền bóng nét phác / bị hở: vùng trắng tràn ra giấy trắng bên ngoài -> bị loại -> rơi về chế độ "free",
        chữ dịch được nới ra ngoài bóng, còn việc xoá chữ thì sót hoặc lấn sang tranh;
      * chữ gốc dính viền / nằm sát chỗ hở: vùng trắng khuyết một mảng quanh chữ -> sót chữ.
    Ở đây dùng chính khung làm ràng buộc:
      1. nét tối nhỏ (chữ) được coi là một phần của lòng bóng; chỉ nét tối LỚN (viền, tranh) mới là "tường";
      2. lòng bóng = vùng sáng nối với tâm khung, bị cắt trong khung (+3 px);
      3. nếu vùng này tràn ra sát mép khung (viền hở) thì cắt thêm bằng hình elip nội tiếp khung;
      4. phải có viền thật quanh biên vùng thì mới nhận (chữ trên giấy trắng không viền thì không phải bóng).
    Trả về (lòng bóng đã co 'erode', method) với method = "boxed" | "boxed-ellipse", hoặc None để dùng cách cũ.
    """
    h, w = roi.shape
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    if min(bw, bh) < 24:
        return None
    core = roi[by1 + bh // 4: max(by1 + bh // 4 + 1, by2 - bh // 4), bx1 + bw // 4: max(bx1 + bw // 4 + 1, bx2 - bw // 4)]
    bright = core[core >= 170]
    if bright.size < 0.25 * core.size:             # lõi khung không sáng: không phải bóng trắng / nhạt
        return None
    t_light = int(min(a.white_thr + 20, max(170, float(np.median(bright)) - 35)))

    # 1) nét tối: nhỏ = chữ, lớn = viền / tranh
    dark = (roi < a.box_dark_thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) < a.box_glyph_frac * min(bw, bh)
    glyph = small[lab]
    struct = (dark > 0) & ~glyph
    cand = (roi >= t_light) | (cv2.dilate(glyph.astype(np.uint8), ellipse_kernel(2)) > 0)

    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy, rx, ry = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0, bw / 2.0, bh / 2.0
    m = 3
    rect = np.zeros((h, w), bool)
    rect[max(0, by1 - m):min(h, by2 + m), max(0, bx1 - m):min(w, bx2 + m)] = True
    seed = np.zeros((h, w), bool)
    seed[by1 + bh // 3: max(by1 + bh // 3 + 1, by2 - bh // 3), bx1 + bw // 3: max(bx1 + bw // 3 + 1, bx2 - bw // 3)] = True

    def grow(limit: np.ndarray):
        _, lb = cv2.connectedComponents((cand & limit).astype(np.uint8), connectivity=4)
        ids, cnt = np.unique(lb[seed & (lb > 0)], return_counts=True)
        return None if not len(ids) else lb == ids[np.argmax(cnt)]

    # 2) lòng bóng trong khung
    comp = grow(rect)
    if comp is None:
        return None
    # 3) viền hở: vùng sáng chạm mép khung -> cắt bằng elip nội tiếp
    edge = rect & ~(cv2.erode(rect.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    method = "boxed"
    if edge.any() and float(comp[edge].mean()) > 0.05:
        comp = grow(rect & (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0))
        method = "boxed-ellipse"
        if comp is None:
            return None
    filled = fill_holes(comp) & ~struct              # fill_holes có thể nuốt mẩu viền nằm trong lòng: bỏ nét viền / tranh ra
    area = int(filled.sum())
    if not (a.box_min_area * bw * bh <= area <= 1.12 * bw * bh):
        return None
    # 4) phải có viền thật quanh biên (nét tối lớn trong phạm vi box_evidence_reach)
    near = cv2.dilate(struct.astype(np.uint8), ellipse_kernel(a.box_evidence_reach)) > 0
    contour = filled & ~(cv2.erode(filled.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if not contour.any() or float(near[contour].mean()) < a.box_outline_cov:
        return None
    return cv2.erode(filled.astype(np.uint8), ellipse_kernel(a.erode)) > 0, method


def white_fill_stats(roi: np.ndarray, box: tuple, a) -> tuple[float, float]:
    """(diện tích vùng trắng khép kín / diện tích khung, phần khung được vùng trắng đó phủ). (0, 0) nếu không có."""
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    white = cv2.morphologyEx((roi > a.white_thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(white, connectivity=4)
    core = lab[by1 + bh // 4: max(by1 + bh // 4 + 1, by2 - bh // 4), bx1 + bw // 4: max(bx1 + bw // 4 + 1, bx2 - bw // 4)]
    ids, cnt = np.unique(core[core > 0], return_counts=True)
    if not len(ids):
        return 0.0, 0.0
    filled = fill_holes((lab == ids[np.argmax(cnt)]).astype(np.uint8))
    return float(filled.sum()) / max(1, bw * bh), float(filled[by1:by2, bx1:bx2].sum()) / max(1, bw * bh)


def halo_radius(mask: np.ndarray) -> float:
    if mask.sum() < 20:
        return 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.percentile(dt[mask], 90) * 2)


def busy_fraction(g: np.ndarray, sel: np.ndarray | None = None, thr: float = 20.0) -> float:
    """Share of pixels whose 5x5 neighbourhood has std > thr (hatching, screentone, line art)."""
    f = g.astype(np.float32)
    mu = cv2.blur(f, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (5, 5)) - mu * mu, 0))
    v = sd[sel] if sel is not None else sd.ravel()
    return float((v > thr).mean()) if v.size else 0.0


def stroke_thickness(roi: np.ndarray, box: tuple, thr: int) -> tuple[float, float]:
    """(p75, p90) độ dày nét tối trong khung (đường kính, px), ước lượng bằng đường sống của distance transform.

    Bỏ qua các thành phần là vòng kín (viền bóng thoại, khung tranh: bên trong rỗng, diện tích lỗ lớn) để
    viền dày không bị tính như nét chữ.
    """
    bx1, by1, bx2, by2 = box
    sub = (roi[by1:by2, bx1:bx2] < thr).astype(np.uint8)
    if sub.sum() < 30:
        return 0.0, 0.0
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    keep = np.zeros_like(sub)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if area < 12:
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        holes = int(fill_holes(comp).sum()) - int(comp.sum())
        if holes > 0.3 * w * h:
            continue
        keep[y:y + h, x:x + w] |= comp
    if keep.sum() < 30:
        return 0.0, 0.0
    dt = cv2.distanceTransform(keep, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1.5)
    v = dt[ridge] * 2
    if v.size < 20:
        return 0.0, 0.0
    return float(np.percentile(v, 75)), float(np.percentile(v, 90))


def sever_big_masses(cand: np.ndarray, radius: int) -> np.ndarray:
    """Remove thick dark/bright masses (hair, panel borders, black panels) from the candidate mask.

    Glyph strokes are thin; anything that survives an opening with a big kernel is artwork.
    Removing it (plus a 1px margin) cuts the glyphs loose from the artwork they touch, so the
    component filter can see them as separate, small components again.
    """
    k = ellipse_kernel(radius)
    big = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k)
    big = cv2.dilate(big, ellipse_kernel(1))
    return cand & ~(big > 0)


def collect_core(roi: np.ndarray, pol: str, a, max_side: int, keep_box: tuple, box: tuple | None = None) -> np.ndarray:
    if pol == "white":
        cand = roi >= a.white_text_thr
        cand = sever_big_masses(cand, a.big_radius)
        return pick_components(cand, roi, lambda v: v < a.white_ring_thr, a.ring_min, max_side, keep_box)
    thr = int(a.free_dark_thr)
    if int(getattr(a, "free_adaptive", 1)) and roi.size:
        # Ngưỡng cố định (100) chỉ bắt lõi nét ĐEN: bản scan mềm / font mảnh có lõi nét chỉ xám 110-170 nên không bắt được nét nào
        # (chữ không bị xoá) hoặc sót rải rác thành "viền chữ". Khi vùng là GIẤY TRẮNG PHẲNG (đa số điểm gần trắng) thì nới ngưỡng theo mức
        # giấy; vùng screentone / tranh (ít điểm trắng) giữ ngưỡng cũ để không xoá nhầm hạt tone.
        paper = float(np.percentile(roi, 80))
        if paper >= 235 and float((roi >= 230).mean()) >= 0.55:
            # Điều kiện chặt: phần ROI NẰM NGOÀI khung chữ cũng phải là giấy sạch (>= 95% gần trắng). Screentone / hạt tone / nét tranh quanh
            # chữ làm vùng ngoài khung không sạch -> giữ ngưỡng cũ, tránh bắt hạt tone thành "chữ" rồi xoá.
            outside = np.ones(roi.shape, bool)
            kx1, ky1, kx2, ky2 = (int(v) for v in (box if box is not None else keep_box))
            outside[max(0, ky1):max(0, ky2), max(0, kx1):max(0, kx2)] = False
            if int(outside.sum()) >= 200 and float((roi[outside] >= 225).mean()) >= 0.95:
                thr = int(np.clip(paper - float(getattr(a, "free_ink_delta", 80)), thr, 175))
    cand = roi < thr
    cand = sever_big_masses(cand, a.big_radius)
    return pick_components(cand, roi, lambda v: v >= 170, a.ring_min, max_side, keep_box)


def koharu_interior(roi: np.ndarray, bal: np.ndarray, a) -> np.ndarray | None:
    """Lòng bóng từ mask bóng thoại của koharu (ROI-local, bool). Khác detect_bubble_boxed: không cần vùng trắng khép kín nên
    chạy được với bóng hở viền / bóng gai / bóng nền screentone. Mask của model sai số vài px ở mép nên:
      * bỏ nét tối LỚN (viền, tranh) bằng đúng quy tắc của detect_bubble_boxed (nét nhỏ = chữ, nét lớn = tường);
      * co thêm k_inset px (và a.erode) để `guard` của bubble không chạm vào viền bóng.
    Trả về None nếu kết quả vô lý -> analyze dùng các bộ nhận diện cũ."""
    ys, xs = np.nonzero(bal)
    if len(xs) < 64:
        return None
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    if min(bw, bh) < 20:
        return None
    dark = (roi < a.box_dark_thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= a.box_glyph_frac * min(bw, bh)
    struct = big[lab]
    inner = bal & ~(cv2.dilate(struct.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    r = max(int(a.erode), int(a.k_inset))
    inner = cv2.erode(inner.astype(np.uint8), ellipse_kernel(r)) > 0
    if inner.sum() < 0.3 * bal.sum():
        return None
    if float(np.median(roi[inner])) < 100:        # lòng tối: để detect_dark / các bộ cũ xử lý
        return None
    return inner


def _analyze_once(gray: np.ndarray, b: Item, a) -> None:
    H, W = gray.shape
    x1, y1, x2, y2 = b.bbox
    # Khung lớn (chữ cỡ lớn) hay bị khung OCR cắt lệch vài chục px: mở rộng vùng phân tích theo cỡ khung
    pad = a.pad
    if a.thick_stroke > 0 and min(x2 - x1, y2 - y1) >= 60:
        pad = max(a.pad, min(40, int(0.25 * min(x2 - x1, y2 - y1))))
    ox, oy = max(0, x1 - pad), max(0, y1 - pad)
    ex, ey = min(W, x2 + pad), min(H, y2 + pad)
    b.ox, b.oy = ox, oy
    roi = gray[oy:ey, ox:ex]
    h, w = roi.shape
    box = (x1 - ox, y1 - oy, x2 - ox, y2 - oy)
    bw, bh = x2 - x1, y2 - y1
    forced = b.type
    kind, interior = None, None
    kz = b.kz if getattr(a, "koharu", 0) else None
    kzone = None
    if kz is not None and kz.usable:
        from app.services import koharu_detector as kd      # import trễ: stage_a vẫn chạy độc lập không cần koharu
        kzone = kd.rasterize(kz, (h, w), ox, oy, int(a.k_zone_pad), int(a.k_box_pad))
    if forced == "mask" and kzone is not None and kzone.sfx is not None:
        # SFX theo mask koharu: xoá nguyên đa giác (nở k_sfx_grow px), để LaMa / texture tái tạo tranh bên dưới
        sm = cv2.dilate(kzone.sfx.astype(np.uint8), ellipse_kernel(int(a.k_sfx_grow))) > 0
        b.kind, b.method, b.polarity, b.strokes, b.thickness = "free", "koharu-sfx", "black", sm, 0.0
        ys, xs = np.nonzero(sm)
        b.occ = (int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + ox + 1, int(ys.max()) + oy + 1) if len(xs) else \
            (x1, y1, x2, y2)
        return
    # Chữ tiêu đề / chữ cỡ lớn (nét dày, thường dính nhau thành cả cột, nằm trên nền trắng hoặc tranh):
    # bộ lọc "khối dày = hình vẽ" sẽ bỏ sót chúng, còn chính nét chữ lại tạo ra "viền" giả khiến bị nhận nhầm là bóng thoại.
    t75 = t90 = 0.0
    if a.thick_stroke > 0:
        t75, t90 = stroke_thickness(roi, box, a.free_dark_thr)
    b.large = bool(a.thick_stroke > 0 and a.thick_stroke <= t75 <= 0.3 * min(bw, bh))
    if b.large and forced is None:
        # Bóng thoại thật có viền dày (>= 7 px) cũng cho độ dày lớn: nhưng vùng trắng của nó lấp gần kín khung.
        # Vùng "trắng" do nét chữ lớn tạo ra thì tràn ra ngoài khung hoặc chỉ phủ một phần.
        ratio, cover = white_fill_stats(roi, box, a)
        if 0.8 <= ratio <= 1.15 and cover >= a.bubble_cover_thick:
            b.large = False
    if b.large and forced is None:
        forced = "free"
        b.method = "large text"

    if forced in (None, "dark"):
        r = detect_dark(roi, box, a)
        if r is not None:
            kind, interior, b.method = "dark", r, "blob"
    if kind is None and forced in (None, "bubble") and kzone is not None and kzone.interior is not None:
        r = koharu_interior(roi, kzone.interior, a)
        if r is not None:
            kind, interior, b.method = "bubble", r, "boxed-koharu"       # tên bắt đầu bằng "boxed": decide_fill / edge_reach coi như bóng nhận theo khung
    if kind is None and forced in (None, "bubble") and getattr(a, "box_bubble", 0):
        r = detect_bubble_boxed(roi, box, a)          # khung = khung bao bóng thoại (xem docstring)
        if r is not None:
            kind, (interior, b.method) = "bubble", r
    if kind is None and forced in (None, "bubble"):
        r = detect_bubble(roi, box, a, force=forced == "bubble")
        if r is not None:
            kind, (interior, b.method) = "bubble", r
    if kind is None and forced == "dark":      # forced but no blob found -> use the bbox
        kind, b.method = "dark", "bbox"
        interior = np.zeros((h, w), bool)
        interior[box[1]:box[3], box[0]:box[2]] = True
    if kind is None and forced is None and float(np.median(roi[box[1]:box[3], box[0]:box[2]])) >= a.auto_bubble_med \
            and busy_fraction(roi[box[1]:box[3], box[0]:box[2]]) < a.auto_bubble_busy:
        a2 = copy.copy(a)
        a2.bubble_cover, a2.outline_frac = a.auto_bubble_cover, 0.25
        r = detect_bubble(roi, box, a2, force=False)
        if r is not None:
            kind, (interior, b.method) = "bubble", r
    if kind is None:
        kind = "sfx" if forced == "sfx" else "free"
        b.method = forced or "auto"
    b.kind = kind

    max_side = int(np.clip(0.28 * a.max_side_mult * max(bw, bh), 40, 110 * a.max_side_mult))
    if b.large:
        a = copy.copy(a)
        a.big_radius = max(a.big_radius, int(round(1.6 * t90)))
        max_side = max(max_side, int(1.3 * max(bw, bh)))
    rp = a.ruby_pad
    kb = max(4, pad - 2) if b.large else 4          # chữ lớn: thành phần chữ chỉ cần có tâm gần khung
    keep_box = (box[0] - kb - rp, box[1] - kb, box[2] + kb + rp, box[3] + kb)
    strokes = np.zeros((h, w), bool)
    core_for_t = strokes
    pol = b.polarity or ("white" if kind == "dark" else "black")

    if kind == "bubble":
        thr_b = a.bubble_thr
        if interior.any():
            thr_b = int(min(a.bubble_thr, max(120, float(np.median(roi[interior])) - 14)))
        dist_in = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, 3)
        guard = dist_in > a.bubble_guard           # keeps the outline and its anti-aliased halo intact
        kgate = None
        if kzone is not None and kzone.text is not None and a.k_gate:
            # koharu: chỉ xoá nét nằm trong vùng chữ. Lưới an toàn: còn nhiều mực ngoài vùng chữ => koharu có thể sót chữ -> xoá cả lòng bóng như cũ
            ink = (roi <= thr_b) & guard
            ink_in, ink_out = int((ink & kzone.text).sum()), int((ink & ~kzone.text).sum())
            if ink_out > 30 and ink_out > a.k_ink_ratio * max(1, ink_in):
                b.method += "-nogate"
            else:
                kgate = kzone.text
        strokes = (roi <= thr_b) & guard
        if kgate is not None:
            strokes &= kgate
        strokes = (cv2.dilate(strokes.astype(np.uint8), ellipse_kernel(a.bubble_grow)) > 0) & guard
        if kgate is not None:
            strokes &= kgate
        core_in = strokes.copy()
        b.guard = guard
        pol = "black"
        b.interior = interior
        if a.bubble_edge_pass:
            # glyphs that touch the bubble outline / hair are cut off from `interior`; rescue the
            # dark strokes next to it. Thick artwork is severed first so hair is not taken for text.
            cb = collect_core(roi, "black", a, max_side, keep_box, box)
            # bóng nhận theo khung: chữ chỉ có thể thò ra vài px quanh lòng bóng; với tầm xa 14 px nét tranh (gạch bóng,
            # tóc) sát bóng bị nhận nhầm là chữ dính viền và bị xoá lố
            reach = min(a.edge_reach, 6) if b.method.startswith("boxed") else a.edge_reach
            near_i = cv2.dilate(interior.astype(np.uint8), ellipse_kernel(reach)) > 0
            cb &= near_i & ~interior
            lim_e = np.zeros((h, w), bool)
            lim_e[max(0, box[1] - 4):box[3] + 4, max(0, box[0] - 4 - rp):box[2] + 4 + rp] = True
            cb &= lim_e
            if kgate is not None:
                cb &= kgate
            if cb.any() and core_in.any():
                seed = cv2.dilate(core_in.astype(np.uint8), ellipse_kernel(a.edge_link)) > 0
                nlab, lab = cv2.connectedComponents(cv2.dilate(cb.astype(np.uint8), ellipse_kernel(1)), connectivity=8)
                hit = np.unique(lab[seed & cb])
                cb = np.isin(lab, hit[hit > 0]) & cb
                if cb.any():
                    cb = cv2.dilate(cb.astype(np.uint8), ellipse_kernel(a.dilation + 1)) > 0
                    strokes |= cb
            else:
                pass
    elif kind == "dark":
        inner = cv2.erode(interior.astype(np.uint8), ellipse_kernel(2)) > 0
        cand = (roi >= a.white_text_thr) & inner
        strokes = pick_components(cand, roi, lambda v: v < 110, a.ring_min, max_side, keep_box)
        strokes = cv2.dilate(strokes.astype(np.uint8), ellipse_kernel(a.dilation + 1)) > 0
        strokes &= inner
        pol = "white"
        b.interior = interior
    elif kind == "free":
        zt = kzone.text if (kzone is not None and kzone.text is not None and a.k_gate) else None
        if zt is not None and zt.any():
            # Lưới an toàn (giống nhánh bóng thoại): vùng chữ koharu hay chỉ phủ MỘT PHẦN chữ cách điệu / chữ giãn chữ rộng (VD: tag
            # "＃まのさばネタバレ" các chữ cách xa nhau) -> mọi nét nằm ngoài vùng bị bỏ sót, hiện ra thành bóng mờ chữ gốc quanh bản dịch.
            # Nếu trong KHUNG còn nhiều mực ngoài vùng chữ thì coi vùng koharu không đủ tin cậy: bỏ cổng, xoá theo khung như cũ.
            pol0 = b.polarity
            c_in = c_out = 0
            if pol0 is None:
                cb0 = collect_core(roi, "black", a, max_side, keep_box, box)
                cw0 = collect_core(roi, "white", a, max_side, keep_box, box)
                bg0 = float(np.median(roi[box[1]:box[3], box[0]:box[2]]))
                core0 = cw0 if (bg0 < a.auto_pol_bg and cw0.sum() > a.auto_pol_ratio * max(1, cb0.sum())) else cb0
            else:
                core0 = collect_core(roi, pol0, a, max_side, keep_box, box)
            c_in, c_out = int((core0 & zt).sum()), int((core0 & ~zt).sum())
            if c_out > 30 and c_out > a.k_ink_ratio * max(1, c_in):
                zt = None
                b.method += "-nogate"
        if zt is not None and zt.any():
            zy, zx = np.nonzero(zt)                 # chữ tự do: thành phần chữ phải có tâm trong vùng chữ koharu (+4 px)
            keep_box = (int(zx.min()) - 4, int(zy.min()) - 4, int(zx.max()) + 4, int(zy.max()) + 4)
        else:
            zt = None
        if b.polarity is None:                  # auto: pick the polarity that explains more glyph pixels
            cb = collect_core(roi, "black", a, max_side, keep_box, box)
            cw = collect_core(roi, "white", a, max_side, keep_box, box)
            med_bg = float(np.median(roi[box[1]:box[3], box[0]:box[2]]))
            pol = "white" if (med_bg < a.auto_pol_bg and cw.sum() > a.auto_pol_ratio * max(1, cb.sum())) else "black"
            core = cw if pol == "white" else cb
        else:
            core = collect_core(roi, pol, a, max_side, keep_box, box)
        t = halo_radius(core)
        r = int(np.clip(round(0.6 * t), 3, 9))
        ring = cv2.dilate(core.astype(np.uint8), ellipse_kernel(r)) > 0
        halo = ring & ~core & ((roi >= a.halo_thr) if pol == "black" else (roi < 110))
        for kx1, ky1, kx2, ky2 in b.keep:      # "keep" blocks the halo only; glyph pixels are still erased
            halo[max(0, ky1 - oy):max(0, ky2 - oy), max(0, kx1 - ox):max(0, kx2 - ox)] = False
        strokes = cv2.dilate((core | halo).astype(np.uint8), ellipse_kernel(a.dilation)) > 0
        # never reach outside box + pad margin
        lim = np.zeros((h, w), bool)
        lm = pad if b.large else 8                  # chữ lớn: xoá trọn thành phần, không cắt cụt ở mép khung
        lim[max(0, box[1] - lm):box[3] + lm, max(0, box[0] - lm - rp):box[2] + lm + rp] = True
        strokes &= lim
        if zt is not None:
            strokes &= zt                       # koharu: không bao giờ xoá ngoài vùng chữ
        core_for_t = core
    if kind != "free":                          # bubble / dark: protected areas are never erased
        for kx1, ky1, kx2, ky2 in b.keep:
            strokes[max(0, ky1 - oy):max(0, ky2 - oy), max(0, kx1 - ox):max(0, kx2 - ox)] = False
    b.polarity = pol
    b.strokes = strokes

    core_mask = strokes if kind != "free" else core_for_t
    if kind == "bubble":
        core_mask = (roi <= 140) & interior
    b.thickness = halo_radius(core_mask) if core_mask.sum() > 30 else 0.0

    # occupied rectangle (global)
    if kind in ("bubble", "dark") and b.interior is not None and b.interior.any():
        ys, xs = np.nonzero(b.interior)
    elif strokes.any():
        ys, xs = np.nonzero(strokes)
    else:
        ys, xs = np.array([box[1], box[3]]), np.array([box[0], box[2]])
    b.occ = (int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + ox + 1, int(ys.max()) + oy + 1)


def _grow_box_to_ink(gray: np.ndarray, b: Item, a, others=None):
    """Chữ tự do (không phải bóng) mà khung dò được CẮT mất đầu/cuối dòng (VD: khung bắt đầu giữa dòng, bỏ sót dấu ◆ và vài chữ đầu)
    thì phần chữ nằm ngoài khung không bao giờ được xoá -> sót chữ gốc đè lên bản dịch.

    Nới khung dọc theo HƯỚNG ĐỌC (khung ngang -> nới trái / phải, khung dọc -> nới lên / xuống) qua các thành phần nét chữ nối tiếp nhau
    (khoảng hở <= grow_gap_mult x cỡ chữ). Chỉ nhận thành phần nằm trong dải của khung (không ăn sang dòng / đoạn khác), không nhận thành phần
    thuộc khung thoại khác, và nới tối đa ~35% cạnh dài mỗi phía. Trả về khung mới hoặc None nếu không có gì để nới."""
    H, W = gray.shape
    x1, y1, x2, y2 = b.bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8 or b.polarity not in ("white", "black"):
        return None
    horiz = bw >= bh
    G = int(max(60, 0.35 * max(bw, bh)))
    wx1, wy1 = (max(0, x1 - G), y1) if horiz else (x1, max(0, y1 - G))
    wx2, wy2 = (min(W, x2 + G), y2) if horiz else (x2, min(H, y2 + G))
    roi = gray[wy1:wy2, wx1:wx2]
    if roi.size == 0:
        return None
    max_side = int(np.clip(0.28 * a.max_side_mult * max(bw, bh), 40, 110 * a.max_side_mult))
    whole = (-10, -10, roi.shape[1] + 10, roi.shape[0] + 10)
    core = collect_core(roi, b.polarity, a, max_side, whole)
    # Vùng quanh chữ phải PHẲNG: loại trừ nét chữ (nở 3px), phần còn lại phải gần như một màu (đen kín / giấy trắng). Chữ cách điệu / SFX nằm
    # trên nền tranh (nét, hatching) thì các nét tranh liền nhau sẽ bị nối nhầm thành "chữ" làm khung lan hàng trăm px -> không nới.
    txt_zone = cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rest = roi[~txt_zone]
    if rest.size < 200:
        return None
    rmed = float(np.median(rest))
    if float((np.abs(rest.astype(np.int16) - rmed) <= 30).mean()) < 0.97:
        return None
    n, lab, st, cen = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    comps = []
    for i in range(1, n):
        cx0, cy0, cw, ch, area = (int(v) for v in st[i])
        comps.append((cx0 + wx1, cy0 + wy1, cx0 + wx1 + cw, cy0 + wy1 + ch))
    inside = [c for c in comps if x1 <= (c[0] + c[2]) / 2 <= x2 and y1 <= (c[1] + c[3]) / 2 <= y2]
    if not inside:
        return None
    size = float(np.median([(c[3] - c[1]) if horiz else (c[2] - c[0]) for c in inside]))   # cỡ chữ ~ bề cao ký tự (khung ngang) / bề rộng (khung dọc)
    gap = max(14.0, float(getattr(a, "grow_gap_mult", 1.6)) * max(size, 8.0))
    owned = [tuple(o) for o in (others or [])]

    def foreign(c) -> bool:
        cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
        return any(o[0] + 2 <= cx <= o[2] - 2 and o[1] + 2 <= cy <= o[3] - 2 for o in owned)

    lo, hi = (x1, x2) if horiz else (y1, y2)
    ca, cb_ = (y1, y2) if horiz else (x1, x2)          # dải ngang (khung ngang) / dọc (khung dọc) mà thành phần phải nằm trong
    todo = [c for c in comps if c not in inside and not foreign(c)
            and ca - 4 <= ((c[1] + c[3]) / 2 if horiz else (c[0] + c[2]) / 2) <= cb_ + 4]
    changed = True
    while changed:
        changed = False
        for c in list(todo):
            c0, c1 = (c[0], c[2]) if horiz else (c[1], c[3])
            if c0 < lo and c1 >= lo - gap:
                lo, changed = min(lo, c0), True
            elif c1 > hi and c0 <= hi + gap:
                hi, changed = max(hi, c1), True
            else:
                continue
            todo.remove(c)
    pad = 2
    if horiz:
        nb = (max(0, int(lo) - pad), y1, min(W, int(hi) + pad), y2)
    else:
        nb = (x1, max(0, int(lo) - pad), x2, min(H, int(hi) + pad))
    grew = (x1 - nb[0]) + (nb[2] - x2) + (y1 - nb[1]) + (nb[3] - y2)
    return nb if grew >= 8 else None


def analyze(gray: np.ndarray, b: Item, a, others=None) -> None:
    """Phân tích 1 khung thoại (xem _analyze_once). Với chữ tự do / chữ trắng trên nền tối: nếu khung bị cắt mất một phần dòng chữ thì nới
    khung theo nét chữ rồi phân tích lại để xoá trọn dòng (a.grow_ink = 0 để tắt). others: danh sách khung (x1,y1,x2,y2) của các khung
    thoại KHÁC trên trang (không nuốt chữ của chúng)."""
    _analyze_once(gray, b, a)
    if not int(getattr(a, "grow_ink", 1)) or b.kind not in ("free", "dark") or b.region is not None or b.type in ("mask", "sfx"):
        return
    old_box, old_type = b.bbox, b.type
    cap = int(0.5 * max(old_box[2] - old_box[0], old_box[3] - old_box[1]))   # trần TỔNG mỗi phía (cộng dồn các vòng), tránh nới lan
    for _ in range(3):                      # mỗi vòng nới tối đa ~35% cạnh dài mỗi phía; khung bị cắt nhiều thì cần vài vòng
        nb = _grow_box_to_ink(gray, b, a, others)
        if nb is None:
            break
        nb = (max(nb[0], old_box[0] - cap), max(nb[1], old_box[1] - cap), min(nb[2], old_box[2] + cap), min(nb[3], old_box[3] + cap))
        if nb == tuple(b.bbox):
            break
        b.bbox = nb
        b.type = old_type or b.kind         # phân tích lại đúng nhánh cũ (không dò lại bóng thoại trong khung đã nới)
        try:
            _analyze_once(gray, b, a)
        finally:
            b.type = old_type
        if b.kind not in ("free", "dark"):
            break
    if b.bbox != old_box:
        log.info("Nới khung chữ %s: %s -> %s (nét chữ nằm ngoài khung dò được)", b.ids, old_box, b.bbox)


def decide_fill(image: np.ndarray, gray: np.ndarray, b: Item, a) -> None:
    """Pick the erase engine + fill mode for a block, from the ORIGINAL image.

    OpenCV (flat) only when the background is trivially reproducible: a closed, clean, plain
    white bubble, a solid dark blob, or flat white/black paper. Everything else goes to LaMa.
    """
    h, w = b.strokes.shape
    roi = image[b.oy:b.oy + h, b.ox:b.ox + w]
    groi = gray[b.oy:b.oy + h, b.ox:b.ox + w]
    st = b.strokes
    auto, why = "inpaint", "no strokes"
    if st.any():
        if b.kind == "bubble":
            # Chỉ đo nền ở phần lòng bóng nằm xa viền (guard): mẩu viền / vệt chì sát biên không phải "nền không đồng đều"
            # (đo cả vùng sát viền làm bóng trắng trơn bị đẩy sang LaMa / inpaint không cần thiết)
            use = b.interior & ~st
            if b.guard is not None and int((use & b.guard).sum()) > 50:
                use &= b.guard
            bg = roi[use]
            med = np.median(bg, axis=0) if len(bg) > 50 else np.array([255, 255, 255])
            sd = float(bg.std(axis=0).max()) if len(bg) > 50 else 99
            b.flat_color = tuple(int(v) for v in med)
            if med.min() >= 244:                # near-white paper -> pure white, no ghost outlines
                b.flat_color = (255, 255, 255)
            sel = b.guard & ~(cv2.dilate(st.astype(np.uint8), ellipse_kernel(3)) > 0)
            b.busy = busy_fraction(groi, sel) if int(sel.sum()) > 50 else 0.0
            if not b.method.startswith(("shape", "boxed")):
                why = "bubble outline not closed (ellipse fallback)"
            elif med.min() < 225:
                why = f"bubble is not white (median {med.min():.0f})"
            elif sd >= 14:
                why = f"background not uniform (std {sd:.0f})"
            elif b.busy >= a.simple_busy:
                why = f"texture inside bubble (busy {b.busy:.2f})"
            else:
                auto, why = "flat", "plain white bubble"
        elif b.kind == "dark":
            bg = roi[b.interior & ~st & (groi < a.dark_thr)]
            med = np.median(bg, axis=0) if len(bg) > 50 else np.array([0, 0, 0])
            auto, why = "flat", "solid dark blob"
            b.flat_color = tuple(int(v) for v in med)
        else:
            ring = (cv2.dilate(st.astype(np.uint8), ellipse_kernel(12)) > 0) & \
                   (cv2.dilate(st.astype(np.uint8), ellipse_kernel(3)) == 0)
            bg = roi[ring]
            g = groi[ring]
            if len(bg) > 50:
                med = np.median(bg, axis=0)
                gm, gs = float(np.median(g)), float(g.std())
                b.flat_color = tuple(int(v) for v in med)
                if (gm >= 228 and gs < 14) or (gm <= 30 and gs < 10):
                    auto, why = "flat", f"flat paper/black (grey {gm:.0f}, std {gs:.0f})"
                else:
                    why = f"complex background (grey {gm:.0f}, std {gs:.0f})"
            else:
                why = "no background ring around text"

    # busy art (hatching / screentone): LaMa by default, v6 patch-match texture fill if asked
    want_tex = False
    if auto == "inpaint":
        if b.kind == "free":
            ring_t = (cv2.dilate(st.astype(np.uint8), ellipse_kernel(14)) > 0) & \
                     (cv2.dilate(st.astype(np.uint8), ellipse_kernel(4)) == 0)
            b.busy = busy_fraction(groi, ring_t)
            want_tex = a.textured_fill == "texture" or (a.textured_fill == "auto" and b.busy >= a.texture_busy)
        else:
            want_tex = a.textured_fill == "texture"
    lama_pref = a.lama_ok and a.busy_engine == "lama"
    if want_tex and (a.textured_fill == "texture" or not lama_pref):
        auto, why = "texture", f"busy art (busy {b.busy:.2f}) -> patch-match texture"

    # ---- engine routing
    mode = b.fill or auto
    force = b.engine_force or (a.force_engine if a.force_engine != "auto" else None)
    if force == "lama" and mode == "flat" and a.lama_ok and not b.fill:
        mode, why = "inpaint", "forced lama"
    if force == "opencv":
        why += " | forced opencv"
    if mode in ("flat", "texture"):
        engine = "opencv"
    elif a.lama_ok and force != "opencv":
        engine = "lama"
    else:
        engine = "opencv"
        why += " | cv2.inpaint (LaMa off/unavailable)"
    if mode == "flat" and b.kind == "bubble" and b.interior is not None and a.lama_ok and force != "opencv":
        if int((st & ~b.interior).sum()) >= 4:     # strokes touching the outline: LaMa rebuilds the line
            engine, why = "opencv+lama", why + " + glyphs on the outline -> LaMa"
    b.fill_mode, b.engine, b.why = mode, engine, why


# --------------------------------------------------------------------------- erase
class Lama:
    """Optional LaMa ONNX inpainter (crops around the mask, runs on CPU)."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        self.img_name = next((i.name for i in ins if "image" in i.name.lower()), ins[0].name)
        self.mask_name = next((i.name for i in ins if "mask" in i.name.lower()), ins[1].name)
        shape = next(i.shape for i in ins if i.name == self.img_name)
        self.size = int(shape[-1]) if isinstance(shape[-1], int) else 512

    def __call__(self, img: np.ndarray, mask: np.ndarray, ctx: int = 48) -> np.ndarray:
        H, W = mask.shape
        ys, xs = np.nonzero(mask)
        x1, x2, y1, y2 = xs.min() - ctx, xs.max() + ctx + 1, ys.min() - ctx, ys.max() + ctx + 1
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1, x2, y1, y2 = cx - side // 2, cx + side // 2 + 1, cy - side // 2, cy + side // 2 + 1
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        crop, mcrop = img[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        S = self.size
        ci = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (S, S), interpolation=cv2.INTER_AREA)
        mi = cv2.resize((mcrop > 0).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        feed = {self.img_name: (ci.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
                self.mask_name: mi.astype(np.float32)[None, None]}
        out = self.sess.run(None, feed)[0][0].transpose(1, 2, 0)
        if out.max() <= 2.0:
            out = out * 255.0
        out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
        res = img.copy()
        res[y1:y2, x1:x2][mcrop > 0] = out[mcrop > 0]
        return res


_LAMA_QUEUE: np.ndarray | None = None      # global uint8 mask of everything LaMa has to rebuild (flushed once)


def _lama_apply(out: np.ndarray, full: np.ndarray, a, lama) -> bool:
    """Run LaMa on the global mask `full` and blend the result into `out`. False if it failed."""
    H, W = out.shape[:2]
    try:
        t0 = time.perf_counter()
        ml = cv2.dilate(full, ellipse_kernel(a.lama_grow))
        res = lama(out, ml)
        ys, xs = np.nonzero(ml)
        pad = 8
        y1, y2 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
        x1, x2 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)
        soft = cv2.GaussianBlur(ml[y1:y2, x1:x2].astype(np.float32), (0, 0), 1.5)[..., None]
        soft = np.clip(soft * 1.5, 0, 1)
        blend = res[y1:y2, x1:x2].astype(np.float32) * soft + out[y1:y2, x1:x2].astype(np.float32) * (1 - soft)
        out[y1:y2, x1:x2] = np.clip(blend, 0, 255).astype(np.uint8)
        print(f"  LaMa call: {int((ml > 0).sum())} px, bbox {x2 - x1}x{y2 - y1}, {time.perf_counter() - t0:.1f}s")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  LaMa failed ({e}); falling back to cv2.inpaint", file=sys.stderr)
        return False


def _cv2_apply(out: np.ndarray, full: np.ndarray, a) -> None:
    H, W = out.shape[:2]
    ys, xs = np.nonzero(full)
    pad = 16
    y1, y2 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
    x1, x2 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)
    crop = np.ascontiguousarray(out[y1:y2, x1:x2])
    res = cv2.inpaint(crop, full[y1:y2, x1:x2] * 255, a.inpaint_radius, cv2.INPAINT_TELEA)
    sel = full[y1:y2, x1:x2] > 0
    out[y1:y2, x1:x2][sel] = res[sel]


def flush_lama(out: np.ndarray, a, lama) -> None:
    """Run the queued LaMa mask of the whole page in a single pass."""
    global _LAMA_QUEUE
    q, _LAMA_QUEUE = _LAMA_QUEUE, None
    if q is None or not q.any():
        return
    if lama is None or not _lama_apply(out, q, a, lama):
        _cv2_apply(out, q, a)


def inpaint_region(out: np.ndarray, oy: int, ox: int, m: np.ndarray, a, lama, overlay: bool = False) -> str:
    """Inpaint the ROI-local mask `m` (origin ox, oy). LaMa (queued if a queue is active) else cv2.inpaint."""
    H, W = out.shape[:2]
    h, w = m.shape
    if lama is not None and m.any():
        if _LAMA_QUEUE is not None:
            _LAMA_QUEUE[oy:oy + h, ox:ox + w] |= m
            return "queued"
        full = np.zeros((H, W), np.uint8)
        full[oy:oy + h, ox:ox + w] = m
        if _lama_apply(out, full, a, lama):
            return "lama"
    roi = out[oy:oy + h, ox:ox + w]
    res = cv2.inpaint(np.ascontiguousarray(roi), m * 255, a.inpaint_radius, cv2.INPAINT_TELEA)
    if overlay:
        soft = cv2.GaussianBlur(res, (0, 0), 2.0)
        grow = cv2.dilate(m, ellipse_kernel(2)) > 0
        res = np.where(grow[..., None], soft, res)
        roi[grow] = res[grow]
    else:
        roi[m > 0] = res[m > 0]
    return "cv2"


def _erase_main(out: np.ndarray, b: Item, a, lama: Lama | None) -> np.ndarray:
    full = np.zeros(out.shape[:2], np.uint8)
    h, w = b.strokes.shape
    m = b.strokes.astype(np.uint8)
    if not m.any():
        return full
    full[b.oy:b.oy + h, b.ox:b.ox + w] = m
    roi = out[b.oy:b.oy + h, b.ox:b.ox + w]
    mode = b.fill_mode
    lm = lama if b.engine in ("lama", "opencv+lama") else None
    if mode == "texture":          # done once for all items in texture_fill()
        return full
    if mode == "flat":
        if b.kind == "bubble" and b.interior is not None:
            inner, outer = (m > 0) & b.interior, (m > 0) & ~b.interior
        else:
            inner, outer = m > 0, np.zeros((h, w), bool)
        roi[inner] = np.array(b.flat_color, np.uint8)
        if b.kind == "bubble" and b.guard is not None and min(b.flat_color) >= 250:
            # JPEG halo around the glyphs sits at 225-254: invisible per pixel, but a faint ghost of
            # the text as a whole. Snap it to paper white, only inside the guard so the outline stays.
            near = cv2.dilate(m, ellipse_kernel(a.snap_reach)) > 0
            g = cv2.cvtColor(np.ascontiguousarray(roi), cv2.COLOR_BGR2GRAY)
            snap = near & b.guard & (g >= a.snap_thr)
            roi[snap] = 255
        elif b.kind == "free" and int(getattr(a, "halo_sweep", 1)):
            # Quét viền mờ cho CHỮ TỰ DO trên giấy phẳng (đã kiểm chứng bằng mô phỏng scan mềm): quanh nét đã xoá còn dải khử răng cưa xám nhạt
            # hiện thành "viền chữ" hình ma -> ép các điểm sáng-vừa sát nét (<= 2 px) về màu nền. Không áp dụng cho bóng thoại (giữ logic gốc).
            near = cv2.dilate(m, ellipse_kernel(min(int(a.snap_reach), 2))) > 0
            g = cv2.cvtColor(np.ascontiguousarray(roi), cv2.COLOR_BGR2GRAY)
            if min(b.flat_color) >= 200:
                snap = near & (g >= int(getattr(a, "snap_thr_low", 150)))
            elif max(b.flat_color) <= 40:
                snap = near & (g <= 255 - int(getattr(a, "snap_thr_low", 150)))
            else:
                snap = None
            if snap is not None:
                roi[snap] = np.array(b.flat_color, np.uint8)

        if outer.any():             # glyphs over the outline: a flat fill would erase the line too
            inpaint_region(out, b.oy, b.ox, outer.astype(np.uint8), a, lm)
        return full
    inpaint_region(out, b.oy, b.ox, m, a, lm, overlay=(mode == "overlay"))
    return full


def erase_item(out: np.ndarray, b: Item, a, lama: Lama | None) -> np.ndarray:
    full = _erase_main(out, b, a, lama)
    sw = b.strokes_w
    if sw is not None and sw.any():            # white glyphs on artwork: always a "complex" case
        h, w = sw.shape
        inpaint_region(out, b.oy, b.ox, sw.astype(np.uint8), a, lama if b.engine != "opencv" else None)
        fw = np.zeros(out.shape[:2], np.uint8)
        fw[b.oy:b.oy + h, b.ox:b.ox + w] = sw.astype(np.uint8)
        full = full | fw
    return full


# --------------------------------------------------------------------------- texture fill
def texture_fill(out: np.ndarray, mask: np.ndarray, forbid: np.ndarray, a, report: list | None = None) -> int:
    """Rebuild masked pixels by boundary-matched patch copy. Modifies `out`; returns #failed tiles.

    mask   : uint8, pixels to rebuild
    forbid : uint8, pixels that must never be used as a source (other text, kept areas)
    """
    H, W = mask.shape
    T, C, R = a.tex_tile, a.tex_ctx, a.tex_radius
    todo = mask > 0
    if not todo.any():
        return 0
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bad = (todo | (forbid > 0)).astype(np.uint8)
    S = cv2.integral(bad)                                  # (H+1, W+1)
    ys, xs = np.nonzero(todo)
    x0b, x1b, y0b, y1b = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    remaining = {(tx, ty) for ty in range(y0b, y1b, T) for tx in range(x0b, x1b, T)
                 if todo[ty:ty + T, tx:tx + T].any()}
    failed = np.zeros((H, W), bool)
    nfail = 0

    def ctx_win(tx, ty):
        return max(0, tx - C), max(0, ty - C), min(W, tx + T + C), min(H, ty + T + C)

    def known_ratio(t):
        x0, y0, x1, y1 = ctx_win(*t)
        return 1.0 - float(todo[y0:y1, x0:x1].mean())

    while remaining:
        t = max(remaining, key=known_ratio)          # most surrounding context first (peel inwards)
        remaining.discard(t)
        tx, ty = t
        cx1, cy1 = min(W, tx + T), min(H, ty + T)
        core_todo = todo[ty:cy1, tx:cx1]
        if not core_todo.any():
            continue
        x0, y0, x1, y1 = ctx_win(tx, ty)
        templ = gray[y0:y1, x0:x1]
        known = (~todo[y0:y1, x0:x1]).astype(np.float32)
        th, tw = templ.shape
        sx0, sy0, sx1, sy1 = max(0, x0 - R), max(0, y0 - R), min(W, x1 + R), min(H, y1 + R)
        if sx1 - sx0 < tw or sy1 - sy0 < th or known.sum() < 1:
            failed[ty:cy1, tx:cx1] |= core_todo
            nfail += 1
            if report is not None:
                report.append(dict(rect=[tx, ty, cx1, cy1], reason="no_context", score=None))
            continue
        res = cv2.matchTemplate(gray[sy0:sy1, sx0:sx1], templ, cv2.TM_SQDIFF, mask=known)
        rh, rw = res.shape
        ox_, oy_ = tx - x0, ty - y0                  # core offset inside the window
        cw, ch = cx1 - tx, cy1 - ty
        Ys, Xs = sy0 + oy_, sx0 + ox_
        win_bad = (S[Ys + ch:Ys + ch + rh, Xs + cw:Xs + cw + rw] - S[Ys:Ys + rh, Xs + cw:Xs + cw + rw]
                   - S[Ys + ch:Ys + ch + rh, Xs:Xs + rw] + S[Ys:Ys + rh, Xs:Xs + rw])
        dy = (np.arange(rh) + sy0 - y0)[:, None].astype(np.float32)
        dx = (np.arange(rw) + sx0 - x0)[None, :].astype(np.float32)
        dist = np.hypot(dy, dx)
        score = res / float(known.sum()) + a.tex_lam * dist
        score[(win_bad > 0) | (dist < 0.6 * T)] = np.inf
        j, i = np.unravel_index(int(np.argmin(score)), score.shape)
        if not np.isfinite(score[j, i]):
            failed[ty:cy1, tx:cx1] |= core_todo
            nfail += 1
            if report is not None:
                report.append(dict(rect=[tx, ty, cx1, cy1], reason="no_source", score=None))
            continue
        if report is not None:       
            kv = templ[known > 0]       # normalise by the context variance so flat and hatched areas compare
            report.append(dict(rect=[tx, ty, cx1, cy1], reason="ok",
                               score=float(res[j, i] / max(1.0, float(known.sum())) / max(25.0, float(kv.var())))))
        sY, sX = Ys + j, Xs + i
        src = out[sY:sY + ch, sX:sX + cw]
        tgt = out[ty:cy1, tx:cx1]
        tgt[core_todo] = src[core_todo]
        todo[ty:cy1, tx:cx1] = False
        gray[ty:cy1, tx:cx1] = cv2.cvtColor(np.ascontiguousarray(tgt), cv2.COLOR_BGR2GRAY)
    if failed.any():
        res = cv2.inpaint(out, failed.astype(np.uint8) * 255, a.inpaint_radius, cv2.INPAINT_TELEA)
        out[failed] = res[failed]
    return nfail


# --------------------------------------------------------------------------- residual check
def residual_pass(out: np.ndarray, items: list, a, lama, mask_full: np.ndarray) -> list[dict]:
    """Re-detect text in the CLEANED image around every erased block; erase what is left.

    Only pixels within a few px of the first-pass mask count, so artwork that was never touched
    cannot be mistaken for leftovers.
    """
    report = []
    for b in items:
        if not b.cleanup or b.kind == "sfx" or not b.strokes.any():
            continue
        if b.busy > a.retry_max_busy:          # hatching/screentone: re-detection only finds the art itself
            continue
        gnow = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        near = cv2.dilate(b.strokes.astype(np.uint8), ellipse_kernel(6)) > 0
        a2 = copy.copy(a)
        a2.dilation, a2.ring_min, a2.bubble_thr = a.dilation + 1, a.ring_min * 0.6, a.bubble_thr + 3
        a2.grow_ink = 0                         # dò lại trong ĐÚNG khung đã nới của lượt 1 (không nới thêm: đổi khung ROI làm lệch mảng near)
        t = Item(gid=b.gid, ids=b.ids, bbox=b.bbox, text=b.text, type=b.kind, polarity=b.polarity, keep=b.keep)
        analyze(gnow, t, a2)
        h, w = t.strokes.shape
        # near thuộc hệ toạ độ ROI của b, còn t có thể có ROI khác -> căn theo toạ độ TRANG rồi cắt đúng khung của t
        Hp, Wp = gnow.shape
        near_full = np.zeros((Hp, Wp), bool)
        bh, bw = near.shape
        near_full[b.oy:b.oy + bh, b.ox:b.ox + bw] = near[:Hp - b.oy, :Wp - b.ox]
        win = np.zeros((h, w), bool)
        ys, xs = max(0, t.oy), max(0, t.ox)
        ye, xe = min(Hp, t.oy + h), min(Wp, t.ox + w)
        if ye > ys and xe > xs:
            win[ys - t.oy:ye - t.oy, xs - t.ox:xe - t.ox] = near_full[ys:ye, xs:xe]
        t.strokes = t.strokes & win
        left = int(t.strokes.sum())
        entry = dict(id=b.gid, residual_px=left, retried=False)
        if left > a.residual_px:
            t.fill_mode = "flat" if b.fill_mode == "flat" else "inpaint"
            t.flat_color = b.flat_color
            t.engine = b.engine
            for _ in range(a.retry):
                m = erase_item(out, t, a2, lama)
                mask_full |= m
            entry["retried"] = True
            entry["rect"] = list(b.bbox)
        report.append(entry)
    return report


# --------------------------------------------------------------------------- free-text region
def page_gutters(gray: np.ndarray) -> np.ndarray:
    """Rows that are white across (almost) the whole page width = gutters between panels."""
    return (gray > 235).mean(axis=1) >= 0.985


def allowed_rect(b: Item, items: list[Item], gray: np.ndarray, gutters: np.ndarray, a) -> tuple:
    """Grow the original text rect sideways/up/down until something is in the way."""
    H, W = gray.shape
    x1, y1, x2, y2 = b.occ
    w, h = x2 - x1, y2 - y1
    cap_x = int(max(0.5 * (a.h_expand - 1) * w, 24))
    cap_y = int(max(0.5 * (a.v_expand - 1) * h, 16))
    others = [(o.occ[0] - 6, o.occ[1] - 6, o.occ[2] + 6, o.occ[3] + 6) for o in items if o is not b]
    # Chữ trắng nằm trên nền đen: nền tối là chỗ ĐƯỢC PHÉP nới ra; rào cản là điểm sáng (viền panel, bóng thoại, tranh).
    # Trước đây luôn coi điểm tối là rào cản nên với nền đen khung chữ không nới được, chữ dịch bị ép vào cột hẹp rất nhỏ.
    dark = (gray > 200) if b.polarity == "white" else (gray < 70)

    def blocked(r) -> bool:
        return any(rects_intersect(r, o) for o in others)

    # Chỉ nới vào chỗ còn là NỀN: cột / hàng chứa nhiều điểm khác màu nền (mặt, tóc, nét tranh dày) là rào cản. Trước đây chỉ dừng khi gặp
    # vùng gần như đen hẳn nên chữ dịch của khung tự do nới ra đè lên nhân vật / hoạ tiết (nền = màu trung vị của vùng chữ gốc).
    flat_frac = float(getattr(a, "flat_frac", 0.15))
    bg_med = float(np.median(gray[max(0, y1):min(H, y2), max(0, x1):min(W, x2)])) if x2 > x1 and y2 > y1 else 255.0
    off_bg = np.abs(gray.astype(np.int16) - bg_med) > 60

    def busy_col(x, ya, yb) -> bool:
        col = off_bg[ya:yb, x]
        return flat_frac > 0 and col.size > 0 and col.mean() > flat_frac

    def busy_row(y, xa, xb) -> bool:
        row = off_bg[y, xa:xb]
        return flat_frac > 0 and row.size > 0 and row.mean() > flat_frac

    def border_col(x, ya, yb) -> bool:
        col = dark[ya:yb, x]
        return (col.size > 0 and col.mean() > 0.8) or busy_col(x, ya, yb)

    def border_row(y, xa, xb) -> bool:
        if gutters[y]:
            return True
        row = dark[y, xa:xb]
        return (row.size > 0 and row.mean() > 0.8) or busy_row(y, xa, xb)

    L = R = U = D = 0
    while R < cap_x and x2 + R + 2 < W and not blocked((x1, y1, x2 + R + 2, y2)) \
            and not border_col(min(W - 1, x2 + R + 2), y1, y2):
        R += 2
    while L < cap_x and x1 - L - 2 > 0 and not blocked((x1 - L - 2, y1, x2 + R, y2)) \
            and not border_col(max(0, x1 - L - 2), y1, y2):
        L += 2
    ax1, ax2 = x1 - L, x2 + R
    while U < cap_y and y1 - U - 2 > 0 and not blocked((ax1, y1 - U - 2, ax2, y2)) \
            and not border_row(y1 - U - 2, ax1, ax2):
        U += 2
    while D < cap_y and y2 + D + 2 < H and not blocked((ax1, y1, ax2, y2 + D + 2)) \
            and not border_row(y2 + D + 2, ax1, ax2):
        D += 2
    res = (ax1, y1 - U, ax2, y2 + D)
    if getattr(a, "box_bubble", 0):
        # Khung lớn hơn hẳn vùng chữ gốc = khung bao bóng thoại / vùng chữ do người dùng đặt: không nới chữ dịch ra ngoài khung
        # (nới 1.7 lần theo chiều ngang làm chữ tràn khỏi bóng khi không nhận ra được viền bóng)
        bx1, by1, bx2, by2 = b.bbox
        if w * h < 0.65 * max(1, (bx2 - bx1) * (by2 - by1)):
            m = 4
            res = (min(max(res[0], bx1 - m), x1), min(max(res[1], by1 - m), y1),
                   max(min(res[2], bx2 + m), x2), max(min(res[3], by2 + m), y2))
    return res


# --------------------------------------------------------------------------- export
def polys_of(b: Item) -> list | None:
    if b.interior is None or b.kind not in ("bubble", "dark"):
        return None
    cs, _ = cv2.findContours(b.interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = b.interior.shape
    polys = [[[int(x) + b.ox, int(y) + b.oy] for x, y in c.reshape(-1, 2)] for c in cs if len(c) >= 3]
    return dict(roi=[b.ox, b.oy, w, h], polys=polys)     # roi = clip window v4 analysed (needed to reproduce erode/borders)


def export_block(b: Item) -> dict:
    tc = None if b.text_color is None else [b.text_color[2], b.text_color[1], b.text_color[0]]  # BGR -> RGB
    return dict(
        id=b.gid, ids=b.ids, text=b.text, kind=b.kind, method=b.method, polarity=b.polarity,
        fill=b.fill_mode, engine=b.engine, engine_why=b.why, bold=bool(b.bold), thickness=round(float(b.thickness), 2),
        occ=list(b.occ), allowed=list(b.allowed) if b.allowed else None,
        region=list(b.region) if b.region else None,
        font_size=b.font_size, font_path=b.font_path, text_color=tc, shape=polys_of(b))


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    """All stage-A options with their defaults (also used in-process by the app: see engine.make_args)."""
    ap = argparse.ArgumentParser(description="Stage A: erase text, export layout (v5).")
    ap.add_argument("--json", required=True)
    ap.add_argument("--page", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--stem", default=None, help="output prefix (default: image stem)")
    ap.add_argument("--lama", default=None)
    ap.add_argument("--pad", type=int, default=14)
    ap.add_argument("--white-thr", type=int, default=200)
    ap.add_argument("--dark-thr", type=int, default=70)
    ap.add_argument("--dark-frac", type=float, default=0.6)
    ap.add_argument("--white-text-thr", type=int, default=140)
    ap.add_argument("--free-dark-thr", type=int, default=100)
    ap.add_argument("--halo-thr", type=int, default=190)
    ap.add_argument("--ring-min", type=float, default=0.4)
    ap.add_argument("--outline-frac", type=float, default=0.4)
    ap.add_argument("--bubble-cover", type=float, default=0.55)
    ap.add_argument("--erode", type=int, default=2)
    ap.add_argument("--dilation", type=int, default=2)
    ap.add_argument("--white-ring-thr", type=int, default=150, help="background must be darker than this around white glyphs (110 in v5)")
    ap.add_argument("--max-side-mult", type=float, default=1.0, help="allow bigger glyph components (touching characters)")
    ap.add_argument("--bubble-edge-pass", type=int, default=1,
                    help="also erase dark glyph strokes that touch the bubble outline / hair; 0=off")
    ap.add_argument("--bubble-guard", type=float, default=5.0,
                    help="px from the bubble outline that are never erased (protects the outline + its halo)")
    ap.add_argument("--snap-thr", type=int, default=225, help="near-white pixels (>= this) around erased text become pure white")
    ap.add_argument("--snap-reach", type=int, default=8, help="how far around the erased strokes the snap applies (px)")
    ap.add_argument("--edge-link", type=int, default=3, help="edge-pass strokes must come within this many px of glyphs inside the bubble")
    ap.add_argument("--edge-reach", type=int, default=14, help="how far outside the bubble interior (px) to look for glyph strokes")
    ap.add_argument("--ruby-pad", type=int, default=14, help="extra side margin so furigana beside a column is included")
    ap.add_argument("--big-radius", type=int, default=7, help="opening radius that identifies artwork masses (hair, borders)")
    ap.add_argument("--thick-stroke", type=float, default=0.0,
                    help="TUỲ CHỌN (mặc định 0 = tắt, giữ nguyên hành vi gốc của v8): nhận diện chữ cỡ lớn theo độ dày nét "
                         "(px, phân vị 75 của đường sống); từ mức này trở lên thì nới --big-radius / max-side, mở rộng vùng "
                         "phân tích và không tự nhận là bóng thoại / khối đen. Gợi ý nếu bật: 7")
    ap.add_argument("--box-bubble", type=int, default=1,
                    help="1 = khung thoại là khung bao của bóng thoại (toạ độ bóng đã chính xác): nhận diện lòng bóng theo khung "
                         "(chịu được viền nét phác / hở), không xoá lấn ra ngoài khung, chữ dịch không nới ra ngoài khung; 0 = cách cũ")
    ap.add_argument("--box-dark-thr", type=int, default=150, help="điểm ảnh tối hơn mức này là nét (chữ / viền / tranh)")
    ap.add_argument("--box-glyph-frac", type=float, default=0.45,
                    help="nét tối có cạnh dài < tỉ lệ này của cạnh ngắn khung là CHỮ; dài hơn là viền / tranh")
    ap.add_argument("--box-min-area", type=float, default=0.30, help="lòng bóng phải phủ >= tỉ lệ này của khung")
    ap.add_argument("--box-outline-cov", type=float, default=0.5, help="tỉ lệ biên lòng bóng phải có viền thật quanh nó")
    ap.add_argument("--box-evidence-reach", type=int, default=5, help="viền phải nằm trong phạm vi này (px) quanh biên lòng bóng")
    ap.add_argument("--koharu", type=int, default=0, help="1 = dùng bóng thoại / vùng chữ của koharu-yolo26s (engine gán Item.kz)")
    ap.add_argument("--k-gate", type=int, default=1, help="chỉ xoá nét nằm trong vùng chữ koharu (có lưới an toàn --k-ink-ratio)")
    ap.add_argument("--k-zone-pad", type=int, default=6, help="nở đa giác vùng chữ (px): đa giác koharu khá thô")
    ap.add_argument("--k-box-pad", type=int, default=4, help="nở hộp vùng chữ (px), hợp với đa giác để không cắt cụt cột chữ")
    ap.add_argument("--k-inset", type=int, default=3, help="co lòng bóng koharu (px): mask có sai số vài px ở mép / viền bóng")
    ap.add_argument("--k-ink-ratio", type=float, default=0.35,
                    help="mực trong bóng nằm NGOÀI vùng chữ > tỉ lệ này so với mực trong vùng chữ -> koharu có thể sót chữ, bỏ gate")
    ap.add_argument("--k-sfx-grow", type=int, default=3, help="nở đa giác SFX (px) khi xoá bằng type=mask")
    ap.add_argument("--bubble-cover-thick", type=float, default=0.72,
                    help="khung nét dày vẫn là bóng thoại thường nếu vùng trắng khép kín phủ >= tỉ lệ này của khung")
    ap.add_argument("--auto-pol-ratio", type=float, default=1.5)
    ap.add_argument("--auto-pol-bg", type=float, default=140, help="auto white polarity only when the box median grey is below this")
    ap.add_argument("--auto-bubble-med", type=float, default=215, help="free block with this median grey is retried as a (relaxed) bubble")
    ap.add_argument("--auto-bubble-cover", type=float, default=0.3)
    ap.add_argument("--bubble-thr", type=int, default=244, help="pixels darker than this inside a bubble are text/halo")
    ap.add_argument("--bubble-grow", type=int, default=2)
    ap.add_argument("--lama-grow", type=int, default=3)
    ap.add_argument("--retry", type=int, default=1, help="extra passes that re-detect text left in the clean image (0=off)")
    ap.add_argument("--retry-max-busy", type=float, default=0.45,
                    help="residual retry only for blocks whose surrounding art is calmer than this (busy fraction)")
    ap.add_argument("--residual-px", type=int, default=25, help="strokes left in a block that trigger a retry")
    ap.add_argument("--inpaint-radius", type=int, default=4)
    ap.add_argument("--textured-fill", choices=["texture", "inpaint", "auto"], default="auto",
                    help="auto: texture only where the surrounding art is busy (hatching/screentone), else inpaint/LaMa")
    ap.add_argument("--texture-busy", type=float, default=0.4)
    ap.add_argument("--auto-bubble-busy", type=float, default=0.35)
    ap.add_argument("--force-engine", choices=["auto", "opencv", "lama"], default="auto",
                    help="auto: OpenCV for plain white bubbles, LaMa for complex backgrounds")
    ap.add_argument("--simple-busy", type=float, default=0.08,
                    help="a white bubble with more than this share of textured pixels is 'complex' -> LaMa")
    ap.add_argument("--busy-engine", choices=["lama", "texture"], default="lama",
                    help="engine for hatching/screentone backgrounds when LaMa is loaded (texture = v6 patch-match)")
    ap.add_argument("--tex-tile", type=int, default=32)
    ap.add_argument("--tex-ctx", type=int, default=12)
    ap.add_argument("--tex-radius", type=int, default=120)
    ap.add_argument("--tex-lam", type=float, default=0.05)
    ap.add_argument("--tex-grow", type=int, default=2)
    ap.add_argument("--qa-score", type=float, default=1.8,
                    help="flag texture tiles whose match error / context variance exceeds this "
                         "(1.0 = no better than the context mean; 1.8 is about the worst 10% on page 011, adjust by eye)")
    ap.add_argument("--h-expand", type=float, default=1.7)
    ap.add_argument("--v-expand", type=float, default=1.3)
    ap.add_argument("--merge-thr", type=float, default=0.2)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--no-auto-bold", action="store_true")
    ap.add_argument("--bold-ratio", type=float, default=1.35)
    ap.add_argument("--no-clean", action="store_true")
    return ap


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    a = build_parser().parse_args()
    t_all = time.perf_counter()
    marks: list[tuple[str, float]] = []

    def mark(name: str) -> None:
        marks.append((name, time.perf_counter()))

    mark("start")

    data = json.loads(Path(a.json).read_text(encoding="utf-8"))
    image = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {a.image}")
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw, manual, page_name = load_items(data, a.page, Path(a.image).name)
    stem = a.stem or Path(a.image).stem
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lama = None
    if a.lama:
        try:
            if a.lama.lower().endswith(".safetensors"):      # mayocream/lama-manga (needs torch + safetensors)
                try:
                    from .lama_manga import LamaManga
                except ImportError:            # run as a plain script
                    from lama_manga import LamaManga
                lama = LamaManga(a.lama)
            else:
                lama = Lama(a.lama)
            print(f"LaMa loaded ({getattr(lama, 'size', '?')}px)")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: cannot load LaMa ({e}); using cv2.inpaint", file=sys.stderr)

    a.lama_ok = lama is not None
    mark("load")
    items = build_items(raw, W, H, a.merge_thr, not a.no_merge, manual)
    print(f"Page {page_name or '-'}: {len(raw)} boxes -> {len(items)} text groups")

    # 1) classify + stroke masks (all from the ORIGINAL image)
    for b in items:
        analyze(gray, b, a)
        if b.kind == "sfx":
            b.cleanup = False
        decide_fill(image, gray, b, a)
    th = [b.thickness for b in items if b.thickness > 0 and b.kind != "sfx"]
    med_th = float(np.median(th)) if th else 0.0
    for b in items:
        if b.style:
            b.bold = b.style.lower() == "bold"
        elif not a.no_auto_bold and med_th > 0 and b.kind in ("bubble", "free"):
            b.bold = b.thickness > a.bold_ratio * med_th

    mark("analyze")
    # 2) erase
    out = image.copy()
    mask_full = np.zeros((H, W), np.uint8)
    tex_mask = np.zeros((H, W), np.uint8)
    cv_mask = np.zeros((H, W), np.uint8)
    lama_mask = np.zeros((H, W), np.uint8)
    keep_map = np.zeros((H, W), np.uint8)
    for b in items:
        for kx1, ky1, kx2, ky2 in b.keep:
            keep_map[max(0, ky1):ky2, max(0, kx1):kx2] = 1
        if b.kind == "sfx":
            x1, y1, x2, y2 = b.bbox
            keep_map[y1:y2, x1:x2] = 1
    tiles: list[dict] = []
    global _LAMA_QUEUE
    if not a.no_clean:
        _LAMA_QUEUE = np.zeros((H, W), np.uint8) if lama is not None else None
        for b in items:
            if b.cleanup and b.kind != "sfx":
                m = erase_item(out, b, a, lama)
                mask_full |= m
                if b.fill_mode == "texture":
                    tex_mask |= m
                elif b.engine == "opencv":
                    cv_mask |= m
                else:
                    lama_mask |= m
        flush_lama(out, a, lama)            # one LaMa pass for every queued block of the page
        if tex_mask.any():
            if a.tex_grow > 0:
                tex_mask = cv2.dilate(tex_mask, ellipse_kernel(a.tex_grow))
            thin = tex_mask & keep_map
            tex_mask[keep_map > 0] = 0
            mask_full |= tex_mask | thin
            forbid = cv2.dilate(mask_full, ellipse_kernel(3)) | keep_map
            t0 = time.time()
            nf = texture_fill(out, tex_mask, forbid, a, report=tiles)
            if thin.any():
                res = cv2.inpaint(out, thin * 255, 3, cv2.INPAINT_TELEA)
                out[thin > 0] = res[thin > 0]
            print(f"Texture fill: {int(tex_mask.sum())} px, {nf} tile(s) fell back to inpaint, "
                  f"{time.time() - t0:.1f}s")

    mark("erase+fill")
    residual: list[dict] = []
    if not a.no_clean and a.retry > 0:
        residual = residual_pass(out, items, a, lama, mask_full)
        for r in residual:
            if r["residual_px"] > a.residual_px:
                print(f"  residual g{r['id']}: {r['residual_px']} px left -> re-erased")

    mark("residual")
    # 3) text regions for free text (stage B uses them as the layout area)
    gutters = page_gutters(gray)
    for b in items:
        if b.kind != "sfx" and b.region is None and b.kind == "free":
            b.allowed = allowed_rect(b, items, gray, gutters, a)

    mark("layout")
    # outputs
    clean_name = f"{stem}_clean.png"
    if not cv2.imwrite(str(outdir / clean_name), out):
        raise RuntimeError("Failed to write clean image")
    cv2.imwrite(str(outdir / f"{stem}_mask.png"), mask_full)

    flagged = []
    for t in tiles:
        if t["reason"] != "ok":
            flagged.append(dict(t, flag="fallback"))
        elif t["score"] is not None and t["score"] > a.qa_score:
            flagged.append(dict(t, flag="weak_match"))
    qa = image.copy()
    tint = qa.copy()
    tint[mask_full > 0] = (255, 120, 0)        # texture fill / residual / other
    tint[cv_mask > 0] = (0, 200, 0)            # OpenCV
    tint[lama_mask > 0] = (0, 0, 255)          # LaMa
    qa = cv2.addWeighted(qa, 0.55, tint, 0.45, 0)
    for t in flagged:
        x1, y1, x2, y2 = t["rect"]
        col = (0, 140, 255) if t["flag"] == "fallback" else (255, 0, 255)
        cv2.rectangle(qa, (x1, y1), (x2, y2), col, 2)
    for r in residual:
        if r.get("retried"):
            x1, y1, x2, y2 = r["rect"]
            cv2.rectangle(qa, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.imwrite(str(outdir / f"{stem}_qa.png"), qa)

    layout = dict(schema=SCHEMA, page=page_name, image=Path(a.image).name, clean=clean_name,
                  size=[W, H], blocks=[export_block(b) for b in items],
                  qa_tiles=flagged)
    (outdir / f"{stem}_layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
    nweak = sum(1 for t in flagged if t["flag"] == "weak_match")
    print(f"QA: {len(flagged) - nweak} fallback tile(s), {nweak} weak-match tile(s) -> {stem}_qa.png")
    for b in items:
        print(f"  g{b.gid:<3}{str(b.ids):<8} {b.kind:<6} {b.polarity:<5} engine={b.engine:<11} fill={b.fill_mode:<8} "
              f"busy={b.busy:.2f} | {b.why}")
    cnt = {}
    for b in items:
        if b.kind != "sfx" and b.cleanup:
            cnt[b.engine] = cnt.get(b.engine, 0) + 1
    print("Engines: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    mark("write")
    parts = ", ".join(f"{n} {marks[i][1] - marks[i - 1][1]:.2f}s" for i, (n, _) in enumerate(marks) if i > 0)
    print(f"Time: {time.perf_counter() - t_all:.2f}s total ({parts})")
    print(f"Done: {outdir / clean_name}, {stem}_layout.json")


if __name__ == "__main__":
    main()
