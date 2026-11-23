"""Đũa thần (Magic Wand) chọn bóng thoại: bấm vào trong bóng -> ra vùng bóng thoại (giống Magic Wand của Photoshop).

Giai đoạn 1: bóng thoại NÉT LIỀN (viền đen khép kín). Cách làm:
  1. Điểm bấm trúng nét chữ / viền thì dời tới điểm sáng gần nhất (bấm đâu trong bóng cũng được).
  2. Các điểm tối (viền, chữ) được coi là "tường", giãn nhẹ 1 px và lan theo 4 hướng nên khe hở li ti không làm vùng chọn tràn ra ngoài.
  3. Flood fill (cv2.floodFill, dải cố định quanh màu điểm bấm) theo Dung sai (Tolerance).
  4. Đóng hình + lấp lỗ (chữ nằm trong bóng, kể cả chữ dính viền) thành một vùng đặc.
  5. Phát hiện tràn (vùng quá lớn: viền hở nên lan ra cả trang) -> từ chối và báo lý do thay vì chọn nhầm.
Không phụ thuộc giao diện: nhận ảnh xám numpy, trả mặt nạ bool.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

MIN_AREA = 150            # nhỏ hơn: chắc chắn không phải bóng thoại
MAX_AREA_FRAC = 0.12      # lớn hơn tỉ lệ này của TRANG đầy đủ: coi là tràn ra ngoài (viền hở); bóng lớn nhất trên các trang thử ~7%
FULL_PAGE_PX = 600_000    # ảnh nhỏ hơn (ảnh cắt, panel đơn lẻ): bóng chiếm tỉ lệ lớn là bình thường nên nới giới hạn
MAX_AREA_FRAC_CROP = 0.40
DARK_THR = 140            # dưới ngưỡng này là nét tối (viền / chữ) = tường chặn flood fill
SEED_LIGHT = 200          # điểm bấm phải sáng hơn mức này (nếu không, dời tới điểm sáng gần nhất)
SEED_RADIUS = 14
MIN_CLOSED = 0.94         # tỉ lệ biên vùng chọn có nét viền tối bao sát ngoài: bóng nét liền ~1.00, nét đứt / viền hở < 0.92


def _refine_seed(gray: np.ndarray, x: int, y: int) -> Optional[Tuple[int, int]]:
    """Điểm bấm trúng nét tối (chữ, viền) -> điểm sáng gần nhất trong bán kính SEED_RADIUS; None nếu không có."""
    h, w = gray.shape
    if int(np.median(gray[max(0, y - 2):y + 3, max(0, x - 2):x + 3])) >= SEED_LIGHT:
        return x, y
    x0, x1, y0, y1 = max(0, x - SEED_RADIUS), min(w, x + SEED_RADIUS + 1), max(0, y - SEED_RADIUS), min(h, y + SEED_RADIUS + 1)
    win = gray[y0:y1, x0:x1]
    ys, xs = np.nonzero(win >= SEED_LIGHT)
    if len(xs) == 0:
        return None
    d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    i = int(np.argmin(d2))
    if d2[i] > SEED_RADIUS ** 2:
        return None
    return int(xs[i] + x0), int(ys[i] + y0)


def closed_ratio(gray: np.ndarray, mask: np.ndarray, thr: int = DARK_THR, band: int = 3) -> float:
    """Tỉ lệ điểm biên của vùng chọn có nét tối nằm sát ngay bên ngoài (tức là có viền thật sự bao quanh). 1.0 = viền liền khép kín."""
    m8 = mask.astype(np.uint8)
    boundary = (m8 > 0) & (cv2.erode(m8, np.ones((3, 3), np.uint8)) == 0)
    k = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    dark_ring = (cv2.dilate(m8, k) > 0) & (m8 == 0) & (gray < thr)
    near_dark = cv2.dilate(dark_ring.astype(np.uint8), k) > 0
    return float((boundary & near_dark).sum()) / max(1, int(boundary.sum()))


# Các mức "nhạy" thử lần lượt, mức đầu tiên cho kết quả đạt chuẩn được dùng:
#   thr  : ngưỡng xám coi là nét tối (viền mảnh / nhạt cần ngưỡng cao hơn)
#   big  : cạnh tối thiểu (tỉ lệ so với cạnh dài của trang) để một nét tối là TƯỜNG; nhỏ hơn là chữ.
#          Viền bị đứt thành nhiều đoạn ngắn cần `big` nhỏ để không bị nhầm là chữ
#   dil  : độ giãn tường (bịt khe hở của viền)   open: bán kính "mở" cắt các lối rò hẹp (tỉ lệ so với trang 1400 px)
_PROFILES = (
    dict(thr=140, big=0.05, dil=3, open=3.0),
    dict(thr=140, big=0.02, dil=3, open=3.0),
    dict(thr=175, big=0.02, dil=5, open=4.0),
    dict(thr=205, big=0.015, dil=5, open=5.0),
)


def _flood(gray: np.ndarray, seed: Tuple[int, int], base: int, tol: int, thr: int, big: float, dil: int) -> np.ndarray:
    """Một lượt flood fill từ điểm gieo với bộ tham số `profile`; trả mặt nạ uint8 (chưa làm sạch)."""
    h, w = gray.shape
    sx, sy = seed
    dark = (gray < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    big_side = max(14, int(big * max(h, w)))
    side = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    small_ids = np.nonzero(side <= big_side)[0]
    small_ids = small_ids[small_ids > 0]                              # nhãn 0 = nền không tối
    small = np.isin(lab, small_ids)
    wall = cv2.dilate(((dark > 0) & ~small).astype(np.uint8), np.ones((dil, dil), np.uint8))
    text_zone = cv2.dilate(small.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0           # gồm cả viền mờ (khử răng cưa) quanh chữ
    work = gray.copy()
    work[text_zone & (wall == 0)] = 255
    work[wall > 0] = 0
    if work[sy, sx] == 0:                     # điểm gieo sát nét tối bị tường nuốt: lùi về vùng sáng thật
        work[sy, sx] = base
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(work, ff_mask, (sx, sy), 255, (tol,) * 3, (tol,) * 3, flags)
    return (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8)


def _cut_channels(mask: np.ndarray, seed: Tuple[int, int], r: int) -> np.ndarray:
    """Phép "mở" giữ thành phần chứa điểm gieo: cắt các lối rò hẹp (khe hở viền) nối bóng với vùng trắng bên ngoài."""
    if r < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(mask, k)
    n, lab = cv2.connectedComponents(core, connectivity=4)
    if n <= 1:
        return mask
    sx, sy = seed
    label = int(lab[sy, sx])
    if label == 0:                                         # điểm gieo nằm sát mép bị erode ăn mất: lấy mảnh lõi lớn nhất
        areas = np.bincount(lab.ravel())[1:]
        label = int(np.argmax(areas)) + 1
    return (cv2.dilate((lab == label).astype(np.uint8), k) & mask).astype(np.uint8)


def _fill_shape(mask: np.ndarray) -> np.ndarray:
    """Đóng hình + lấp lỗ: chữ trong bóng (cả chữ dính viền) thành một khối đặc."""
    ys, xs = np.nonzero(mask)
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    k = int(max(3, min(15, round(0.04 * min(bw, bh)))) | 1)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    return filled


def _sides_touched(mask: np.ndarray) -> int:
    """Số cạnh của ảnh mà vùng chọn chạm vào (bóng thật chạm tối đa 2 cạnh; chạm 3-4 cạnh = đã tràn ra cả trang)."""
    return int(mask[0, :].any()) + int(mask[-1, :].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def select_bubble(gray: np.ndarray, seed: Tuple[int, int], tolerance: int = 24,
                  max_area_frac: Optional[float] = None, min_closed: float = MIN_CLOSED) -> Tuple[Optional[np.ndarray], str]:
    """Trả về (mặt nạ bool HxW, "") hoặc (None, lý do) nếu không chọn được. min_closed = 0 để bỏ kiểm tra viền liền.
    Thử lần lượt các mức nhạy trong _PROFILES; mức đầu tiên cho vùng hợp lệ được dùng."""
    h, w = gray.shape
    if max_area_frac is None:
        max_area_frac = MAX_AREA_FRAC if h * w >= FULL_PAGE_PX else MAX_AREA_FRAC_CROP
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < w and 0 <= y < h):
        return None, "Điểm bấm nằm ngoài ảnh."
    pt = _refine_seed(gray, x, y)
    if pt is None:
        return None, "Không có vùng sáng gần điểm bấm: hãy bấm vào phần trắng bên trong bóng thoại."
    sx, sy = pt
    base = int(np.median(gray[max(0, sy - 2):sy + 3, max(0, sx - 2):sx + 3]))
    tol = int(max(1, min(120, tolerance)))
    scale = max(h, w) / 1400.0
    first_msg = ""
    for prof in _PROFILES:
        mask = _flood(gray, (sx, sy), base, tol, prof["thr"], prof["big"], prof["dil"])
        if int(mask.sum()) < MIN_AREA:
            msg = "Vùng chọn quá nhỏ: tăng Dung sai hoặc bấm vào phần trắng của bóng thoại."
        else:
            mask = _cut_channels(mask, (sx, sy), int(round(prof["open"] * scale)))
            if int(mask.sum()) < MIN_AREA:
                msg = "Vùng chọn quá nhỏ: tăng Dung sai hoặc bấm vào phần trắng của bóng thoại."
            else:
                filled = _fill_shape(mask)
                area = int(filled.sum())
                if area > max_area_frac * h * w or _sides_touched(filled) >= 3:
                    msg = "Vùng chọn tràn ra ngoài bóng thoại (viền bóng bị hở?). Giảm Dung sai hoặc thử bóng có viền liền."
                elif min_closed > 0 and closed_ratio(gray, filled, thr=min(prof["thr"], 200)) < min_closed:
                    msg = "Viền bóng không liền (nét đứt hoặc bị hở): giai đoạn này chỉ hỗ trợ bóng thoại có viền liền."
                else:
                    return filled > 0, ""
        first_msg = first_msg or msg
    return None, first_msg


def combine(prev: Optional[np.ndarray], new: np.ndarray, mode: str) -> Optional[np.ndarray]:
    """Gộp vùng mới vào vùng đang chọn: replace (thay), add (cộng, Shift), subtract (bớt, Alt; có thể ra rỗng)."""
    if mode == "subtract":
        return None if prev is None else prev & ~new
    if mode == "add" and prev is not None:
        return prev | new
    return new


def bbox(mask: np.ndarray) -> Optional[List[int]]:
    """[x1, y1, x2, y2] (x2, y2 không gồm) của vùng chọn; None nếu rỗng."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def polygons(mask: np.ndarray, epsilon: float = 0.8) -> List[np.ndarray]:
    """Đường viền vùng chọn dạng đa giác (cho nét đứt chuyển động); bỏ các mảnh vụn."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 20:
            continue
        out.append(cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2))
    return out


def apply_ops(gray: np.ndarray, ops: Sequence[Tuple[str, Tuple[int, int]]], tolerance: int) -> Tuple[Optional[np.ndarray], str]:
    """Tính lại vùng chọn từ danh sách thao tác [(mode, seed)...] theo Dung sai hiện tại (kéo thanh trượt là tính lại tức thì).
    Thao tác lỗi bị bỏ qua nhưng lý do của thao tác CUỐI được trả về để báo người dùng."""
    mask: Optional[np.ndarray] = None
    last_msg = ""
    for mode, seed in ops:
        m, msg = select_bubble(gray, seed, tolerance)
        last_msg = msg
        if m is None:
            continue
        mask = combine(mask, m, mode)
    return mask, last_msg
