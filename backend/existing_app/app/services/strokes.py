"""Nét vẽ tay trên trang (bút, bút dạ, dạ quang): cấu trúc dữ liệu và vẽ lên ảnh khi xuất. Không phụ thuộc Qt.

Một nét là dict: {"id": int, "tool": "pen"|"marker"|"highlighter", "color": "r,g,b", "width": px, "opacity": 0..1,
"pts": [[x, y], ...]} với toạ độ là pixel của ảnh gốc.
"""
from __future__ import annotations

import cv2
import numpy as np

TOOL_PRESETS = {  # tool: (độ dày px, độ đục)
    "pen": (5.0, 1.0),
    "marker": (14.0, 1.0),
    "highlighter": (22.0, 0.4),
}


def make_stroke(tool: str, color: str, width: float, pts, opacity: float | None = None) -> dict:
    """Tạo nét mới (chưa có id): làm tròn toạ độ về số nguyên và bỏ các điểm trùng liên tiếp."""
    clean: list = []
    for x, y in pts:
        p = [int(round(x)), int(round(y))]
        if not clean or p != clean[-1]:
            clean.append(p)
    op = TOOL_PRESETS.get(tool, (0, 1.0))[1] if opacity is None else float(opacity)
    return dict(tool=tool, color=color, width=round(float(width), 1), opacity=round(op, 2), pts=clean)


def parse_rgb(color: str) -> tuple:
    try:
        r, g, b = (max(0, min(255, int(v))) for v in color.split(","))
        return r, g, b
    except (ValueError, AttributeError):
        return 20, 20, 20


def render_strokes(img: np.ndarray, strokes: list) -> np.ndarray:
    """Vẽ mọi nét lên bản sao của ảnh BGR (khử răng cưa, đầu nét tròn; nét bán trong suốt đục đều trong 1 nét)."""
    out = img.copy()
    H, W = out.shape[:2]
    for s in strokes or []:
        pts = np.asarray(s.get("pts") or [], np.int32)
        if pts.size == 0:
            continue
        th = max(1, int(round(float(s.get("width", 5)))))
        op = min(max(float(s.get("opacity", 1.0)), 0.0), 1.0)
        x1, y1 = max(0, int(pts[:, 0].min()) - th - 2), max(0, int(pts[:, 1].min()) - th - 2)
        x2, y2 = min(W, int(pts[:, 0].max()) + th + 3), min(H, int(pts[:, 1].max()) + th + 3)
        if x2 <= x1 or y2 <= y1:
            continue
        local = (pts - np.array([x1, y1], np.int32)).reshape(-1, 1, 2)
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        if len(local) > 1:
            cv2.polylines(mask, [local], False, 255, th, cv2.LINE_AA)
        for p in local.reshape(-1, 2)[:: max(1, len(local) // 400)]:      # đầu nét / khớp nối tròn
            cv2.circle(mask, (int(p[0]), int(p[1])), max(1, th // 2), 255, -1, cv2.LINE_AA)
        r, g, b = parse_rgb(s.get("color", "20,20,20"))
        a = (mask.astype(np.float32) / 255.0 * op)[..., None]
        roi = out[y1:y2, x1:x2].astype(np.float32)
        out[y1:y2, x1:x2] = np.clip(roi * (1 - a) + np.array([b, g, r], np.float32) * a, 0, 255).astype(np.uint8)
    return out
