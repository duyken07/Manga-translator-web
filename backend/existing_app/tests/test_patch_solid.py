"""Lớp vá không được thủng: mọi điểm ảnh trong vùng chữ phải có alpha 255 (tránh lộ viền chữ gốc trên viewer)."""
from types import SimpleNamespace as NS

import numpy as np

from app.services.redraw import engine


def test_patch_has_no_holes():
    H, W = 120, 160
    image = np.full((H, W, 3), 255, np.uint8)
    image[30:50, 40:80] = 120                      # chữ gốc
    clean = image.copy()
    clean[30:50, 40:80] = 255                      # đã xoá
    keep = np.random.RandomState(0).rand(20, 40) < 0.1
    clean[30:50, 40:80][keep] = 120                # điểm ảnh "không đổi" -> lỗ trong mặt nạ diff
    b = NS(strokes=np.ones((20, 40), bool), ox=40, oy=30, bbox=(40, 30, 80, 50), ids=[1])
    patches = engine._build_patches(image, clean, [b], None)
    assert len(patches) == 1
    p = patches[0]
    x0, y0 = 40 - p["x"], 30 - p["y"]
    region = p["bgra"][y0:y0 + 20, x0:x0 + 40, 3]
    assert (region == 255).all(), "còn lỗ alpha trong vùng chữ"
    # ghép lại phải trùng clean
    out = engine.compose_layers(image, patches, [])
    assert (out == clean).all()


if __name__ == "__main__":
    test_patch_has_no_holes()
    print("OK")
