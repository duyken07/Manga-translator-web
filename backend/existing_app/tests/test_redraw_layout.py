"""Test xoá chữ gốc + chia vùng + khung chữ dịch (không cần torch / Qt): python -m unittest discover -s tests -v"""
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.redraw import engine as eng          # noqa: E402
from app.services.redraw import stage_a as sa          # noqa: E402
from app.services.redraw import typeset as ts          # noqa: E402
from app.services.redraw.stage_a import Item, allowed_rect, analyze   # noqa: E402


def args(H=1600, **kw):
    a = eng.make_args(eng.RedrawOptions(), H, False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def put(img, text, org, scale=1.0, color=255, th=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


class GrowToInk(unittest.TestCase):
    """Trang đen chữ trắng (như trang cảnh báo): khung dò bắt đầu GIỮA dòng, bỏ sót phần đầu dòng."""

    def setUp(self):
        self.g = np.zeros((300, 1000), np.uint8)
        put(self.g, "# NOI DUNG CO CHUA SPOILER NHE", (40, 100), 1.0)
        put(self.g, "   KHUYEN KHICH XEM SAU KHI XONG", (40, 150), 1.0)
        ys, xs = np.nonzero(self.g > 140)
        self.full = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        self.cut = (self.full[0] + 160, self.full[1] - 4, self.full[2] - 30, self.full[3] + 4)   # khung cắt 160px đầu dòng + 30px cuối

    def covered(self, a, **item_kw):
        b = Item(gid=1, ids=[1], bbox=self.cut, text="x", polarity="white", **item_kw)
        analyze(self.g, b, a)
        ink = self.g > 140
        gone = (ink & b.strokes_full()).sum() if hasattr(b, "strokes_full") else None
        full = np.zeros_like(ink)
        full[b.oy:b.oy + b.strokes.shape[0], b.ox:b.ox + b.strokes.shape[1]] = b.strokes
        return float((ink & full).sum()) / float(ink.sum()), b

    def test_line_fully_erased_after_grow(self):
        frac, b = self.covered(args())
        self.assertGreater(frac, 0.97, "phải xoá trọn dòng chữ dù khung dò bị cắt")
        self.assertLessEqual(b.bbox[0], self.full[0] + 4)

    def test_old_behaviour_leaves_remnants(self):
        frac, _ = self.covered(args(grow_ink=0))
        self.assertLess(frac, 0.85, "(đối chứng) tắt grow thì còn sót chữ")

    def test_does_not_eat_other_items(self):
        other = (self.full[0] - 5, self.full[1] - 5, self.full[0] + 150, self.full[3] + 5)   # phần đầu dòng thuộc khung khác
        b = Item(gid=1, ids=[1], bbox=self.cut, text="x", polarity="white")
        analyze(self.g, b, args(), others=[other])
        self.assertGreaterEqual(b.bbox[0], other[2] - 60, "không được nuốt chữ của khung khác")

    def test_does_not_cross_to_other_paragraph(self):
        # Đoạn thứ hai nằm dưới (khác dải): khung của đoạn 1 không được nới xuống đó
        b = Item(gid=1, ids=[1], bbox=(self.full[0], self.full[1] - 4, self.full[2], self.full[1] + 40), text="x", polarity="white")
        analyze(self.g, b, args())
        self.assertLessEqual(b.bbox[3], self.full[1] + 60)


class KoharuGateSafetyNet(unittest.TestCase):
    """Chữ đen trên nền trắng, các chữ cách xa nhau (tag giãn chữ): vùng chữ koharu chỉ phủ phần giữa."""

    def test_zone_covering_only_middle_is_ignored(self):
        g = np.full((200, 900), 255, np.uint8)
        xs = list(range(60, 840, 70))
        for i, x in enumerate(xs):
            put(g, "ABCDEFGHIJKL"[i], (x, 110), 1.8, 0, 4)
        box = (40, 40, 880, 160)

        class FakeKz:
            usable = True

        zone = types.SimpleNamespace(text=None, interior=None, sfx=None)

        def fake_raster(kz, shape, ox, oy, zp, bp):
            m = np.zeros(shape, bool)
            m[:, 300 - ox:600 - ox] = True            # vùng chữ chỉ phủ khoảng giữa
            zone.text = m
            return zone

        from app.services import koharu_detector as kd
        orig = kd.rasterize
        kd.rasterize = fake_raster
        try:
            def covered(gate_net):
                b = Item(gid=1, ids=[1], bbox=box, text="x", kz=FakeKz())
                a = args(koharu=1, k_gate=1)
                if not gate_net:
                    a.k_ink_ratio = 1e9                # đối chứng: tắt lưới an toàn
                analyze(g, b, a)
                full = np.zeros(g.shape, bool)
                full[b.oy:b.oy + b.strokes.shape[0], b.ox:b.ox + b.strokes.shape[1]] = b.strokes
                ink = g < 100
                return float((ink & full).sum()) / float(ink.sum())
            self.assertGreater(covered(True), 0.95, "vùng koharu phủ thiếu -> vẫn phải xoá hết chữ trong khung")
            self.assertLess(covered(False), 0.7, "(đối chứng) không có lưới an toàn thì sót chữ ngoài vùng")
        finally:
            kd.rasterize = orig


class AllowedRectFlat(unittest.TestCase):
    def test_does_not_expand_over_artwork(self):
        g = np.full((600, 900), 255, np.uint8)
        rng = np.random.default_rng(1)                             # nhân vật bên phải: nét tranh mật độ vừa (~35% điểm tối), KHÔNG đen kín
        g[100:500, 470:700] = np.where(rng.random((400, 230)) < 0.35, 30, 255).astype(np.uint8)
        occ = (300, 250, 450, 330)                                 # khung chữ gốc rộng 150 -> được nới tối đa ~50px sang phải
        b = Item(gid=1, ids=[1], bbox=occ, text="x", kind="free", occ=occ)
        res = allowed_rect(b, [b], g, ts.page_gutters(g), args())
        self.assertLessEqual(res[2], 474, "không được nới vào vùng tranh")
        res0 = allowed_rect(b, [b], g, ts.page_gutters(g), args(flat_frac=0.0))
        self.assertGreater(res0[2], 480, "(đối chứng) tắt kiểm tra nền thì nới đè lên tranh")

    def test_background_noise_ok(self):
        g = np.full((600, 900), 255, np.uint8)
        g[::9, ::9] = 120                                          # vài chấm screentone thưa (< 15%) không chặn
        occ = (380, 250, 420, 330)
        b = Item(gid=1, ids=[1], bbox=occ, text="x", kind="free", occ=occ)
        res = allowed_rect(b, [b], g, ts.page_gutters(g), args())
        self.assertGreater(res[2] - res[0], 40)


def soft_text_page(blur, tone=False):
    """Giấy trắng + 3 cột chữ mảnh, làm mờ như bản scan mềm (lõi nét chỉ xám, không đen hẳn)."""
    g = np.full((520, 420), 255, np.uint8)
    for i, word in enumerate(["ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZ"]):
        for k, ch in enumerate(word):
            cv2.putText(g, ch, (130 + i * 34, 120 + k * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    ink = g < 200
    g = cv2.GaussianBlur(g, (0, 0), blur)
    if tone:
        t = np.full(g.shape, 255, np.uint8)
        t[::5, ::5] = 110
        t[2::5, 2::5] = 110
        g = np.minimum(g, cv2.GaussianBlur(t, (0, 0), 0.8))
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), ink


def erase_free(bgr, bbox):
    H = bgr.shape[0]
    a = args(H)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = Item(gid=1, ids=[1], bbox=bbox, text="x")
    analyze(gray, b, a, others=[])
    sa.decide_fill(bgr, gray, b, a)
    clean = bgr.copy()
    eng._erase(clean, bgr, [b], [b], a, None)
    return clean


class SoftScanErase(unittest.TestCase):
    BOX = (110, 90, 250, 330)

    def test_thin_soft_text_fully_erased(self):
        """Nét mảnh + scan mềm: ngưỡng cố định <100 không bắt được lõi nét -> còn nguyên / sót chữ gốc."""
        bgr, ink = soft_text_page(1.4)
        clean = erase_free(bgr, self.BOX)
        g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(int)
        region = np.zeros(g.shape, bool)
        region[self.BOX[1]:self.BOX[3], self.BOX[0]:self.BOX[2]] = True
        left = int(((255 - g) > 25)[region].sum())
        self.assertEqual(left, 0, f"còn {left} điểm sẫm sau khi xoá")

    def test_screentone_far_from_text_untouched(self):
        """Nền screentone: ngưỡng thích nghi / quét viền không được xoá hạt tone ở xa chữ."""
        bgr, ink = soft_text_page(0.8, tone=True)
        clean = erase_free(bgr, self.BOX)
        far = ~(cv2.dilate(ink.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0)
        changed = int(((cv2.absdiff(clean, bgr).max(axis=2) > 8) & far).sum())
        self.assertEqual(changed, 0, f"{changed} điểm tone xa chữ bị đổi")



class GrowSafety(unittest.TestCase):
    def test_no_runaway_growth_on_artwork(self):
        """Chữ SFX cách điệu nằm trên nền tranh (nét hatching): khung không được lan hàng trăm px theo các nét tranh."""
        rng = np.random.default_rng(3)
        g = np.full((1400, 760), 255, np.uint8)
        for _ in range(260):                                   # hatching dày quanh dải chữ
            x, y = int(rng.integers(430, 700)), int(rng.integers(250, 1300))
            cv2.line(g, (x, y), (x + int(rng.integers(-18, 18)), y + int(rng.integers(14, 40))), 0, int(rng.integers(2, 5)))
        box = (474, 766, 644, 988)
        g[box[1]:box[3], box[0]:box[2]] = 255
        cv2.putText(g, "HO", (490, 900), cv2.FONT_HERSHEY_SIMPLEX, 3.2, 0, 14, cv2.LINE_AA)
        b = Item(gid=1, ids=[1], bbox=box, text="x", type="free")
        analyze(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)[:, :, 0], b, args())
        grew = (box[1] - b.bbox[1]) + (b.bbox[3] - box[3])
        self.assertLessEqual(grew, 20, f"khung lan thêm {grew}px trên nền tranh")

    def test_cap_on_total_growth(self):
        """Một dòng chữ rất dài, khung chỉ phủ đoạn giữa: tổng mức nới mỗi phía <= 50% cạnh dài của khung gốc."""
        g = np.zeros((200, 1400), np.uint8)
        put(g, "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z", (40, 100), 1.0)
        box = (600, 70, 680, 120)
        b = Item(gid=1, ids=[1], bbox=box, text="x", polarity="white")
        analyze(g, b, args())
        self.assertLessEqual(box[0] - b.bbox[0], 0.5 * 80 + 4)
        self.assertLessEqual(b.bbox[2] - box[2], 0.5 * 80 + 4)

    def test_erase_with_growth_does_not_crash_residual_pass(self):
        """Khung bị nới nhiều lần: residual_pass dò lại không được lệch kích thước mảng (ValueError: operands could not be broadcast)."""
        g = np.zeros((220, 1400), np.uint8)
        put(g, "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z", (40, 100), 1.0)
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        a = args(220)
        b = Item(gid=1, ids=[1], bbox=(600, 70, 660, 120), text="x", polarity="white")
        analyze(g, b, a, others=[])
        sa.decide_fill(bgr, g, b, a)
        b.cleanup = True
        clean = bgr.copy()
        mask = np.zeros(g.shape, np.uint8)
        sa.residual_pass(clean, [b], a, None, mask)            # trước đây: ValueError ở residual_pass (shapes (78,707) (78,342))


class CacheInvalidation(unittest.TestCase):
    def test_old_versions_are_rejected(self):
        """Cache do bản cũ ghi (STATE_VERSION / LAYOUT_VERSION cũ) phải bị bỏ để xoá lại / xếp lại bằng thuật toán mới."""
        import json
        import tempfile
        cdir = Path(tempfile.mkdtemp())
        img = cdir / "p.png"
        cv2.imwrite(str(img), np.full((40, 60, 3), 255, np.uint8))
        st = img.stat()
        (cdir / "state.json").write_text(json.dumps(dict(version=eng.STATE_VERSION - 1, size=[60, 40], image=[st.st_size, st.st_mtime_ns])), encoding="utf-8")
        self.assertIsNone(eng._load_state(cdir, str(img), 60, 40))
        (cdir / "layouts.json").write_text(json.dumps(dict(version=eng.LAYOUT_VERSION - 1)), encoding="utf-8")
        self.assertIsNone(eng._load_layouts(cdir))
        (cdir / "layouts.json").write_text(json.dumps(dict(version=eng.LAYOUT_VERSION)), encoding="utf-8")
        self.assertIsNotNone(eng._load_layouts(cdir))


if __name__ == "__main__":
    unittest.main()
