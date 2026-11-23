"""Test cỡ chữ / khung chữ của typeset (không cần torch / Qt): python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.redraw import engine as eng          # noqa: E402
from app.services.redraw import typeset as ts          # noqa: E402
from app.services.redraw.stage_a import Item, allowed_rect   # noqa: E402

REG, BOLD = ts.resolve_fonts("", "")


def bubble(gid, cx, cy, rx, ry, text):
    interior = np.zeros((2 * ry, 2 * rx), np.uint8)
    cv2.ellipse(interior, (rx, ry), (rx - 4, ry - 4), 0, 0, 360, 1, -1)
    return Item(gid=gid, ids=[gid], bbox=(cx - rx, cy - ry, cx + rx, cy + ry), text=text, kind="bubble",
                ox=cx - rx, oy=cy - ry, interior=interior.astype(bool), occ=(cx - 25, cy - 60, cx + 25, cy + 60))


def run(items, H=2400, W=1600, tweak=None):
    a = eng.make_args(eng.RedrawOptions(), H, False)
    if tweak:
        tweak(a)
    out = np.full((H, W, 3), 255, np.uint8)
    return ts.typeset_page(out, items, cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), a, REG, BOLD, draw=False), a


@unittest.skipIf(REG is None, "không có font hỗ trợ tiếng Việt")
class BubbleSizing(unittest.TestCase):
    def page(self):
        return [bubble(1, 420, 500, 150, 230, "Sau này chúng ta sống chung nhé?"),     # bóng to
                bubble(2, 900, 1000, 55, 120, "Hanna-san"),                            # các bóng hẹp kéo trung vị cả trang xuống
                bubble(3, 700, 1500, 50, 90, "Hả?"),
                bubble(4, 1200, 1900, 55, 100, "Không thể tha thứ được đâu")]

    def test_big_bubble_not_capped_by_page_median(self):
        res, _ = run(self.page())
        big = res["blocks"][0]
        r = res["blocks"][0]["rects"]
        height = max(x[3] for x in r) - min(x[1] for x in r)
        self.assertGreaterEqual(height / 460, 0.5, "chữ trong bóng to phải lấp ít nhất ~50% chiều cao")
        self.assertGreater(big["size"], res["caps"]["shaped"], "bóng to được phép vượt trần theo trung vị")

    def test_geo_cap_disabled_reproduces_old_behaviour(self):
        res_old, _ = run(self.page(), tweak=lambda a: setattr(a, "geo_k", 0.0))
        res_new, _ = run(self.page())
        self.assertGreater(res_new["blocks"][0]["size"], res_old["blocks"][0]["size"])

    def test_small_bubbles_unchanged_by_geo_cap(self):
        res_old, _ = run(self.page(), tweak=lambda a: setattr(a, "geo_k", 0.0))
        res_new, _ = run(self.page())
        for i in (1, 2):   # bóng hẹp: cỡ chữ do bề rộng quyết định, không được đổi
            self.assertEqual(res_old["blocks"][i]["size"], res_new["blocks"][i]["size"])

    def test_hyphenated_name_breaks_only_when_needed(self):
        for rx, must_fit in ((45, True), (55, True)):
            res, _ = run([bubble(1, 800, 800, rx, 120, "Hanna-san")])
            b = res["blocks"][0]
            self.assertTrue(b["fits"] == must_fit and b["lines"] == 2, (rx, b["fits"], b["lines"]))
        res, _ = run([bubble(1, 800, 800, 65, 120, "Hanna-san")])
        self.assertEqual(res["blocks"][0]["lines"], 1, "vừa 1 dòng thì không được ngắt")

    def test_paragraph_helpers(self):
        self.assertEqual(ts._split_hyphen("Hanna-san"), ["Hanna-", ts.GLUE + "san"])
        self.assertEqual(ts._split_hyphen("-x"), ["-x"])
        self.assertEqual(ts.join_line(ts.paragraphs("Hanna-san ơi", True)[0]), "Hanna-san ơi")
        self.assertEqual(ts.paragraphs("a  b\n\nc"), [["a", "b"], ["c"]])   # hành vi cũ giữ nguyên


class AllowedRect(unittest.TestCase):
    def test_white_text_on_black_can_expand(self):
        gray = np.full((800, 800), 200, np.uint8)
        gray[100:700, 300:700] = 25                                  # panel đen
        a = eng.make_args(eng.RedrawOptions(), 800, False)
        occ = (480, 300, 520, 420)
        mk = lambda pol: Item(gid=1, ids=[1], bbox=occ, text="x", kind="free", polarity=pol, occ=occ)
        bw, bd = mk("white"), mk(None)
        white = allowed_rect(bw, [bw], gray, ts.page_gutters(gray), a)
        dark = allowed_rect(bd, [bd], gray, ts.page_gutters(gray), a)
        self.assertGreater(white[2] - white[0], dark[2] - dark[0], "chữ trắng trên nền đen phải nới được khung")
        self.assertGreaterEqual(white[0], 300 - 2)                   # nhưng không tràn khỏi panel đen
        self.assertLessEqual(white[2], 700 + 2)


if __name__ == "__main__":
    unittest.main()
