"""Test phần logic thuần (không cần torch / manga-ocr / Gemini / Qt): chạy `python -m unittest discover -s tests -v` từ thư mục gốc project."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import koharu_detector as kd          # noqa: E402
from app.services import manga_ocr_pipeline as mp       # noqa: E402
from app.services import ocr_engine as oe               # noqa: E402

# Trang mẫu của người dùng (preview.jpg 1100x1600): hộp chữ đọc từ ảnh preview, khung tranh ước lượng bằng mắt từ ảnh đó.
TEXTS = {
    "d1": (778, 170, 830, 375), "d2": (282, 150, 420, 450), "d3": (980, 410, 1005, 508),
    "d4": (915, 685, 1005, 830), "d5": (138, 705, 168, 775), "d6": (806, 832, 843, 910),
    "d7": (548, 790, 625, 945), "d8": (200, 953, 255, 1028),
    "d9": (693, 1250, 845, 1448), "d10": (530, 1192, 607, 1275), "d11": (225, 1200, 250, 1327), "d12": (108, 1295, 145, 1385),
}
FRAMES = [(95, 118, 1030, 545), (583, 575, 1030, 1100), (95, 575, 572, 885), (95, 890, 572, 1100), (0, 1128, 1100, 1600)]


def page_layout(with_frames=True):
    dets = [kd.Det(kd.C_TEXT, 0.9, tuple(map(float, b))) for b in TEXTS.values()]
    if with_frames:
        dets += [kd.Det(kd.C_FRAME, 0.9, tuple(map(float, f))) for f in FRAMES]
    return kd.PageLayout((1100, 1600), dets)


def names(units):
    inv = {tuple(map(int, b)): n for n, b in TEXTS.items()}
    return [inv[(u.box[0], u.box[1], u.box[0] + u.box[2], u.box[1] + u.box[3])] for u in units]


class FrameOrder(unittest.TestCase):
    def test_panel_order_staggered_layout(self):
        # hàng 1 | cột phải cao (582..1030) trước | hai khung trái trên -> dưới | khung dưới cùng
        self.assertEqual(kd.order_frames(FRAMES), [0, 1, 2, 3, 4])

    def test_panel_order_shuffled_input(self):
        order = kd.order_frames([FRAMES[i] for i in (4, 2, 0, 3, 1)])
        self.assertEqual([(4, 2, 0, 3, 1)[i] for i in order], [0, 1, 2, 3, 4])

    def test_units_follow_panels(self):
        got = names(kd.detection_units(page_layout()))
        # mọi câu của khung trước đứng trước mọi câu của khung sau
        panel_of = {"d1": 0, "d2": 0, "d3": 0, "d4": 1, "d6": 1, "d5": 2, "d8": 3, "d9": 4, "d10": 4, "d11": 4, "d12": 4}
        ranks = [panel_of[n] for n in got if n in panel_of]
        self.assertEqual(ranks, sorted(ranks))
        self.assertLess(got.index("d1"), got.index("d2"))      # phải trước trái trong cùng hàng
        self.assertLess(got.index("d2"), got.index("d3"))      # d3 thấp hơn nên sau
        self.assertEqual(len(got), 12)

    def test_old_band_order_interleaves_panels(self):
        """Cách cũ (dải 100 px, phải -> trái) xen câu của khung phải (d6) vào giữa các câu khung trái: đây là lỗi cần sửa."""
        old = names(sorted(kd.detection_units(page_layout(with_frames=False)), key=lambda u: kd._reading_key(u.box)))
        self.assertGreater(old.index("d6"), old.index("d5"))   # d6 (khung phải) bị xếp SAU d5 (khung trái)
        new = names(kd.detection_units(page_layout()))
        self.assertLess(new.index("d6"), new.index("d5"))

    def test_no_frames_falls_back(self):
        units = kd.detection_units(page_layout(with_frames=False))
        self.assertTrue(all(u.frame == -1 for u in units))
        self.assertEqual(len(units), 12)


class CropForOcr(unittest.TestCase):
    def setUp(self):
        self.img = np.full((200, 300, 3), 255, np.uint8)
        self.img[60:140, 100:110] = 0          # cột chữ của đơn vị
        self.img[60:140, 230:240] = 0          # chữ bóng khác
        self.img[10:15, 100:200] = 0           # nét tranh / viền
        text = kd.Det(kd.C_TEXT, 0.9, (98.0, 58.0, 112.0, 142.0))
        # đa giác bóng THÔ: cố tình cắt vào mép phải cột chữ (x < 105)
        poly = np.array([[40, 30], [105, 30], [105, 170], [40, 170]], np.float32)
        self.unit = kd.Unit("balloon", [40, 30, 65, 140], (92, 52, 118, 148), kd.Det(kd.C_BALLOON, 0.9, (40., 30., 105., 170.), poly), [text])

    def test_default_never_clips_own_text(self):
        roi = kd.crop_for_ocr(self.img, self.unit, variant="default")
        self.assertEqual(int((roi[:, :, 0] == 0).sum()), 10 * 80)      # cột chữ còn nguyên dù đa giác bóng thô cắt qua nó

    def test_old_behaviour_would_clip(self):
        # chứng minh bản cũ (chỉ nở đa giác 5 px) làm mất nét: nở 5x5 từ x<=104 chỉ tới ~106, nhưng ở trường hợp đa giác hẹp hơn thì mất
        x1, y1, x2, y2 = self.unit.crop
        roi = self.img[y1:y2, x1:x2].copy()
        poly = np.array([[40, 30], [101, 30], [101, 170], [40, 170]], np.float32)
        inside = kd.poly_mask(poly, roi.shape[:2], x1, y1)
        import cv2
        inside = cv2.dilate(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        roi[~inside] = 255
        self.assertLess(int((roi[:, :, 0] == 0).sum()), 10 * 80)

    def test_text_only_removes_art(self):
        u = self.unit
        u.crop = (90, 5, 250, 150)
        roi = kd.crop_for_ocr(self.img, u, variant="text_only")
        self.assertEqual(int((roi[:, :, 0] == 0).sum()), 10 * 80)      # chỉ còn cột chữ: nét tranh y=10..15 và chữ x=230 bị tô trắng

    def test_others_whitened_in_all_variants(self):
        u = self.unit
        u.crop = (90, 52, 250, 148)
        u.others = [kd.Det(kd.C_TEXT, 0.9, (228.0, 58.0, 242.0, 142.0))]
        for v in ("default", "text_only", "wide"):
            roi = kd.crop_for_ocr(self.img, u, variant=v)
            x_off = u.crop[0] if v != "wide" else max(0, u.crop[0] - 12)
            self.assertTrue((roi[:, 230 - x_off:241 - x_off] == 255).all(), v)

    def test_wide_is_larger(self):
        a = kd.crop_for_ocr(self.img, self.unit, variant="default")
        b = kd.crop_for_ocr(self.img, self.unit, variant="wide")
        self.assertGreater(b.shape[0] * b.shape[1], a.shape[0] * a.shape[1])


class TextNormalize(unittest.TestCase):
    def test_ellipsis(self):
        self.assertEqual(oe.normalize_ocr_text("うう．．．ごめんね．．．"), "うう…ごめんね…")
        self.assertEqual(oe.normalize_ocr_text("ギター探しもあれからまったく進展ないし．．．"), "ギター探しもあれからまったく進展ないし…")
        self.assertEqual(oe.normalize_ocr_text("ええ......"), "ええ……")
        self.assertEqual(oe.normalize_ocr_text("ちょっと ！？"), "ちょっと！？")

    def test_plain_text_untouched(self):
        for t in ("師匠！！ベンケー殿！！見つけたっ", "もーっ！", "！！", "ちょつちょっと何をっ？"):
            self.assertEqual(oe.normalize_ocr_text(t), t)

    def test_suspicious(self):
        self.assertTrue(oe.looks_suspicious(""))
        self.assertTrue(oe.looks_suspicious("ああああああ"))
        self.assertTrue(oe.looks_suspicious("abc.xyz"))
        self.assertFalse(oe.looks_suspicious("！！"))
        self.assertFalse(oe.looks_suspicious("見つけた！！"))

    def test_quality(self):
        self.assertTrue(oe.is_low_quality("もーっ！", 0.4, 0.6))
        self.assertFalse(oe.is_low_quality("もーっ！", 0.9, 0.6))
        self.assertTrue(oe.is_low_quality("", None, 0.6))
        self.assertFalse(oe.is_low_quality("もーっ！", None, 0.6))
        self.assertLess(oe.quality_score("ああああああ", 0.99, 0.6), 0.5)


class OcrBest(unittest.TestCase):
    def setUp(self):
        self.orig = oe.read
        self.calls = 0

    def tearDown(self):
        oe.read = self.orig

    def _fake(self, seq):
        it = iter(seq)

        def read(img, cfg=None):
            self.calls += 1
            return next(it)
        oe.read = read

    def crop(self):
        return np.full((40, 40, 3), 255, np.uint8)

    def test_stops_at_first_confident(self):
        self._fake([oe.OcrResult("もーっ！", 0.95)])
        r = mp.ocr_best([self.crop, self.crop, self.crop], oe.OcrConfig())
        self.assertEqual((r["text"], r["low_conf"], self.calls), ("もーっ！", False, 1))

    def test_retry_picks_best(self):
        self._fake([oe.OcrResult("きゃうけーっ！！", 0.30), oe.OcrResult("ぎゃあーっ！！", 0.72), oe.OcrResult("x", 0.1)])
        r = mp.ocr_best([self.crop, self.crop, self.crop], oe.OcrConfig())
        self.assertEqual((r["text"], r["low_conf"], self.calls), ("ぎゃあーっ！！", False, 2))

    def test_all_low_flags(self):
        self._fake([oe.OcrResult("いい", 0.2), oe.OcrResult("ぃぃ", 0.3), oe.OcrResult("い", 0.25)])
        r = mp.ocr_best([self.crop, self.crop, self.crop], oe.OcrConfig())
        self.assertTrue(r["low_conf"])
        self.assertEqual(r["text"], "ぃぃ")

    def test_without_confidence(self):
        self._fake([oe.OcrResult("ああああああ", None), oe.OcrResult("もーっ！", None)])
        r = mp.ocr_best([self.crop, self.crop], oe.OcrConfig())
        self.assertEqual((r["text"], r["low_conf"]), ("もーっ！", False))


class GeminiPayload(unittest.TestCase):
    ITEMS = [
        {"id": 1, "raw_text": "もーっ！", "kind": "balloon", "panel": 0, "low_conf": False, "ocr_conf": 0.9},
        {"id": 2, "raw_text": "きゃうけーっ！！", "kind": "free", "panel": 0, "low_conf": True, "ocr_conf": 0.3, "crop_png": b"x"},
        {"id": 3, "raw_text": "いい", "kind": "free", "panel": -1, "low_conf": True, "ocr_conf": 0.1, "crop_png": b"y"},
        {"id": 4, "raw_text": "？", "low_conf": True, "ocr_conf": 0.2},
    ]

    def test_payload_fields(self):
        p = mp.build_cleanup_payload(self.ITEMS, {2})
        self.assertEqual(p[0], {"id": 1, "text": "もーっ！", "kind": "balloon", "panel": 1})
        self.assertEqual(p[1], {"id": 2, "text": "きゃうけーっ！！", "kind": "free", "panel": 1, "low_confidence": True, "crop_attached": True})
        self.assertNotIn("panel", p[2])
        self.assertNotIn("crop_attached", p[2])

    def test_rescue_priority_and_cap(self):
        self.assertEqual(mp.pick_rescue_ids(self.ITEMS, 8), [3, 2])      # conf thấp nhất trước; id 4 không có ảnh nên bị bỏ
        self.assertEqual(mp.pick_rescue_ids(self.ITEMS, 1), [3])
        self.assertEqual(mp.pick_rescue_ids(self.ITEMS, 0), [])

    def test_prompt_formats(self):
        txt = mp.GEMINI_CLEANUP_TRANSLATE_PROMPT_TEMPLATE.format(
            vision_section=mp.GEMINI_VISION_SECTION_TEMPLATE, source_lang="ja", target_lang="vi", context_section="",
            romaji_instruction="", honorifics_instruction="", payload_json="[]")
        self.assertIn("ẢNH ĐỐI CHIẾU", txt)
        self.assertIn('"glossary_updates"', txt)


class FusedBalloons(unittest.TestCase):
    """Hai bóng liền nhau (viền đen ngăn cách) nhưng model trả MỘT đa giác bóng / MỘT hộp chữ trùm cả hai."""

    def setUp(self):
        import cv2
        self.cv2 = cv2
        img = np.full((260, 420, 3), 255, np.uint8)
        # hai bóng ellip chạm nhau, mỗi bóng vẽ viền đen dày 3 px (viền chung ở x ~ 205)
        for cx in (120, 290):
            cv2.ellipse(img, (cx, 130), (88, 100), 0, 0, 360, (0, 0, 0), 3)
        # cột chữ dọc: bóng trái 3 cột, bóng phải 2 cột
        for x in (85, 110, 135):
            img[70:190, x:x + 8] = 0
        for x in (270, 295):
            img[80:180, x:x + 8] = 0
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        union = np.zeros(self.gray.shape, np.uint8)
        for cx in (120, 290):
            cv2.ellipse(union, (cx, 130), (88, 100), 0, 0, 360, 1, -1)
        cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))        # đa giác THÔ của model: bao lồi, không có cổ thắt
        self.poly = hull.reshape(-1, 2).astype(np.float32)
        self.balloon = kd.Det(kd.C_BALLOON, 0.9, (30.0, 30.0, 378.0, 230.0), self.poly)

    def layout(self, texts):
        return kd.PageLayout((420, 260), [self.balloon] + texts)

    def test_two_text_boxes_two_balloons(self):
        t1 = kd.Det(kd.C_TEXT, 0.9, (80.0, 66.0, 148.0, 194.0))
        t2 = kd.Det(kd.C_TEXT, 0.9, (264.0, 76.0, 310.0, 184.0))
        units = kd.detection_units(self.layout([t1, t2]), gray=self.gray)
        self.assertEqual(len(units), 2)
        self.assertEqual([len(u.texts) for u in units], [1, 1])
        self.assertLess(units[1].box[0] + units[1].box[2] / 2, units[0].box[0] + units[0].box[2] / 2)   # phải -> trái: bóng phải đọc trước

    def test_one_text_box_spanning_both_balloons_is_cut(self):
        t = kd.Det(kd.C_TEXT, 0.9, (80.0, 66.0, 310.0, 194.0))           # MỘT hộp chữ trùm cả hai bóng
        units = kd.detection_units(self.layout([t]), gray=self.gray)
        self.assertEqual(len(units), 2)
        xs = sorted((u.texts[0].box[0], u.texts[0].box[2]) for u in units)
        self.assertLess(xs[0][1], 205)                                  # nửa trái nằm trong bóng trái
        self.assertGreater(xs[1][0], 205)                               # nửa phải nằm trong bóng phải

    def test_without_gray_keeps_old_behaviour(self):
        t = kd.Det(kd.C_TEXT, 0.9, (80.0, 66.0, 310.0, 194.0))
        old = kd.detection_units(self.layout([t]))                       # không có ảnh xám -> không tách theo vùng trắng
        self.assertEqual(len(old), 1)

    def test_region_split_is_what_fixes_spanning_box(self):
        t = kd.Det(kd.C_TEXT, 0.9, (80.0, 66.0, 310.0, 194.0))
        kd.SPLIT_BY_IMAGE = False
        try:
            self.assertEqual(len(kd.detection_units(self.layout([t]), gray=self.gray)), 1)   # cách cũ: gom làm một
        finally:
            kd.SPLIT_BY_IMAGE = True
        self.assertEqual(len(kd.detection_units(self.layout([t]), gray=self.gray)), 2)

    def test_single_balloon_many_columns_not_split(self):
        import cv2
        img = np.full((260, 260, 3), 255, np.uint8)
        cv2.ellipse(img, (130, 130), (110, 110), 0, 0, 360, (0, 0, 0), 3)
        for x in range(70, 200, 14):
            img[60:200, x:x + 10] = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = np.zeros(gray.shape, np.uint8)
        cv2.ellipse(m, (130, 130), (110, 110), 0, 0, 360, 1, -1)
        cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        b = kd.Det(kd.C_BALLOON, 0.9, (20.0, 20.0, 240.0, 240.0), max(cn, key=cv2.contourArea).reshape(-1, 2).astype(np.float32))
        texts = [kd.Det(kd.C_TEXT, 0.9, (float(x - 1), 56.0, float(x + 11), 204.0)) for x in range(70, 200, 14)]
        units = kd.detection_units(kd.PageLayout((260, 260), [b] + texts), gray=gray)
        self.assertEqual(len(units), 1)
        self.assertEqual(len(units[0].texts), len(texts))

    def test_grey_balloon_is_left_alone(self):
        g = self.gray.copy()
        g[g > 200] = 120                                                # nền xám: không đủ trắng để phán đoán
        t = kd.Det(kd.C_TEXT, 0.9, (80.0, 66.0, 310.0, 194.0))
        self.assertIsNone(kd._split_by_regions(g, self.balloon, [t]))


class DetectorSettings(unittest.TestCase):
    def test_use_yolo_default_keeps_old_behaviour(self):
        self.assertTrue(kd.KoharuConfig().use_yolo)


if __name__ == "__main__":
    unittest.main()
