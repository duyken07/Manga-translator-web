"""Test nối câu về 1 dòng (không cần Qt): python tests/test_flatten_lines.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Dialogue, flatten_source_text  # noqa: E402


class Flatten(unittest.TestCase):
    def test_japanese_columns_join_without_space(self):
        self.assertEqual(flatten_source_text("私の詞がないと\n曲ができない…"), "私の詞がないと曲ができない…")

    def test_latin_and_mixed_join_with_space(self):
        self.assertEqual(flatten_source_text("watashi no shi ga nai to\nkyoku ga dekinai…"), "watashi no shi ga nai to kyoku ga dekinai…")
        self.assertEqual(flatten_source_text("私の\nOK"), "私の OK")

    def test_dialogue_all_three_fields(self):
        d = Dialogue(dialogue_id=1, box_location=[0, 0, 1, 1], original_text="ずっとここに\nいるわけにも\nいかない…",
                     romanization="zutto koko ni\niru wake ni mo\nikanai shi…", translated_text="Cũng không thể cứ mãi ở\nđây\nđược…").flatten_lines()
        self.assertEqual(d.original_text, "ずっとここにいるわけにもいかない…")
        self.assertEqual(d.romanization, "zutto koko ni iru wake ni mo ikanai shi…")
        self.assertEqual(d.translated_text, "Cũng không thể cứ mãi ở đây được…")

    def test_from_dict_does_not_flatten_but_helper_does(self):
        d = Dialogue.from_dict({"dialogue_id": 1, "original_text": "a\nb", "translated_text": "x\ny", "romanization": "r\ns"})
        self.assertEqual(d.translated_text, "x\ny")          # nạp project giữ nguyên xuống dòng người dùng gõ
        self.assertEqual(d.flatten_lines().translated_text, "x y")

    def test_legacy_romaji_split_then_flatten(self):
        d = Dialogue.from_dict({"dialogue_id": 1, "original_text": "ねえ\nね\nnee", "translated_text": "Này"}).flatten_lines()
        self.assertEqual((d.original_text, d.romanization), ("ねえね", "nee"))


if __name__ == "__main__":
    unittest.main()
