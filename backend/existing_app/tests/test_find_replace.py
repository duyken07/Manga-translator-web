"""Test tìm / thay thế: python -m unittest discover -s tests -v (không cần Qt)"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import find_replace as fr  # noqa: E402


def D(i, tr, org=""):
    return NS(dialogue_id=i, translated_text=tr, original_text=org)


PAGES = [
    NS(page_name="001.webp", dialogues=[D(1, "Xin chào Tomori"), D(2, "Không có gì")]),
    NS(page_name="002.webp", dialogues=[D(1, "TOMORI à, tomori ơi!\nTomori"), D(2, "Tomorin")]),
    NS(page_name="003.webp", dialogues=[D(7, "", "ねえ")]),
]


class FindReplace(unittest.TestCase):
    def test_case_insensitive_counts(self):
        ms = fr.find_matches(PAGES, fr.build_pattern("tomori"))
        self.assertEqual([(m.page_index, m.dialogue_id, m.count) for m in ms], [(0, 1, 1), (1, 1, 3), (1, 2, 1)])

    def test_match_case(self):
        ms = fr.find_matches(PAGES, fr.build_pattern("Tomori", match_case=True))
        self.assertEqual([(m.page_index, m.dialogue_id, m.count) for m in ms], [(0, 1, 1), (1, 1, 1), (1, 2, 1)])

    def test_whole_word(self):
        ms = fr.find_matches(PAGES, fr.build_pattern("tomori", whole_word=True))
        self.assertEqual([(m.page_index, m.dialogue_id) for m in ms], [(0, 1), (1, 1)])    # "Tomorin" bị loại

    def test_vietnamese_unicode_case(self):
        pages = [NS(page_name="a", dialogues=[D(1, "ĐƯỢC rồi, được thôi")])]
        self.assertEqual(fr.find_matches(pages, fr.build_pattern("được"))[0].count, 2)

    def test_regex_chars_are_literal(self):
        pages = [NS(page_name="a", dialogues=[D(1, "Hả?! (gì cơ) a.b")])]
        self.assertEqual(fr.find_matches(pages, fr.build_pattern("?!"))[0].count, 1)
        self.assertEqual(fr.find_matches(pages, fr.build_pattern("a.b"))[0].count, 1)
        self.assertEqual(fr.find_matches(pages, fr.build_pattern("(gì"))[0].count, 1)

    def test_original_field_and_page_filter(self):
        self.assertEqual(len(fr.find_matches(PAGES, fr.build_pattern("ねえ"), field="original")), 1)
        self.assertEqual(len(fr.find_matches(PAGES, fr.build_pattern("tomori"), page_indices=[1])), 2)

    def test_empty_query(self):
        self.assertEqual(fr.find_matches(PAGES, fr.build_pattern("")), [])

    def test_replace_is_literal(self):
        p = fr.build_pattern("tomori")
        self.assertEqual(fr.replace_text("Tomori và tomori", p, r"Tô\1 mô"), (r"Tô\1 mô và Tô\1 mô", 2))

    def test_plan(self):
        p = fr.build_pattern("tomori", whole_word=True)
        plan = fr.plan_replacements(fr.find_matches(PAGES, p), p, "Tomo")
        self.assertEqual(plan, {0: {1: "Xin chào Tomo"}, 1: {1: "Tomo à, Tomo ơi!\nTomo"}})

    def test_snippet_marks_hit_and_flattens(self):
        m = fr.find_matches(PAGES, fr.build_pattern("ơi"))[0]
        self.assertIn("«ơi»", m.snippet())
        self.assertNotIn("\n", m.snippet())


if __name__ == "__main__":
    unittest.main()
