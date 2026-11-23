"""Model chính lỗi -> tự dùng model phụ; lỗi API key thì không chuyển."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import gemini_models as gm  # noqa: E402


class _Client:
    def __init__(self, behaviour):
        self.calls = []
        self.models = types.SimpleNamespace(generate_content=self._gen)
        self.behaviour = behaviour

    def _gen(self, model, **kw):
        self.calls.append(model)
        r = self.behaviour(model)
        if isinstance(r, Exception):
            raise r
        return r


class Err(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class FallbackTests(unittest.TestCase):
    def test_primary_ok(self):
        c = _Client(lambda m: types.SimpleNamespace(text="ok"))
        self.assertEqual(gm.generate_content(c, model="x", contents="a").text, "ok")
        self.assertEqual(c.calls, [gm.PRIMARY_MODEL])      # model người gọi truyền bị bỏ qua

    def test_error_uses_fallback(self):
        c = _Client(lambda m: Err("429 quota", 429) if m == gm.PRIMARY_MODEL else types.SimpleNamespace(text="B"))
        self.assertEqual(gm.generate_content(c, contents="a").text, "B")
        self.assertEqual(c.calls, [gm.PRIMARY_MODEL, gm.FALLBACK_MODEL])

    def test_empty_text_uses_fallback_only_when_required(self):
        c = _Client(lambda m: types.SimpleNamespace(text="" if m == gm.PRIMARY_MODEL else "B"))
        self.assertEqual(gm.generate_content(c, require_text=True, contents="a").text, "B")
        c2 = _Client(lambda m: types.SimpleNamespace(text=""))
        gm.generate_content(c2, contents="a")
        self.assertEqual(c2.calls, [gm.PRIMARY_MODEL])

    def test_key_error_no_fallback(self):
        c = _Client(lambda m: Err("API key not valid. Please pass a valid API key.", 400))
        with self.assertRaises(Err):
            gm.generate_content(c, contents="a")
        self.assertEqual(c.calls, [gm.PRIMARY_MODEL])

    def test_both_fail_raises_fallback_error(self):
        c = _Client(lambda m: Err("boom " + m))
        with self.assertRaises(Err) as cm:
            gm.generate_content(c, contents="a")
        self.assertIn(gm.FALLBACK_MODEL, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
