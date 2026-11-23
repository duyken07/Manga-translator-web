"""Kiểm thử chat_service: kho lịch sử theo phiên, ngữ cảnh gửi kèm, cắt lịch sử, gọi API (giả lập google.genai)."""
import sys
import types

try:  # môi trường không có PySide6 (CI không GUI): translation_service -> config chỉ cần QSettings
    import PySide6.QtCore  # noqa: F401
except ModuleNotFoundError:
    _qt = types.ModuleType("PySide6")
    _qtc = types.ModuleType("PySide6.QtCore")
    _qtc.QSettings = type("QSettings", (), {"__init__": lambda self, *a, **k: None})
    _qt.QtCore = _qtc
    sys.modules.update({"PySide6": _qt, "PySide6.QtCore": _qtc})

from app.services import chat_service as cs


def test_store_threads_and_ordering():
    st = cs.ChatSessionStore()
    a = st.thread_for_sentence("後悔に浮足立って", "Bồn chồn trong sự hối hận", [("x", "y")])
    b = st.thread_for_sentence("破った言いつけ", "Lời dặn dò đã phá vỡ")
    assert st.thread_for_sentence("後悔に浮足立って") is a            # cùng câu -> cùng cuộc trò chuyện
    assert a.translation == "Bồn chồn trong sự hối hận"              # không bị ghi đè bằng chuỗi rỗng
    assert [t.key for t in st.threads()] == [cs.FREE_THREAD_KEY]     # chưa có tin nhắn -> chưa liệt kê
    a.messages.append(cs.ChatMessage("user", "hi", ts=1))
    b.messages.append(cs.ChatMessage("user", "hi", ts=2))
    assert [t.key for t in st.threads()] == [cs.FREE_THREAD_KEY, "破った言いつけ", "後悔に浮足立って"]
    st.clear_thread("破った言いつけ")
    assert b.messages == []
    assert st.thread_for_sentence("   ").is_free                     # câu rỗng -> hội thoại tự do


def test_system_instruction_contains_context():
    st = cs.ChatSessionStore()
    t = st.thread_for_sentence("後悔に浮足立って", "Bồn chồn", [("破った言いつけ", "Lời dặn dò đã phá vỡ")])
    analysis = {"words": [{"word": "後悔", "reading": "こうかい", "part_of_speech": "danh từ", "meaning": "hối hận"}],
                "grammar_notes": "に chỉ nguyên nhân"}
    s = cs.build_system_instruction("Nhật", "Việt", t, analysis)
    for needle in ("後悔に浮足立って", "Bồn chồn", "こうかい", "に chỉ nguyên nhân", "破った言いつけ"):
        assert needle in s
    free = cs.build_system_instruction("Nhật", "Việt", st.free_thread(), None)
    assert "Câu gốc đang bàn" not in free


def test_trim_history_starts_with_user():
    msgs = [cs.ChatMessage("user" if i % 2 == 0 else "model", str(i)) for i in range(60)]
    out = cs.trim_history(msgs, 25)
    assert out[0].role == "user" and len(out) <= 25 and out[-1].text == "59"
    assert cs.trim_history([cs.ChatMessage("model", "x")]) == []


class ClientError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def _install_fake_genai(response_text="Xin chào", raise_exc=None):
    calls = {}

    class _Models:
        def generate_content(self, **kw):
            calls.update(kw)
            if raise_exc:
                raise raise_exc
            return types.SimpleNamespace(text=response_text)

    class Client:
        def __init__(self, api_key):
            self.models = _Models()

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    errors = types.ModuleType("google.genai.errors")
    genai.Client, genai.errors, errors.ClientError = Client, errors, ClientError
    google.genai = genai
    sys.modules.update({"google": google, "google.genai": genai, "google.genai.errors": errors})
    return calls, ClientError


def test_chat_reply_ok_and_errors():
    from app.services.translation_service import TranslationServiceError, TranslationRateLimitError
    calls, _ = _install_fake_genai("Trả lời")
    msgs = [cs.ChatMessage("user", "Vì sao dùng に?")]
    assert cs.chat_reply("KEY", "m", "SYS", msgs) == "Trả lời"
    assert calls["config"]["system_instruction"] == "SYS"
    assert calls["contents"] == [{"role": "user", "parts": [{"text": "Vì sao dùng に?"}]}]

    for bad in ([], [cs.ChatMessage("model", "x")]):
        try:
            cs.chat_reply("KEY", "m", "SYS", bad); assert False
        except TranslationServiceError:
            pass
    try:
        cs.chat_reply("", "m", "SYS", msgs); assert False
    except TranslationServiceError as e:
        assert "API Key" in str(e)

    _install_fake_genai(raise_exc=ClientError("429 quota", code=429))
    try:
        cs.chat_reply("KEY", "m", "SYS", msgs); assert False
    except TranslationRateLimitError:
        pass
    _install_fake_genai(response_text="")
    try:
        cs.chat_reply("KEY", "m", "SYS", msgs); assert False
    except TranslationServiceError as e:
        assert "không trả về" in str(e)




def test_time_divider_and_analysis_message():
    import time as _t
    ts = _t.mktime((2026, 11, 13, 14, 14, 0, 0, 0, -1))          # thứ Sáu
    assert cs.time_divider_text(ts) == "14:14 T6"
    assert cs.time_divider_text(_t.mktime((2026, 11, 15, 9, 5, 0, 0, 0, -1))) == "09:05 CN"
    assert cs.needs_time_divider(None, ts) and not cs.needs_time_divider(ts, ts + 599) and cs.needs_time_divider(ts, ts + 600)
    msg = cs.format_analysis_message("後悔に浮足立って", "Bồn chồn", {
        "words": [{"word": "後悔", "reading": "こうかい", "romaji": "koukai", "part_of_speech": "danh từ", "meaning": "hối hận"}],
        "grammar_notes": "に chỉ nguyên nhân"})
    for needle in ("後悔に浮足立って", "Bồn chồn", "**後悔** (こうかい · koukai) _danh từ_ — hối hận", "**Ngữ pháp:** に chỉ nguyên nhân"):
        assert needle in msg


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)
