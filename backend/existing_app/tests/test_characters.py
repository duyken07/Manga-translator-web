"""Kiểm thử hệ thống nhân vật (không cần Qt): validator, CharacterManager (nạp / thêm / sửa / xoá / mặc định), dữ liệu đi kèm app."""
import json
import os
import shutil
import tempfile
from pathlib import Path

from app.characters import (
    Character, CharacterError, CharacterManager, CharacterNotFoundError, CharacterValidationError, validate_character_data,
)

SHIPPED = Path(__file__).resolve().parent.parent / "characters"


def _profile(cid="hikawa_hina", **kw):
    d = {"schema_version": 1, "id": cid, "name": "Hikawa Hina", "description": "Pastel", "avatar": "avatar.png",
         "enabled": True, "is_default": False, "mode": "roleplay", "persona": "Hina vui vẻ.",
         "model": {"provider": "gemini", "model_name": "", "temperature": 0.8}}
    d.update(kw)
    return d


def _write(root: Path, data, folder=None, raw=None):
    d = root / (folder or data["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.json").write_text(raw if raw is not None else json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return d


def _mgr(tmp, builtin=(), user=()):
    b, u = Path(tmp) / "builtin", Path(tmp) / "user"
    b.mkdir(), u.mkdir()
    for p in builtin:
        _write(b, p)
    for p in user:
        _write(u, p)
    m = CharacterManager(b, u)
    m.load_characters()
    return m, b, u


class _Tmp:
    def __enter__(self):
        self.d = tempfile.mkdtemp()
        return self.d

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"không ném {exc.__name__}")


# -- validator ------------------------------------------------------------------------------------------------
def test_validator_rules():
    assert validate_character_data(_profile()) == []
    for bad_id in ("", "Hina", "hi na", "hina-x", "../x", "a/b", "x" * 49, None, 5):
        assert validate_character_data(_profile(cid=bad_id)), bad_id
    assert validate_character_data(_profile(name=" "))
    assert validate_character_data(_profile(mode="bot"))
    assert validate_character_data(_profile(persona=""))                         # roleplay cần persona / system_prompt
    assert validate_character_data(_profile(persona="", system_prompt="You are X.")) == []
    assert validate_character_data(_profile(mode="assistant", persona="")) == []  # assistant không cần persona
    assert validate_character_data(_profile(avatar="../a.png"))
    assert validate_character_data(_profile(avatar="a.exe"))
    assert validate_character_data(_profile(enabled="yes"))
    assert validate_character_data(_profile(model={"provider": "openai"}))
    assert validate_character_data(_profile(model={"temperature": 3}))
    assert validate_character_data(_profile(model={"temperature": True}))
    assert validate_character_data(_profile(style_examples=[{"ja": 1}]))
    assert validate_character_data(_profile(starters=[{"label": "x"}]))
    assert validate_character_data(_profile(model={"provider": "gemini", "api_key": "AIza..."}))   # không lưu key
    assert validate_character_data(["x"])


# -- manager ----------------------------------------------------------------------------------------------------
def test_ut01_02_03_load_get():
    with _Tmp() as t:
        m, _, _ = _mgr(t, builtin=[_profile("a_one"), _profile("b_two", name="Two")])
        assert [c.id for c in m.get_all_characters()] == ["a_one", "b_two"]
        assert m.get_character("b_two").name == "Two"
        _raises(CharacterNotFoundError, m.get_character, "nope")
        assert m.find("nope") is None


def test_ut04_05_add_and_duplicate():
    with _Tmp() as t:
        m, b, u = _mgr(t, builtin=[_profile("a_one")])
        c = m.add_character(_profile("new_one", name="New"))
        assert c.id == "new_one" and (u / "new_one" / "profile.json").is_file() and c.source == "user"
        assert m.get_character("new_one").name == "New"
        before = (u / "new_one" / "profile.json").read_text(encoding="utf-8")
        _raises(CharacterError, m.add_character, _profile("new_one", name="Other"))     # trùng ID người dùng
        _raises(CharacterError, m.add_character, _profile("a_one", name="Other"))       # trùng ID có sẵn
        assert (u / "new_one" / "profile.json").read_text(encoding="utf-8") == before   # không bị ghi đè
        e = _raises(CharacterValidationError, m.add_character, _profile("Bad Id"))
        assert e.errors
        assert not list(u.glob("**/*.tmp"))                                              # không để lại file tạm


def test_add_with_avatar_and_rollback():
    with _Tmp() as t:
        m, _, u = _mgr(t)
        img = Path(t) / "pic.PNG"
        img.write_bytes(b"\x89PNG fake")
        c = m.add_character(_profile("with_av", avatar=""), avatar_src=img)
        assert c.avatar == "avatar.png" and c.avatar_path and c.avatar_path.is_file()
        _raises(CharacterError, m.add_character, _profile("bad_av", avatar=""), avatar_src=Path(t) / "missing.png")
        assert not (u / "bad_av").exists()                                               # hoàn tác thư mục dở dang
        txt = Path(t) / "x.txt"
        txt.write_text("x")
        _raises(CharacterError, m.add_character, _profile("bad_av2", avatar=""), avatar_src=txt)
        assert not (u / "bad_av2").exists()


def test_ut06_broken_json_does_not_crash():
    with _Tmp() as t:
        m, b, u = _mgr(t, builtin=[_profile("good_one")])
        _write(b, None, folder="broken", raw="{not json")
        _write(b, _profile("other_id"), folder="mismatch")
        _write(u, _profile("bad id"), folder="user_bad")
        m.load_characters()
        assert [c.id for c in m.get_all_characters()] == ["good_one"]
        assert len(m.load_errors) == 3 and "broken" in m.load_errors


def test_ut07_missing_avatar():
    with _Tmp() as t:
        m, _, _ = _mgr(t, builtin=[_profile("a_one", avatar="avatar.png"), _profile("b_two", avatar="")])
        assert m.get_character("a_one").avatar_path is None       # file không có -> UI dùng ảnh mặc định
        assert m.get_character("b_two").avatar_path is None


def test_ut08_09_delete_rules():
    with _Tmp() as t:
        m, b, u = _mgr(t, builtin=[_profile("a_one", is_default=True)])
        m.add_character(_profile("mine_a"))
        m.add_character(_profile("mine_b"))
        _raises(CharacterError, m.delete_character, "a_one")                  # có sẵn: không xoá được
        m.set_default_character("mine_a")
        e = _raises(CharacterError, m.delete_character, "mine_a")             # mặc định: bắt buộc chọn thay thế
        assert "mặc định" in str(e)
        _raises(CharacterError, m.delete_character, "mine_a", "mine_a")        # thay thế không được là chính nó
        m.delete_character("mine_a", new_default_id="mine_b")
        assert not (u / "mine_a").exists() and m.get_default_character().id == "mine_b"
        m.delete_character("mine_b", new_default_id="a_one")
        assert m.find("mine_b") is None and m.get_default_character().id == "a_one"
        _raises(CharacterNotFoundError, m.delete_character, "mine_b")


def test_ut10_disabled_cannot_be_selected():
    with _Tmp() as t:
        m, _, _ = _mgr(t, builtin=[_profile("a_one"), _profile("b_two", enabled=False)])
        _raises(CharacterError, m.get_enabled_character, "b_two")
        assert [c.id for c in m.get_enabled_characters()] == ["a_one"]
        _raises(CharacterError, m.set_default_character, "b_two")


def test_default_resolution_and_persistence():
    with _Tmp() as t:
        m, b, u = _mgr(t, builtin=[_profile("a_one"), _profile("b_two", is_default=True)])
        assert m.get_default_character().id == "b_two"                         # theo gợi ý trong profile
        assert [c.id for c in m.get_enabled_characters()][0] == "b_two"        # mặc định đứng đầu
        m.set_default_character("a_one")
        assert sum(c.is_default for c in m.get_all_characters()) == 1
        m2 = CharacterManager(b, u)
        m2.load_characters()
        assert m2.get_default_character().id == "a_one"                         # nhớ qua lần mở sau
        m2.update_character("a_one", {"enabled": False})                        # mặc định bị tắt -> tự chọn lại
        assert m2.get_default_character().id == "b_two"
    with _Tmp() as t:
        m, _, _ = _mgr(t, builtin=[_profile("a_one"), _profile("b_two")])
        assert m.get_default_character().id == "a_one"                         # không gợi ý: nhân vật bật đầu tiên
    with _Tmp() as t:
        m, _, _ = _mgr(t)
        assert m.get_default_character() is None and m.get_enabled_characters() == []   # rỗng: UI hiện hướng dẫn thêm nhân vật


def test_update_override_and_restore():
    with _Tmp() as t:
        m, b, u = _mgr(t, builtin=[_profile("a_one", persona="Gốc.")])
        _raises(CharacterError, m.update_character, "a_one", {"id": "other"})  # không đổi ID
        _raises(CharacterValidationError, m.update_character, "a_one", {"name": ""})
        c = m.update_character("a_one", {"persona": "Sửa.", "model": {"temperature": 1.1}})
        assert c.source == "override" and c.persona == "Sửa." and c.temperature == 1.1 and c.model_name == ""
        assert "Gốc." in (b / "a_one" / "profile.json").read_text(encoding="utf-8")     # bản có sẵn không bị đụng
        assert (u / "a_one" / "profile.json").is_file()
        c = m.restore_default("a_one")
        assert c.source == "builtin" and c.persona == "Gốc." and not (u / "a_one").exists()
        _raises(CharacterError, m.restore_default, "a_one")


def test_cannot_disable_last_enabled():
    with _Tmp() as t:
        m, _, _ = _mgr(t, builtin=[_profile("a_one"), _profile("b_two")])
        m.update_character("a_one", {"enabled": False})
        _raises(CharacterError, m.update_character, "b_two", {"enabled": False})
        assert m.get_character("b_two").enabled


def test_extra_fields_survive_roundtrip_and_no_secrets_written():
    with _Tmp() as t:
        m, _, u = _mgr(t, builtin=[_profile("a_one", future_field={"x": [1, 2]})])
        m.update_character("a_one", {"description": "Mới"})
        data = json.loads((u / "a_one" / "profile.json").read_text(encoding="utf-8"))
        assert data["future_field"] == {"x": [1, 2]} and data["description"] == "Mới"


def test_read_only_manager_and_path_safety():
    with _Tmp() as t:
        b = Path(t) / "b"
        b.mkdir()
        _write(b, _profile("a_one"))
        m = CharacterManager(b, None)
        m.load_characters()
        _raises(CharacterError, m.add_character, _profile("x_new"))
        _raises(CharacterError, m.update_character, "a_one", {"description": "x"})
    with _Tmp() as t:
        m, _, u = _mgr(t)
        for evil in ("../evil", "..", "a/b"):
            _raises(CharacterError, m._dir_for, evil)
        assert not (Path(t) / "evil").exists()


def test_effective_temperature_and_model():
    a = Character.from_dict(_profile(mode="assistant", persona="", model={}))
    r = Character.from_dict(_profile(model={}))
    assert a.effective_temperature() == 0.5 and r.effective_temperature() == 0.9
    assert Character.from_dict(_profile()).effective_temperature() == 0.8
    assert r.effective_model("gem") == "gem" and Character.from_dict(_profile(model={"model_name": "x"})).effective_model("gem") == "x"


def test_shipped_characters_are_valid():
    m = CharacterManager(SHIPPED)
    m.load_characters()
    assert m.load_errors == {}
    ids = {c.id for c in m.get_all_characters()}
    assert {"assistant", "sengoku_yuno"} <= ids
    assert m.get_character("assistant").mode == "assistant"
    y = m.get_character("sengoku_yuno")
    assert y.mode == "roleplay" and y.persona and y.style_examples and y.avatar_path and y.avatar_path.is_file()
    assert m.get_default_character().id == "sengoku_yuno"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
