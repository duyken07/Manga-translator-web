"""
CharacterManager: đọc / thêm / sửa / xoá nhân vật, quản lý nhân vật mặc định. Không phụ thuộc Qt, không gọi Gemini, không đụng nội dung prompt.

Hai nguồn dữ liệu:
  * builtin_dir  - nhân vật đi kèm app (chỉ đọc). Gốc: APP_ROOT/characters, cạnh .exe giống models/ và fonts/.
  * user_dir     - nhân vật của người dùng (AppData, ghi được). Trùng ID với bản có sẵn = BẢN GHI ĐÈ: xoá bản ghi đè là khôi phục bản gốc.

Nhân vật mặc định lưu riêng trong `user_dir/manager_state.json` (không sửa profile của nhân vật có sẵn). Thứ tự chọn mặc định:
nhân vật người dùng đã đặt -> nhân vật có `is_default: true` trong profile -> nhân vật đang bật đầu tiên.

Mọi thao tác ghi: kiểm tra trước, ghi file tạm rồi `os.replace` (không để lại file nửa vời), lỗi thì dọn thư mục vừa tạo.
Mọi lỗi người dùng gặp được là `CharacterError` với thông điệp tiếng Việt hiển thị thẳng được. Một profile hỏng không làm cả danh sách hỏng:
nó bị bỏ qua và ghi vào `load_errors`.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.characters.character_model import (
    AVATAR_EXTENSIONS,
    SOURCE_BUILTIN,
    SOURCE_OVERRIDE,
    SOURCE_USER,
    Character,
)
from app.characters.character_validator import is_valid_id, validate_character_data

log = logging.getLogger("character_manager")

PROFILE_NAME = "profile.json"
STATE_NAME = "manager_state.json"
MAX_AVATAR_BYTES = 5 * 1024 * 1024


class CharacterError(Exception):
    """Lỗi có thể hiển thị cho người dùng."""


class CharacterNotFoundError(CharacterError):
    pass


class CharacterValidationError(CharacterError):
    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, encoding="utf-8") as f:       # kiểm tra file tạm đọc lại được rồi mới thay file cũ
            json.load(f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class CharacterManager:
    def __init__(self, builtin_dir, user_dir=None) -> None:
        self.builtin_dir = Path(builtin_dir)
        self.user_dir: Optional[Path] = Path(user_dir) if user_dir else None
        self._chars: Dict[str, Character] = {}
        self.load_errors: Dict[str, str] = {}           # tên thư mục -> lý do không nạp được

    # ------------------------------------------------------------------ nạp
    def load_characters(self) -> List[Character]:
        """Đọc lại toàn bộ nhân vật từ đĩa. Không bao giờ ném lỗi vì một profile hỏng."""
        self._chars = {}
        self.load_errors = {}
        for c in self._read_dir(self.builtin_dir, SOURCE_BUILTIN):
            self._chars[c.id] = c
        if self.user_dir is not None:
            for c in self._read_dir(self.user_dir, SOURCE_USER):
                if c.id in self._chars:
                    c.source = SOURCE_OVERRIDE
                self._chars[c.id] = c
        for c in self._chars.values():
            c.avatar_path = self._resolve_avatar(c)
        self._apply_default()
        return self.get_all_characters()

    def _read_dir(self, root: Optional[Path], source: str) -> List[Character]:
        out: List[Character] = []
        if root is None or not root.is_dir():
            return out
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            f = d / PROFILE_NAME
            if not f.is_file():
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:         # JSON hỏng / không đọc được
                self._record_error(d.name, source, f"Không đọc được {PROFILE_NAME}: {exc}")
                continue
            errors = validate_character_data(data)
            if not errors and data.get("id") != d.name:
                errors = [f"ID '{data.get('id')}' không khớp tên thư mục '{d.name}'."]
            if errors:
                self._record_error(d.name, source, "; ".join(errors))
                continue
            c = Character.from_dict(data)
            c.source, c.directory = source, d
            out.append(c)
        return out

    def _record_error(self, folder: str, source: str, msg: str) -> None:
        key = folder if source == SOURCE_BUILTIN else f"{folder} (người dùng)"
        self.load_errors[key] = msg
        log.warning("Bỏ qua nhân vật '%s': %s", key, msg)

    def _resolve_avatar(self, c: Character) -> Optional[Path]:
        if not c.avatar:
            return None
        for root in (self.user_dir, self.builtin_dir):    # bản ghi đè không có avatar riêng thì dùng avatar của bản gốc
            if root is None:
                continue
            p = root / c.id / c.avatar
            if p.is_file():
                return p
        return None

    # ------------------------------------------------------------------ trạng thái mặc định
    def _state_path(self) -> Optional[Path]:
        return self.user_dir / STATE_NAME if self.user_dir else None

    def _read_state(self) -> Dict[str, Any]:
        p = self._state_path()
        if p is None or not p.is_file():
            return {}
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _apply_default(self) -> None:
        enabled = [c for c in self._chars.values() if c.enabled]
        hinted = [c for c in enabled if c.is_default]
        chosen = None
        want = self._read_state().get("default_character_id")
        if isinstance(want, str) and want in self._chars and self._chars[want].enabled:
            chosen = self._chars[want]
        elif hinted:
            chosen = hinted[0]
        elif enabled:
            chosen = enabled[0]
        for c in self._chars.values():
            c.is_default = c is chosen

    # ------------------------------------------------------------------ truy vấn
    def find(self, cid: str) -> Optional[Character]:
        return self._chars.get(cid)

    def get_character(self, cid: str) -> Character:
        c = self._chars.get(cid)
        if c is None:
            raise CharacterNotFoundError(f"Không tìm thấy nhân vật '{cid}'.")
        return c

    def get_all_characters(self) -> List[Character]:
        return list(self._chars.values())

    def get_enabled_characters(self) -> List[Character]:
        """Nhân vật đang bật (dùng cho danh sách chọn), mặc định đứng đầu."""
        out = [c for c in self._chars.values() if c.enabled]
        out.sort(key=lambda c: not c.is_default)
        return out

    def get_default_character(self) -> Optional[Character]:
        return next((c for c in self._chars.values() if c.is_default), None)

    def get_enabled_character(self, cid: str) -> Character:
        """Dùng khi chuyển nhân vật: phải tồn tại VÀ đang bật."""
        c = self.get_character(cid)
        if not c.enabled:
            raise CharacterError(f"Nhân vật '{c.title}' đang bị vô hiệu hóa.")
        return c

    def validate_character(self, data: Any) -> List[str]:
        return validate_character_data(data)

    # ------------------------------------------------------------------ ghi
    def _require_writable(self) -> Path:
        if self.user_dir is None:
            raise CharacterError("Chưa có thư mục dữ liệu người dùng nên không thể lưu nhân vật.")
        return self.user_dir

    def _dir_for(self, cid: str) -> Path:
        root = self._require_writable()
        if not is_valid_id(cid):                          # ID là tên thư mục: chặn '..', '/', ...
            raise CharacterError("ID nhân vật không hợp lệ.")
        return root / cid

    @staticmethod
    def _copy_avatar(src, dest_dir: Path) -> str:
        src = Path(src)
        ext = src.suffix.lower()
        if not src.is_file():
            raise CharacterError(f"Không tìm thấy file avatar: {src}")
        if ext not in AVATAR_EXTENSIONS:
            raise CharacterError("Avatar phải là ảnh: " + ", ".join(AVATAR_EXTENSIONS) + ".")
        if src.stat().st_size > MAX_AVATAR_BYTES:
            raise CharacterError("Avatar quá lớn (tối đa 5 MB).")
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = "avatar" + ext
        for old in dest_dir.glob("avatar.*"):             # tránh để lại avatar cũ khác đuôi
            if old.name != name:
                old.unlink(missing_ok=True)
        shutil.copyfile(src, dest_dir / name)
        return name

    def add_character(self, data: Dict[str, Any], avatar_src=None) -> Character:
        data = copy.deepcopy(data) if isinstance(data, dict) else data
        errors = validate_character_data(data)
        if errors:
            raise CharacterValidationError(errors)
        cid = data["id"]
        d = self._dir_for(cid)
        if cid in self._chars or d.exists():
            raise CharacterError(f"ID '{cid}' đã tồn tại, hãy chọn ID khác.")   # không bao giờ ghi đè
        want_default = data.get("is_default") is True
        created = False
        try:
            d.mkdir(parents=True, exist_ok=False)
            created = True
            if avatar_src:
                data["avatar"] = self._copy_avatar(avatar_src, d)
            char = Character.from_dict(data)
            char.is_default = False                       # mặc định do manager_state quyết định
            _write_json_atomic(d / PROFILE_NAME, char.to_dict())
        except CharacterError:
            if created:
                shutil.rmtree(d, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001
            if created:
                shutil.rmtree(d, ignore_errors=True)
            raise CharacterError(f"Không thể tạo nhân vật: {exc}") from exc
        self.load_characters()
        if want_default and self._chars[cid].enabled:
            self.set_default_character(cid)
        return self.get_character(cid)

    def update_character(self, cid: str, changes: Dict[str, Any], avatar_src=None) -> Character:
        """Sửa nhân vật. Không đổi được ID. Sửa nhân vật có sẵn sẽ tạo bản ghi đè trong thư mục người dùng."""
        cur = self.get_character(cid)
        d = self._dir_for(cid)
        changes = copy.deepcopy(changes)
        if "id" in changes and changes["id"] != cid:
            raise CharacterError("Không được đổi ID nhân vật sau khi tạo.")
        want_default = changes.pop("is_default", None) is True
        merged = cur.to_dict()
        for k, v in changes.items():
            if k == "model" and isinstance(v, dict) and isinstance(merged.get("model"), dict):
                merged["model"] = {**merged["model"], **v}
            else:
                merged[k] = v
        merged["id"] = cid
        errors = validate_character_data(merged)
        if errors:
            raise CharacterValidationError(errors)
        if cur.enabled and merged.get("enabled") is False and len(self.get_enabled_characters()) <= 1:
            raise CharacterError("Phải còn ít nhất một nhân vật đang bật.")
        created = not d.exists()
        try:
            d.mkdir(parents=True, exist_ok=True)
            if avatar_src:
                merged["avatar"] = self._copy_avatar(avatar_src, d)
            _write_json_atomic(d / PROFILE_NAME, Character.from_dict(merged).to_dict())
        except CharacterError:
            if created:
                shutil.rmtree(d, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001
            if created:
                shutil.rmtree(d, ignore_errors=True)
            raise CharacterError(f"Không thể lưu nhân vật: {exc}") from exc
        self.load_characters()
        if want_default and self._chars[cid].enabled:
            self.set_default_character(cid)
        return self.get_character(cid)

    def set_default_character(self, cid: str) -> Character:
        c = self.get_enabled_character(cid)
        root = self._require_writable()
        state = self._read_state()
        state["default_character_id"] = cid
        try:
            _write_json_atomic(root / STATE_NAME, state)
        except OSError as exc:
            raise CharacterError(f"Không thể lưu nhân vật mặc định: {exc}") from exc
        self._apply_default()
        return c

    def delete_character(self, cid: str, new_default_id: Optional[str] = None) -> None:
        """Xoá nhân vật của người dùng. Với bản ghi đè = khôi phục bản gốc. Nhân vật có sẵn không xoá được (chỉ vô hiệu hóa).
        Xoá nhân vật mặc định phải chỉ định `new_default_id`. Việc 'đang dùng thì không xoá' do tầng giao diện kiểm tra."""
        c = self.get_character(cid)
        if c.source == SOURCE_BUILTIN:
            raise CharacterError(f"'{c.title}' là nhân vật có sẵn của ứng dụng nên không xóa được. Bạn có thể vô hiệu hóa nó.")
        d = self._dir_for(cid)
        if d.resolve().parent != self._require_writable().resolve():
            raise CharacterError("Đường dẫn nhân vật không hợp lệ.")
        if c.source == SOURCE_USER:
            others = [x for x in self.get_enabled_characters() if x.id != cid]
            if not others:
                raise CharacterError("Phải còn ít nhất một nhân vật đang bật.")
            if c.is_default and (new_default_id is None or all(x.id != new_default_id for x in others)):
                raise CharacterError("Đây là nhân vật mặc định: hãy chọn một nhân vật đang bật khác làm mặc định thay thế.")
        try:
            shutil.rmtree(d)
        except OSError as exc:
            raise CharacterError(f"Không thể xóa nhân vật: {exc}") from exc
        self.load_characters()
        if new_default_id and c.source == SOURCE_USER and new_default_id in self._chars:
            self.set_default_character(new_default_id)

    def restore_default(self, cid: str) -> Character:
        """Bỏ bản chỉnh sửa của người dùng, trả về nhân vật có sẵn."""
        c = self.get_character(cid)
        if c.source != SOURCE_OVERRIDE:
            raise CharacterError("Nhân vật này chưa bị chỉnh sửa nên không có gì để khôi phục.")
        self.delete_character(cid)
        return self.get_character(cid)
