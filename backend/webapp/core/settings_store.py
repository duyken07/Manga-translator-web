"""Cài đặt toàn cục (SQLite) + API key Gemini (mã hóa Fernet, không bao giờ trả về client).

Chỉ cho client sửa các trường trong whitelist (UserSettings). Mọi đường dẫn (model, font) do server tự suy ra -> không nhận path từ client.
Giữ hành vi Desktop: model Gemini CỐ ĐỊNH (DEFAULT_GEMINI_MODEL) + model dự phòng tự động (gemini_models.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..database.models import SettingRow
from ..database.session import session_scope
from .config import get_settings
from .security import decrypt_secret, encrypt_secret

_APP_KEY = "app"
_KEY_KEY = "gemini_api_key"
DEFAULT_FONT_NAME = "000 WildWords2 TB Roman.ttf"     # font mặc định của Desktop (DEFAULT_FONT_REL)
FONT_EXTS = {".ttf", ".otf", ".ttc"}


class UserSettings(BaseModel):
    """Trường người dùng được chỉnh. Tên trùng AppSettings của Desktop để dễ đối chiếu."""
    source_lang: str = "ja"
    target_lang: str = "vi"
    concurrency_limit: int = Field(4, ge=1, le=16)
    translation_backend: Literal["yolo_ocr_gemini", "yolo_gemini_vision"] = "yolo_gemini_vision"
    sequential_context_mode: bool = True
    keep_honorifics: bool = False
    enable_stray_text_removal: bool = False
    enable_redraw: bool = True
    paint_auto_redraw: bool = False
    paint_direct: bool = False
    lama_device: Literal["cpu", "cuda", "mps"] = "cpu"
    lama_threads: int = Field(0, ge=0, le=64)
    redraw_font: str = ""                 # TÊN file font trong thư mục font của server ("" = mặc định)
    redraw_text_color: str = "20,20,20"
    redraw_line_spacing: float = Field(1.06, ge=0.7, le=2.5)
    theme: Literal["light", "dark", "system"] = "system"
    use_koharu: bool = False
    koharu_imgsz: int = Field(0, ge=0, le=2048)
    koharu_erase_sfx: bool = False
    koharu_use_yolo: bool = True
    ocr_force_cpu: bool = False
    ocr_gemini_rescue: bool = True
    ocr_low_conf: float = Field(0.60, ge=0.0, le=1.0)
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    custom_glossary: Dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


# Trường có thể ghi đè theo từng project
PROJECT_OVERRIDABLE = ("source_lang", "target_lang", "translation_backend", "sequential_context_mode", "keep_honorifics", "enable_redraw")


def list_fonts() -> List[str]:
    d = get_settings().font_dir
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in FONT_EXTS)


def resolve_font(name: str) -> str:
    """Tên font -> đường dẫn tuyệt đối trong font_dir; không hợp lệ/không tồn tại -> ''. Không bao giờ nhận path từ client."""
    d = get_settings().font_dir
    if not name:
        name = DEFAULT_FONT_NAME if (d / DEFAULT_FONT_NAME).is_file() else ""
    if not name or Path(name).name != name:
        return ""
    p = d / name
    return str(p) if p.is_file() and p.suffix.lower() in FONT_EXTS else ""


def get_user_settings() -> UserSettings:
    with session_scope() as s:
        row = s.get(SettingRow, _APP_KEY)
        data = dict(row.value) if row and row.value else {}
    known = {k: v for k, v in data.items() if k in UserSettings.model_fields}
    try:
        return UserSettings(**known)
    except Exception:  # noqa: BLE001 - dữ liệu cũ/hỏng: về mặc định thay vì sập
        return UserSettings()


def update_user_settings(patch: Dict[str, Any]) -> UserSettings:
    cur = get_user_settings().model_dump()
    cur.update(patch)
    new = UserSettings(**cur)                      # validate (ném ValidationError nếu sai)
    with session_scope() as s:
        row = s.get(SettingRow, _APP_KEY) or SettingRow(key=_APP_KEY, value={})
        row.value = new.model_dump()
        s.merge(row)
    return new


def effective_settings(project_overrides: Optional[dict] = None) -> UserSettings:
    base = get_user_settings().model_dump()
    for k, v in (project_overrides or {}).items():
        if k in PROJECT_OVERRIDABLE and v is not None:
            base[k] = v
    return UserSettings(**base)


# ------------------------------------------------------------------ API key
def set_api_key(key: str) -> None:
    key = key.strip()
    if not key:
        raise ValueError("API key rỗng")
    with session_scope() as s:
        s.merge(SettingRow(key=_KEY_KEY, value={}, secret=encrypt_secret(key)))


def get_api_key() -> Optional[str]:
    with session_scope() as s:
        row = s.get(SettingRow, _KEY_KEY)
        return decrypt_secret(row.secret) if row and row.secret else None


def has_api_key() -> bool:
    return bool(get_api_key())


def delete_api_key() -> None:
    with session_scope() as s:
        row = s.get(SettingRow, _KEY_KEY)
        if row:
            s.delete(row)


def masked_api_key() -> str:
    k = get_api_key()
    return "" if not k else (k[:3] + "…" + k[-2:] if len(k) > 8 else "••••")
