"""
Mô hình dữ liệu nhân vật (Character Profile). Không phụ thuộc Qt nên test được độc lập.

Một nhân vật = một thư mục `<id>/` chứa `profile.json` (+ avatar). `Character` là bản đã nạp trong bộ nhớ; `from_dict` / `to_dict`
chuyển qua lại với JSON. Trường lạ (do phiên bản mới hơn thêm vào) được giữ trong `extra` để không bị mất khi ghi lại.

Hai chế độ (`mode`):
  - "assistant": chatbot trợ lý gốc của app (luật phạm vi cũ, bộ lọc cục bộ cũ). Không dùng persona.
  - "roleplay":  nhập vai nhân vật theo persona / speech_style / ...; không bị bộ lọc "ngoài phạm vi" cục bộ chặn.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
DEFAULT_CHARACTER_ID = "assistant"     # nhân vật 'Trợ lý' đi kèm app: hành vi chatbot cũ

MODE_ASSISTANT = "assistant"
MODE_ROLEPLAY = "roleplay"
VALID_MODES = (MODE_ASSISTANT, MODE_ROLEPLAY)

SOURCE_BUILTIN = "builtin"             # chỉ có bản đi kèm app
SOURCE_USER = "user"                   # do người dùng tạo (AppData)
SOURCE_OVERRIDE = "override"           # bản của người dùng ghi đè nhân vật có sẵn (còn bản gốc để khôi phục)

ID_RE = re.compile(r"^[a-z0-9_]{1,48}$")
ASSISTANT_TEMPERATURE = 0.5            # trùng giá trị cũ gắn cứng trong chat_reply
ROLEPLAY_TEMPERATURE = 0.9
AVATAR_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

_KNOWN_KEYS = {
    "schema_version", "id", "name", "display_name", "description", "avatar", "enabled", "is_default", "mode", "persona",
    "speech_style", "style_examples", "scenario", "knowledge_boundary", "system_prompt", "greeting", "starters", "model",
}


def _s(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _list_of_dicts(v: Any) -> List[Dict[str, Any]]:
    return [dict(x) for x in v if isinstance(x, dict)] if isinstance(v, list) else []


@dataclass
class Character:
    id: str
    name: str
    display_name: str = ""
    description: str = ""
    avatar: str = ""                                   # CHỈ tên file (vd 'avatar.png'), nằm trong thư mục của nhân vật
    enabled: bool = True
    is_default: bool = False                           # lúc chạy: do CharacterManager tính (đúng một nhân vật là True)
    mode: str = MODE_ROLEPLAY
    persona: str = ""                                  # tính cách, tiểu sử cốt lõi
    speech_style: str = ""                             # cách nói, xưng hô
    style_examples: List[Dict[str, str]] = field(default_factory=list)   # [{"ja": ..., "vi": ...}] chỉ để bắt giọng
    scenario: str = ""                                 # vai trò của nhân vật trong app (đang cùng người dùng đọc / dịch truyện)
    knowledge_boundary: str = ""                       # nhân vật biết / không biết gì
    system_prompt: str = ""                            # ghi chú tự do thêm vào prompt (tuỳ chọn)
    greeting: str = ""                                 # lời mở đầu ở màn hình trống
    starters: List[Dict[str, Any]] = field(default_factory=list)         # [{"label", "text", "send"}] nút gợi ý
    provider: str = "gemini"
    model_name: str = ""                               # rỗng = dùng model đang chọn trong Settings
    temperature: Optional[float] = None                # None = mặc định theo mode
    schema_version: int = SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)
    # -- chỉ có lúc chạy, không ghi vào JSON --
    source: str = SOURCE_USER
    directory: Optional[Path] = None
    avatar_path: Optional[Path] = None                 # đường dẫn tuyệt đối tới avatar nếu tồn tại, ngược lại None (UI dùng ảnh mặc định)

    # -- tiện ích ------------------------------------------------------------------------------------------
    @property
    def is_roleplay(self) -> bool:
        return self.mode == MODE_ROLEPLAY

    @property
    def title(self) -> str:
        return self.name or self.id

    def effective_temperature(self) -> float:
        if self.temperature is not None:
            return float(self.temperature)
        return ROLEPLAY_TEMPERATURE if self.is_roleplay else ASSISTANT_TEMPERATURE

    def effective_model(self, app_model: str) -> str:
        return self.model_name.strip() or app_model

    # -- JSON ---------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Character":
        """Nạp khoan dung: kiểu sai -> giá trị mặc định (việc kiểm tra chặt là của `character_validator`)."""
        model = data.get("model") if isinstance(data.get("model"), dict) else {}
        temp = model.get("temperature")
        temp = float(temp) if isinstance(temp, (int, float)) and not isinstance(temp, bool) else None
        examples = []
        for x in _list_of_dicts(data.get("style_examples")):
            ex = {k: v for k, v in x.items() if isinstance(k, str) and isinstance(v, str)}
            if any(v.strip() for v in ex.values()):
                examples.append(ex)
        starters = []
        for x in _list_of_dicts(data.get("starters")):
            if _s(x.get("label")).strip() and _s(x.get("text")).strip():
                starters.append({"label": x["label"].strip(), "text": x["text"], "send": bool(x.get("send", False))})
        mode = _s(data.get("mode")) or MODE_ROLEPLAY
        return cls(
            id=_s(data.get("id")), name=_s(data.get("name")), display_name=_s(data.get("display_name")),
            description=_s(data.get("description")), avatar=_s(data.get("avatar")),
            enabled=data.get("enabled", True) is not False, is_default=data.get("is_default", False) is True,
            mode=mode, persona=_s(data.get("persona")), speech_style=_s(data.get("speech_style")),
            style_examples=examples, scenario=_s(data.get("scenario")), knowledge_boundary=_s(data.get("knowledge_boundary")),
            system_prompt=_s(data.get("system_prompt")), greeting=_s(data.get("greeting")), starters=starters,
            provider=_s(model.get("provider")) or "gemini", model_name=_s(model.get("model_name")), temperature=temp,
            schema_version=data.get("schema_version") if isinstance(data.get("schema_version"), int) else SCHEMA_VERSION,
            extra={k: copy.deepcopy(v) for k, v in data.items() if k not in _KNOWN_KEYS},
        )

    def to_dict(self) -> Dict[str, Any]:
        model: Dict[str, Any] = {"provider": self.provider or "gemini", "model_name": self.model_name}
        if self.temperature is not None:
            model["temperature"] = self.temperature
        out: Dict[str, Any] = {
            "schema_version": self.schema_version, "id": self.id, "name": self.name, "display_name": self.display_name,
            "description": self.description, "avatar": self.avatar, "enabled": self.enabled, "is_default": self.is_default,
            "mode": self.mode, "persona": self.persona, "speech_style": self.speech_style,
            "style_examples": copy.deepcopy(self.style_examples), "scenario": self.scenario,
            "knowledge_boundary": self.knowledge_boundary, "system_prompt": self.system_prompt, "greeting": self.greeting,
            "starters": copy.deepcopy(self.starters), "model": model,
        }
        for k, v in self.extra.items():
            out.setdefault(k, copy.deepcopy(v))
        return out
