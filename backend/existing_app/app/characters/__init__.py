"""Hệ thống nhiều nhân vật (Multi-Character): dữ liệu, kiểm tra, quản lý. Toàn bộ không phụ thuộc Qt."""
from app.characters.character_model import (  # noqa: F401
    DEFAULT_CHARACTER_ID,
    MODE_ASSISTANT,
    MODE_ROLEPLAY,
    Character,
)
from app.characters.character_validator import is_valid_id, validate_character_data  # noqa: F401
from app.characters.character_manager import (  # noqa: F401
    CharacterError,
    CharacterManager,
    CharacterNotFoundError,
    CharacterValidationError,
)
