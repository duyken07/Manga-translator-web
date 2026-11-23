"""
Bộ thuật ngữ tùy chỉnh do người dùng TỰ SOẠN (khác với `Project.glossary` -
bảng thuật ngữ Gemini TỰ ĐỘNG phát hiện/tích luỹ trong lúc dịch, chỉ gắn với
1 project cụ thể). Bộ thuật ngữ này:
- Do người dùng tự nhập (VD: tên nhân vật hay bị dịch sai)
- Lưu thành 1 file JSON RIÊNG, độc lập với project -> tái sử dụng được cho
  nhiều project/bộ truyện khác nhau (VD: 1 bộ truyện = 1 file thuật ngữ)
- Được ưu tiên hơn glossary tự động của project khi có xung đột, vì đây là
  nguồn do chính người dùng xác nhận là đúng.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_custom_glossary(path: str) -> Dict[str, str]:
    """Đọc bộ thuật ngữ từ file JSON. Trả về {} nếu path rỗng, file không tồn tại, hoặc lỗi."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_custom_glossary(path: str, glossary: Dict[str, str]) -> None:
    """Ghi bộ thuật ngữ ra file JSON (ghi đè toàn bộ - khác với Project.glossary vốn merge)."""
    Path(path).write_text(json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8")
