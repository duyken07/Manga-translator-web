"""
Kiểm tra cấu hình nhân vật (dữ liệu JSON thô). Trả về DANH SÁCH lỗi tiếng Việt có thể hiện thẳng cho người dùng; rỗng = hợp lệ.
Không phụ thuộc Qt.

Quy tắc chính: ID chỉ gồm chữ thường, số, gạch dưới (vì ID là tên thư mục nên cũng chặn luôn path traversal); `avatar` chỉ là tên file,
không chứa đường dẫn; nhân vật 'roleplay' phải có persona hoặc system_prompt; không lưu API key trong profile.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.characters.character_model import AVATAR_EXTENSIONS, ID_RE, VALID_MODES, MODE_ROLEPLAY

MAX_NAME = 80
MAX_DESC = 200
MAX_PERSONA = 6000
MAX_STYLE = 3000
MAX_SCENARIO = 2000
MAX_BOUNDARY = 2000
MAX_SYSTEM_PROMPT = 8000
MAX_GREETING = 500
MAX_EXAMPLES = 12
MAX_EXAMPLE_CHARS = 300
MAX_STARTERS = 6
_SECRET_KEYS = ("api_key", "apikey", "api-key", "secret", "token")


def is_valid_id(cid: Any) -> bool:
    return isinstance(cid, str) and bool(ID_RE.match(cid))


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_text(data: Dict[str, Any], key: str, label: str, limit: int, errors: List[str]) -> None:
    v = data.get(key, "")
    if v is None:
        return
    if not isinstance(v, str):
        errors.append(f"'{label}' phải là chuỗi văn bản.")
    elif len(v) > limit:
        errors.append(f"'{label}' quá dài ({len(v)} ký tự, tối đa {limit}).")


def _find_secret_key(obj: Any) -> bool:
    if isinstance(obj, dict):
        return any(str(k).lower() in _SECRET_KEYS or _find_secret_key(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_find_secret_key(x) for x in obj)
    return False


def validate_character_data(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return ["Cấu hình nhân vật phải là một đối tượng JSON."]
    errors: List[str] = []

    cid = data.get("id")
    if cid is None or (isinstance(cid, str) and not cid.strip()):
        errors.append("ID nhân vật không được để trống.")
    elif not is_valid_id(cid):
        errors.append("ID nhân vật chỉ gồm chữ thường (a-z), số và dấu gạch dưới, tối đa 48 ký tự (ví dụ: hikawa_hina).")

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("Tên nhân vật không được để trống.")
    elif len(name) > MAX_NAME:
        errors.append(f"Tên nhân vật quá dài (tối đa {MAX_NAME} ký tự).")

    mode = data.get("mode", MODE_ROLEPLAY)
    if mode not in VALID_MODES:
        errors.append(f"Chế độ (mode) phải là một trong: {', '.join(VALID_MODES)}.")

    for key in ("enabled", "is_default"):
        if key in data and not isinstance(data[key], bool):
            errors.append(f"'{key}' phải là true hoặc false.")
    if "schema_version" in data and (not isinstance(data["schema_version"], int) or isinstance(data["schema_version"], bool)):
        errors.append("'schema_version' phải là số nguyên.")

    _check_text(data, "display_name", "Tên hiển thị phụ", MAX_NAME, errors)
    _check_text(data, "description", "Mô tả", MAX_DESC, errors)
    _check_text(data, "persona", "Persona", MAX_PERSONA, errors)
    _check_text(data, "speech_style", "Cách nói", MAX_STYLE, errors)
    _check_text(data, "scenario", "Bối cảnh", MAX_SCENARIO, errors)
    _check_text(data, "knowledge_boundary", "Giới hạn kiến thức", MAX_BOUNDARY, errors)
    _check_text(data, "system_prompt", "System prompt", MAX_SYSTEM_PROMPT, errors)
    _check_text(data, "greeting", "Lời chào", MAX_GREETING, errors)

    if mode == MODE_ROLEPLAY:
        has_text = any(isinstance(data.get(k), str) and data.get(k, "").strip() for k in ("persona", "system_prompt"))
        if not has_text:
            errors.append("Nhân vật nhập vai cần có Persona hoặc System prompt.")

    avatar = data.get("avatar", "")
    if avatar not in (None, ""):
        if not isinstance(avatar, str):
            errors.append("'avatar' phải là tên file.")
        elif "/" in avatar or "\\" in avatar or ".." in avatar or ":" in avatar:
            errors.append("'avatar' chỉ được là tên file (vd avatar.png), không chứa đường dẫn.")
        elif not avatar.lower().endswith(AVATAR_EXTENSIONS):
            errors.append("Avatar phải là ảnh: " + ", ".join(AVATAR_EXTENSIONS) + ".")

    ex = data.get("style_examples", [])
    if ex is not None:
        if not isinstance(ex, list):
            errors.append("'style_examples' phải là danh sách.")
        else:
            if len(ex) > MAX_EXAMPLES:
                errors.append(f"Quá nhiều câu mẫu giọng nói (tối đa {MAX_EXAMPLES}).")
            for i, item in enumerate(ex, 1):
                if not isinstance(item, dict) or not item or not all(isinstance(v, str) for v in item.values()):
                    errors.append(f"Câu mẫu #{i} phải là đối tượng gồm các chuỗi (vd {{\"ja\": \"...\", \"vi\": \"...\"}}).")
                elif any(len(v) > MAX_EXAMPLE_CHARS for v in item.values()):
                    errors.append(f"Câu mẫu #{i} quá dài (tối đa {MAX_EXAMPLE_CHARS} ký tự mỗi trường).")

    st = data.get("starters", [])
    if st is not None:
        if not isinstance(st, list):
            errors.append("'starters' phải là danh sách.")
        else:
            if len(st) > MAX_STARTERS:
                errors.append(f"Quá nhiều nút gợi ý (tối đa {MAX_STARTERS}).")
            for i, item in enumerate(st, 1):
                if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"].strip() \
                        or not isinstance(item.get("text"), str) or not item["text"].strip():
                    errors.append(f"Nút gợi ý #{i} cần có 'label' và 'text' không rỗng.")
                elif "send" in item and not isinstance(item["send"], bool):
                    errors.append(f"Nút gợi ý #{i}: 'send' phải là true hoặc false.")

    model = data.get("model", {})
    if model is not None:
        if not isinstance(model, dict):
            errors.append("'model' phải là một đối tượng.")
        else:
            prov = model.get("provider", "gemini")
            if prov != "gemini":
                errors.append("Hiện chỉ hỗ trợ provider 'gemini'.")
            mn = model.get("model_name", "")
            if mn is not None and not isinstance(mn, str):
                errors.append("'model.model_name' phải là chuỗi.")
            if "temperature" in model and model["temperature"] is not None:
                t = model["temperature"]
                if not _is_num(t) or not 0.0 <= float(t) <= 2.0:
                    errors.append("'model.temperature' phải là số từ 0 đến 2.")

    if _find_secret_key(data):
        errors.append("Không lưu API Key / khóa bí mật trong profile.json.")
    return errors
