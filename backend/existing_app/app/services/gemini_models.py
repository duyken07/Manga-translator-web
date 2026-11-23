"""Model Gemini cố định của app: model chính + model phụ tự động dùng khi model chính gặp lỗi.

Người dùng không còn chọn model (bỏ ô chọn trong Thiết lập) -> mọi lệnh gọi `generate_content` đi qua `generate_content()` ở đây.
Model chính lỗi (429 / 5xx / 404 / trả rỗng...) -> gọi ngay model phụ với cùng nội dung. Chỉ KHÔNG chuyển sang model phụ khi lỗi do API key
(sai key thì model nào cũng sai). Cả hai cùng lỗi -> ném lỗi của model phụ để các lớp phía trên xử lý (429 -> backoff / thử lại như cũ).
"""
from __future__ import annotations

import logging

PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

log = logging.getLogger(__name__)

_KEY_ERROR_MARKERS = ("api key not valid", "api_key_invalid", "api key expired", "api key must be set")


class EmptyResponseError(RuntimeError):
    """Model trả về nhưng không có nội dung chữ."""


def is_key_error(exc: BaseException) -> bool:
    """Lỗi do API key (sai / hết hạn / thiếu): đổi model không giúp được gì."""
    if getattr(exc, "code", None) == 401:
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _KEY_ERROR_MARKERS)


def generate_content(client, *, require_text: bool = False, **kwargs):
    """client.models.generate_content với model chính, lỗi thì thử model phụ. `model=` do người gọi truyền sẽ bị bỏ qua.
    require_text=True: phản hồi không có chữ cũng coi là lỗi (để thử model phụ)."""
    kwargs.pop("model", None)
    try:
        response = client.models.generate_content(model=PRIMARY_MODEL, **kwargs)
        if require_text and not getattr(response, "text", None):
            raise EmptyResponseError("model chính trả về nội dung rỗng")
        return response
    except Exception as exc:  # noqa: BLE001
        if is_key_error(exc):
            raise
        log.warning("Model chính %s lỗi (%s: %s) -> thử model phụ %s", PRIMARY_MODEL, type(exc).__name__, str(exc)[:200], FALLBACK_MODEL)
    response = client.models.generate_content(model=FALLBACK_MODEL, **kwargs)
    log.info("Đã dùng model phụ %s", FALLBACK_MODEL)
    return response
