"""
Tính năng "Tra từ điển / Phân tích ngữ pháp": chọn 1 câu gốc trong bảng thoại,
phân tích nghĩa từng từ + cấu trúc ngữ pháp của câu, hỗ trợ học ngôn ngữ.

Độc lập với backend dịch đang chọn (YOLO+OCR+Gemini / YOLO+Gemini Vision) -
luôn gọi trực tiếp Gemini text-only, vì đây là tác vụ phân tích ngôn ngữ dựa
trên 1 câu text đã có sẵn, không cần OCR/vision. Dùng chung lớp lỗi với
`translation_service.py` (TranslationServiceError/TranslationRateLimitError)
để đồng bộ cách báo lỗi trong toàn app.
"""
from __future__ import annotations

import json
import logging

from app.services import gemini_models

log = logging.getLogger("grammar_analysis")

GRAMMAR_ANALYSIS_PROMPT_TEMPLATE = """\
Bạn là giáo viên ngôn ngữ, hỗ trợ người học phân tích câu sau đây (ngôn ngữ
nguồn: "{source_lang}"), để giải thích bằng tiếng Việt ("{target_lang}"):

"{sentence}"

Nhiệm vụ:
1. TÁCH TỪ: chia câu thành các từ/cụm từ có nghĩa theo đúng thứ tự xuất hiện
   trong câu (không bỏ sót trợ từ, đuôi chia động từ...). Với MỖI từ/cụm từ:
   - "word": từ gốc đúng như trong câu
   - "reading": cách đọc theo hệ chữ bản ngữ (furigana dạng hiragana nếu là
     tiếng Nhật có kanji; pinyin có dấu thanh nếu tiếng Trung; để trống "" nếu
     ngôn ngữ dùng chữ Latin hoặc từ đó đã thuần hiragana)
   - "romaji": phiên âm La-tinh hoá của từ đó (Romaji hệ Hepburn nếu tiếng
     Nhật; Pinyin không dấu/La-tinh nếu tiếng Trung; để trống "" nếu ngôn ngữ
     nguồn vốn đã dùng chữ Latin)
   - "part_of_speech": loại từ bằng tiếng Việt (danh từ, động từ, tính từ,
     trợ từ, trợ động từ, liên từ, thán từ, đại từ, v.v. - ghi ngắn gọn)
   - "meaning": nghĩa tiếng Việt của riêng từ/cụm từ đó trong ngữ cảnh câu này
2. GIẢI THÍCH NGỮ PHÁP: viết 1 đoạn ngắn (3-6 câu) giải thích cấu trúc ngữ
   pháp tổng thể của câu - mẫu câu đang dùng, điểm ngữ pháp đáng chú ý (chia
   động từ, kính ngữ, thể phủ định/nghi vấn, trợ từ nối, v.v.), và vì sao câu
   lại mang nghĩa như bản dịch. Giải thích dễ hiểu cho người đang học ngôn ngữ.

CHỈ trả về JSON đúng theo schema sau, không thêm bất kỳ văn bản nào khác:
{{
  "words": [
    {{"word": "...", "reading": "...", "romaji": "...", "part_of_speech": "...", "meaning": "..."}}
  ],
  "grammar_notes": "<đoạn giải thích ngữ pháp>"
}}
"""

GRAMMAR_ANALYSIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "reading": {"type": "string"},
                    "romaji": {"type": "string"},
                    "part_of_speech": {"type": "string"},
                    "meaning": {"type": "string"},
                },
                "required": ["word", "part_of_speech", "meaning"],
            },
        },
        "grammar_notes": {"type": "string"},
    },
    "required": ["words", "grammar_notes"],
}


def analyze_sentence(
    api_key: str,
    model_name: str,
    sentence: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    """
    Gọi Gemini để phân tích 1 câu: tách từ (kèm cách đọc, loại từ, nghĩa) +
    giải thích cấu trúc ngữ pháp. Trả về {"words": [...], "grammar_notes": "..."}.

    Dùng cùng SDK `google-genai` và cùng cách bắt lỗi như phần còn lại của
    `translation_service.py` để đồng bộ hành vi (retry 429, thông báo lỗi).
    """
    from google import genai
    from google.genai import errors as genai_errors
    from app.services.translation_service import TranslationServiceError, TranslationRateLimitError

    if not api_key:
        raise TranslationServiceError("Thiếu Gemini API Key. Vào Settings (⚙️) để nhập API Key.")
    if not sentence or not sentence.strip():
        raise TranslationServiceError("Câu gốc đang trống, không có gì để phân tích.")

    client = genai.Client(api_key=api_key)
    prompt = GRAMMAR_ANALYSIS_PROMPT_TEMPLATE.format(
        source_lang=source_lang, target_lang=target_lang, sentence=sentence.strip()
    )

    log.info("Gửi câu lên Gemini để phân tích ngữ pháp (model=%s): %r", model_name, sentence[:50])
    try:
        response = gemini_models.generate_content(
            client,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": GRAMMAR_ANALYSIS_RESPONSE_SCHEMA,
                "temperature": 0.2,
                "thinking_config": {"thinking_level": "high"},
            },
        )
    except genai_errors.ClientError as exc:
        if getattr(exc, "code", None) == 429 or "429" in str(exc):
            raise TranslationRateLimitError(f"Vượt giới hạn Rate Limit: {exc}") from exc
        raise TranslationServiceError(f"Lỗi Gemini API khi phân tích câu: {exc}") from exc
    except Exception as exc:
        log.exception("Lỗi không xác định khi phân tích ngữ pháp")
        raise TranslationServiceError(f"Lỗi không xác định khi gọi Gemini API: {exc}") from exc

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise TranslationServiceError("Gemini không trả về nội dung phân tích.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        log.error("JSON không hợp lệ từ bước phân tích ngữ pháp. Raw: %s", raw_text[:500])
        raise TranslationServiceError(f"Gemini trả về JSON không hợp lệ: {exc}") from exc

    return data
