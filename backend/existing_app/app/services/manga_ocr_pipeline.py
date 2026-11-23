"""
Pipeline phát hiện & cắt bóng thoại bằng YOLO + xử lý mask bằng OpenCV, và
bước lọc lỗi OCR + dịch bằng Gemini (text-only, không gửi ảnh).

Port từ script `test.py` do người dùng cung cấp, giữ nguyên thuật toán gốc
(Otsu threshold + contour + watershed để tách bóng thoại dính liền nhau),
chỉ chỉnh sửa để:
- Không dùng biến toàn cục (DEBUG_SAVE_CROPS, DEBUG_DIR) mà truyền qua tham số
- Cache model YOLO theo đường dẫn để không phải load lại mỗi lần dịch 1 trang
  (load YOLO tốn vài giây, dịch nhiều trang liên tiếp sẽ rất chậm nếu load lại)

QUAN TRỌNG: hàm gọi Gemini (`cleanup_and_translate_texts`) dùng ĐÚNG cùng SDK
và cách gọi như `app/services/translation_service.py` (SDK chính thức
`google-genai`, `from google import genai`, bắt lỗi qua
`google.genai.errors.ClientError`) để đồng bộ hành vi (retry 429, log,
thông báo lỗi) giữa các backend - không dùng thư viện `google.generativeai`
(cũ, đã deprecate) hay bất kỳ cách gọi nào khác.

Các thư viện ultralytics/opencv-python/numpy/google-genai chỉ được import LÚC
GỌI HÀM (import trễ), để người dùng không bật backend này vẫn chạy app bình
thường mà không cần cài các thư viện nặng này.
"""
from __future__ import annotations

import difflib
import logging
import os
import threading

from app.services import gemini_models
from typing import List, Optional, Tuple

# Ngưỡng độ giống (0-1, theo difflib) giữa raw OCR và cleaned_text của Gemini.
# Thấp hơn mức này -> khả năng cao Gemini đã diễn giải/viết lại thay vì chỉ
# sửa lỗi OCR đơn thuần -> log cảnh báo để người dùng biết dòng nào cần xem lại.
CLEANUP_FIDELITY_WARNING_THRESHOLD = 0.5

log = logging.getLogger("manga_ocr_pipeline")

# ---------------------------------------------------------------------------
# Prompt + schema cho bước Gemini lọc lỗi OCR + dịch (text-only, không kèm ảnh)
# ---------------------------------------------------------------------------
GEMINI_CLEANUP_TRANSLATE_PROMPT_TEMPLATE = """\
Bạn là một chuyên gia dịch thuật Manga/Comic và làm sạch dữ liệu OCR.
{context_section}
Dưới đây là danh sách các câu thoại thu được từ Manga-OCR theo đúng thứ tự
xuất hiện trên trang truyện (ngôn ngữ nguồn: "{source_lang}"):
{payload_json}

Ý nghĩa các trường phụ: "panel" = số thứ tự khung tranh theo thứ tự đọc (các câu
CÙNG panel thuộc cùng một cảnh/đoạn hội thoại - dùng để suy ra người nói, xưng
hô, đại từ bị lược); "kind" = "balloon" (chữ trong bóng thoại) hoặc "free" (chữ
tự do ngoài bóng: lời dẫn, độc thoại, chữ nhỏ...); "low_confidence" = true nghĩa
là Manga-OCR đọc CHƯA CHẮC câu này.
{vision_section}
Nhiệm vụ của bạn:
1. LỌC VÀ SỬA LỖI OCR - CHỈ sửa LỖI NHẬN DẠNG KÝ TỰ THỰC SỰ RÕ RÀNG, TUYỆT ĐỐI
   KHÔNG được diễn giải lại, viết lại cho "mượt" hơn, đoán ý, hay thêm/bớt bất
   kỳ từ nào so với OCR gốc:
   - CHỈ ĐƯỢC PHÉP sửa: (a) 1-2 ký tự bị Manga-OCR đọc nhầm thành ký tự khác
     có hình dạng gần giống (lỗi nhận dạng mặt chữ); (b) phần chữ bị LẶP LẠI
     y hệt một đoạn của câu trước/sau (do cắt ảnh lem dính 2 bóng thoại); (c)
     1 vài ký tự rác/vô nghĩa rõ ràng không thuộc ngôn ngữ nguồn bị lẫn vào.
   - TUYỆT ĐỐI KHÔNG ĐƯỢC: thêm từ/cụm từ không có trong OCR gốc; đổi trật tự
     từ; viết lại câu theo cách hiểu/suy luận riêng của bạn; "sửa" một câu chỉ
     vì nó nghe lạ hoặc ngữ pháp hơi khác thường - tiếng Nhật/Trung/Hàn trong
     manga thường dùng văn nói, tỉnh lược, hoặc cố ý viết sai ngữ pháp để tạo
     hiệu ứng, KHÔNG được "chuẩn hoá" lại những chỗ đó.
   - NẾU KHÔNG CHẮC CHẮN một đoạn có phải lỗi OCR hay không, GIỮ NGUYÊN y như
     bản OCR gốc, KHÔNG đoán/sửa. cleaned_text PHẢI giống hệt raw OCR trong
     tuyệt đại đa số trường hợp; chỉ khác đi khi có lỗi OCR THỰC SỰ rõ ràng
     như mô tả ở trên.
2. DỊCH sang ngôn ngữ đích "{target_lang}":
   - Dịch tự nhiên, phù hợp văn phong manga/comic, nhưng PHẢI bám sát nghĩa
     của cleaned_text - KHÔNG được thêm tình tiết/chi tiết không có trong câu
     gốc, KHÔNG được bỏ sót ý trong câu gốc, KHÔNG được đoán hộ nhân vật đang
     nghĩ gì ngoài những gì câu chữ thể hiện. Được phép thêm đại từ nhân xưng
     (tôi/em/anh/chị...) hoặc trợ từ cho câu tiếng Việt tự nhiên hơn (vì tiếng
     Nhật/Trung/Hàn hay lược bỏ chủ ngữ), nhưng không được thêm Ý MỚI.
   - Xưng hô nhất quán dựa trên ngữ cảnh toàn trang. Nếu có BẢNG THUẬT NGỮ từ
     các trang trước (xem bên trên nếu có), PHẢI dùng đúng bản dịch đã quy
     ước cho tên riêng/chức danh đó, không tự ý đổi cách dịch khác đi.
3. Sau khi dịch xong, nếu có tên riêng/chức danh/thuật ngữ quan trọng LẶP LẠI
   NHIỀU LẦN mà CHƯA có trong bảng thuật ngữ đã cho, liệt kê vào
   "glossary_updates" kèm bản dịch đã chọn, để trang SAU dùng lại đúng cách
   dịch này. Không liệt kê từ thông thường.
{romaji_instruction}
(Lưu ý: "cleaned_text" CHỈ chứa câu gốc đã làm sạch, không kèm romaji. Phiên âm nằm riêng trong "romanization".)
{honorifics_instruction}

CHỈ trả về JSON đúng theo schema sau, không kèm lời dẫn, không markdown thừa:
{{
  "dialogues": [
    {{"id": <int, giữ nguyên id đầu vào>, "cleaned_text": "<câu gốc đã làm sạch, KHÔNG kèm phiên âm>", "romanization": "<phiên âm Romaji của cleaned_text, hoặc \"\" nếu ngôn ngữ nguồn không phải tiếng Nhật>", "translated_text": "<bản dịch>"}}
  ],
  "glossary_updates": [
    {{"term": "<tên riêng/thuật ngữ gốc MỚI>", "translation": "<bản dịch đã chọn>"}}
  ]
}}
Nếu không có thuật ngữ mới, để "glossary_updates" là mảng rỗng [].
"""

GEMINI_CLEANUP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "dialogues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "cleaned_text": {"type": "string"},
                    "romanization": {"type": "string"},
                    "translated_text": {"type": "string"},
                },
                "required": ["id", "cleaned_text", "romanization", "translated_text"],
            },
        },
        "glossary_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "translation": {"type": "string"},
                },
                "required": ["term", "translation"],
            },
        },
    },
    "required": ["dialogues"],
}


GEMINI_VISION_SECTION_TEMPLATE = """
ẢNH ĐỐI CHIẾU: kèm theo prompt là ảnh crop của các câu có "crop_attached": true
(mỗi ảnh có dòng chữ ghi rõ id đứng ngay trước). CHỈ với những câu này (OCR đọc
chưa chắc), ảnh là NGUỒN SỰ THẬT: đọc lại chữ trong ảnh và đặt cleaned_text = chữ
đọc được từ ảnh (quy tắc "không diễn giải / không viết lại" vẫn áp dụng: chỉ chép
đúng chữ nhìn thấy). Nếu ảnh không có chữ đọc được (hình hiệu ứng, hoạ tiết, nét
tranh) thì cleaned_text = "" và translated_text = "". Với các câu KHÔNG có ảnh đính
kèm, giữ nguyên quy tắc nghiêm ngặt ở mục 1.
"""


def _thinking_level() -> str:
    from app.services import ocr_engine
    return ocr_engine.current().thinking_level or "low"


def build_cleanup_payload(ocr_results: list, rescue_ids: Optional[set] = None) -> list:
    """Danh sách gửi Gemini: id + text + (kind, panel nếu có) + low_confidence / crop_attached khi liên quan."""
    payload = []
    for item in ocr_results:
        entry = {"id": item["id"], "text": item["raw_text"]}
        if item.get("kind"):
            entry["kind"] = item["kind"]
        if item.get("panel", -1) is not None and item.get("panel", -1) >= 0:
            entry["panel"] = item["panel"] + 1
        if item.get("low_conf"):
            entry["low_confidence"] = True
        if rescue_ids and item["id"] in rescue_ids:
            entry["crop_attached"] = True
        payload.append(entry)
    return payload


def pick_rescue_ids(ocr_results: list, max_images: int) -> list:
    """Các câu đọc chưa chắc có ảnh crop, ưu tiên conf thấp nhất, tối đa max_images (giữ chi phí token)."""
    cands = [it for it in ocr_results if it.get("low_conf") and it.get("crop_png")]
    cands.sort(key=lambda it: (it["ocr_conf"] if it.get("ocr_conf") is not None else -1.0))
    return [it["id"] for it in cands[: max(0, max_images)]]


def cleanup_and_translate_texts(
    client,
    model_name: str,
    ocr_results: list,
    source_lang: str,
    target_lang: str,
    context: Optional[dict] = None,
    rescue_images: Optional[dict] = None,
) -> Tuple[dict, dict]:
    """
    Gửi danh sách text đã OCR (KHÔNG kèm ảnh) lên Gemini để lọc lỗi OCR + dịch.

    `client` phải là một `google.genai.Client` đã được khởi tạo sẵn (dùng
    chung instance với nơi gọi, ví dụ HybridOcrTranslationService.client),
    để đảm bảo đồng bộ 100% cách gọi API với các backend Gemini khác trong
    `translation_service.py` (cùng SDK `google-genai`, cùng kiểu bắt lỗi
    `ClientError`/429, cùng cấu hình `thinking_config`).

    context: ngữ cảnh trang trước (glossary + recent_dialogues) - xem docstring
    TranslationService.translate_page để biết cấu trúc đầy đủ.

    Trả về tuple (cleaned_map, glossary_updates):
        cleaned_map = {id: {"id":, "cleaned_text":, "romanization":, "translated_text":}}
        glossary_updates = {"<thuật ngữ gốc>": "<bản dịch>", ...} (thuật ngữ MỚI phát hiện trên trang này)
    """
    import json
    from app.services.json_utils import parse_gemini_json
    from google.genai import errors as genai_errors
    # Import trễ để tránh phụ thuộc vòng khi module này được import trước
    # translation_service.py; đây là lớp lỗi + hàm dùng chung toàn app.
    from app.services.translation_service import (
        TranslationServiceError,
        TranslationRateLimitError,
        build_context_section,
        build_romaji_instruction,
        build_honorifics_instruction,
    )

    rescue_images = rescue_images or {}
    input_payload = build_cleanup_payload(ocr_results, set(rescue_images))
    context_section = build_context_section(context)
    prompt = GEMINI_CLEANUP_TRANSLATE_PROMPT_TEMPLATE.format(
        vision_section=GEMINI_VISION_SECTION_TEMPLATE if rescue_images else "",
        source_lang=source_lang,
        target_lang=target_lang,
        context_section=context_section,
        romaji_instruction=build_romaji_instruction(source_lang),
        honorifics_instruction=build_honorifics_instruction((context or {}).get("keep_honorifics", False)),
        payload_json=json.dumps(input_payload, ensure_ascii=False, indent=2),
    )

    contents: object = prompt
    if rescue_images:
        from google.genai import types
        parts: list = [prompt]
        for rid, png in rescue_images.items():
            parts += [f"Ảnh crop của câu id={rid}:", types.Part.from_bytes(data=png, mime_type="image/png")]
        contents = parts

    log.info("Gửi %d câu OCR lên Gemini để lọc lỗi + dịch (model=%s, %d ảnh đối chiếu)...",
             len(ocr_results), model_name, len(rescue_images))
    try:
        response = gemini_models.generate_content(
            client, require_text=True,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": GEMINI_CLEANUP_RESPONSE_SCHEMA,
                "temperature": 0.2,
                "thinking_config": {"thinking_level": _thinking_level()},
            },
        )
    except genai_errors.ClientError as exc:
        if getattr(exc, "code", None) == 429 or "429" in str(exc):
            raise TranslationRateLimitError(f"Vượt giới hạn Rate Limit: {exc}") from exc
        raise TranslationServiceError(f"Lỗi Gemini API khi lọc/dịch text: {exc}") from exc
    except Exception as exc:
        log.exception("Lỗi không xác định khi gọi Gemini để lọc/dịch text")
        raise TranslationServiceError(f"Lỗi không xác định khi gọi Gemini API: {exc}") from exc

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise TranslationServiceError("Gemini không trả về nội dung khi lọc/dịch text.")

    try:
        data = parse_gemini_json(raw_text, expect_ids=[it["id"] for it in ocr_results])
    except json.JSONDecodeError as exc:
        log.error("JSON không hợp lệ từ bước lọc/dịch text. Raw: %s", raw_text[:500])
        raise TranslationServiceError(f"Gemini trả về JSON không hợp lệ: {exc}") from exc

    cleaned_map = {item["id"]: item for item in data.get("dialogues", []) if "id" in item}

    # Kiểm tra độ trung khớp giữa OCR gốc (Manga-OCR, đáng tin cậy vì là model
    # chuyên dụng) và bản "đã làm sạch" của Gemini - nếu lệch quá nhiều, khả
    # năng cao Gemini đã diễn giải/viết lại thay vì chỉ sửa lỗi OCR như yêu
    # cầu trong prompt. KHÔNG tự động ghi đè lại (Gemini vẫn có thể đúng khi
    # sửa lỗi OCR thực sự nghiêm trọng), chỉ log cảnh báo để người dùng biết
    # dòng nào nên đối chiếu lại thủ công.
    raw_text_by_id = {item["id"]: item["raw_text"] for item in ocr_results}
    for dialogue_id, cleaned_item in cleaned_map.items():
        raw = raw_text_by_id.get(dialogue_id, "")
        cleaned_first_line = "".join(str(cleaned_item.get("cleaned_text", "")).split())   # cleaned_text là văn bản gốc thuần; bỏ xuống dòng / khoảng trắng để so với OCR
        raw = "".join(str(raw).split())
        if raw and cleaned_first_line:
            similarity = difflib.SequenceMatcher(None, raw, cleaned_first_line).ratio()
            if similarity < CLEANUP_FIDELITY_WARNING_THRESHOLD:
                log.warning(
                    "⚠️ Câu id=%s: cleaned_text lệch nhiều so với OCR gốc (độ giống=%.0f%%) "
                    "- có thể Gemini đã diễn giải/viết lại thay vì chỉ sửa lỗi OCR. "
                    "OCR gốc: %r | Đã làm sạch: %r",
                    dialogue_id, similarity * 100, raw, cleaned_first_line,
                )

    glossary_updates = {
        item["term"]: item["translation"]
        for item in data.get("glossary_updates", [])
        if item.get("term") and item.get("translation")
    }
    if glossary_updates:
        log.info("Phát hiện %d thuật ngữ mới: %s", len(glossary_updates), glossary_updates)

    return cleaned_map, glossary_updates

# Cache model YOLO theo đường dẫn file .pt - load 1 lần, dùng lại cho mọi trang/vùng ảnh
_yolo_model_cache: dict = {}

# Hàng đợi dịch chạy nhiều trang song song (mặc định 4 luồng) mà các model local là MỘT đối tượng dùng chung:
#  * nạp model: nếu không khoá, 4 luồng cùng thấy "chưa có" và nạp 4 bản cùng lúc (RAM vọt lên, lần đầu còn cùng tải
#    model về một thư mục cache) -> máy 8 GB dễ bị hệ điều hành đóng app đột ngột;
#  * suy luận: model YOLO (ultralytics) và MangaOcr giữ trạng thái nội bộ, không an toàn khi nhiều luồng gọi cùng lúc.
# Nên: khoá nạp, và tuần tự hoá suy luận (phần gọi Gemini qua mạng vẫn chạy song song nên tốc độ gần như không đổi).
_LOAD_LOCK = threading.RLock()
from app.threading_utils import TORCH_INFER_LOCK as _INFER_LOCK  # khoá chung với LaMa (xem threading_utils)


class _SerializedModel:
    """Bọc model dùng chung: mọi lời gọi suy luận (predict / gọi trực tiếp) xếp hàng qua _INFER_LOCK."""

    def __init__(self, model):
        self._model = model

    def __call__(self, *args, **kwargs):
        with _INFER_LOCK:
            return self._model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with _INFER_LOCK:
            return self._model.predict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def get_yolo_model(model_path: str):
    """Load model YOLO (cache theo đường dẫn để không load lại nhiều lần). An toàn khi gọi từ nhiều luồng."""
    cached = _yolo_model_cache.get(model_path)
    if cached is not None:
        return cached
    with _LOAD_LOCK:
        if model_path not in _yolo_model_cache:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Không tìm thấy file model YOLO tại: {model_path}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "Chưa cài thư viện ultralytics. Chạy: pip install ultralytics"
                ) from exc
            log.info("Đang load model YOLO từ %s...", model_path)
            _yolo_model_cache[model_path] = _SerializedModel(YOLO(model_path))
            log.info("Đã load xong model YOLO.")
        return _yolo_model_cache[model_path]


def get_manga_ocr():
    """Manga-OCR dùng chung toàn app (model theo ocr_engine.configure; mặc định mayocream/manga-ocr). Trả đối tượng gọi được: mocr(PIL) -> str."""
    from app.services import ocr_engine
    return _SerializedModel(ocr_engine.get_mocr())


def get_bubble_boxes(yolo_model, img_path: str, conf: float = 0.35) -> List[List[int]]:
    """Phát hiện các bóng thoại và trả về danh sách [x, y, w, h]."""
    results = yolo_model.predict(source=img_path, conf=conf, verbose=False)
    boxes: List[List[int]] = []

    for result in results:
        for box in result.boxes.xywh:
            x_center, y_center, w, h = box.tolist()
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)
            boxes.append([max(0, x), max(0, y), int(w), int(h)])

    return boxes


def draw_annotated_page(image_path: str, boxes: List[List[int]]) -> bytes:
    """
    Vẽ viền đỏ + số thứ tự lên đúng từng vùng bóng thoại YOLO đã phát hiện
    (theo đúng thứ tự đọc đã sắp xếp), trả về ảnh PNG (bytes) để gửi cho
    Gemini Vision - kỹ thuật "Set-of-Mark prompting": đánh dấu trực quan rõ
    ràng lên ảnh để Gemini biết CHÍNH XÁC vùng nào cần đọc/dịch, KHÔNG cần tự
    dò lại toạ độ (tránh lỗi toạ độ sai khi để Gemini tự xác định box_2d).

    boxes: list [x, y, w, h] theo đúng thứ tự đọc - số thứ tự vẽ lên ảnh sẽ
    là index + 1 (1-based) tương ứng 1-1 với thứ tự này.
    """
    import io
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Cỡ chữ số đủ lớn để Gemini đọc rõ (mặc định PIL không truyền size rất nhỏ)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()  # Pillow cũ hơn không hỗ trợ tham số size

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=max(2, w // 60))
        label = str(idx)
        label_w = 16 + 18 * len(label)
        label_h = 30
        label_y0 = max(0, y - label_h)
        draw.rectangle([x, label_y0, x + label_w, label_y0 + label_h], fill=(255, 0, 0))
        draw.text((x + 5, label_y0 + 2), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sort_reading_order(boxes: List[List[int]], vertical_threshold: int = 100) -> List[List[int]]:
    """
    Sắp xếp bóng thoại theo thứ tự đọc Manga:
    1. Ưu tiên hàng từ trên xuống dưới (y).
    2. Trong cùng một hàng (chênh lệch y < vertical_threshold), sắp xếp từ phải sang trái (-x).
    """
    return sorted(boxes, key=lambda b: (b[1] // vertical_threshold, -b[0]))


# Tỉ lệ diện tích tối thiểu so với cụm chữ chính để 1 component KHÔNG bị coi
# Ngưỡng diện tích: chỉ XEM XÉT xoá những component nhỏ hơn mức này so với
# cụm lớn nhất (lọc sơ bộ ứng viên). Để khá thấp vì quyết định XOÁ THẬT SỰ
# còn phải qua bước kiểm tra an toàn thứ 2 (xem docstring bên dưới).
STRAY_TEXT_AREA_RATIO = 0.15
STRAY_TEXT_DILATE_ITERATIONS = 3
# Mức dãn nở MẠNH hơn hẳn, dùng để KIỂM TRA AN TOÀN: nếu sau khi dãn nở mạnh
# mà ứng viên vẫn KHÔNG dính liền với cụm chữ chính, mới thực sự coi là chữ
# lạc tách biệt. Nếu nó dính liền (dù phải dãn nở mạnh hơn mức 1) nghĩa là nó
# chỉ cách cụm chính 1 khoảng trống hơi lớn (VD: giữa 2 dòng/cột trong CÙNG 1
# bóng thoại) chứ không phải đến từ bóng thoại/khung tranh khác -> PHẢI giữ lại.
STRAY_TEXT_SAFETY_DILATE_ITERATIONS = 14


def remove_stray_text_fragments(
    cropped_bgr,
    dilate_iterations: int = STRAY_TEXT_DILATE_ITERATIONS,
    safety_dilate_iterations: int = STRAY_TEXT_SAFETY_DILATE_ITERATIONS,
):
    """
    Loại bỏ các mảnh chữ/ký tự nằm TÁCH BIỆT THẬT SỰ, cách một khoảng rất rõ
    rệt so với cụm chữ chính trong vùng ảnh đã cắt - thường là chữ bị "lọt"
    từ bóng thoại hoặc khung tranh lân cận vào trong margin của bbox YOLO.

    Thuật toán 2 bước (để tránh xoá nhầm phần chữ hợp lệ bị cắt mất):
      BƯỚC 1 (tìm ứng viên): nhị phân hoá lấy vùng nét chữ, dãn nở NHẸ, tìm
        các component nhỏ hơn hẳn cụm lớn nhất -> đây mới chỉ là ỨNG VIÊN,
        CHƯA xoá vội.
      BƯỚC 2 (xác nhận an toàn): dãn nở MẠNH HƠN NHIỀU. Nếu sau khi dãn nở
        mạnh, ứng viên đó HÒA LÀM MỘT với cụm chữ chính -> nghĩa là nó chỉ
        cách cụm chính 1 khoảng trống hơi lớn (VD: giữa 2 dòng/cột chữ dọc
        trong CÙNG 1 câu/bóng thoại) -> GIỮ NGUYÊN, không xoá. CHỈ xoá khi
        ứng viên đó VẪN tách biệt dù đã dãn nở rất mạnh - lúc này mới đủ tin
        cậy để kết luận đó là chữ lạc đến từ bóng thoại/khung tranh khác.

    An toàn: nếu ảnh chỉ có 1 cụm chữ, hoặc không có component nào tách biệt
    thật sự sau bước 2, hàm trả về ảnh gốc nguyên vẹn - không xoá gì cả.
    """
    import cv2
    import numpy as np

    if cropped_bgr is None or cropped_bgr.size == 0:
        return cropped_bgr

    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY) if cropped_bgr.ndim == 3 else cropped_bgr.copy()

    # Nhị phân hoá ngược: nét chữ (tối) -> 255, nền bóng thoại (sáng) -> 0
    _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if cv2.countNonZero(text_mask) == 0:
        return cropped_bgr  # không có nét chữ nào (ảnh trắng/trống) -> bỏ qua

    kernel = np.ones((3, 3), np.uint8)

    # -- BƯỚC 1: tìm ứng viên (dãn nở nhẹ) --------------------------------------
    dilated = cv2.dilate(text_mask, kernel, iterations=dilate_iterations)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n_labels <= 2:
        return cropped_bgr  # chỉ có 1 cụm chữ (+ nền) -> không có gì để lọc

    areas = stats[1:, cv2.CC_STAT_AREA]  # bỏ qua label 0 (nền)
    main_label = 1 + int(np.argmax(areas))
    main_area = float(areas[main_label - 1])

    candidate_labels = [
        label for label in range(1, n_labels)
        if label != main_label and stats[label, cv2.CC_STAT_AREA] < main_area * STRAY_TEXT_AREA_RATIO
    ]
    if not candidate_labels:
        return cropped_bgr  # không có ứng viên nào nhỏ hơn hẳn cụm chính

    # -- BƯỚC 2: dãn nở MẠNH để xác nhận ứng viên có thực sự tách biệt ------------
    heavy_dilated = cv2.dilate(text_mask, kernel, iterations=safety_dilate_iterations)
    n_labels_h, labels_h, _, _ = cv2.connectedComponentsWithStats(heavy_dilated, connectivity=8)

    # Cụm chính (label ở bước 1) tương ứng với label nào bên bản đồ dãn nở mạnh
    main_mask_step1 = (labels == main_label)
    heavy_main_label = int(np.bincount(labels_h[main_mask_step1]).argmax()) if main_mask_step1.any() else 0

    result = cropped_bgr.copy()
    removed_count = 0
    for label in candidate_labels:
        candidate_mask_step1 = (labels == label)
        # Label tương ứng của ứng viên này trên bản đồ dãn nở MẠNH
        heavy_candidate_label = int(np.bincount(labels_h[candidate_mask_step1]).argmax())
        if heavy_candidate_label == heavy_main_label:
            # Sau khi dãn nở mạnh đã hoà vào cụm chính -> chỉ là khoảng cách
            # bình thường giữa các dòng/cột trong CÙNG bóng thoại -> GIỮ LẠI
            continue
        # Vẫn tách biệt dù đã dãn nở rất mạnh -> thực sự là chữ lạc -> xoá
        result[candidate_mask_step1] = 255
        removed_count += 1

    if removed_count:
        log.info(
            "Đã loại bỏ %d mảnh chữ lạc (xác nhận tách biệt thật sự qua 2 bước kiểm tra)",
            removed_count,
        )

    return result


def extract_bubble_mask(
    cv_img,
    x: int,
    y: int,
    w: int,
    h: int,
    margin: int = 15,
    debug_dir: Optional[str] = None,
    debug_id: Optional[int] = None,
    remove_stray_fragments: bool = False,
):
    """
    Thay vì crop nguyên bbox chữ nhật (dễ dính chữ của bong bóng bên cạnh),
    hàm này:
      1. Crop vùng ROI rộng hơn bbox gốc một chút (có margin).
      2. Threshold để tách vùng sáng (bong bóng) khỏi nền/viền tối.
      3. Tìm contour bao quanh điểm tâm bbox gốc -> đó chính là bong bóng thật.
      4. Tô trắng (xóa) mọi pixel NẰM NGOÀI contour đó (chữ của bong bóng
         hàng xóm bị lem vào bbox sẽ biến mất).
      5. Crop lại theo bounding rect của contour để loại viền thừa.

    debug_dir/debug_id: nếu cả 2 được truyền, lưu ảnh mask/roi/result để debug.

    remove_stray_fragments: MẶC ĐỊNH TẮT (False). Khi bật, chạy thêm bước lọc
    ký tự/mảnh chữ lạc tách biệt (xem remove_stray_text_fragments()) - đã ghi
    nhận NHIỀU LẦN xoá nhầm chữ hợp lệ khi khoảng cách giữa các cột chữ dọc
    trong cùng 1 bóng thoại lớn hơn mức an toàn, nên mặc định KHÔNG chạy bước
    này trừ khi người dùng chủ động bật trong Settings.
    """
    import cv2
    import numpy as np

    img_h, img_w = cv_img.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(img_w, x + w + margin)
    y1 = min(img_h, y + h + margin)

    roi = cv_img[y0:y1, x0:x1]
    if roi.size == 0:
        return cv_img[y:y + h, x:x + w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()

    # Otsu threshold: tự động chọn ngưỡng sáng/tối phù hợp cho từng ảnh,
    # robust hơn ngưỡng cố định khi nền có hoa văn/độ tương phản khác nhau.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Nếu Otsu "đoán ngược" (nền được coi là sáng nhiều hơn bong bóng),
    # đảo lại để đảm bảo bong bóng luôn là vùng giá trị 255.
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Đóng các khe hở nhỏ trên viền bong bóng (nét vẽ tay không liền mạch)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Tâm của bbox gốc (do YOLO trả về), quy đổi sang toạ độ trong roi
    cx = (x - x0) + w // 2
    cy = (y - y0) + h // 2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return cv_img[y:y + h, x:x + w]

    target = None
    for c in contours:
        if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0:
            target = c
            break

    if target is None:
        target = max(contours, key=cv2.contourArea)

    # Nếu contour tìm được LỚN HƠN NHIỀU so với bbox gốc -> khả năng cao là
    # 2 bong bóng đang dính liền nhau (contour gộp cả 2). Dùng watershed để
    # tách chúng ra dựa trên khoảng cách tới viền (2 bong bóng tròn dính nhau
    # sẽ có 2 "đỉnh" cực đại riêng biệt trong distance transform).
    contour_area = cv2.contourArea(target)
    bbox_area = w * h
    if contour_area > 1.8 * bbox_area:
        binary_fg = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(binary_fg, [target], -1, 255, thickness=cv2.FILLED)

        dist = cv2.distanceTransform(binary_fg, cv2.DIST_L2, 5)
        # Ngưỡng lấy các "tâm" bong bóng (local maxima vùng) làm marker cho watershed
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        n_markers, markers = cv2.connectedComponents(sure_fg)

        if n_markers > 2:  # >2 nghĩa là có từ 2 vùng tâm trở lên (1 là nền)
            sure_bg = cv2.dilate(binary_fg, kernel, iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            roi_bgr = roi if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(roi_bgr, markers)

            # Chọn marker (vùng) nào chứa tâm bbox gốc -> đó là bong bóng đang cần
            marker_at_center = markers[cy, cx] if 0 <= cy < markers.shape[0] and 0 <= cx < markers.shape[1] else -1

            if marker_at_center > 1:
                split_mask = np.uint8(markers == marker_at_center) * 255
                contours2, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours2:
                    target = max(contours2, key=cv2.contourArea)

    # Bỏ qua nếu contour tìm được quá nhỏ so với bbox gốc (khả năng cao là nhiễu)
    if cv2.contourArea(target) < 0.15 * (w * h):
        return cv_img[y:y + h, x:x + w]

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [target], -1, 255, thickness=cv2.FILLED)

    # Nới mask ra vài pixel để không cắt mất nét chữ sát viền bong bóng
    mask = cv2.dilate(mask, kernel, iterations=1)

    result = roi.copy()
    result[mask == 0] = 255  # tô trắng mọi thứ ngoài bong bóng thật

    x_c, y_c, w_c, h_c = cv2.boundingRect(target)
    # Padding nhỏ quanh contour để manga-ocr không bị cắt sát chữ
    pad = 4
    x_c0 = max(0, x_c - pad)
    y_c0 = max(0, y_c - pad)
    x_c1 = min(result.shape[1], x_c + w_c + pad)
    y_c1 = min(result.shape[0], y_c + h_c + pad)

    cropped = result[y_c0:y_c1, x_c0:x_c1]

    if cropped.size == 0:
        return cv_img[y:y + h, x:x + w]

    # Lọc hậu kỳ (TẮT MẶC ĐỊNH - xem docstring): xoá các ký tự/mảnh chữ lạc
    # nằm tách biệt, cách xa cụm chữ chính trong vùng vừa cắt.
    if remove_stray_fragments:
        cropped = remove_stray_text_fragments(cropped)

    if debug_dir and debug_id is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_id:02d}_roi.png"), roi)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_id:02d}_mask.png"), mask)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_id:02d}_result.png"), cropped)

    return cropped


# ---------------------------------------------------------------------------
# OCR một trang: đo độ tin cậy, đọc lại bằng biến thể crop khác khi chưa chắc, chuẩn bị ảnh đối chiếu cho Gemini
# ---------------------------------------------------------------------------
def _bgr_to_pil(bgr):
    import cv2
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _png_bytes(bgr) -> Optional[bytes]:
    import cv2
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else None


def ocr_best(makers, cfg=None) -> dict:
    """makers: các hàm không tham số trả crop BGR, theo thứ tự thử. Dừng ở crop đầu tiên đọc chắc; nếu không có thì lấy kết quả điểm cao nhất.
    Trả {text, conf, low_conf, tries}."""
    from app.services import ocr_engine as oe
    cfg = cfg or oe.current()
    best = None
    tries = 0
    for mk in makers:
        crop = mk()
        if crop is None or crop.size == 0:
            continue
        tries += 1
        res = oe.read(_bgr_to_pil(crop), cfg)
        text = oe.normalize_ocr_text(res.text)
        score = oe.quality_score(text, res.conf, cfg.low_conf)
        if best is None or score > best["score"]:
            best = {"text": text, "conf": res.conf, "score": score}
        if not oe.is_low_quality(text, res.conf, cfg.low_conf):
            break
    if best is None:
        return {"text": "", "conf": None, "low_conf": True, "tries": 0}
    low = oe.is_low_quality(best["text"], best["conf"], cfg.low_conf)
    return {"text": best["text"], "conf": best["conf"], "low_conf": low, "tries": tries}


def ocr_units(cv_img, units, cfg=None) -> List[dict]:
    """Đơn vị koharu -> [{id, box, raw_text, kind, panel, ocr_conf, low_conf, crop_png?}] theo đúng thứ tự `units`."""
    from app.services import koharu_detector as kd
    from app.services import ocr_engine as oe
    cfg = cfg or oe.current()
    results: List[dict] = []
    for idx, u in enumerate(units, start=1):
        r = ocr_best([lambda u=u, v=v: kd.crop_for_ocr(cv_img, u, variant=v) for v in ("default", "text_only", "wide")], cfg)
        item = {"id": idx, "box": u.box, "raw_text": r["text"], "kind": u.kind, "panel": u.frame,
                "ocr_conf": r["conf"], "low_conf": r["low_conf"]}
        if r["low_conf"] and cfg.gemini_rescue:
            item["crop_png"] = _png_bytes(kd.crop_for_ocr(cv_img, u, variant="wide"))
        log.info("OCR #%d %s conf=%s tries=%d low=%s: %s", idx, u.kind,
                 "n/a" if r["conf"] is None else f"{r['conf']:.2f}", r["tries"], r["low_conf"], r["text"])
        results.append(item)
    return results


def ocr_yolo_boxes(cv_img, boxes, remove_stray_fragments: bool = False, cfg=None) -> List[dict]:
    """Đường cũ (không koharu): hộp YOLO [x, y, w, h] đã sắp thứ tự -> cùng định dạng như ocr_units."""
    from app.services import ocr_engine as oe
    cfg = cfg or oe.current()
    results: List[dict] = []
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        makers = [
            lambda: extract_bubble_mask(cv_img, x, y, w, h, margin=15, remove_stray_fragments=remove_stray_fragments),
            lambda: cv_img[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12],
        ]
        r = ocr_best(makers, cfg)
        item = {"id": idx, "box": [x, y, w, h], "raw_text": r["text"], "kind": "balloon", "panel": -1,
                "ocr_conf": r["conf"], "low_conf": r["low_conf"]}
        if r["low_conf"] and cfg.gemini_rescue:
            item["crop_png"] = _png_bytes(makers[1]())
        results.append(item)
    return results
