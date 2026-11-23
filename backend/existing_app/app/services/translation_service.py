"""
Lớp trừu tượng cho dịch vụ dịch thuật (OCR + Translate 1 trang ảnh).

Thiết kế theo Strategy Pattern để dễ dàng chuyển từ Mock sang Gemini API
thật mà không cần sửa bất kỳ đâu ở UI / Queue Manager.

Đã tích hợp Gemini API thật qua SDK chính thức `google-genai`
(https://pypi.org/project/google-genai/, import `from google import genai`).
Để dùng:
    1. `pip install google-genai` (bỏ comment trong requirements.txt)
    2. Nhập API Key thật ở màn hình Settings và bỏ tick "Dùng chế độ Mock"
"""
from __future__ import annotations

import os
import json
import logging
import mimetypes
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from PIL import Image

from app.models import Dialogue, Page
from app.services import gemini_models
from app.services.json_utils import parse_gemini_json

# google-genai log 1 cảnh báo vô hại mỗi lần gọi generate_content() trực tiếp
# (khuyên dùng Chat.send_message thay thế) - ẩn đi để log gọn hơn, không ảnh
# hưởng chức năng vì ta không dùng tính năng function-calling.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

log = logging.getLogger("translation_service")


def build_context_section(context: Optional[dict]) -> str:
    """
    Dựng đoạn text ngữ cảnh (glossary + vài câu thoại cuối trang trước) để
    chèn vào prompt, dùng chung cho cả GeminiTranslationService lẫn
    manga_ocr_pipeline.cleanup_and_translate_texts() - đảm bảo 2 backend
    diễn giải "ngữ cảnh trang trước" theo đúng 1 định dạng nhất quán.

    context (xem docstring TranslationService.translate_page để biết cấu trúc).
    Trả về chuỗi rỗng nếu không có context hoặc context rỗng.
    """
    if not context:
        return ""

    parts: List[str] = []

    glossary = context.get("glossary") or {}
    if glossary:
        glossary_lines = "\n".join(f'- "{term}" → "{translation}"' for term, translation in glossary.items())
        parts.append(
            "BẢNG THUẬT NGỮ ĐÃ DÙNG Ở CÁC TRANG TRƯỚC (giữ NGUYÊN cách dịch "
            "này nếu gặp lại đúng thuật ngữ, để nhất quán xuyên suốt truyện):\n"
            f"{glossary_lines}"
        )

    recent_dialogues = context.get("recent_dialogues") or []
    if recent_dialogues:
        recent_lines = "\n".join(
            f'- "{d.get("original_text", "")}" → "{d.get("translated_text", "")}"' for d in recent_dialogues
        )
        parts.append(
            "MỘT SỐ CÂU THOẠI CUỐI CÙNG CỦA TRANG NGAY TRƯỚC (chỉ để tham khảo "
            "mạch truyện, xưng hô, văn phong đang dùng - KHÔNG dịch lại các câu này):\n"
            f"{recent_lines}"
        )

    if not parts:
        return ""

    return "\n\n" + "\n\n".join(parts) + "\n"


def build_honorifics_instruction(keep_honorifics: bool) -> str:
    """
    Khi bật (người dùng chọn "Giữ nguyên hậu tố" trong Bộ thuật ngữ), trả về
    hướng dẫn giữ nguyên hậu tố xưng hô (-san, -chan, -kun, -senpai, -sama...)
    thay vì dịch/bỏ đi. Trả về chuỗi rỗng nếu tắt (hành vi mặc định trước đây:
    để Gemini tự quyết định dịch hậu tố sang tiếng Việt hoặc bỏ đi tuỳ ngữ cảnh).
    """
    if not keep_honorifics:
        return ""
    return (
        "\nGIỮ NGUYÊN HẬU TỐ XƯNG HÔ - QUAN TRỌNG: khi dịch tên riêng có kèm hậu "
        "tố xưng hô của ngôn ngữ nguồn (VD tiếng Nhật: -san, -chan, -kun, "
        "-senpai, -sama, -sensei...), GIỮ NGUYÊN hậu tố đó y như bản gốc "
        "(viết liền sau tên, ngăn cách bằng dấu gạch nối), TUYỆT ĐỐI KHÔNG dịch "
        "hậu tố sang tiếng Việt (vd không dịch thành be, anh, chi, tien boi...) "
        "và KHÔNG được bỏ hậu tố đi. Ví dụ đúng: chữ Nhật co dạng ...chan dịch "
        "thành Sayo-chan (giữ nguyên hậu tố -chan), không dịch thành be Sayo "
        "hay chỉ Sayo.\n"
    )


def build_romaji_instruction(source_lang: str) -> str:
    """
    Hướng dẫn về field "romanization" (phiên âm) trong JSON trả về. Phiên âm là field RIÊNG, tách khỏi
    original_text (original_text chỉ chứa văn bản gốc thật, không lẫn phiên âm). App tự ghép hai field thành
    "câu gốc\\nphiên âm" khi hiển thị. Dùng chung cho mọi prompt đọc văn bản gốc (Gemini lẫn manga_ocr_pipeline).
    Ngôn ngữ nguồn không phải tiếng Nhật: yêu cầu để "romanization" là chuỗi rỗng.
    """
    if source_lang != "ja":
        return (
            "\nTRƯỜNG \"romanization\": luôn để chuỗi rỗng \"\" (chỉ dùng cho tiếng Nhật).\n"
        )
    return (
        "\nNGÔN NGỮ NGUỒN LÀ TIẾNG NHẬT - QUAN TRỌNG: ngoài văn bản gốc, với mỗi câu hãy điền thêm "
        "field \"romanization\" = phiên âm Romaji (hệ Hepburn, chữ thường, có dấu gạch nối giữa các "
        "âm tiết của tên riêng nếu cần cho dễ đọc) của TOÀN BỘ câu gốc đó. Mỗi field chứa ĐÚNG phần của "
        "nó: field văn bản gốc CHỈ chứa chữ gốc (tuyệt đối KHÔNG chèn romaji vào đó, không xuống dòng thêm "
        "romaji), còn romaji CHỈ nằm trong \"romanization\". Romaji viết LIỀN trên MỘT dòng, kể cả khi "
        "câu gốc có nhiều dòng / nhiều cột chữ. Ví dụ đúng: văn bản gốc \"りみちゃん…！\", romanization "
        "\"Rimi-chan...!\". Áp dụng cho MỌI câu, không bỏ sót câu nào.\n"
    )


# Prompt hệ thống dùng khi gọi Gemini thật (giữ sẵn để dùng sau này)
GEMINI_SYSTEM_PROMPT_TEMPLATE = """\
Bạn là công cụ OCR và dịch thuật manga/comic chuyên nghiệp, có khả năng phân
tích bố cục trang truyện tranh (panel layout) với độ chính xác cao.
{context_section}
QUY TRÌNH BẮT BUỘC (thực hiện tuần tự):
BƯỚC 1 - Xác định các KHUNG TRANH (panel) trên trang, theo đúng thứ tự đọc
manga Nhật Bản: khung ở trên cùng bên PHẢI đọc trước, sau đó đọc sang TRÁI
trong cùng hàng khung, rồi mới xuống hàng khung DƯỚI. Đây là thứ tự panel,
KHÔNG phải thứ tự toạ độ pixel thô của từng bóng thoại trên toàn trang.

BƯỚC 2 - Trong TỪNG khung tranh (theo đúng thứ tự đã xác định ở Bước 1),
liệt kê TẤT CẢ bóng thoại (speech bubble), hộp thoại tường thuật (narration
box), và đoạn chữ rời có nghĩa xuất hiện bên trong khung đó, theo thứ tự đọc
tự nhiên của khung (thường là phải sang trái, trên xuống dưới trong nội bộ
khung). Đánh số dialogue_id tăng dần 1, 2, 3... nối tiếp qua các khung theo
đúng thứ tự Bước 1 rồi Bước 2 - KHÔNG được nhảy lộn xộn giữa các khung.

BƯỚC 3 - Với mỗi bóng thoại:
- original_text PHẢI là bản ghi CHÍNH XÁC những ký tự nhìn thấy trong ảnh,
  đọc đúng những gì THỰC SỰ hiển thị. TUYỆT ĐỐI KHÔNG được tự diễn giải, đoán
  ý, viết lại cho "hợp lý" hơn, hay bịa thêm chữ nếu một phần khó đọc - nếu
  không chắc chắn đọc được chính xác, hãy đọc kỹ lại ảnh thêm lần nữa thay vì
  đoán theo suy luận riêng.
- translated_text PHẢI bám sát nghĩa của original_text - KHÔNG được thêm
  tình tiết/chi tiết không có trong câu gốc, KHÔNG được bỏ sót ý trong câu
  gốc, KHÔNG được đoán hộ nhân vật đang nghĩ gì ngoài những gì câu chữ thể
  hiện. Được phép thêm đại từ nhân xưng (tôi/em/anh/chị...) hoặc trợ từ cho
  câu tiếng Việt tự nhiên hơn (vì tiếng Nhật/Trung/Hàn hay lược bỏ chủ ngữ),
  nhưng không được thêm Ý MỚI không có trong nguyên tác.
- Nếu có BẢNG THUẬT NGỮ từ các trang trước (xem bên trên nếu có), PHẢI dùng
  đúng bản dịch đã quy ước cho tên riêng/chức danh đó, không tự ý đổi cách
  dịch khác đi giữa chừng truyện.
{romaji_instruction}
{honorifics_instruction}

BƯỚC 4 - Sau khi dịch xong, rà lại toàn trang: nếu có tên riêng (tên nhân
vật), chức danh, hoặc thuật ngữ quan trọng LẶP LẠI NHIỀU LẦN mà CHƯA có trong
bảng thuật ngữ đã cho, liệt kê chúng vào "glossary_updates" kèm bản dịch đã
chọn, để các trang SAU dùng lại đúng cách dịch này. Không liệt kê từ thông
thường, chỉ liệt kê tên riêng/danh xưng/thuật ngữ đặc thù của truyện.

YÊU CẦU VỀ ĐỘ ĐẦY ĐỦ (rất quan trọng):
- Phải phát hiện 100% số bóng thoại có trên trang, kể cả bóng nhỏ, bóng ở
  góc, hộp thoại tường thuật, hoặc bóng thoại bị chồng lấn giữa các khung.
- KHÔNG được bỏ sót bất kỳ đoạn hội thoại nào chỉ vì nó nhỏ hoặc khó đọc -
  hãy cố gắng đọc hết khả năng, nếu thực sự không đọc được thì vẫn tạo entry
  với original_text là phần đọc được nhiều nhất có thể.
- Không tự ý gộp 2 bóng thoại riêng biệt thành 1 entry, trừ khi chúng thực
  sự là 1 bóng thoại duy nhất bị ngắt dòng.
- QUAN TRỌNG: với MỖI bóng thoại, original_text phải là TOÀN BỘ văn bản có
  trong bóng đó, đọc hết tất cả các dòng/cột chữ bên trong (kể cả chữ dọc
  nhiều cột, chữ nhỏ ở rìa bóng). KHÔNG được dừng lại giữa chừng chỉ vì gặp
  dấu "…", dấu chấm lửng, hoặc hết một dòng - dấu "…" thường chỉ là 1 phần
  của câu văn đang bỏ lửng theo văn phong, không phải là kết thúc bóng thoại.
  Nếu bóng thoại có 3 dòng chữ dọc, original_text phải chứa đủ nội dung của
  cả 3 dòng đó ghép lại theo đúng thứ tự đọc, không chỉ lấy 1 dòng đầu.

VỀ TỌA ĐỘ BOUNDING BOX:
Cung cấp tọa độ dạng chuẩn hóa trên thang điểm 0 - 1000: [ymin, xmin, ymax, xmax]
- Bounding box phải bao SÁT viền ngoài của riêng bóng thoại/hộp chữ đó,
  KHÔNG bao trùm cả khung tranh xung quanh hay các bóng thoại khác.
- ymin: Mép trên của bóng thoại (0 - 1000)
- xmin: Mép trái của bóng thoại (0 - 1000)
- ymax: Mép dưới của bóng thoại (0 - 1000)
- xmax: Mép phải của bóng thoại (0 - 1000)

ĐỊNH DẠNG CHỮ: original_text, romanization và translated_text của mỗi bóng thoại đều phải nằm trên MỘT DÒNG
DUY NHẤT - TUYỆT ĐỐI KHÔNG chèn ký tự xuống dòng (\\n) vào các trường này, kể cả khi chữ gốc trong ảnh bị ngắt
thành nhiều dòng / nhiều cột. App tự xuống dòng khi xếp chữ lên bóng thoại.

CHỈ trả về JSON theo đúng schema sau, không thêm bất kỳ văn bản giải thích nào khác:
{{
  "page_name": "<tên_file_ảnh>",
  "dialogues": [
    {{
      "dialogue_id": <số nguyên 1, 2, 3... theo đúng thứ tự Khung tranh -> Bóng thoại trong khung>,
      "box_2d": [ymin, xmin, ymax, xmax],
      "original_text": "<văn bản gốc, KHÔNG kèm phiên âm>",
      "romanization": "<phiên âm Romaji của original_text, hoặc \"\" nếu ngôn ngữ nguồn không phải tiếng Nhật>",
      "translated_text": "<văn bản dịch>"
    }}
  ],
  "glossary_updates": [
    {{"term": "<tên riêng/thuật ngữ gốc MỚI>", "translation": "<bản dịch đã chọn>"}}
  ]
}}
Nếu không có thuật ngữ mới nào, để "glossary_updates" là mảng rỗng [].
"""

# JSON Schema ép Gemini trả về đúng cấu trúc Page/Dialogue (Structured Output).
# Khớp 1-1 với app.models.Dialogue / app.models.Page.
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_name": {"type": "string"},
        "dialogues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dialogue_id": {"type": "integer"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "[ymin, xmin, ymax, xmax] dạng chuẩn hóa 0-1000",
                    },
                    "original_text": {"type": "string"},
                    "romanization": {"type": "string"},
                    "translated_text": {"type": "string"},
                },
                "required": ["dialogue_id", "box_2d", "original_text", "romanization", "translated_text"],
            },
        },
        "glossary_updates": {
            "type": "array",
            "description": "Tên riêng/thuật ngữ MỚI phát hiện trên trang này, chưa có trong bảng thuật ngữ đã cho",
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
    "required": ["page_name", "dialogues"],
}

"""
Prompt dùng khi dịch MỘT VÙNG ẢNH đã được cắt sẵn (crop) do người dùng chọn
bằng chuột trên Workspace - không cần dò lại bounding box vì người dùng đã
tự khoanh vùng, chỉ cần OCR + dịch đúng nội dung trong vùng đó.
"""
GEMINI_REGION_PROMPT_TEMPLATE = """\
Đây là một vùng ảnh được cắt (crop) từ một trang truyện tranh, chứa văn bản
trong một bóng thoại hoặc khung chữ. Nhiệm vụ:
1. Đọc chính xác toàn bộ văn bản trong ảnh (ngôn ngữ nguồn "{source_lang}").
   Nếu có nhiều dòng thuộc cùng một bóng thoại, ghép lại thành 1 đoạn hoàn chỉnh.
2. Dịch sang ngôn ngữ đích "{target_lang}".
Nếu ảnh không chứa văn bản đọc được, trả về original_text và translated_text là chuỗi rỗng "".
{romaji_instruction}

CHỈ trả về JSON đúng theo schema sau, không thêm bất kỳ văn bản nào khác:
{{
  "original_text": "<văn bản gốc đọc được, KHÔNG kèm phiên âm>",
  "romanization": "<phiên âm Romaji của original_text, hoặc \"\" nếu ngôn ngữ nguồn không phải tiếng Nhật>",
  "translated_text": "<văn bản đã dịch>"
}}
"""

GEMINI_REGION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "original_text": {"type": "string"},
        "romanization": {"type": "string"},
        "translated_text": {"type": "string"},
    },
    "required": ["original_text", "romanization", "translated_text"],
}

# LƯU Ý: prompt/schema cho bước "lọc lỗi OCR + dịch" của backend YOLO+Manga-OCR+Gemini
# đã được chuyển sang app/services/manga_ocr_pipeline.py (cùng với hàm gọi Gemini
# cleanup_and_translate_texts()), để toàn bộ logic của backend này nằm chung 1 chỗ.

# Model mặc định lấy từ config.py (có thể đổi trong Settings UI, không cần sửa code).
from app import config as _config

DEFAULT_GEMINI_MODEL = _config.DEFAULT_GEMINI_MODEL


class TranslationServiceError(Exception):
    pass


class TranslationRateLimitError(TranslationServiceError):
    """Riêng cho lỗi 429/rate-limit, để Queue Manager có thể nhận diện và retry."""

    pass


class TranslationService(ABC):
    @abstractmethod
    def translate_page(
        self, image_path: str, source_lang: str, target_lang: str, context: Optional[dict] = None
    ) -> Page:
        """
        Nhận đường dẫn ảnh 1 trang, trả về Page đã có dialogues điền sẵn.

        context (tuỳ chọn): ngữ cảnh từ (các) trang trước, dùng để giữ nhất
        quán tên riêng/thuật ngữ/văn phong khi dịch tuần tự nhiều trang liên
        tiếp. Gồm:
            {
                "glossary": {"<thuật ngữ gốc>": "<bản dịch đã dùng>", ...},
                "recent_dialogues": [{"original_text":, "translated_text":}, ...]
                    # vài câu thoại cuối của trang ngay trước đó
            }
        Service không bắt buộc phải dùng context (bỏ qua nếu None hoặc rỗng),
        nhưng NÊN tận dụng nếu có để tăng độ nhất quán giữa các trang.
        Page trả về có thể set thêm `page.glossary_updates` (dict) nếu phát
        hiện thuật ngữ MỚI trên trang này, để nơi gọi gộp vào glossary chung.
        """
        raise NotImplementedError

    @abstractmethod
    def translate_region(
        self, image_path: str, crop_box: List[int], source_lang: str, target_lang: str
    ) -> Dialogue:
        """
        Dịch MỘT vùng ảnh do người dùng tự khoanh (crop) trên trang hiện tại.

        crop_box: [x1, y1, x2, y2] theo pixel trên ẢNH GỐC (không phải toạ độ màn hình).
        Trả về 1 Dialogue mới (dialogue_id tạm = 0, AppState sẽ gán ID thật khi thêm vào Page).
        """
        raise NotImplementedError

    def validate_api_key(self, api_key: str) -> bool:
        """Ping nhẹ để kiểm tra Key hợp lệ trước khi vào giao diện chính."""
        return True


class MockTranslationService(TranslationService):
    """
    Giả lập OCR + dịch mà không cần gọi API thật.
    Hữu ích để dựng và test toàn bộ luồng UI/Queue/Export trước khi có key thật.
    """

    # Câu gốc và phiên âm là HAI field riêng (giống định dạng thật Gemini trả về khi source_lang là tiếng Nhật),
    # để test UI hiển thị / sao chép đúng ngay cả ở Mock.
    _SAMPLE_ORIGINAL = [
        "おい、何やってんだよ？",
        "べ、別に…",
        "早く行こう！",
        "本当にごめんなさい。",
        "これで終わりだ。",
    ]
    _SAMPLE_ROMANIZATION = [
        "Oi, nani yattenda yo?",
        "Be, betsu ni...",
        "Hayaku ikou!",
        "Hontou ni gomennasai.",
        "Kore de owari da.",
    ]
    _SAMPLE_TRANSLATED = [
        "Này, cậu đang làm cái trò gì đấy?",
        "K... Không có gì...",
        "Đi nhanh lên nào!",
        "Tôi thực sự xin lỗi.",
        "Đến đây là kết thúc rồi.",
    ]

    def __init__(self, simulate_delay: bool = True, simulate_failure_rate: float = 0.0):
        self.simulate_delay = simulate_delay
        self.simulate_failure_rate = simulate_failure_rate

    def validate_api_key(self, api_key: str) -> bool:
        # Mock: coi mọi key không rỗng là hợp lệ
        return bool(api_key and api_key.strip())

    def translate_page(
        self, image_path: str, source_lang: str, target_lang: str, context: Optional[dict] = None
    ) -> Page:
        if self.simulate_delay:
            time.sleep(random.uniform(0.4, 1.2))

        if self.simulate_failure_rate and random.random() < self.simulate_failure_rate:
            raise TranslationServiceError(f"Mock lỗi giả lập khi dịch {image_path}")

        page_name = Path(image_path).name
        n_dialogues = random.randint(2, 4)
        dialogues: List[Dialogue] = []
        for i in range(1, n_dialogues + 1):
            idx = (i - 1) % len(self._SAMPLE_ORIGINAL)
            dialogues.append(
                Dialogue(
                    dialogue_id=i,
                    box_location=[
                        random.randint(20, 400),
                        random.randint(20, 600),
                        random.randint(100, 250),
                        random.randint(60, 120),
                    ],
                    original_text=self._SAMPLE_ORIGINAL[idx],
                    romanization=self._SAMPLE_ROMANIZATION[idx],
                    translated_text=self._SAMPLE_TRANSLATED[idx],
                )
            )
        return Page(page_name=page_name, image_path=image_path, dialogues=dialogues, status="done")

    def translate_region(
        self, image_path: str, crop_box: List[int], source_lang: str, target_lang: str
    ) -> Dialogue:
        if self.simulate_delay:
            time.sleep(random.uniform(0.3, 0.8))
        if self.simulate_failure_rate and random.random() < self.simulate_failure_rate:
            raise TranslationServiceError(f"Mock lỗi giả lập khi dịch vùng ảnh {image_path}")

        x1, y1, x2, y2 = crop_box
        idx = random.randrange(len(self._SAMPLE_ORIGINAL))
        return Dialogue(
            dialogue_id=0,  # placeholder, AppState sẽ gán ID thật
            box_location=[x1, y1, x2 - x1, y2 - y1],
            original_text=self._SAMPLE_ORIGINAL[idx],
            romanization=self._SAMPLE_ROMANIZATION[idx],
            translated_text=self._SAMPLE_TRANSLATED[idx],
            edited=False,
        ).flatten_lines()


class GeminiTranslationService(TranslationService):
    """
    Tích hợp Gemini API thật qua SDK `google-genai` (Structured Output).

    Yêu cầu: `pip install google-genai`
    """

    def __init__(self, api_key: str, model_name: str = DEFAULT_GEMINI_MODEL):
        if not api_key:
            raise TranslationServiceError("Thiếu Gemini API Key.")
        try:
            from google import genai  # import trễ để app vẫn chạy được ở chế độ Mock
        except ImportError as exc:
            raise TranslationServiceError(
                "Chưa cài thư viện google-genai. Chạy: pip install google-genai"
            ) from exc

        self.api_key = api_key
        self.model_name = gemini_models.PRIMARY_MODEL   # model cố định (không còn chọn); lỗi thì tự dùng model phụ
        self.client = _make_client(genai, api_key)

    def validate_api_key(self, api_key: str) -> bool:
        """Xác nhận Key + tên model trước khi vào giao diện chính (không sinh nội dung -> nhanh, không tốn token)."""
        return _ping_gemini(self.client, self.model_name)

    def translate_page(
        self, image_path: str, source_lang: str, target_lang: str, context: Optional[dict] = None
    ) -> Page:
        from google.genai import types  # import trễ, cùng lý do như trên
        from google.genai import errors as genai_errors

        image_path_obj = Path(image_path)
        try:
            image_bytes = image_path_obj.read_bytes()
        except OSError as exc:
            raise TranslationServiceError(f"Không đọc được file ảnh {image_path}: {exc}") from exc

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as exc:
            raise TranslationServiceError(f"Không thể đọc kích thước ảnh {image_path}: {exc}") from exc
        
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        context_section = build_context_section(context)
        romaji_instruction = build_romaji_instruction(source_lang)
        honorifics_instruction = build_honorifics_instruction((context or {}).get("keep_honorifics", False))
        prompt = GEMINI_SYSTEM_PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            context_section=context_section,
            romaji_instruction=romaji_instruction,
            honorifics_instruction=honorifics_instruction,
        )

        log.info("Gửi request dịch trang %s (model=%s)...", image_path_obj.name, self.model_name)
        try:
            from app.threading_utils import warn_if_ui_thread; warn_if_ui_thread("Gemini generate_content")
            response = gemini_models.generate_content(
                self.client, require_text=True,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                        # Ảnh trang truyện thường có chữ nhỏ + nhiều bóng thoại chồng lấn -
                        # bật độ phân giải cao để Gemini đọc chữ & xác định bounding box chính xác hơn.
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                    ),
                    prompt,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GEMINI_RESPONSE_SCHEMA,
                    # Nhiệt độ thấp để kết quả nhận diện bóng thoại/toạ độ ổn định, ít "sáng tạo" sai lệch
                    "temperature": 0.1,
                    # Trang có nhiều thoại cần budget token đủ lớn để JSON không bị cắt cụt giữa chừng
                    # (tăng thêm vì bật thinking_level=high cũng tốn thêm token suy luận nội bộ)
                    "max_output_tokens": 16384,
                    # Đẩy mức độ "suy nghĩ" lên cao nhất - đánh đổi tốc độ lấy độ chính xác khi
                    # đọc/định vị chữ trong ảnh (đặc biệt hữu ích với các model dòng "lite" vốn
                    # mặc định suy luận ở mức MINIMAL để tối ưu tốc độ/chi phí).
                    # LƯU Ý: tham số này chỉ áp dụng cho model Gemini 3.x trở lên; model cũ hơn
                    # sẽ báo lỗi nếu nhận tham số này.
                    "thinking_config": {"thinking_level": _thinking_level()},
                },
            )
        except genai_errors.ClientError as exc:
            log.error("Gemini API lỗi (ClientError) khi dịch %s: %s", image_path_obj.name, exc)
            # Gemini trả 429 khi vượt Rate Limit (RPM) -> báo riêng để Queue có thể retry/backoff
            if getattr(exc, "code", None) == 429 or "429" in str(exc):
                raise TranslationRateLimitError(f"Vượt giới hạn Rate Limit khi dịch {image_path_obj.name}: {exc}") from exc
            raise TranslationServiceError(f"Lỗi Gemini API khi dịch {image_path_obj.name}: {exc}") from exc
        except Exception as exc:
            log.exception("Lỗi không xác định khi gọi Gemini API cho trang %s", image_path_obj.name)
            raise TranslationServiceError(f"Lỗi không xác định khi gọi Gemini API: {exc}") from exc

        # Log finish_reason/usage LUÔN LUÔN (không chỉ khi rỗng) để dễ phát hiện
        # trường hợp bị cắt cụt do MAX_TOKENS dù JSON vẫn "parse được" một phần.
        try:
            candidates = getattr(response, "candidates", None) or []
            finish_reasons = [getattr(c, "finish_reason", None) for c in candidates]
            usage = getattr(response, "usage_metadata", None)
            log.info(
                "Trang %s: finish_reason=%s | usage=%s",
                image_path_obj.name, finish_reasons, usage,
            )
            if any(str(fr) not in ("STOP", "FinishReason.STOP", "None") for fr in finish_reasons):
                log.warning(
                    "Trang %s có finish_reason bất thường (%s) - kết quả có thể bị THIẾU do "
                    "vượt max_output_tokens hoặc bị safety filter chặn giữa chừng.",
                    image_path_obj.name, finish_reasons,
                )
        except Exception:
            pass

        raw_text = getattr(response, "text", None)
        log.info(
            "Đã nhận phản hồi cho trang %s (độ dài text=%s)",
            image_path_obj.name,
            len(raw_text) if raw_text else 0,
        )
        if not raw_text:
            # In thêm finish_reason/prompt_feedback nếu có, để biết vì sao rỗng
            # (VD: bị chặn bởi safety filter, hết max_output_tokens, v.v.)
            try:
                candidates = getattr(response, "candidates", None) or []
                finish_reasons = [getattr(c, "finish_reason", None) for c in candidates]
                prompt_feedback = getattr(response, "prompt_feedback", None)
                log.warning(
                    "Response rỗng cho %s | finish_reason=%s | prompt_feedback=%s",
                    image_path_obj.name,
                    finish_reasons,
                    prompt_feedback,
                )
            except Exception:
                pass
            raise TranslationServiceError(
                f"Gemini không trả về nội dung cho trang {image_path_obj.name}. "
                f"Xem log terminal để biết finish_reason/safety filter."
            )

        try:
            data = parse_gemini_json(raw_text)
        except json.JSONDecodeError as exc:
            log.error("JSON không hợp lệ cho %s. Raw text: %s", image_path_obj.name, raw_text[:500])
            raise TranslationServiceError(
                f"Gemini trả về JSON không hợp lệ cho trang {image_path_obj.name}: {exc}"
            ) from exc

        dialogues: List[Dialogue] = []
        for d in data.get("dialogues", []):
            try:
                box_2d = d.get("box_2d", [])
                if len(box_2d) == 4:
                    ymin, xmin, ymax, xmax = box_2d
                    
                    # Quy đổi từ thang 0-1000 sang pixel thực tế [x, y, w, h]
                    x = int((xmin / 1000.0) * img_w)
                    y = int((ymin / 1000.0) * img_h)
                    w = int(((xmax - xmin) / 1000.0) * img_w)
                    h = int(((ymax - ymin) / 1000.0) * img_h)

                    dialogue_dict = {
                        "dialogue_id": d.get("dialogue_id"),
                        "box_location": [x, y, w, h],
                        "original_text": d.get("original_text", ""),
                        "translated_text": d.get("translated_text", ""),
                    }
                    if "romanization" in d:          # model trả theo định dạng mới; thiếu khoá -> from_dict tự tách kiểu cũ "câu gốc\nromaji"
                        dialogue_dict["romanization"] = d.get("romanization") or ""
                    dialogues.append(Dialogue.from_dict(dialogue_dict).flatten_lines())   # mỗi trường đúng 1 dòng
            except (KeyError, TypeError) as exc:
                log.warning("Bỏ qua 1 dialogue lỗi ở trang %s: %s", image_path_obj.name, exc)
                continue  # bỏ qua dialogue lỗi thay vì làm hỏng cả trang

        glossary_updates = {
            item["term"]: item["translation"]
            for item in data.get("glossary_updates", [])
            if item.get("term") and item.get("translation")
        }
        if glossary_updates:
            log.info("Trang %s phát hiện %d thuật ngữ mới: %s", image_path_obj.name, len(glossary_updates), glossary_updates)

        log.info("Trang %s dịch xong: %d dialogue(s)", image_path_obj.name, len(dialogues))
        return Page(
            page_name=image_path_obj.name,
            image_path=image_path,
            dialogues=dialogues,
            status="done",
            glossary_updates=glossary_updates,
        )

    def translate_region(
        self, image_path: str, crop_box: List[int], source_lang: str, target_lang: str
    ) -> Dialogue:
        from google.genai import types
        from google.genai import errors as genai_errors
        import io

        image_path_obj = Path(image_path)
        x1, y1, x2, y2 = crop_box

        # Cắt đúng vùng người dùng đã khoanh, chỉ gửi phần ảnh nhỏ này lên Gemini
        # (chính xác hơn và rẻ hơn so với gửi lại nguyên trang + yêu cầu dò lại toạ độ).
        try:
            with Image.open(image_path) as img:
                cropped = img.convert("RGB").crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                cropped.save(buf, format="PNG")
                image_bytes = buf.getvalue()
        except Exception as exc:
            raise TranslationServiceError(f"Không thể cắt vùng ảnh từ {image_path_obj.name}: {exc}") from exc

        prompt = GEMINI_REGION_PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            romaji_instruction=build_romaji_instruction(source_lang),
        )

        log.info(
            "Gửi request dịch vùng ảnh [%d,%d,%d,%d] trong %s (model=%s)...",
            x1, y1, x2, y2, image_path_obj.name, self.model_name,
        )
        try:
            from app.threading_utils import warn_if_ui_thread; warn_if_ui_thread("Gemini generate_content")
            response = gemini_models.generate_content(
                self.client, require_text=True,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                    ),
                    prompt,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GEMINI_REGION_RESPONSE_SCHEMA,
                    "temperature": 0.1,
                    "thinking_config": {"thinking_level": _thinking_level()},
                },
            )
        except genai_errors.ClientError as exc:
            log.error("Gemini API lỗi (ClientError) khi dịch vùng ảnh %s: %s", image_path_obj.name, exc)
            if getattr(exc, "code", None) == 429 or "429" in str(exc):
                raise TranslationRateLimitError(f"Vượt giới hạn Rate Limit khi dịch vùng ảnh: {exc}") from exc
            raise TranslationServiceError(f"Lỗi Gemini API khi dịch vùng ảnh: {exc}") from exc
        except Exception as exc:
            log.exception("Lỗi không xác định khi dịch vùng ảnh trong %s", image_path_obj.name)
            raise TranslationServiceError(f"Lỗi không xác định khi gọi Gemini API: {exc}") from exc

        raw_text = getattr(response, "text", None)
        log.info("Đã nhận phản hồi vùng ảnh (độ dài text=%s)", len(raw_text) if raw_text else 0)
        if not raw_text:
            raise TranslationServiceError(
                "Gemini không trả về nội dung cho vùng ảnh đã chọn. Xem log terminal để biết chi tiết."
            )

        try:
            data = parse_gemini_json(raw_text)
        except json.JSONDecodeError as exc:
            log.error("JSON không hợp lệ cho vùng ảnh. Raw text: %s", raw_text[:500])
            raise TranslationServiceError(f"Gemini trả về JSON không hợp lệ: {exc}") from exc

        return Dialogue(
            dialogue_id=0,  # placeholder, AppState sẽ gán ID thật khi thêm vào Page
            box_location=[x1, y1, x2 - x1, y2 - y1],
            original_text=data.get("original_text", ""),
            romanization=data.get("romanization", "") or "",
            translated_text=data.get("translated_text", ""),
            edited=False,
        ).flatten_lines()


GEMINI_TIMEOUT_MS = 240_000   # mỗi request Gemini tối đa 4 phút: trước đây không có timeout -> luồng nền có thể treo vô hạn trong lúc dịch,
                              # khiến app không đóng được ("Not Responding", \"QThreadPool destroyed while threads are running\")


def _make_client(genai, api_key: str):
    try:
        return genai.Client(api_key=api_key, http_options={"timeout": GEMINI_TIMEOUT_MS})
    except Exception:  # noqa: BLE001 - bản google-genai không nhận http_options dạng dict
        return genai.Client(api_key=api_key)


_KEY_ERR = ("api key", "api_key", "401", "403", "permission")


def _ping_gemini(client, model_name: str = "") -> bool:
    from app.threading_utils import warn_if_ui_thread
    warn_if_ui_thread("Gemini models.get")
    """True = dùng được (hoặc chưa rõ do lỗi tạm thời); False = Key sai; ném TranslationServiceError nếu CẢ model chính lẫn model phụ
    đều không tồn tại (404). Dùng models.get (chỉ đọc metadata, ~0,3 s); không gọi generate_content trên luồng giao diện vì model 3.x
    suy luận nên có thể mất hàng chục giây."""
    not_found = []
    for name in (gemini_models.PRIMARY_MODEL, gemini_models.FALLBACK_MODEL):
        try:
            client.models.get(model=name)
            return True
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "404" in msg or "not found" in msg:
                not_found.append(name)
                continue
            if any(k in msg for k in _KEY_ERR):
                try:   # Key hạn chế quyền đọc metadata nhưng vẫn sinh nội dung được: thử một request cực nhỏ trước khi kết luận Key sai
                    client.models.generate_content(model=name, contents="hi",
                                                   config={"max_output_tokens": 8, "thinking_config": {"thinking_level": "minimal"}})
                    return True
                except Exception as exc2:  # noqa: BLE001
                    m2 = str(exc2).lower()
                    return not any(k in m2 for k in _KEY_ERR)
            log.warning("Không kiểm tra được model/Key (%s) - vẫn cho tiếp tục, lỗi thật sẽ báo khi dịch", exc)
            return True
    raise TranslationServiceError(f"Cả hai model {', '.join(not_found)} đều không tồn tại (404) - có thể Google đã khai tử; cần cập nhật app.")


def _thinking_level() -> str:
    """Mức suy luận Gemini do người dùng chọn trong Settings (mặc định low; high chậm ~100 s/trang)."""
    from app.services import ocr_engine
    return ocr_engine.current().thinking_level or "low"


def _require_detector(yolo_model_path: str) -> None:
    """Cần ít nhất một bộ dò: koharu-yolo26s (bật trong Settings) hoặc file YOLO .pt."""
    from app.services import koharu_detector as kd
    if not yolo_model_path and not kd.current().enabled:
        raise TranslationServiceError(
            "Chưa có bộ dò bóng thoại: bật koharu-yolo26s hoặc chọn file model YOLO (.pt) trong Settings.")


class HybridOcrTranslationService(TranslationService):
    """
    Backend chính xác cao hơn: YOLO phát hiện vị trí bóng thoại (thay vì để
    Gemini tự dò), Manga-OCR đọc chữ Nhật trong từng bóng (thay vì để Gemini
    tự OCR qua ảnh), Gemini CHỈ nhận text để lọc lỗi OCR + dịch - không cần
    gửi ảnh lên Gemini, nhanh hơn và rẻ hơn nhiều so với GeminiTranslationService.

    Yêu cầu cài thêm: pip install ultralytics opencv-python manga-ocr numpy
    Yêu cầu: 1 file model YOLO đã huấn luyện để phát hiện bóng thoại (.pt).
    """

    def __init__(
        self,
        yolo_model_path: str,
        api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL,
        yolo_conf: float = 0.35,
        enable_stray_text_removal: bool = False,
    ):
        _require_detector(yolo_model_path)
        if not api_key:
            raise TranslationServiceError("Thiếu Gemini API Key (vẫn cần để lọc lỗi OCR + dịch text).")
        try:
            from google import genai  # import trễ, cùng lý do như GeminiTranslationService
        except ImportError as exc:
            raise TranslationServiceError(
                "Chưa cài thư viện google-genai. Chạy: pip install google-genai"
            ) from exc

        self.yolo_model_path = yolo_model_path
        self.api_key = api_key
        self.model_name = gemini_models.PRIMARY_MODEL   # model cố định (không còn chọn); lỗi thì tự dùng model phụ
        self.yolo_conf = yolo_conf
        # TẮT MẶC ĐỊNH - đã ghi nhận nhiều lần xoá nhầm chữ hợp lệ khi khoảng
        # cách giữa các cột chữ dọc trong cùng 1 bóng thoại lớn hơn mức an
        # toàn. Chỉ bật nếu người dùng chủ động chọn trong Settings.
        self.enable_stray_text_removal = enable_stray_text_removal
        self.client = _make_client(genai, api_key)

    def validate_api_key(self, api_key: str) -> bool:
        return _ping_gemini(self.client, self.model_name)

    def _yolo_boxes_for_hybrid(self, image_path: str):
        """Koharu bật + có model YOLO bóng thoại: trả hộp YOLO (đã sắp theo thứ tự đọc) để kết hợp; ngược lại None (koharu tự quyết định)."""
        return _yolo_boxes_for_koharu_hybrid(self.yolo_model_path, getattr(self, "yolo_conf", 0.35), image_path)

    def _run_ocr_on_page(self, image_path: str):
        """Dò vùng thoại + Manga-OCR trên 1 trang -> list[dict{id, box, raw_text, kind, panel, ocr_conf, low_conf, crop_png?}]."""
        from app.services import koharu_detector as kd
        from app.services import manga_ocr_pipeline as mp

        from app.services import ocr_engine
        _kc = kd.current()
        log.info("Bộ dò: %s | Manga-OCR: %s | Gemini thinking: %s",
                 ("koharu-yolo26s" + (" + YOLO" if _kc.use_yolo and self.yolo_model_path else "")) if _kc.enabled else "YOLO",
                 ocr_engine.current().model, ocr_engine.current().thinking_level)
        cv_img = kd._imread(image_path)      # np.fromfile + imdecode: đọc được đường dẫn Unicode trên Windows (cv2.imread thì không)
        if cv_img is None:
            raise TranslationServiceError(f"Không đọc được ảnh: {image_path}")

        got = kd.units_for_page(image_path, yolo_boxes=self._yolo_boxes_for_hybrid(image_path))   # None = koharu tắt / lỗi -> YOLO cũ
        if got is not None:
            _layout, units = got
            log.info("koharu phát hiện %d vùng thoại (bóng + chữ tự do) trong %s", len(units), Path(image_path).name)
            return mp.ocr_units(cv_img, units)

        if not self.yolo_model_path:
            raise TranslationServiceError("koharu không chạy được và chưa chọn model YOLO (.pt) để dự phòng.")
        yolo_model = mp.get_yolo_model(self.yolo_model_path)
        sorted_boxes = mp.sort_reading_order(mp.get_bubble_boxes(yolo_model, image_path, conf=self.yolo_conf))
        log.info("YOLO phát hiện %d bóng thoại trong %s", len(sorted_boxes), Path(image_path).name)
        return mp.ocr_yolo_boxes(cv_img, sorted_boxes, remove_stray_fragments=self.enable_stray_text_removal)

    def translate_page(
        self, image_path: str, source_lang: str, target_lang: str, context: Optional[dict] = None
    ) -> Page:
        from app.services.manga_ocr_pipeline import cleanup_and_translate_texts, pick_rescue_ids

        image_path_obj = Path(image_path)
        ocr_results = self._run_ocr_on_page(image_path)

        if not ocr_results:
            log.warning("Không phát hiện bóng thoại nào trong %s (YOLO trả về 0 box)", image_path_obj.name)
            return Page(page_name=image_path_obj.name, image_path=image_path, dialogues=[], status="done")

        from app.services import ocr_engine
        ocfg = ocr_engine.current()
        rescue = {}
        if ocfg.gemini_rescue:
            by_id = {it["id"]: it for it in ocr_results}
            rescue = {i: by_id[i]["crop_png"] for i in pick_rescue_ids(ocr_results, ocfg.max_rescue)}
            if rescue:
                log.info("%d câu đọc chưa chắc -> gửi kèm ảnh crop cho Gemini đối chiếu: ids=%s", len(rescue), sorted(rescue))
        import time as _t
        _t0 = _t.perf_counter()
        cleaned_map, glossary_updates = cleanup_and_translate_texts(
            self.client, self.model_name, ocr_results, source_lang, target_lang, context=context,
            rescue_images=rescue,
        )
        log.info("Gemini lọc lỗi + dịch mất %.1fs (%d câu)", _t.perf_counter() - _t0, len(ocr_results))

        dialogues: List[Dialogue] = []
        for item in ocr_results:
            cleaned = cleaned_map.get(item["id"], {})
            x, y, w, h = item["box"]
            dialogues.append(
                Dialogue(
                    dialogue_id=item["id"],
                    box_location=[x, y, w, h],
                    original_text=cleaned.get("cleaned_text", item["raw_text"]),
                    romanization=cleaned.get("romanization", "") or "",
                    translated_text=cleaned.get("translated_text", ""),
                    edited=False,
                ).flatten_lines()
            )

        _append_sfx_dialogues(dialogues, image_path)
        log.info("Trang %s dịch xong (YOLO+OCR+Gemini): %d dialogue(s)", image_path_obj.name, len(dialogues))
        return Page(
            page_name=image_path_obj.name,
            image_path=image_path,
            dialogues=dialogues,
            status="done",
            glossary_updates=glossary_updates,
        )

    def translate_region(
        self, image_path: str, crop_box: List[int], source_lang: str, target_lang: str
    ) -> Dialogue:
        """
        Vùng do người dùng tự khoanh -> không cần YOLO dò lại, chỉ cần chạy
        Manga-OCR trực tiếp trên vùng đã chọn rồi lọc/dịch qua Gemini.
        """
        from app.services.manga_ocr_pipeline import get_manga_ocr, cleanup_and_translate_texts

        x1, y1, x2, y2 = crop_box
        try:
            with Image.open(image_path) as img:
                cropped = img.convert("RGB").crop((x1, y1, x2, y2))
        except Exception as exc:
            raise TranslationServiceError(f"Không thể cắt vùng ảnh: {exc}") from exc

        from app.services.ocr_engine import normalize_ocr_text
        mocr = get_manga_ocr()
        text_jp = normalize_ocr_text(mocr(cropped))

        if not text_jp.strip():
            return Dialogue(
                dialogue_id=0,
                box_location=[x1, y1, x2 - x1, y2 - y1],
                original_text="",
                translated_text="",
                edited=False,
            ).flatten_lines()

        cleaned_map, _glossary_updates = cleanup_and_translate_texts(
            self.client,
            self.model_name,
            [{"id": 1, "box": [x1, y1, x2 - x1, y2 - y1], "raw_text": text_jp}],
            source_lang,
            target_lang,
        )
        cleaned = cleaned_map.get(1, {})

        return Dialogue(
            dialogue_id=0,
            box_location=[x1, y1, x2 - x1, y2 - y1],
            original_text=cleaned.get("cleaned_text", text_jp),
            romanization=cleaned.get("romanization", "") or "",
            translated_text=cleaned.get("translated_text", ""),
            edited=False,
        ).flatten_lines()


"""
Prompt dùng cho backend YOLO + Gemini Vision: YOLO đã xác định sẵn chính xác
vị trí từng bóng thoại (vẽ viền đỏ + số thứ tự trực tiếp lên ảnh - kỹ thuật
"Set-of-Mark prompting"), Gemini CHỈ cần đọc + dịch nội dung trong từng vùng
đã đánh số đó, KHÔNG cần tự xác định lại toạ độ (box_2d) như backend Gemini
Vision thuần tuý - tránh được lỗi toạ độ sai, vẫn giữ chất lượng OCR/dịch tốt
của Gemini (so với Manga-OCR có thể gặp khó với chữ cách điệu/hiệu ứng).
"""
GEMINI_YOLO_VISION_PROMPT_TEMPLATE = """\
Đây là 1 trang truyện tranh đã được đánh dấu sẵn CHÍNH XÁC {n_boxes} vùng bóng
thoại cần đọc + dịch (mỗi vùng có viền đỏ bao quanh và 1 số thứ tự ở góc trên
bên trái vùng đó), theo ĐÚNG thứ tự đọc manga (phải→trái, trên→dưới theo khung
tranh). KHÔNG cần tự xác định lại toạ độ hay tìm thêm bóng thoại nào khác -
chỉ xử lý đúng {n_boxes} vùng đã được đánh số sẵn.

Với MỖI vùng đã đánh số (từ số 1 đến {n_boxes}):
1. Nhìn kỹ vào ĐÚNG vùng ảnh nằm trong viền đỏ mang số đó, đọc CHÍNH XÁC văn
   bản nhìn thấy (ngôn ngữ nguồn "{source_lang}"). Đây phải là bản ghi chính
   xác những gì nhìn thấy, KHÔNG được tự diễn giải, đoán ý, hay viết lại.
2. Dịch sang ngôn ngữ đích "{target_lang}": bám sát nghĩa, không thêm tình
   tiết không có trong câu gốc, không bỏ sót ý. Được phép thêm đại từ nhân
   xưng/trợ từ cho câu tự nhiên hơn (vì ngôn ngữ nguồn hay lược chủ ngữ).
{context_section}
{romaji_instruction}
{honorifics_instruction}
Sau khi dịch xong, nếu có tên riêng/chức danh/thuật ngữ quan trọng LẶP LẠI
NHIỀU LẦN mà CHƯA có trong bảng thuật ngữ đã cho, liệt kê vào "glossary_updates".

CHỈ trả về JSON đúng theo schema sau, không thêm bất kỳ văn bản giải thích nào khác:
{{
  "dialogues": [
    {{"id": <số thứ tự vùng, 1 đến {n_boxes}>, "original_text": "<văn bản gốc, KHÔNG kèm phiên âm>", "romanization": "<phiên âm Romaji, hoặc \"\">", "translated_text": "<bản dịch>"}}
  ],
  "glossary_updates": [
    {{"term": "<thuật ngữ gốc MỚI>", "translation": "<bản dịch đã chọn>"}}
  ]
}}
"""

GEMINI_YOLO_VISION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "dialogues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "original_text": {"type": "string"},
                    "romanization": {"type": "string"},
                    "translated_text": {"type": "string"},
                },
                "required": ["id", "original_text", "romanization", "translated_text"],
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


class YoloGeminiVisionTranslationService(TranslationService):
    """
    Kết hợp điểm mạnh của 2 backend kia: YOLO xác định toạ độ bóng thoại
    (chính xác hơn để Gemini tự dò), Gemini Vision đọc + dịch trực tiếp trên
    ảnh (chất lượng OCR/hiểu ngữ cảnh tốt hơn Manga-OCR với chữ cách điệu).

    Cách làm: vẽ số thứ tự + viền đỏ lên đúng từng vùng YOLO phát hiện (ngay
    trên ảnh gốc), gửi ảnh ĐÃ ĐÁNH DẤU đó cho Gemini kèm yêu cầu đọc+dịch
    đúng nội dung trong từng vùng theo số, không cần Gemini tự tìm vị trí.

    Yêu cầu: cùng thư viện với backend YOLO+Manga-OCR (pip install ultralytics
    opencv-python numpy) NGOẠI TRỪ không cần manga-ocr (Gemini tự OCR).
    """

    def __init__(
        self,
        yolo_model_path: str,
        api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL,
        yolo_conf: float = 0.35,
    ):
        _require_detector(yolo_model_path)
        if not api_key:
            raise TranslationServiceError("Thiếu Gemini API Key.")
        try:
            from google import genai
        except ImportError as exc:
            raise TranslationServiceError(
                "Chưa cài thư viện google-genai. Chạy: pip install google-genai"
            ) from exc

        self.yolo_model_path = yolo_model_path
        self.api_key = api_key
        self.model_name = gemini_models.PRIMARY_MODEL   # model cố định (không còn chọn); lỗi thì tự dùng model phụ
        self.yolo_conf = yolo_conf
        self.client = _make_client(genai, api_key)
        # Tái dùng GeminiTranslationService cho translate_region() (khoanh
        # vùng thủ công không cần YOLO, giống hệt backend Gemini Vision thuần).
        self._region_helper = GeminiTranslationService(api_key=api_key, model_name=model_name)

    def validate_api_key(self, api_key: str) -> bool:
        return self._region_helper.validate_api_key(api_key)

    def translate_page(
        self, image_path: str, source_lang: str, target_lang: str, context: Optional[dict] = None
    ) -> Page:
        from google.genai import types
        from google.genai import errors as genai_errors
        from app.services.manga_ocr_pipeline import (
            get_yolo_model,
            get_bubble_boxes,
            sort_reading_order,
            draw_annotated_page,
        )

        image_path_obj = Path(image_path)
        from app.services import koharu_detector as kd
        got = kd.units_for_page(image_path, yolo_boxes=_yolo_boxes_for_koharu_hybrid(
            self.yolo_model_path, self.yolo_conf, image_path))   # None = koharu tắt / lỗi -> dùng YOLO cũ
        if got is not None:
            sorted_boxes = [u.box for u in got[1]]
            log.info("koharu phát hiện %d vùng thoại trong %s", len(sorted_boxes), image_path_obj.name)
        else:
            if not self.yolo_model_path:
                raise TranslationServiceError("koharu không chạy được và chưa chọn model YOLO (.pt) để dự phòng.")
            yolo_model = get_yolo_model(self.yolo_model_path)
            raw_boxes = get_bubble_boxes(yolo_model, image_path, conf=self.yolo_conf)
            sorted_boxes = sort_reading_order(raw_boxes)
            log.info("YOLO phát hiện %d bóng thoại trong %s", len(sorted_boxes), image_path_obj.name)

        if not sorted_boxes:
            log.warning("Không phát hiện bóng thoại nào trong %s (YOLO trả về 0 box)", image_path_obj.name)
            return Page(page_name=image_path_obj.name, image_path=image_path, dialogues=[], status="done")

        annotated_bytes = draw_annotated_page(image_path, sorted_boxes)

        context_section = build_context_section(context)
        romaji_instruction = build_romaji_instruction(source_lang)
        honorifics_instruction = build_honorifics_instruction((context or {}).get("keep_honorifics", False))
        prompt = GEMINI_YOLO_VISION_PROMPT_TEMPLATE.format(
            n_boxes=len(sorted_boxes),
            source_lang=source_lang,
            target_lang=target_lang,
            context_section=context_section,
            romaji_instruction=romaji_instruction,
            honorifics_instruction=honorifics_instruction,
        )

        log.info("Gửi ảnh đã đánh số (%d vùng) lên Gemini (model=%s)...", len(sorted_boxes), self.model_name)
        try:
            from app.threading_utils import warn_if_ui_thread; warn_if_ui_thread("Gemini generate_content")
            response = gemini_models.generate_content(
                self.client, require_text=True,
                contents=[
                    types.Part.from_bytes(
                        data=annotated_bytes,
                        mime_type="image/png",
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                    ),
                    prompt,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GEMINI_YOLO_VISION_RESPONSE_SCHEMA,
                    "temperature": 0.1,
                    "max_output_tokens": 16384,
                    "thinking_config": {"thinking_level": _thinking_level()},
                },
            )
        except genai_errors.ClientError as exc:
            log.error("Gemini API lỗi (ClientError) khi dịch %s: %s", image_path_obj.name, exc)
            if getattr(exc, "code", None) == 429 or "429" in str(exc):
                raise TranslationRateLimitError(f"Vượt giới hạn Rate Limit khi dịch {image_path_obj.name}: {exc}") from exc
            raise TranslationServiceError(f"Lỗi Gemini API khi dịch {image_path_obj.name}: {exc}") from exc
        except Exception as exc:
            log.exception("Lỗi không xác định khi gọi Gemini API cho trang %s", image_path_obj.name)
            raise TranslationServiceError(f"Lỗi không xác định khi gọi Gemini API: {exc}") from exc

        raw_text = getattr(response, "text", None)
        if not raw_text:
            raise TranslationServiceError(
                f"Gemini không trả về nội dung cho trang {image_path_obj.name}. Xem log terminal để biết chi tiết."
            )

        try:
            data = parse_gemini_json(raw_text)
        except json.JSONDecodeError as exc:
            log.error("JSON không hợp lệ cho %s. Raw text: %s", image_path_obj.name, raw_text[:500])
            raise TranslationServiceError(f"Gemini trả về JSON không hợp lệ cho trang {image_path_obj.name}: {exc}") from exc

        result_by_id = {item["id"]: item for item in data.get("dialogues", []) if "id" in item}

        dialogues: List[Dialogue] = []
        for idx, (x, y, w, h) in enumerate(sorted_boxes, start=1):
            item = result_by_id.get(idx, {})
            dialogues.append(
                Dialogue(
                    dialogue_id=idx,
                    box_location=[x, y, w, h],
                    original_text=item.get("original_text", ""),
                    romanization=item.get("romanization", "") or "",
                    translated_text=item.get("translated_text", ""),
                    edited=False,
                ).flatten_lines()
            )

        glossary_updates = {
            item["term"]: item["translation"]
            for item in data.get("glossary_updates", [])
            if item.get("term") and item.get("translation")
        }

        _append_sfx_dialogues(dialogues, image_path)
        log.info("Trang %s dịch xong (YOLO+Gemini Vision): %d dialogue(s)", image_path_obj.name, len(dialogues))
        return Page(
            page_name=image_path_obj.name,
            image_path=image_path,
            dialogues=dialogues,
            status="done",
            glossary_updates=glossary_updates,
        )

    def translate_region(
        self, image_path: str, crop_box: List[int], source_lang: str, target_lang: str
    ) -> Dialogue:
        # Khoanh vùng thủ công không cần YOLO - dùng lại y hệt backend Gemini Vision thuần.
        return self._region_helper.translate_region(image_path, crop_box, source_lang, target_lang)


def _yolo_boxes_for_koharu_hybrid(yolo_model_path: str, conf: float, image_path: str):
    """Chế độ kết hợp YOLO + koharu: trả danh sách [x, y, w, h] của bóng thoại YOLO theo thứ tự đọc.
    None nếu không cần / không thể kết hợp (koharu tắt, tắt HYBRID_WITH_YOLO, không có model YOLO, lỗi) -> nơi gọi dùng đường cũ."""
    from app.services import koharu_detector as kd
    if not (kd.current().enabled and kd.current().use_yolo and kd.HYBRID_WITH_YOLO and yolo_model_path and os.path.isfile(yolo_model_path)):
        return None
    try:
        from app.services.manga_ocr_pipeline import get_yolo_model, get_bubble_boxes, sort_reading_order
        boxes = get_bubble_boxes(get_yolo_model(yolo_model_path), image_path, conf=conf)
        log.info("Kết hợp YOLO + koharu: %d bóng thoại từ YOLO trong %s", len(boxes), Path(image_path).name)
        return sort_reading_order(boxes)
    except Exception as exc:  # noqa: BLE001
        log.warning("Không chạy được YOLO để kết hợp với koharu, dùng koharu thuần: %s", exc)
        return None


def _append_sfx_dialogues(dialogues: List[Dialogue], image_path: str) -> None:
    """Tuỳ chọn koharu_erase_sfx: mỗi SFX ngoài bóng thoại thành một vùng 'chỉ redraw' (xoá theo mask, không dịch / chèn chữ)."""
    from app.services import koharu_detector as kd
    cfg = kd.current()
    if not (cfg.enabled and cfg.erase_sfx):
        return
    try:
        layout = kd.get_layout(image_path, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Không lấy được SFX từ koharu: %s", exc)
        return
    next_id = max((d.dialogue_id for d in dialogues), default=0) + 1
    # SFX nằm trong khung thoại đã dịch (kể cả hộp YOLO) không phải SFX ngoài bóng
    for box in kd.sfx_units(layout, exclude_boxes=[d.box_location for d in dialogues if not d.redraw_only]):
        dialogues.append(Dialogue(dialogue_id=next_id, box_location=box, original_text="", translated_text="",
                                  redraw_only=True, overrides={"type": "mask"}))
        next_id += 1


def get_translation_service(
    backend: str,
    api_key: str = "",
    model_name: str = DEFAULT_GEMINI_MODEL,
    yolo_model_path: str = "",
    enable_stray_text_removal: bool = False,
) -> TranslationService:
    """Factory: chọn service theo backend cấu hình trong Settings."""
    from app import config as _cfg

    if backend == _cfg.BACKEND_YOLO_OCR_GEMINI:
        return HybridOcrTranslationService(
            yolo_model_path=yolo_model_path,
            api_key=api_key,
            model_name=model_name,
            enable_stray_text_removal=enable_stray_text_removal,
        )
    # còn lại (kể cả cài đặt cũ mock / gemini_vision đã bỏ) -> YOLO + Gemini Vision
    return YoloGeminiVisionTranslationService(
        yolo_model_path=yolo_model_path, api_key=api_key, model_name=model_name
    )
