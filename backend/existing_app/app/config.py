"""
Quản lý cấu hình ứng dụng: API Key, thư mục làm việc, tuỳ chọn ngôn ngữ.

- API Key được lưu qua `keyring` (hệ thống keychain của OS) khi có sẵn.
  Nếu môi trường không hỗ trợ keyring (VD: Linux server không có DBus/Secret
  Service), ứng dụng sẽ fallback về QSettings (registry/plist/ini) - kém an
  toàn hơn nhưng vẫn hoạt động được, kèm cảnh báo cho người dùng.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.qt_compat import QSettings

# Thư mục gốc của ứng dụng (chứa main.py). Đường dẫn mặc định bên dưới tính theo thư mục này,
# nên chạy `python main.py` từ đâu cũng được (không phụ thuộc thư mục làm việc hiện tại).
if getattr(sys, "frozen", False):
    # Đã đóng gói bằng PyInstaller: models/ và fonts/ nằm CẠNH file .exe (không nằm trong thư mục giải nén tạm / _internal)
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OCR_MODEL = "mayocream/manga-ocr"  # trùng app.services.ocr_engine.DEFAULT_OCR_MODEL (không import để config không kéo thêm module)
DEFAULT_YOLO_MODEL_REL = "models/comic-speech-bubble-detector.pt"
DEFAULT_LAMA_MODEL_REL = "models/lama-manga.safetensors"
DEFAULT_FONT_REL = "fonts/000 WildWords2 TB Roman.ttf"


def default_asset_path(rel: str) -> str:
    """Đường dẫn tuyệt đối tới file mặc định đi kèm app (./models, ./fonts). Trả về "" nếu file chưa có
    để app tự rơi về hành vi cũ (OpenCV / tự tìm font / hỏi chọn YOLO) thay vì báo lỗi đường dẫn hỏng."""
    p = APP_ROOT / rel
    return os.path.normpath(str(p)) if p.is_file() else ""


def characters_builtin_dir() -> Path:
    """Nhân vật chatbot đi kèm app: ./characters cạnh main.py (hoặc cạnh .exe), cùng quy ước với models/ và fonts/."""
    return APP_ROOT / "characters"


def characters_user_dir() -> Path:
    """Nhân vật do người dùng thêm / chỉnh sửa: thư mục dữ liệu người dùng (AppData trên Windows), không ghi vào thư mục cài đặt."""
    from app.qt_compat import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or str(Path.home() / ".manga_translator")
    return Path(base) / "characters"


ORG_NAME = "MangaTranslatorApp"
APP_NAME = "MangaTranslator"
KEYRING_SERVICE = "manga_translator_gemini_api_key"
KEYRING_USERNAME = "default_user"

SUPPORTED_SOURCE_LANGS = [
    ("ja", "Tiếng Nhật (JA)"),
    ("en", "Tiếng Anh (EN)"),
    ("zh", "Tiếng Trung (ZH)"),
    ("ko", "Tiếng Hàn (KO)"),
]
DEFAULT_TARGET_LANG = ("vi", "Tiếng Việt (VI)")

# Số ảnh dịch song song tối đa cùng lúc (Concurrency Limit) để tránh
# vượt quá Rate Limit (RPM) của Gemini API.
DEFAULT_CONCURRENCY_LIMIT = 4

# Danh sách model Gemini gợi ý hiển thị trong Settings (dropdown có thể gõ tự do).
# LƯU Ý: Google thường xuyên khai tử model cũ (VD: gemini-2.5-flash sẽ tắt
# 16/10/2026). Danh sách này chỉ mang tính gợi ý tại thời điểm viết code -
# luôn kiểm tra model mới nhất tại https://ai.google.dev/gemini-api/docs/models
# rồi gõ trực tiếp tên model vào ô combo nếu danh sách bên dưới đã lỗi thời.
# Model Gemini CỐ ĐỊNH (không còn chọn trong Thiết lập): model chính + model phụ tự dùng khi model chính lỗi - xem app/services/gemini_models.py
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODEL = "gemini-3.1-flash-lite"

# ---------------------------------------------------------------------------
# Backend dịch thuật: 2 lựa chọn hiển thị trong Settings UI
# ---------------------------------------------------------------------------
# Hai backend cũ (Mock giả lập, Gemini Vision thuần) đã bỏ khỏi giao diện; hằng số giữ lại CHỈ để đổi cài đặt đã lưu sang backend mới.
_REMOVED_BACKENDS = ("mock", "gemini_vision")
BACKEND_YOLO_OCR_GEMINI = "yolo_ocr_gemini"  # YOLO phát hiện bóng thoại + manga-ocr đọc chữ, Gemini chỉ lọc & dịch text
BACKEND_YOLO_GEMINI_VISION = "yolo_gemini_vision"  # YOLO xác định toạ độ (đánh số lên ảnh) + Gemini đọc/dịch trực tiếp trên ảnh đã đánh số

TRANSLATION_BACKEND_OPTIONS = [
    (BACKEND_YOLO_OCR_GEMINI, "YOLO + Manga-OCR + Gemini (YOLO định vị, Manga-OCR đọc chữ)"),
    (BACKEND_YOLO_GEMINI_VISION, "YOLO + Gemini Vision (YOLO định vị, Gemini đọc + dịch trên ảnh đã đánh số)"),
]
DEFAULT_TRANSLATION_BACKEND = BACKEND_YOLO_GEMINI_VISION


def normalize_backend(backend) -> str:
    """Backend không còn hỗ trợ (mock / gemini_vision) hoặc không hợp lệ -> backend mặc định."""
    return backend if backend in dict(TRANSLATION_BACKEND_OPTIONS) else DEFAULT_TRANSLATION_BACKEND


def _settings() -> QSettings:
    return QSettings(ORG_NAME, APP_NAME)


def save_api_key(api_key: str) -> None:
    """Lưu API key an toàn nhất có thể trên máy hiện tại."""
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, api_key)
        _settings().setValue("api_key_storage", "keyring")
    except Exception:
        # Fallback: lưu trong QSettings (ít an toàn hơn keyring hệ thống)
        _settings().setValue("api_key_storage", "qsettings")
        _settings().setValue("api_key_fallback", api_key)


def load_api_key() -> Optional[str]:
    storage = _settings().value("api_key_storage", None)
    if storage == "keyring":
        try:
            import keyring

            return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            return None
    if storage == "qsettings":
        return _settings().value("api_key_fallback", None)
    return None


def clear_api_key() -> None:
    storage = _settings().value("api_key_storage", None)
    if storage == "keyring":
        try:
            import keyring

            keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            pass
    _settings().remove("api_key_fallback")
    _settings().remove("api_key_storage")


@dataclass
class AppSettings:
    """Tuỳ chọn phiên làm việc hiện tại (không nhạy cảm, có thể lưu QSettings)."""

    source_lang: str = "ja"
    target_lang: str = "vi"
    concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT
    last_project_dir: Optional[str] = None
    gemini_model: str = DEFAULT_GEMINI_MODEL  # Chọn được trong Settings UI, không cần sửa code
    translation_backend: str = DEFAULT_TRANSLATION_BACKEND
    yolo_model_path: str = default_asset_path(DEFAULT_YOLO_MODEL_REL)  # Đường dẫn file .pt của model YOLO phát hiện bóng thoại
    # Khi bật: dịch TUẦN TỰ theo đúng thứ tự trang (thay vì song song), mang
    # glossary + vài câu thoại cuối trang trước sang làm ngữ cảnh cho trang
    # sau -> nhất quán tên riêng/thuật ngữ hơn, đổi lại chậm hơn đáng kể.
    sequential_context_mode: bool = True
    # Đường dẫn file JSON chứa bộ thuật ngữ tùy chỉnh do người dùng tự soạn
    # (tên riêng hay bị dịch sai...), độc lập với project, tái sử dụng được
    # giữa nhiều project/bộ truyện khác nhau.
    custom_glossary_path: str = ""
    # Khi bật: giữ nguyên hậu tố xưng hô (-san, -chan, -senpai...) thay vì
    # dịch/bỏ đi khi dịch tên riêng.
    keep_honorifics: bool = False
    # Backend YOLO+Manga-OCR: lọc ký tự/mảnh chữ bị "lọt" từ bóng thoại khác
    # vào vùng crop. MẶC ĐỊNH TẮT vì đã 2 lần ghi nhận xoá nhầm chữ hợp lệ khi
    # khoảng cách giữa các cột chữ dọc trong CÙNG 1 bóng thoại lớn hơn mức an
    # toàn - rủi ro mất chữ hợp lệ lớn hơn lợi ích loại bỏ hiếm khi xảy ra của
    # ký tự lạc. Chỉ bật nếu bạn CHỦ ĐỘNG gặp vấn đề chữ bị lẫn từ bóng khác.
    enable_stray_text_removal: bool = False
    # -- Redraw: xoá chữ gốc (OpenCV / lama-manga) + chèn bản dịch lên ảnh --------------------
    enable_redraw: bool = True
    paint_auto_redraw: bool = False  # Vẽ / Fill / bút ma thuật có TỰ ĐỘNG redraw + xếp lại khung chữ không (mặc định TẮT: chỉ redraw khi bấm tay)
    paint_direct: bool = False  # Vẽ / Fill thẳng lên ảnh (không tạo lớp, chỉ chỉnh trên bản làm việc trong app); file gốc của người dùng không bao giờ bị sửa
    lama_model_path: str = default_asset_path(DEFAULT_LAMA_MODEL_REL)  # lama-manga.safetensors; trống = chỉ dùng OpenCV
    lama_device: str = "cpu"  # cpu | cuda | mps
    lama_threads: int = 0  # 0 = để torch tự chọn
    redraw_font_path: str = default_asset_path(DEFAULT_FONT_REL)  # font thường (.ttf, có dấu tiếng Việt); trống = tự tìm
    font_only_dir: str = ""  # chỉ dùng font trong thư mục này (danh sách font + font mặc định); trống = ./fonts + font hệ thống
    redraw_text_color: str = "20,20,20"  # R,G,B của chữ tối
    redraw_line_spacing: float = 1.06  # hệ số giãn dòng mặc định (0.7 - 2.5)
    theme: str = "system"  # light | dark | system (theo hệ điều hành)
    # -- koharu-yolo26s (bóng thoại + vùng chữ có mask): dùng cho cả bước dịch (định vị / cắt OCR) và redraw (vùng cần xoá) --
    use_koharu: bool = False
    koharu_model_dir: str = ""  # thư mục có model.safetensors + *.yaml + config.json; trống = tải mayocream/koharu-yolo26s từ HuggingFace
    koharu_imgsz: int = 0  # 0 = theo config.json (1280); CPU chậm thì 960
    koharu_erase_sfx: bool = False  # khi dịch, thêm vùng 'chỉ redraw' cho SFX (mặc định giữ nguyên SFX)
    koharu_use_yolo: bool = True  # có file YOLO bóng thoại thì kết hợp với koharu; False = chỉ koharu (nhanh hơn, 1 lượt model / trang)
    # -- Manga-OCR (xem app/services/ocr_engine.py) --
    manga_ocr_model: str = DEFAULT_OCR_MODEL  # HuggingFace repo hoặc thư mục local; lỗi nạp thì lùi về kha-white/manga-ocr-base
    ocr_force_cpu: bool = False
    ocr_gemini_rescue: bool = True  # câu OCR đọc chưa chắc: gửi kèm ảnh crop cho Gemini đối chiếu
    ocr_low_conf: float = 0.60
    gemini_thinking_level: str = "low"  # bước lọc lỗi OCR + dịch (backend YOLO + Manga-OCR + Gemini); "high" làm mỗi trang ~100 s

    def save(self) -> None:
        s = _settings()
        s.setValue("source_lang", self.source_lang)
        s.setValue("target_lang", self.target_lang)
        s.setValue("concurrency_limit", self.concurrency_limit)
        s.setValue("last_project_dir", self.last_project_dir)
        s.setValue("gemini_model", self.gemini_model)
        s.setValue("translation_backend", self.translation_backend)
        s.setValue("yolo_model_path", self.yolo_model_path)
        s.setValue("sequential_context_mode", self.sequential_context_mode)
        s.setValue("custom_glossary_path", self.custom_glossary_path)
        s.setValue("keep_honorifics", self.keep_honorifics)
        s.setValue("enable_stray_text_removal", self.enable_stray_text_removal)
        s.setValue("enable_redraw", self.enable_redraw)
        s.setValue("paint_direct", self.paint_direct)
        s.setValue("paint_auto_redraw", self.paint_auto_redraw)
        s.setValue("lama_model_path", self.lama_model_path)
        s.setValue("lama_device", self.lama_device)
        s.setValue("lama_threads", self.lama_threads)
        s.setValue("redraw_font_path", self.redraw_font_path)
        s.setValue("font_only_dir", self.font_only_dir)
        s.setValue("redraw_text_color", self.redraw_text_color)
        s.setValue("redraw_line_spacing", self.redraw_line_spacing)
        s.setValue("theme", self.theme)
        s.setValue("use_koharu", self.use_koharu)
        s.setValue("koharu_model_dir", self.koharu_model_dir)
        s.setValue("koharu_imgsz", self.koharu_imgsz)
        s.setValue("koharu_erase_sfx", self.koharu_erase_sfx)
        s.setValue("koharu_use_yolo", self.koharu_use_yolo)
        s.setValue("manga_ocr_model", self.manga_ocr_model)
        s.setValue("ocr_force_cpu", self.ocr_force_cpu)
        s.setValue("ocr_gemini_rescue", self.ocr_gemini_rescue)
        s.setValue("ocr_low_conf", self.ocr_low_conf)
        s.setValue("gemini_thinking_level", self.gemini_thinking_level)

    @classmethod
    def load(cls) -> "AppSettings":
        s = _settings()
        backend = normalize_backend(s.value("translation_backend", None))   # cài mới / backend đã bỏ -> mặc định
        return cls(
            source_lang=s.value("source_lang", "ja"),
            target_lang=s.value("target_lang", "vi"),
            concurrency_limit=int(s.value("concurrency_limit", DEFAULT_CONCURRENCY_LIMIT)),
            last_project_dir=s.value("last_project_dir", None),
            gemini_model=DEFAULT_GEMINI_MODEL,   # bỏ giá trị cũ đã lưu: model cố định
            translation_backend=backend,
            yolo_model_path=s.value("yolo_model_path", default_asset_path(DEFAULT_YOLO_MODEL_REL)),
            sequential_context_mode=s.value("sequential_context_mode", True, type=bool),
            custom_glossary_path=s.value("custom_glossary_path", ""),
            keep_honorifics=s.value("keep_honorifics", False, type=bool),
            enable_stray_text_removal=s.value("enable_stray_text_removal", False, type=bool),
            enable_redraw=s.value("enable_redraw", True, type=bool),
            paint_direct=s.value("paint_direct", False, type=bool),
            paint_auto_redraw=s.value("paint_auto_redraw", False, type=bool),
            lama_model_path=s.value("lama_model_path", default_asset_path(DEFAULT_LAMA_MODEL_REL)),
            lama_device=s.value("lama_device", "cpu"),
            lama_threads=int(s.value("lama_threads", 0)),
            redraw_font_path=s.value("redraw_font_path", default_asset_path(DEFAULT_FONT_REL)),
            font_only_dir=str(s.value("font_only_dir", "")),
            redraw_text_color=s.value("redraw_text_color", "20,20,20"),
            redraw_line_spacing=float(s.value("redraw_line_spacing", 1.06)),
            theme=str(s.value("theme", "system")),
            use_koharu=s.value("use_koharu", False, type=bool),
            koharu_model_dir=s.value("koharu_model_dir", ""),
            koharu_imgsz=int(s.value("koharu_imgsz", 0)),
            koharu_erase_sfx=s.value("koharu_erase_sfx", False, type=bool),
            koharu_use_yolo=s.value("koharu_use_yolo", True, type=bool),
            manga_ocr_model=str(s.value("manga_ocr_model", DEFAULT_OCR_MODEL)) or DEFAULT_OCR_MODEL,
            ocr_force_cpu=s.value("ocr_force_cpu", False, type=bool),
            ocr_gemini_rescue=s.value("ocr_gemini_rescue", True, type=bool),
            ocr_low_conf=float(s.value("ocr_low_conf", 0.60)),
            gemini_thinking_level=str(s.value("gemini_thinking_level", "low")) or "low",
        )
