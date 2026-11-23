"""Dựng RedrawOptions / cấu hình koharu + OCR của Desktop từ cài đặt web + đường dẫn model do ModelManager quản lý.
Tương đương MainWindow._redraw_options() (giữ nguyên thứ tự cấu hình và các trường)."""
from __future__ import annotations

from typing import Optional

from ..core.config import get_settings
from ..core.settings_store import UserSettings, get_user_settings, resolve_font


def apply_legacy_configuration(mm, st: UserSettings):
    """Cấu hình koharu + OCR (trạng thái module toàn cục của Desktop) rồi trả RedrawOptions."""
    from app.services import koharu_detector as kd, ocr_engine
    from app.services.redraw.engine import RedrawOptions

    koharu_dir = str(mm.path("koharu-yolo26s") or "")
    kd.configure(kd.KoharuConfig(enabled=st.use_koharu, model_dir=koharu_dir, imgsz=st.koharu_imgsz, device=st.lama_device or "cpu",
                                 threads=st.lama_threads, erase_sfx=st.koharu_erase_sfx, use_yolo=st.koharu_use_yolo,
                                 cache_dir=str(get_settings().data_dir / "koharu_cache")))
    main, fb = mm.ocr_model_ids()
    ocr_engine.configure(ocr_engine.OcrConfig(model=main, fallback_model=fb, force_cpu=st.ocr_force_cpu, low_conf=st.ocr_low_conf,
                                              gemini_rescue=st.ocr_gemini_rescue, thinking_level=st.gemini_thinking_level))
    lama = mm.path("lama-manga")
    return RedrawOptions(
        lama_path=str(lama) if lama else "",              # "" = chỉ OpenCV (đúng hành vi Desktop khi thiếu LaMa)
        device=st.lama_device, threads=st.lama_threads,
        font_path=resolve_font(st.redraw_font), text_color=st.redraw_text_color, line_spacing=st.redraw_line_spacing,
        koharu=st.use_koharu, koharu_model_dir=koharu_dir, koharu_imgsz=st.koharu_imgsz,
    )


def current_redraw_options(mm, st: Optional[UserSettings] = None):
    return apply_legacy_configuration(mm, st or get_user_settings())
