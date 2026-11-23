"""Manga-OCR dùng chung cho cả app: chọn model (mặc định mayocream/manga-ocr), đo độ tin cậy từng lần đọc, chuẩn hoá text.

Vì sao có module này (trước đây app gọi thẳng `MangaOcr()`):
  * model cố định kha-white/manga-ocr-base, không đổi được; nay cấu hình được + tự lùi về model dự phòng nếu tải/nạp lỗi;
  * Manga-OCR chỉ trả chuỗi, không cho biết đọc chắc hay không -> ở đây tính xác suất token (geometric mean) để biết câu nào
    nên đọc lại bằng biến thể crop khác hoặc gửi ảnh cho Gemini đối chiếu;
  * post_process của manga-ocr đổi "..." thành "．．．" (fullwidth) -> chuẩn hoá về "…" trước khi đưa Gemini.

manga_ocr / torch / transformers chỉ được import LÚC GỌI (app không dùng backend này vẫn chạy bình thường).
Phần đo độ tin cậy dựa trên thuộc tính model/tokenizer/processor của đối tượng MangaOcr; nếu bản manga-ocr đang cài không có
các thuộc tính đó thì tự lùi về gọi thường (conf=None) - không bao giờ làm hỏng bước OCR.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("ocr_engine")

DEFAULT_OCR_MODEL = "mayocream/manga-ocr"
FALLBACK_OCR_MODEL = "kha-white/manga-ocr-base"


@dataclass
class OcrConfig:
    model: str = DEFAULT_OCR_MODEL          # HuggingFace repo hoặc thư mục local
    fallback_model: str = FALLBACK_OCR_MODEL  # dùng khi `model` không nạp được; "" = không lùi
    force_cpu: bool = False                  # False = manga-ocr tự dùng CUDA/MPS nếu có
    low_conf: float = 0.60                   # conf (0-1) dưới mức này = "đọc chưa chắc"
    gemini_rescue: bool = True               # gửi ảnh crop của các câu đọc chưa chắc cho Gemini đối chiếu
    max_rescue: int = 8                      # tối đa số ảnh crop đính kèm mỗi trang (giữ chi phí token)
    thinking_level: str = "low"              # mức suy luận của Gemini cho bước lọc lỗi + dịch: minimal | low | medium | high


_CFG = OcrConfig()


def configure(cfg: OcrConfig) -> None:
    global _CFG
    _CFG = cfg


def current() -> OcrConfig:
    return _CFG


@dataclass
class OcrResult:
    text: str
    conf: Optional[float] = None   # 0-1; None = không đo được


# --------------------------------------------------------------------------- nạp model
_instances: Dict[Tuple[str, bool], Any] = {}


def _locks():
    # dùng CHUNG khoá với YOLO / koharu (manga_ocr_pipeline) để suy luận không chạy chồng nhau giữa các luồng dịch
    from app.services.manga_ocr_pipeline import _INFER_LOCK, _LOAD_LOCK
    return _LOAD_LOCK, _INFER_LOCK


def _cap_threads() -> None:
    """torch mặc định dùng mọi lõi -> giao diện bị bỏ đói CPU. Đặt số luồng một lần, dưới khoá suy luận chung."""
    try:
        from app.threading_utils import apply_torch_threads
        apply_torch_threads(0)
    except Exception:  # noqa: BLE001
        pass


def get_mocr(cfg: Optional[OcrConfig] = None):
    """Đối tượng manga_ocr.MangaOcr (cache theo (model, force_cpu)). An toàn khi gọi từ nhiều luồng."""
    cfg = cfg or _CFG
    key = (cfg.model, cfg.force_cpu)
    got = _instances.get(key)
    if got is not None:
        return got
    load_lock, _ = _locks()
    with load_lock:
        if key in _instances:
            return _instances[key]
        try:
            from manga_ocr import MangaOcr
        except ImportError as exc:
            raise ImportError("Chưa cài thư viện manga-ocr. Chạy: pip install -U manga-ocr") from exc
        last: Optional[Exception] = None
        errors: list = []   # (model_id, lỗi) của MỌI model đã thử: trước đây chỉ giữ lỗi cuối (của model dự phòng) nên che mất lỗi thật
        for model_id in dict.fromkeys(m for m in (cfg.model, cfg.fallback_model) if m):
            try:
                log.info("Đang nạp Manga-OCR '%s' (lần đầu có thể phải tải model, cần mạng)...", model_id)
                inst = MangaOcr(pretrained_model_name_or_path=model_id, force_cpu=cfg.force_cpu)
                _cap_threads()
                if model_id != cfg.model:
                    log.warning("Không nạp được '%s' -> đang dùng model dự phòng '%s'", cfg.model, model_id)
                _instances[key] = inst
                return inst
            except Exception as exc:  # noqa: BLE001
                last = exc
                errors.append((model_id, exc))
                log.warning("Nạp Manga-OCR '%s' lỗi: %s", model_id, exc)
        detail = " | ".join(f"'{m}': {e}" for m, e in errors)
        raise RuntimeError(f"Không nạp được Manga-OCR ('{cfg.model}'): {detail}{_load_hint(errors)}") from last


def _frozen_missing_deps() -> list:
    """Bản đóng gói (PyInstaller): tokenizer của Manga-OCR cần fugashi + từ điển MeCab của unidic_lite. Từ điển là file dữ liệu, không phải
    module Python nên PyInstaller không tự gom -> thiếu thì tokenizer không dựng được (báo lỗi 'vocabulary'). Trả về danh sách thứ thiếu."""
    import sys
    if not getattr(sys, "frozen", False):
        return []
    miss = []
    try:
        import fugashi  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        miss.append(f"fugashi ({exc})")
    try:
        import os
        import unidic_lite
        d = getattr(unidic_lite, "DICDIR", "") or ""
        if not d or not os.path.isdir(d) or not os.listdir(d):
            miss.append(f"từ điển unidic_lite (không thấy dữ liệu tại '{d}')")
    except Exception as exc:  # noqa: BLE001
        miss.append(f"unidic_lite ({exc})")
    return miss


def _load_hint(errors) -> str:
    """Gợi ý cách sửa khi lỗi nạp model liên quan đến vocabulary / tokenizer (cả model chính lẫn dự phòng cùng hỏng thường là do môi trường,
    không phải do tên model)."""
    miss = _frozen_missing_deps()
    if miss:
        return ("\n\nBản đóng gói (.exe) thiếu: " + "; ".join(miss) + ".\n"
                "Cách xử lý: build lại bằng PyInstaller với --collect-all fugashi --collect-all unidic_lite "
                "--collect-all transformers --collect-all tokenizers (xem tools/build_exe.bat).")
    text = " ".join(str(e) for _, e in errors).lower()
    if "vocab" not in text:
        return ""
    tv, major = None, 0
    try:
        from importlib.metadata import version
        tv = version("transformers")
        major = int(tv.split(".")[0])
    except Exception:  # noqa: BLE001
        pass
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as cache
    except Exception:  # noqa: BLE001
        cache = "~/.cache/huggingface/hub"
    lines = ["", "", "Cách xử lý:"]
    if major >= 5:
        lines.append(f"• Bạn đang dùng transformers {tv}: bản 5.x đã bỏ loại tokenizer mà Manga-OCR cần. "
                     'Chạy: pip install "transformers<5" rồi mở lại app.')
    elif tv:
        lines.append(f"• Đang dùng transformers {tv}. Nếu vừa nâng cấp thư viện, thử: pip install -U manga-ocr fugashi unidic-lite")
    lines.append(f"• Nếu không phải do phiên bản: file model trong cache có thể bị hỏng hoặc tải dở. Xoá thư mục "
                 f"models--mayocream--manga-ocr và models--kha-white--manga-ocr-base trong {cache} rồi dịch lại (cần mạng để tải lại, khoảng 400 MB).")
    lines.append("• Đường dẫn cache có ký tự có dấu (tên người dùng Windows tiếng Việt) cũng có thể gây lỗi: đặt biến môi trường "
                 "HF_HOME trỏ tới thư mục không dấu, ví dụ D:\\hf_cache.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- đọc + độ tin cậy
def _post_process(text: str) -> str:
    try:
        from manga_ocr.ocr import post_process      # cùng hàm manga-ocr dùng nội bộ
        return post_process(text)
    except Exception:  # noqa: BLE001
        return "".join(text.split())


def _read_with_conf(mocr, img) -> Optional[OcrResult]:
    """Một lần generate có lấy xác suất token. None nếu bản manga-ocr đang cài không đủ thuộc tính (-> gọi thường)."""
    model = getattr(mocr, "model", None)
    tok = getattr(mocr, "tokenizer", None)
    proc = getattr(mocr, "processor", None)
    if model is None or tok is None or proc is None or not hasattr(model, "compute_transition_scores"):
        return None
    import torch
    pre = getattr(mocr, "_preprocess", None)
    x = pre(img) if callable(pre) else proc(img.convert("L").convert("RGB"), return_tensors="pt").pixel_values.squeeze()
    with torch.no_grad():
        out = model.generate(x[None].to(model.device), max_length=300, output_scores=True, return_dict_in_generate=True)
        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    lp = ts[torch.isfinite(ts)]
    conf = float(torch.exp(lp.mean())) if lp.numel() else None
    text = _post_process(tok.decode(out.sequences[0].cpu(), skip_special_tokens=True))
    return OcrResult(text.strip(), conf)


def read(img, cfg: Optional[OcrConfig] = None) -> OcrResult:
    """OCR một ảnh PIL. Luôn trả kết quả (không ném lỗi do phần đo conf)."""
    mocr = get_mocr(cfg)
    _, infer_lock = _locks()
    with infer_lock:
        try:
            res = _read_with_conf(mocr, img)
            if res is not None:
                return res
        except Exception as exc:  # noqa: BLE001
            log.debug("Không đo được độ tin cậy OCR, đọc thường: %s", exc)
        return OcrResult(str(mocr(img)).strip(), None)


# --------------------------------------------------------------------------- chuẩn hoá + kiểm tra chất lượng
_DOTS = re.compile(r"[．.・]{2,}")
_JP_CHAR = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef々ー〜～…\d]")


def normalize_ocr_text(s: str) -> str:
    """Dãy chấm (．．． / ... / ・・・) -> … (3 chấm = 1 dấu …); bỏ khoảng trắng. Không đụng tới chữ."""
    s = "".join((s or "").split())
    return _DOTS.sub(lambda m: "…" * max(1, round((m.end() - m.start()) / 3)), s)


def looks_suspicious(text: str) -> bool:
    """Dấu hiệu OCR hỏng dù conf cao: rỗng, một ký tự lặp >= 6 lần (ảo giác trên ảnh trắng), hoặc phần lớn không phải chữ Nhật."""
    if not text:
        return True
    if re.search(r"(.)\1{5,}", text) and len(set(text)) <= 2:
        return True
    non_jp = sum(1 for c in text if not _JP_CHAR.match(c))
    return non_jp / len(text) > 0.3


def is_low_quality(text: str, conf: Optional[float], low_conf: float) -> bool:
    return looks_suspicious(text) or (conf is not None and conf < low_conf)


def quality_score(text: str, conf: Optional[float], low_conf: float) -> float:
    """Để chọn kết quả tốt nhất giữa các biến thể crop. Có conf: dùng conf (text đáng ngờ bị chặn ở 0.3); không có: 0 / 1."""
    bad = looks_suspicious(text)
    if conf is None or math.isnan(conf):
        return 0.0 if bad else 1.0
    return min(conf, 0.3) if bad else conf
