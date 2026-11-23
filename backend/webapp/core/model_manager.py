"""Quản lý vòng đời model AI: kiểm tra file, tải từ Hugging Face (một lần), nạp lười + cache instance, trạng thái, tiến trình.

Model nằm ở MODEL_DIR (persistent, cấu hình được; không dùng thư mục tạm). Sau khi tải xong, suy luận chạy offline.
Suy luận torch của TOÀN process đi qua `app.threading_utils.TORCH_INFER_LOCK` (khoá của bản Desktop) -> tối đa 1 suy luận torch
cùng lúc; `inference_lock` expose khoá này. Instance model do các module Desktop tự cache (YOLO/OCR/LaMa/koharu) nên không bị nạp lại
theo từng request; ModelManager chỉ điều phối tải + nạp lần đầu và theo dõi trạng thái.

Trạng thái: not_downloaded -> downloading -> downloaded -> loading -> ready ; lỗi ở bất kỳ bước nào -> error (gọi lại để thử lại).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import ServerSettings

log = logging.getLogger("model_manager")

NOT_DOWNLOADED, DOWNLOADING, DOWNLOADED, LOADING, READY, ERROR = (
    "not_downloaded", "downloading", "downloaded", "loading", "ready", "error")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    repo: str
    kind: str                                   # "file" | "snapshot"
    rel: str                                    # đường dẫn tương đối trong MODEL_DIR (file hoặc thư mục)
    filename: str = ""                          # kind == "file": tên file trong repo
    allow_patterns: tuple = ()
    required: tuple = ()                        # kind == "snapshot": file bắt buộc phải có
    fallback_repo: str = ""                     # chỉ OCR: repo dự phòng
    fallback_rel: str = ""
    optional: bool = False                      # koharu: tuỳ chọn (use_koharu)


SPECS: Dict[str, ModelSpec] = {s.id: s for s in (
    ModelSpec("yolo-bubble", "YOLO – phát hiện bóng thoại", "ogkalu/comic-speech-bubble-detector-yolov8m", "file",
              rel="comic-speech-bubble-detector.pt", filename="comic-speech-bubble-detector.pt"),
    ModelSpec("lama-manga", "LaMa Manga – inpainting", "mayocream/lama-manga", "file",
              rel="lama-manga.safetensors", filename="lama-manga.safetensors"),
    ModelSpec("manga-ocr", "Manga-OCR", "mayocream/manga-ocr", "snapshot", rel="manga-ocr/mayocream--manga-ocr",
              required=("config.json",), fallback_repo="kha-white/manga-ocr-base", fallback_rel="manga-ocr/kha-white--manga-ocr-base"),
    ModelSpec("koharu-yolo26s", "koharu-yolo26s – bóng thoại + vùng chữ", "mayocream/koharu-yolo26s", "snapshot",
              rel="koharu-yolo26s", allow_patterns=("*.safetensors", "*.yaml", "*.json"),
              required=("model.safetensors",), optional=True),
)}


@dataclass
class ModelStatus:
    state: str = NOT_DOWNLOADED
    progress: Optional[float] = None            # 0..1 khi đang tải
    error: str = ""
    version: str = ""
    updated: float = field(default_factory=time.time)


Downloader = Callable[[ModelSpec, str, Path, Path, Callable[[Optional[float]], None]], str]
Loader = Callable[["ModelManager", ModelSpec], None]


def _hf_total_bytes(repo: str, patterns: tuple, filename: str) -> Optional[int]:
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        total = 0
        import fnmatch
        for sib in info.siblings or []:
            if filename and sib.rfilename == filename or (patterns and any(fnmatch.fnmatch(sib.rfilename, p) for p in patterns)) \
                    or (not filename and not patterns):
                total += int(sib.size or 0)
        return total or None
    except Exception:  # noqa: BLE001 - tiến trình chỉ là tiện ích
        return None


def _dir_bytes(d: Path) -> int:
    n = 0
    try:
        for p in d.rglob("*"):
            if p.is_file():
                n += p.stat().st_size
    except OSError:
        pass
    return n


def hf_downloader(spec: ModelSpec, repo: str, dest: Path, model_dir: Path, progress: Callable[[Optional[float]], None]) -> str:
    """Tải thật từ Hugging Face bằng huggingface_hub. Trả commit sha (nếu biết). Tiến trình = dung lượng đã ghi / tổng (poll)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    total = _hf_total_bytes(repo, spec.allow_patterns, spec.filename if spec.kind == "file" else "")
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(0.5):
            if not total:
                continue
            if spec.kind == "snapshot":
                done = _dir_bytes(dest)                                        # gồm cả *.incomplete trong dest/.cache
            else:                                                              # hf_hub_download(local_dir=...) ghi *.incomplete trong model_dir/.cache
                done = dest.stat().st_size if dest.is_file() else _dir_bytes(model_dir / ".cache")
            progress(min(0.99, done / total))

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        if spec.kind == "file":
            hf_hub_download(repo_id=repo, filename=spec.filename, local_dir=str(model_dir))
        else:
            dest.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=list(spec.allow_patterns) or None)
    finally:
        stop.set()
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo).sha or ""
    except Exception:  # noqa: BLE001
        return ""


class ModelManager:
    def __init__(self, settings: ServerSettings, downloader: Optional[Downloader] = None, loaders: Optional[Dict[str, Loader]] = None):
        self.model_dir = settings.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._downloader: Downloader = downloader or hf_downloader
        self._loaders: Dict[str, Loader] = loaders if loaders is not None else default_loaders()
        self._lock = threading.RLock()
        self._status: Dict[str, ModelStatus] = {k: ModelStatus() for k in SPECS}
        self._load_locks: Dict[str, threading.Lock] = {k: threading.Lock() for k in SPECS}
        self._dl_locks: Dict[str, threading.Lock] = {k: threading.Lock() for k in SPECS}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-dl")   # tải tuần tự: tiết kiệm băng thông / đĩa
        self._futures: Dict[str, object] = {}
        self._manifest_path = self.model_dir / "manifest.json"
        self.rescan()

    # ------------------------------------------------------------------ đĩa
    def _present(self, spec: ModelSpec, rel: Optional[str] = None) -> bool:
        p = self.model_dir / (rel or spec.rel)
        if spec.kind == "file":
            return p.is_file() and p.stat().st_size > 0
        return p.is_dir() and all((p / f).is_file() and (p / f).stat().st_size > 0 for f in (spec.required or ())) \
            and (bool(spec.required) or any(p.iterdir()))

    def _manifest(self) -> dict:
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_manifest(self, spec: ModelSpec, repo: str, commit: str) -> None:
        with self._lock:
            m = self._manifest()
            m[spec.id] = {"repo": repo, "commit": commit, "downloaded_at": datetime.now(timezone.utc).isoformat()}
            tmp = self._manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
            tmp.replace(self._manifest_path)

    def rescan(self) -> None:
        """Đồng bộ trạng thái với đĩa (model được cung cấp sẵn bằng tay cũng được nhận)."""
        m = self._manifest()
        with self._lock:
            for sid, spec in SPECS.items():
                st = self._status[sid]
                if st.state in (DOWNLOADING, LOADING, READY):
                    continue
                ok = self._present(spec) or (bool(spec.fallback_rel) and self._present(spec, spec.fallback_rel))
                st.state = DOWNLOADED if ok else NOT_DOWNLOADED
                st.error = ""
                st.progress = 1.0 if ok else None
                st.version = (m.get(sid) or {}).get("commit", "") or ("local" if ok else "")

    # ------------------------------------------------------------------ truy vấn
    def status(self) -> List[dict]:
        with self._lock:
            out = []
            for sid, spec in SPECS.items():
                st = self._status[sid]
                out.append({"id": sid, "label": spec.label, "repo": spec.repo, "state": st.state, "progress": st.progress,
                            "error": st.error, "version": st.version, "optional": spec.optional,
                            "file": spec.filename or spec.rel.split("/")[-1]})    # chỉ tên file, không lộ đường dẫn server
            return out

    def state(self, sid: str) -> str:
        return self._status[sid].state

    def is_available(self, sid: str) -> bool:
        return self._status[sid].state in (DOWNLOADED, LOADING, READY)

    def path(self, sid: str) -> Optional[Path]:
        spec = SPECS[sid]
        p = self.model_dir / spec.rel
        return p if self._present(spec) else None

    def ocr_model_ids(self) -> tuple[str, str]:
        """(model chính, model dự phòng) cho ocr_engine.OcrConfig: thư mục local nếu đã tải, nếu không thì repo id."""
        spec = SPECS["manga-ocr"]
        main = str(self.model_dir / spec.rel) if self._present(spec) else spec.repo
        fb = str(self.model_dir / spec.fallback_rel) if self._present(spec, spec.fallback_rel) else spec.fallback_repo
        return main, fb

    @property
    def inference_lock(self):
        from app.threading_utils import TORCH_INFER_LOCK     # khoá suy luận torch dùng chung của Desktop
        return TORCH_INFER_LOCK

    # ------------------------------------------------------------------ tải
    def _set(self, sid: str, **kw) -> None:
        with self._lock:
            st = self._status[sid]
            for k, v in kw.items():
                setattr(st, k, v)
            st.updated = time.time()

    def download(self, sid: str, wait: bool = False) -> None:
        """Tải model nếu chưa có (idempotent). wait=True: chặn tới khi xong (ném RuntimeError nếu lỗi)."""
        if sid not in SPECS:
            raise KeyError(sid)
        with self._lock:
            st = self._status[sid]
            if st.state in (DOWNLOADED, LOADING, READY):
                return
            if st.state != DOWNLOADING:
                st.state, st.progress, st.error = DOWNLOADING, 0.0, ""
                self._futures[sid] = self._pool.submit(self._do_download, sid)
            fut = self._futures.get(sid)
        if wait and fut is not None:
            fut.result()
            if self._status[sid].state == ERROR:
                raise RuntimeError(self._status[sid].error)

    def _do_download(self, sid: str) -> None:
        spec = SPECS[sid]
        with self._dl_locks[sid]:
            attempts = [(spec.repo, spec.rel)] + ([(spec.fallback_repo, spec.fallback_rel)] if spec.fallback_repo else [])
            errors: list = []
            for repo, rel in attempts:
                try:
                    dest = self.model_dir / rel
                    s2 = spec if repo == spec.repo else ModelSpec(**{**spec.__dict__, "repo": repo, "rel": rel})
                    commit = self._downloader(s2, repo, dest, self.model_dir, lambda p: self._set(sid, progress=p))
                    if not self._present(spec, rel):
                        raise RuntimeError(f"Tải xong nhưng thiếu file bắt buộc của '{repo}'")
                    self._write_manifest(spec, repo, commit or "")
                    self._set(sid, state=DOWNLOADED, progress=1.0, error="", version=commit or "local")
                    log.info("Đã tải model %s từ %s", sid, repo)
                    return
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{repo}: {exc}")
                    log.warning("Tải model %s từ %s lỗi: %s", sid, repo, exc)
            self._set(sid, state=ERROR, progress=None, error=" | ".join(errors)[:600])

    # ------------------------------------------------------------------ nạp
    def load(self, sid: str) -> None:
        """Nạp model vào bộ nhớ (một lần; gọi lại khi đã ready là no-op). Ném RuntimeError nếu chưa tải / nạp lỗi."""
        if sid not in SPECS:
            raise KeyError(sid)
        with self._load_locks[sid]:
            st = self._status[sid]
            if st.state == READY:
                return
            spec = SPECS[sid]
            present = self._present(spec) or (bool(spec.fallback_rel) and self._present(spec, spec.fallback_rel))
            if not present:
                raise RuntimeError(f"Model '{sid}' chưa được tải (trạng thái: {st.state})")
            self._set(sid, state=LOADING, error="")
            try:
                loader = self._loaders.get(sid)
                if loader is None:
                    raise RuntimeError(f"Không có loader cho '{sid}'")
                loader(self, SPECS[sid])
            except Exception as exc:  # noqa: BLE001
                self._set(sid, state=ERROR, error=str(exc)[:600])
                raise RuntimeError(str(exc)) from exc
            self._set(sid, state=READY, progress=1.0)

    def ensure(self, sid: str, load: bool = True) -> None:
        """Đảm bảo model sẵn sàng: tải nếu thiếu rồi nạp (chặn). Dùng trước khi chạy job."""
        self.download(sid, wait=True)
        if load:
            self.load(sid)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------- loader thật (import lười các module Desktop)
def default_loaders() -> Dict[str, Loader]:
    def yolo(mm: ModelManager, spec: ModelSpec) -> None:
        from app.services.manga_ocr_pipeline import get_yolo_model
        get_yolo_model(str(mm.model_dir / spec.rel))

    def lama(mm: ModelManager, spec: ModelSpec) -> None:
        from app.services.redraw import engine
        from webapp.adapters.options import current_redraw_options
        obj, err = engine.get_lama(current_redraw_options(mm))
        if obj is None:
            raise RuntimeError(err or "Không nạp được LaMa")

    def ocr(mm: ModelManager, spec: ModelSpec) -> None:
        from app.services import ocr_engine
        main, fb = mm.ocr_model_ids()
        ocr_engine.get_mocr(ocr_engine.OcrConfig(model=main, fallback_model=fb))

    def koharu(mm: ModelManager, spec: ModelSpec) -> None:
        from app.services import koharu_detector as kd
        kd._load_model(kd.KoharuConfig(enabled=True, model_dir=str(mm.model_dir / spec.rel)))

    return {"yolo-bubble": yolo, "lama-manga": lama, "manga-ocr": ocr, "koharu-yolo26s": koharu}
