"""Cấu hình server, đọc từ biến môi trường (xem .env.example). Không có đường dẫn nào do client gửi lên."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

BACKEND_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = BACKEND_ROOT / "existing_app"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _path(name: str, default: Path) -> Path:
    raw = _env(name)
    p = Path(raw).expanduser() if raw else default
    if not p.is_absolute():
        p = BACKEND_ROOT / p
    return p.resolve()


@dataclass(frozen=True)
class ServerSettings:
    data_dir: Path
    model_dir: Path
    font_dir: Path
    database_url: str
    api_token: str                       # rỗng = tự sinh và lưu trong data_dir/api_token.txt
    cors_origins: List[str] = field(default_factory=list)
    max_upload_mb: int = 40              # kích thước tối đa MỖI ảnh
    max_image_pixels: int = 120_000_000  # chống "decompression bomb"
    translate_concurrency: int = 4       # mặc định như Desktop (DEFAULT_CONCURRENCY_LIMIT)
    hf_offline: bool = False

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"


@lru_cache(maxsize=1)
def get_settings() -> ServerSettings:
    data_dir = _path("MT_DATA_DIR", BACKEND_ROOT / "data")
    model_dir = _path("MODEL_DIR", BACKEND_ROOT / "models")
    s = ServerSettings(
        data_dir=data_dir,
        model_dir=model_dir,
        font_dir=_path("MT_FONT_DIR", BACKEND_ROOT / "fonts"),
        database_url=_env("MT_DATABASE_URL") or f"sqlite:///{(data_dir / 'app.db').as_posix()}",
        api_token=_env("MT_API_TOKEN"),
        cors_origins=[o.strip() for o in _env("MT_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
        max_upload_mb=int(_env("MT_MAX_UPLOAD_MB", "40")),
        max_image_pixels=int(_env("MT_MAX_IMAGE_PIXELS", "120000000")),
        translate_concurrency=max(1, int(_env("MT_TRANSLATE_CONCURRENCY", "4"))),
        hf_offline=_env("MT_HF_OFFLINE") in ("1", "true", "yes"),
    )
    bootstrap_environment(s)
    return s


def bootstrap_environment(s: ServerSettings) -> None:
    """Thư mục persistent + đường dẫn import module Desktop. Cache HuggingFace đặt trong MODEL_DIR để không mất khi restart."""
    for d in (s.data_dir, s.model_dir, s.projects_dir):
        d.mkdir(parents=True, exist_ok=True)
    hf = s.model_dir / ".hf-cache"
    hf.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf))
    os.environ.setdefault("HF_HUB_CACHE", str(hf / "hub"))
    if s.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("MANGA_APP_DATA", str(s.data_dir / "legacy_appdata"))   # qt_compat.QStandardPaths
    if str(LEGACY_ROOT) not in sys.path:
        sys.path.insert(0, str(LEGACY_ROOT))
