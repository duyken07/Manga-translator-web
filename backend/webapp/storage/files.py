"""Lưu trữ file trên đĩa: mỗi project một thư mục riêng, mọi đường dẫn do server tự dựng (không nhận path từ client)."""
from __future__ import annotations

import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Tuple

from PIL import Image, UnidentifiedImageError

from ..core.config import get_settings
from ..core.security import inside, safe_filename

# Cùng danh sách định dạng với Desktop (app/services/file_service.VALID_IMAGE_EXTS)
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_FORMAT_FOR_EXT = {".png": {"PNG"}, ".jpg": {"JPEG"}, ".jpeg": {"JPEG"}, ".webp": {"WEBP"}, ".bmp": {"BMP"}}


class UploadError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def originals(self) -> Path:
        return self.root / "originals"

    @property
    def working(self) -> Path:          # bản làm việc của "vẽ / fill thẳng lên ảnh" (file gốc không bao giờ bị sửa)
        return self.root / "working"

    @property
    def cache(self) -> Path:            # redraw cache (engine.cache_dir_for) - tồn tại qua restart
        return self.root / "redraw_cache"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def thumbs(self) -> Path:
        return self.root / "thumbs"

    def ensure(self) -> "ProjectPaths":
        for d in (self.originals, self.working, self.cache, self.exports, self.thumbs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def resolve_rel(self, rel: str) -> Path | None:
        """Đường dẫn tương đối (lưu trong DB) -> tuyệt đối, chỉ nếu nằm trong thư mục project và tồn tại."""
        if not rel:
            return None
        p = (self.root / rel)
        return p if inside(self.root, p) and p.is_file() else None

    def rel(self, p: str | Path) -> str:
        return Path(p).resolve().relative_to(self.root.resolve()).as_posix()


def project_paths(project_id: str) -> ProjectPaths:
    if not project_id.isalnum():
        raise ValueError("project_id không hợp lệ")
    return ProjectPaths(get_settings().projects_dir / project_id)


def save_upload(paths: ProjectPaths, filename: str, stream: BinaryIO) -> Tuple[str, int, int]:
    """Ghi ảnh upload (stream theo khối, có giới hạn) rồi xác thực bằng Pillow. Trả (page_name, width, height).
    Chống: path traversal, ext không hợp lệ, file quá lớn, nội dung không phải ảnh, ext không khớp định dạng thật, ảnh quá nhiều pixel."""
    s = get_settings()
    try:
        name = safe_filename(filename)
    except ValueError as e:
        raise UploadError(str(e)) from e
    ext = Path(name).suffix.lower()
    if ext not in VALID_IMAGE_EXTS:
        raise UploadError(f"Định dạng '{ext or '?'}' không được hỗ trợ (chấp nhận: {', '.join(sorted(VALID_IMAGE_EXTS))})")
    paths.ensure()
    dest = paths.originals / name
    if not inside(paths.originals, dest):
        raise UploadError("Đường dẫn không hợp lệ")
    limit = s.max_upload_mb * 1024 * 1024
    tmp = dest.with_name(dest.name + ".part")
    size = 0
    try:
        with open(tmp, "wb") as out:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise UploadError(f"Ảnh vượt quá {s.max_upload_mb} MB")
                out.write(chunk)
        if size == 0:
            raise UploadError("File rỗng")
        Image.MAX_IMAGE_PIXELS = s.max_image_pixels
        try:
            with Image.open(tmp) as im:
                fmt, (w, h) = im.format, im.size
                if fmt not in _FORMAT_FOR_EXT[ext]:
                    raise UploadError(f"Nội dung ({fmt}) không khớp đuôi file '{ext}'")
                if w * h > s.max_image_pixels:
                    raise UploadError("Ảnh có quá nhiều điểm ảnh")
                im.verify()
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as e:
            raise UploadError("File không phải ảnh hợp lệ") from e
        os.replace(tmp, dest)
        return name, w, h
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def make_thumbnail(src: Path, dst: Path, max_side: int = 220) -> Path:
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.thumbnail((max_side, max_side))
        im.convert("RGB").save(dst, "JPEG", quality=80)
    return dst


def delete_project_files(project_id: str) -> None:
    shutil.rmtree(project_paths(project_id).root, ignore_errors=True)
