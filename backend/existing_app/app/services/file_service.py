"""Tiện ích thao tác file: quét thư mục ảnh, lọc định dạng hợp lệ."""
from __future__ import annotations

from pathlib import Path
from typing import List

from app.models import natural_sort_key

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def scan_image_folder(folder_path: str) -> List[str]:
    """Trả về danh sách đường dẫn ảnh trong thư mục, đã sắp xếp Natural Sort."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    files = [str(p) for p in folder.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS]
    files.sort(key=lambda p: natural_sort_key(Path(p).name))
    return files


def filter_valid_images(file_paths: List[str]) -> List[str]:
    """Lọc danh sách file do người dùng chọn thủ công, chỉ giữ định dạng ảnh hợp lệ."""
    valid = [p for p in file_paths if Path(p).suffix.lower() in VALID_IMAGE_EXTS]
    valid.sort(key=lambda p: natural_sort_key(Path(p).name))
    return valid


def collect_images(paths: List[str]) -> List[str]:
    """Gom ảnh từ danh sách đường dẫn người dùng chọn / kéo thả: thư mục -> quét ảnh bên trong (Natural Sort),
    tệp -> chỉ giữ định dạng ảnh hợp lệ. Bỏ trùng, giữ thứ tự: ảnh từ thư mục trước rồi tới các tệp lẻ."""
    images: List[str] = []
    loose: List[str] = []
    for p in paths:
        if Path(p).is_dir():
            images.extend(scan_image_folder(p))
        else:
            loose.append(p)
    images.extend(filter_valid_images(loose))
    seen: set = set()
    unique: List[str] = []
    for p in images:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique
