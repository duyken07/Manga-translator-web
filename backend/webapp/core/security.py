"""Xác thực token tĩnh, mã hóa secret (Gemini API key), chuẩn hóa tên file.

Token: MT_API_TOKEN hoặc file data/api_token.txt (tự sinh lần đầu; chỉ log ĐƯỜNG DẪN file, không log nội dung).
Secret: Fernet, khóa ở data/secret.key. Trên Windows chmod không có tác dụng -> bảo vệ thư mục data bằng ACL của hệ điều hành.
"""
from __future__ import annotations

import hmac
import os
import re
import secrets
import logging
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from fastapi import Header, HTTPException, Query, status

from .config import get_settings

log = logging.getLogger("security")
_token_cache: Optional[str] = None
_fernet: Optional[Fernet] = None


def _chmod600(p: Path) -> None:
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def get_api_token() -> str:
    global _token_cache
    if _token_cache:
        return _token_cache
    s = get_settings()
    if s.api_token:
        _token_cache = s.api_token
        return _token_cache
    f = s.data_dir / "api_token.txt"
    if f.is_file() and f.read_text(encoding="utf-8").strip():
        _token_cache = f.read_text(encoding="utf-8").strip()
    else:
        _token_cache = secrets.token_urlsafe(32)
        f.write_text(_token_cache, encoding="utf-8")
        _chmod600(f)
        log.warning("Đã sinh API token mới, lưu tại: %s", f)
    return _token_cache


def reset_security_caches() -> None:
    global _token_cache, _fernet
    _token_cache = None
    _fernet = None


def _check(candidate: Optional[str]) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate.encode(), get_api_token().encode())


def require_token(authorization: Optional[str] = Header(default=None), x_api_token: Optional[str] = Header(default=None)) -> str:
    cand = x_api_token
    if authorization and authorization.lower().startswith("bearer "):
        cand = authorization[7:].strip()
    if not _check(cand):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Thiếu hoặc sai API token", headers={"WWW-Authenticate": "Bearer"})
    return "owner"          # chỗ cắm cho multi-user: trả về user id thật khi có hệ thống tài khoản


def ws_token_ok(token: Optional[str]) -> bool:
    return _check(token)


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        s = get_settings()
        f = s.data_dir / "secret.key"
        if not f.is_file():
            f.write_bytes(Fernet.generate_key())
            _chmod600(f)
        _fernet = Fernet(f.read_bytes().strip())
    return _fernet


def encrypt_secret(value: str) -> str:
    return _get_fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> Optional[str]:
    try:
        return _get_fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError):
        return None


_BAD = re.compile(r"[\x00-\x1f<>:\"|?*]")


def safe_filename(name: str, max_len: int = 120) -> str:
    """Chỉ lấy phần tên file (bỏ thư mục kiểu / và \\), bỏ ký tự điều khiển / ký tự cấm của Windows. Rỗng/'.'/'..' -> ValueError."""
    base = (name or "").replace("\\", "/").split("/")[-1].strip().strip(".")
    base = _BAD.sub("_", base)
    if not base or base in (".", ".."):
        raise ValueError("Tên file không hợp lệ")
    if len(base) > max_len:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: max_len - len(ext) - 1] + "." + ext) if dot else base[:max_len]
    if base.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        base = "_" + base      # tên thiết bị đặc biệt của Windows
    return base


def inside(root: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
