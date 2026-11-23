"""Bảng SQLite. Dữ liệu trang giữ nguyên schema JSON của Desktop (Page.to_dict) trong cột `data` để tương thích 100%."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    owner_id: Mapped[str] = mapped_column(String(64), default="owner", index=True)   # chỗ cắm multi-user
    name: Mapped[str] = mapped_column(String(200))
    source_lang: Mapped[str] = mapped_column(String(16), default="ja")
    target_lang: Mapped[str] = mapped_column(String(16), default="vi")
    work_title: Mapped[str] = mapped_column(String(300), default="")
    glossary: Mapped[dict] = mapped_column(JSON, default=dict)
    grammar_cache: Mapped[dict] = mapped_column(JSON, default=dict)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)           # ghi đè cài đặt theo project (whitelist)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    pages: Mapped[list["PageRow"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class PageRow(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("project_id", "page_name", name="uq_page_name"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    page_name: Mapped[str] = mapped_column(String(200))                  # khoá tương thích JSON Desktop
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="pending")   # pending|translating|done|error
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)               # {"dialogues":[...],"strokes":[...],"fills":[...],"direct_edited":bool}
    redraw_meta: Mapped[dict] = mapped_column(JSON, default=dict)        # {status,out,layers,layout_boxes,layout_info} (đường dẫn TƯƠNG ĐỐI)
    project: Mapped[ProjectRow] = relationship(back_populates="pages")


class JobRow(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    project_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(24))        # translate|redraw|export|model_download|detect
    status: Mapped[str] = mapped_column(String(16), default="pending")   # pending|processing|completed|failed|cancelled
    total: Mapped[int] = mapped_column(Integer, default=0)
    done: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    pages: Mapped[dict] = mapped_column(JSON, default=dict)     # {page_id: {"status","message"}}
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SettingRow(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    secret: Mapped[str | None] = mapped_column(Text, nullable=True)    # đã mã hóa Fernet


class ExportRow(Base):
    __tablename__ = "exports"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fmt: Mapped[str] = mapped_column(String(16))
    rel_path: Mapped[str] = mapped_column(String(300), default="")      # tương đối trong thư mục project
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
