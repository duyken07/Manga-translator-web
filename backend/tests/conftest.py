"""Fixture: server thật (FastAPI + SQLite + luồng trạng thái + AppState/RedrawManager/QueueManager của Desktop) với
 - downloader HF giả (tạo file model giả, đếm số lần gọi), loader giả (đếm số lần nạp),
 - dịch vụ dịch giả (không gọi Gemini).  Redraw dùng engine OpenCV THẬT."""
from __future__ import annotations

import glob
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "existing_app")]

_TMP = Path(tempfile.mkdtemp(prefix="mtweb_"))
os.environ.update({"MT_DATA_DIR": str(_TMP / "data"), "MODEL_DIR": str(_TMP / "models"), "MT_FONT_DIR": str(_TMP / "fonts"),
                   "MT_API_TOKEN": "test-token", "MT_MAX_UPLOAD_MB": "1", "MT_CORS_ORIGINS": "http://localhost:5173"})
(_TMP / "fonts").mkdir(parents=True, exist_ok=True)
_fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
if _fonts:
    shutil.copy(_fonts[0], _TMP / "fonts" / "TestFont.ttf")

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.models import Dialogue, Page
from webapp.core import config, security
from webapp.database import session as dbsession
from webapp.main import create_app

H = {"Authorization": "Bearer test-token"}


class Counters:
    downloads: dict = {}
    loads: dict = {}
    translate_calls = 0
    fail_names: set = set()
    delay = 0.0


def fake_downloader(spec, repo, dest, model_dir, progress):
    Counters.downloads[spec.id] = Counters.downloads.get(spec.id, 0) + 1
    if spec.id == "yolo-bubble" and getattr(Counters, "fail_yolo", False):
        raise ConnectionError("offline")
    progress(0.5)
    if spec.kind == "file":
        (model_dir / spec.filename).write_bytes(b"fake-weights")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for f in (spec.required or ("config.json",)):
            (dest / f).write_text("{}")
    return "abc123"


def _loader(mm, spec):
    Counters.loads[spec.id] = Counters.loads.get(spec.id, 0) + 1


class FakeService:
    def translate_page(self, image_path, source_lang, target_lang, context=None):
        Counters.translate_calls += 1
        time.sleep(Counters.delay)
        name = Path(image_path).name
        if name in Counters.fail_names:
            from app.services.translation_service import TranslationServiceError
            raise TranslationServiceError("lỗi giả lập")
        ds = [Dialogue(dialogue_id=1, box_location=[40, 40, 120, 60], original_text="こんにちは", translated_text="Xin chào", romanization="konnichiwa"),
              Dialogue(dialogue_id=2, box_location=[200, 150, 100, 80], original_text="さようなら", translated_text="Tạm biệt")]
        return Page(page_name=name, image_path=image_path, dialogues=ds, glossary_updates={"太郎": "Taro"})

    def translate_region(self, image_path, crop_box, source_lang, target_lang):
        return Dialogue(dialogue_id=0, box_location=[crop_box[0], crop_box[1], crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]],
                        original_text="領域", translated_text="Vùng")


def make_png(w=400, h=300, name_text=True) -> bytes:
    im = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.ellipse((30, 30, 180, 110), fill="white", outline="black")
    d.text((70, 60), "TEXT", fill="black")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def make_client():
    clients = []

    def _make(fresh: bool = True):
        """fresh=True: xoá model + DB + file project (cô lập test). fresh=False: giữ nguyên (mô phỏng restart server)."""
        s = config.get_settings()
        if fresh:
            for d in (s.model_dir, s.projects_dir):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
            for f in s.data_dir.glob("app.db*"):
                f.unlink(missing_ok=True)
            from webapp.core import events  # noqa: F401
        Counters.downloads, Counters.loads, Counters.translate_calls = {}, {}, 0
        Counters.fail_names, Counters.delay, Counters.fail_yolo = set(), 0.0, False
        app = create_app(downloader=fake_downloader,
                         loaders={k: _loader for k in ("yolo-bubble", "lama-manga", "manga-ocr", "koharu-yolo26s")},
                         service_factory=lambda *a, **k: FakeService())
        c = TestClient(app)
        c.__enter__()
        c.headers.update(H)
        clients.append(c)
        return c

    yield _make
    for c in reversed(clients):
        try:
            c.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def project(client):
    r = client.post("/api/projects", json={"name": "demo"})
    assert r.status_code == 201
    return r.json()["id"]


def upload(client, pid, names, png=None):
    files = [("files", (n, png or make_png(), "image/png")) for n in names]
    return client.post(f"/api/projects/{pid}/pages/upload", files=files)


def wait(pred, timeout=20.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("hết thời gian chờ")


def wait_job(client, jid, timeout=30):
    return wait(lambda: (lambda j: j if j["status"] in ("completed", "failed", "cancelled") else None)(client.get(f"/api/jobs/{jid}").json()), timeout)
