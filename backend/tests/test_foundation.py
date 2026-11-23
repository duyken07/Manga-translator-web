import io
from pathlib import Path

from conftest import Counters, make_png, upload, wait

# ------------------------------------------------------------------ bảo mật / auth
def test_auth_required_and_health_open(client):
    from fastapi.testclient import TestClient
    anon = TestClient(client.app)                       # không có header mặc định
    assert anon.get("/api/health").status_code == 200
    assert anon.get("/api/projects", headers={"Authorization": "Bearer sai"}).status_code == 401
    assert anon.get("/api/projects").status_code == 401
    assert anon.get("/api/models/status").status_code == 401
    assert client.get("/api/projects", headers={"X-API-Token": "test-token"}).status_code == 200


def test_cors_configured(client):
    r = client.options("/api/projects", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = client.options("/api/projects", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers


# ------------------------------------------------------------------ project / upload
def test_project_crud(client):
    pid = client.post("/api/projects", json={"name": "A", "source_lang": "ja", "target_lang": "vi"}).json()["id"]
    assert client.get(f"/api/projects/{pid}").json()["name"] == "A"
    r = client.put(f"/api/projects/{pid}", json={"name": "B", "glossary": {"x": "y"}, "settings": {"keep_honorifics": True}})
    assert r.json()["name"] == "B" and r.json()["glossary"] == {"x": "y"}
    assert client.put(f"/api/projects/{pid}", json={"settings": {"lama_device": "cuda"}}).status_code == 422   # ngoài whitelist
    assert any(p["id"] == pid for p in client.get("/api/projects").json())
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_upload_natural_sort_and_validation(client, project):
    r = upload(client, project, ["p10.png", "p2.png", "p1.png"])
    assert r.status_code == 201 and r.json()["added"] == 3
    names = [p["page_name"] for p in client.get(f"/api/projects/{project}/pages").json()]
    assert names == ["p1.png", "p2.png", "p10.png"]
    p = client.get(f"/api/projects/{project}/pages").json()[0]
    assert (p["width"], p["height"]) == (400, 300)
    # trùng tên: bỏ qua, không ghi đè
    r = upload(client, project, ["p1.png"]).json()
    assert r["added"] == 0 and r["skipped_duplicates"] == ["p1.png"]
    # ext sai, nội dung giả, path traversal, quá lớn, rỗng
    bad = client.post(f"/api/projects/{project}/pages/upload", files=[
        ("files", ("a.exe", b"MZ", "application/octet-stream")),
        ("files", ("fake.png", b"not an image at all", "image/png")),
        ("files", ("big.png", b"\x89PNG" + b"0" * (2 * 1024 * 1024), "image/png")),
        ("files", ("empty.png", b"", "image/png")),
        ("files", ("mismatch.jpg", make_png(), "image/jpeg")),
    ]).json()
    assert bad["added"] == 0 and len(bad["rejected"]) == 5
    r = upload(client, project, ["../../evil.png"]).json()
    assert r["added"] == 1
    root = Path(client.app.state.ctx.server.projects_dir) / project
    assert (root / "originals" / "evil.png").is_file()
    assert not (root.parent.parent / "evil.png").exists()
    # API không lộ đường dẫn server
    detail = client.get(f"/api/pages/{p['id']}").text
    assert str(root) not in detail and "image_path" not in detail
    assert client.get(f"/api/pages/{p['id']}/image").status_code == 200
    assert client.get(f"/api/pages/{p['id']}/thumbnail").headers["content-type"] == "image/jpeg"


def test_delete_page(client, project):
    ids = upload(client, project, ["a.png", "b.png"]).json()["page_ids"]
    assert client.delete(f"/api/pages/{ids[0]}").status_code == 204
    assert [p["page_name"] for p in client.get(f"/api/projects/{project}/pages").json()] == ["b.png"]
    assert client.get(f"/api/pages/{ids[0]}").status_code == 404


# ------------------------------------------------------------------ settings / API key
def test_api_key_never_returned_and_encrypted(client):
    assert client.post("/api/settings/api-key", json={"api_key": "AIzaSyTESTKEY1234567890"}).status_code == 204
    s = client.get("/api/settings")
    assert s.status_code == 200 and "AIzaSyTESTKEY1234567890" not in s.text
    assert s.json()["has_api_key"] is True
    from sqlalchemy import text
    from webapp.database.session import session_scope
    with session_scope() as db:
        raw = db.execute(text("select secret from settings where key='gemini_api_key'")).scalar()
    assert "AIzaSy" not in raw
    assert client.delete("/api/settings/api-key").status_code == 204
    assert client.get("/api/settings").json()["has_api_key"] is False


def test_settings_validation_and_fixed_gemini_model(client):
    assert client.put("/api/settings", json={"gemini_model": "x"}).status_code == 422
    assert client.put("/api/settings", json={"lama_device": "tpu"}).status_code == 422
    assert client.put("/api/settings", json={"redraw_font": "../../etc/passwd"}).status_code == 422
    assert client.put("/api/settings", json={"concurrency_limit": 3}).json()["concurrency_limit"] == 3
    s = client.get("/api/settings").json()
    assert s["gemini_model"] and s["gemini_fallback_model"]


# ------------------------------------------------------------------ model manager
def test_model_status_download_load_once(client):
    st = {m["id"]: m for m in client.get("/api/models/status").json()["models"]}
    assert set(st) == {"yolo-bubble", "lama-manga", "manga-ocr", "koharu-yolo26s"}
    assert all(m["state"] == "not_downloaded" for m in st.values())
    assert client.post("/api/models/load", json={"id": "yolo-bubble"}).status_code == 409     # chưa tải
    client.post("/api/models/download", json={"id": "yolo-bubble"})
    wait(lambda: {m["id"]: m for m in client.get("/api/models/status").json()["models"]}["yolo-bubble"]["state"] == "downloaded")
    mm = client.app.state.ctx.models
    assert (mm.model_dir / "comic-speech-bubble-detector.pt").is_file()          # đúng tên/đường dẫn theo đặc tả
    for _ in range(3):
        j = client.post("/api/models/load", json={"id": "yolo-bubble"}).json()
        wait(lambda: client.get(f"/api/jobs/{j['job_id']}").json()["status"] == "completed")
    assert Counters.loads["yolo-bubble"] == 1 and Counters.downloads["yolo-bubble"] == 1   # không tải/nạp lại mỗi request
    m = {m["id"]: m for m in client.get("/api/models/status").json()["models"]}["yolo-bubble"]
    assert m["state"] == "ready" and m["version"] == "abc123"
    assert str(mm.model_dir) not in client.get("/api/models/status").text            # không lộ đường dẫn


def test_model_download_error_then_retry(client):
    Counters.fail_yolo = True
    client.post("/api/models/download", json={"id": "yolo-bubble"})
    m = wait(lambda: next((x for x in client.get("/api/models/status").json()["models"] if x["id"] == "yolo-bubble" and x["state"] == "error"), None))
    assert "offline" in m["error"]
    Counters.fail_yolo = False
    client.post("/api/models/download", json={"id": "yolo-bubble"})
    wait(lambda: next(x for x in client.get("/api/models/status").json()["models"] if x["id"] == "yolo-bubble")["state"] == "downloaded")


def test_manually_provided_models_detected(make_client):
    c = make_client()
    mm = c.app.state.ctx.models
    (mm.model_dir / "lama-manga.safetensors").write_bytes(b"x")
    mm.rescan()
    assert mm.state("lama-manga") == "downloaded" and mm.path("lama-manga") is not None
