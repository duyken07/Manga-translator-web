import os
import time
from pathlib import Path

from conftest import Counters, upload, wait, wait_job

KEY = {"api_key": "AIzaSyTESTKEY1234567890"}


def setup_project(client, names=("a.png", "b.png", "c.png")):
    client.post("/api/settings/api-key", json=KEY)
    client.put("/api/settings", json={"enable_redraw": False})          # tách redraw khỏi test dịch
    pid = client.post("/api/projects", json={"name": "demo"}).json()["id"]
    upload(client, pid, list(names))
    return pid


def pages(client, pid):
    return client.get(f"/api/projects/{pid}/pages").json()


# ------------------------------------------------------------------ dịch qua hàng đợi nền
def test_translate_requires_api_key(client, project):
    upload(client, project, ["a.png"])
    r = client.post(f"/api/projects/{project}/translate")
    assert r.status_code == 400 and "API key" in r.json()["detail"]


def test_translate_all_pages_background_job(client):
    pid = setup_project(client)
    r = client.post(f"/api/projects/{pid}/translate")
    assert r.status_code == 202
    j = wait_job(client, r.json()["job_id"])
    assert j["status"] == "completed" and j["total"] == 3 and j["done"] == 3 and j["failed"] == 0
    assert all(v["status"] == "completed" for v in j["pages"].values())
    ps = pages(client, pid)
    assert all(p["status"] == "done" and p["dialogue_count"] == 2 for p in ps)
    d = client.get(f"/api/pages/{ps[0]['id']}/dialogues").json()
    assert d[0]["translated_text"] == "Xin chào" and d[0]["romanization"] == "konnichiwa" and d[0]["dialogue_id"] == 1
    assert client.get(f"/api/projects/{pid}").json()["glossary"] == {"太郎": "Taro"}          # glossary tích lũy
    # model YOLO tự tải đúng 1 lần, nạp đúng 1 lần dù dịch lại
    assert Counters.downloads["yolo-bubble"] == 1 and Counters.loads["yolo-bubble"] == 1
    j2 = wait_job(client, client.post(f"/api/projects/{pid}/translate").json()["job_id"])
    assert j2["status"] == "completed"
    assert Counters.downloads["yolo-bubble"] == 1 and Counters.loads["yolo-bubble"] == 1
    assert "manga-ocr" not in Counters.downloads                                               # mode B không cần OCR


def test_mode_a_downloads_ocr_and_koharu_when_enabled(client):
    pid = setup_project(client, ("a.png",))
    client.put("/api/settings", json={"translation_backend": "yolo_ocr_gemini", "use_koharu": True})
    wait_job(client, client.post(f"/api/projects/{pid}/translate").json()["job_id"])
    assert Counters.downloads.keys() >= {"yolo-bubble", "manga-ocr", "koharu-yolo26s"}


def test_failed_page_isolated_and_retry(client):
    pid = setup_project(client)
    Counters.fail_names = {"b.png"}
    j = wait_job(client, client.post(f"/api/projects/{pid}/translate").json()["job_id"])
    assert j["status"] == "completed" and j["failed"] == 1 and "1 trang lỗi" in j["message"]
    st = {p["page_name"]: p for p in pages(client, pid)}
    assert st["b.png"]["status"] == "error" and "lỗi giả lập" in st["b.png"]["error_message"]
    assert st["a.png"]["status"] == "done"
    Counters.fail_names = set()
    j2 = wait_job(client, client.post(f"/api/jobs/{j['id']}/retry-failed").json()["job_id"])
    assert j2["status"] == "completed" and j2["total"] == 1
    assert {p["page_name"]: p["status"] for p in pages(client, pid)}["b.png"] == "done"


def test_cancel_and_busy_guard(client):
    pid = setup_project(client)
    Counters.delay = 0.6
    jid = client.post(f"/api/projects/{pid}/translate").json()["job_id"]
    assert client.post(f"/api/projects/{pid}/translate").status_code == 409                # đang dịch
    wait(lambda: client.get(f"/api/jobs/{jid}").json()["status"] == "processing")
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "cancelled"
    wait(lambda: all(p["status"] != "translating" for p in pages(client, pid)), 10)        # trang dở dang được khôi phục
    n = Counters.translate_calls
    time.sleep(1.5)
    assert Counters.translate_calls <= n + 1                                                 # không dịch tiếp các trang còn lại
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 409
    wait_job(client, client.post(f"/api/projects/{pid}/translate").json()["job_id"]) if False else None


def test_add_delete_pages_blocked_while_translating(client):
    pid = setup_project(client)
    Counters.delay = 0.5
    jid = client.post(f"/api/projects/{pid}/translate").json()["job_id"]
    wait(lambda: client.get(f"/api/jobs/{jid}").json()["status"] == "processing")
    assert upload(client, pid, ["z.png"]).status_code == 409
    client.post(f"/api/jobs/{jid}/cancel")


def test_translate_region_adds_dialogue(client):
    pid = setup_project(client, ("a.png",))
    page = pages(client, pid)[0]["id"]
    r = client.post(f"/api/pages/{page}/translate-region", json={"box": [10, 20, 100, 50]})
    assert r.status_code == 200
    d = client.get(f"/api/pages/{page}/dialogues").json()
    assert len(d) == 1 and d[0]["box_location"] == [10, 20, 100, 50] and d[0]["translated_text"] == "Vùng"
    assert client.post(f"/api/pages/{page}/translate-region", json={"box": [1, 2, -5, 4]}).status_code == 400


def test_retranslate_keeps_manual_dialogues(client):
    pid = setup_project(client, ("a.png",))
    page = pages(client, pid)[0]["id"]
    wait_job(client, client.post(f"/api/pages/{page}/translate").json()["job_id"])
    client.patch(f"/api/pages/{page}/dialogues/1", json={"translated_text": "SỬA TAY"})
    wait_job(client, client.post(f"/api/pages/{page}/translate", json={"keep_manual": True}).json()["job_id"])
    texts = [d["translated_text"] for d in client.get(f"/api/pages/{page}/dialogues").json()]
    assert "SỬA TAY" in texts and "Xin chào" in texts[-2:] or "SỬA TAY" in texts


# ------------------------------------------------------------------ WebSocket
def test_websocket_progress(client):
    pid = setup_project(client, ("a.png", "b.png"))
    from starlette.websockets import WebSocketDisconnect
    import pytest
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/projects/{pid}/progress?token=sai"):
            pass
    with client.websocket_connect(f"/ws/projects/{pid}/progress?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        jid = client.post(f"/api/projects/{pid}/translate").json()["job_id"]
        seen, done = [], False
        for _ in range(200):
            ev = ws.receive_json()
            seen.append(ev["type"])
            if ev["type"] == "job" and ev["job"]["id"] == jid and ev["job"]["status"] == "completed":
                done = True
                break
        assert done and "page_changed" in seen


# ------------------------------------------------------------------ dialogue editor
def test_dialogue_crud_reorder_and_security(client):
    pid = setup_project(client, ("a.png",))
    page = pages(client, pid)[0]["id"]
    base = f"/api/pages/{page}/dialogues"
    r = client.post(base, json={"box_location": [5, 6, 70, 80], "original_text": "原", "translated_text": "gốc"})
    assert r.status_code == 201 and r.json()["dialogue_id"] == 1 and r.json()["edited"] is True       # nhập tay = đã chỉnh sửa
    client.post(base, json={"box_location": [100, 100, 40, 40], "translated_text": "hai"})
    client.post(base, json={"box_location": [200, 100, 40, 40], "translated_text": "ba"})
    r = client.patch(f"{base}/2", json={"translated_text": "HAI", "box_location": [101, 102, 41, 42], "hide_text": True,
                                        "text_rotation": 12.5, "text_frame": [90, 90, 80, 60], "romanization": "ni"})
    d = r.json()
    assert d["translated_text"] == "HAI" and d["box_location"] == [101, 102, 41, 42] and d["hide_text"] is True
    assert d["text_rotation"] == 12.5 and d["text_frame"] == [90, 90, 80, 60] and d["romanization"] == "ni" and d["edited"]
    assert client.patch(f"{base}/2", json={"text_frame": None}).json()["text_frame"] is None
    assert [x["dialogue_id"] for x in client.put(f"{base}/order", json={"order": [3, 1, 2]}).json()] == [1, 2, 3]
    assert client.get(base).json()[0]["translated_text"] == "ba"                                         # thứ tự mới, ID đánh lại 1..N
    assert client.delete(f"{base}/1").status_code == 204 and len(client.get(base).json()) == 2
    assert client.delete(f"{base}/99").status_code == 404
    # chống dữ liệu xấu
    assert client.patch(f"{base}/1", json={"box_location": [1, 2, 0, 5]}).status_code == 400
    assert client.patch(f"{base}/1", json={"overrides": {"evil_key": 1}}).status_code == 400
    assert client.patch(f"{base}/1", json={"overrides": {"font_path": "/etc/passwd"}}).status_code == 400
    assert client.patch(f"{base}/1", json={"unknown": 1}).status_code == 422
    r = client.patch(f"{base}/1", json={"overrides": {"font_size": 30, "style": "bold"}})
    assert r.status_code == 200 and r.json()["overrides"]["font_size"] == 30


# ------------------------------------------------------------------ lưu / mở lại sau restart
def test_project_survives_server_restart(make_client):
    c1 = make_client()
    pid = setup_project(c1, ("a.png", "b.png"))
    wait_job(c1, c1.post(f"/api/projects/{pid}/translate").json()["job_id"])
    pg = pages(c1, pid)[0]["id"]
    c1.patch(f"/api/pages/{pg}/dialogues/1", json={"translated_text": "ĐÃ SỬA", "text_rotation": 5.0, "overrides": {"font_size": 33}})
    c1.put(f"/api/projects/{pid}", json={"work_title": "Truyện X"})
    c1.__exit__(None, None, None)                                           # tắt server (lifespan shutdown -> persist)

    c2 = make_client(fresh=False)
    p = c2.get(f"/api/projects/{pid}").json()
    assert p["glossary"] == {"太郎": "Taro"} and p["work_title"] == "Truyện X" and p["page_count"] == 2
    ps = pages(c2, pid)
    assert [x["page_name"] for x in ps] == ["a.png", "b.png"] and all(x["status"] == "done" for x in ps)
    d = c2.get(f"/api/pages/{ps[0]['id']}/dialogues").json()
    assert d[0]["translated_text"] == "ĐÃ SỬA" and d[0]["edited"] and d[0]["text_rotation"] == 5.0 and d[0]["overrides"]["font_size"] == 33
    assert c2.get("/api/settings").json()["has_api_key"] is True                # API key + settings còn sau restart
    assert c2.get(f"/api/pages/{ps[0]['id']}/image").status_code == 200


def test_orphan_jobs_marked_failed_on_restart(make_client):
    c1 = make_client()
    jid = c1.app.state.ctx.jobs.create(None, "export", total=1)
    c1.app.state.ctx.jobs.start(jid)
    c1.__exit__(None, None, None)
    c2 = make_client(fresh=False)
    j = c2.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "failed" and "khởi động lại" in j["message"]


# ------------------------------------------------------------------ redraw thật (OpenCV) + incremental + cache
def redraw_state(client, page):
    return client.get(f"/api/pages/{page}").json()["redraw"]


def test_redraw_opencv_incremental_and_persisted_cache(make_client):
    c = make_client()
    c.post("/api/settings/api-key", json=KEY)
    pid = c.post("/api/projects", json={"name": "r"}).json()["id"]
    upload(c, pid, ["a.png"])
    page = pages(c, pid)[0]["id"]
    wait_job(c, c.post(f"/api/projects/{pid}/translate").json()["job_id"])           # enable_redraw mặc định True -> redraw tự chạy
    st = wait(lambda: (lambda r: r if r["status"] == "done" and r["has_image"] else (None if r["status"] != "error" else (_ for _ in ()).throw(AssertionError(r))))(redraw_state(c, page)), 60)
    assert st["layout_boxes"].keys() >= {"1", "2"}                                      # layout khối chữ trả về cho frontend
    assert Counters.downloads["lama-manga"] == 1                                       # LaMa tự tải lần đầu
    img = c.get(f"/api/pages/{page}/redraw-preview")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert c.get(f"/api/pages/{page}/redraw-preview?kind=clean").status_code == 200
    assert c.get(f"/api/pages/{page}/redraw-preview?kind=original").status_code == 200

    rt = c.app.state.ctx.registry.get(pid)
    out = Path(rt.run(lambda: rt.project.pages[0].redraw_path))
    clean = out.parent / "clean.png"
    clean_m = clean.stat().st_mtime_ns
    before = img.content
    etag = img.headers["etag"]
    assert c.get(f"/api/pages/{page}/redraw-preview", headers={"If-None-Match": etag}).status_code == 304

    # CHỈ sửa chữ dịch -> chỉ xếp chữ lại (preview đổi), KHÔNG inpaint lại (clean.png giữ nguyên bản ghi)
    c.patch(f"/api/pages/{page}/dialogues/1", json={"translated_text": "Chữ mới hoàn toàn khác hẳn"})
    wait(lambda: c.get(f"/api/pages/{page}/redraw-preview").content != before, 30)
    assert redraw_state(c, page)["status"] in ("done", "running", "queued")
    wait(lambda: redraw_state(c, page)["status"] == "done", 30)
    assert clean.stat().st_mtime_ns == clean_m
    assert c.get(f"/api/pages/{page}/redraw-preview").headers["etag"] != etag
    # công tắc ẩn chữ / ẩn redraw toàn trang
    only_redraw = c.get(f"/api/pages/{page}/redraw-preview?hide_text=true").content
    original = c.get(f"/api/pages/{page}/redraw-preview?hide_text=true&hide_redraw=true").content
    full = c.get(f"/api/pages/{page}/redraw-preview").content
    assert len({only_redraw, original, full}) == 3
    # ẩn chữ từng câu qua API thoại
    c.patch(f"/api/pages/{page}/dialogues/1", json={"hide_text": True, "hide_redraw": True})
    c.patch(f"/api/pages/{page}/dialogues/2", json={"hide_text": True, "hide_redraw": True})
    assert c.get(f"/api/pages/{page}/redraw-preview").content == original

    # restart: cache còn trên đĩa, trang vẫn có ảnh redraw, không phải chạy lại
    c.__exit__(None, None, None)
    c2 = make_client(fresh=False)
    st2 = redraw_state(c2, page)
    assert st2["has_image"] and st2["status"] == "done" and st2["layout_boxes"].keys() >= {"1", "2"}
    assert c2.get(f"/api/pages/{page}/redraw-preview").content == original
    cm = clean.stat().st_mtime_ns
    time.sleep(1.0)
    assert clean.stat().st_mtime_ns == cm                                               # không tự redraw lại khi mở project


def test_redraw_endpoint_and_dialogue_region(make_client):
    c = make_client()
    c.put("/api/settings", json={"enable_redraw": True})
    pid = c.post("/api/projects", json={"name": "r"}).json()["id"]
    upload(c, pid, ["a.png"])
    page = pages(c, pid)[0]["id"]
    c.post(f"/api/pages/{page}/dialogues", json={"box_location": [40, 40, 120, 60], "translated_text": "Chào"})
    wait(lambda: redraw_state(c, page)["status"] == "done", 60)
    assert c.post(f"/api/pages/{page}/redraw", json={"force_full": True}).status_code == 202
    assert c.post(f"/api/pages/{page}/redraw-dialogue", json={"dialogue_id": 1}).status_code == 202
    assert c.post(f"/api/pages/{page}/redraw-dialogue", json={"dialogue_id": 77}).status_code == 404
    wait(lambda: redraw_state(c, page)["status"] == "done", 60)
    lay = c.get(f"/api/pages/{page}/layers")
    assert lay.status_code == 200 and lay.json()["layers"]
