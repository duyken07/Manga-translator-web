"""FastAPI app. Chạy: python run.py  (hoặc uvicorn webapp.main:create_app --factory). Chỉ 1 process (workers=1): model torch dùng chung."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.events import EventHub
from .core.job_manager import JobManager
from .core.model_manager import ModelManager
from .core.security import get_api_token, require_token
from .database.session import dispose_db, init_db


def create_app(downloader=None, loaders=None, service_factory=None) -> FastAPI:
    s = get_settings()                               # cũng cấu hình env + sys.path tới existing_app
    from .adapters.runtime import AppContext, default_service_factory
    from .api import dialogues, jobs, models_api, pages, projects, redraw, settings_api, translation

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        get_api_token()
        ctx.hub.bind_loop(asyncio.get_running_loop())
        n = ctx.jobs.recover_orphans()
        if n:
            logging.getLogger("jobs").warning("Đánh dấu %d job dở dang là failed (server khởi động lại)", n)
        yield
        ctx.registry.shutdown()
        ctx.jobs.shutdown()
        ctx.models.shutdown()
        dispose_db()

    app = FastAPI(title="Manga Translator Web", version="0.1.0", lifespan=lifespan)
    hub = EventHub()
    ctx = AppContext(s, ModelManager(s, downloader=downloader, loaders=loaders), JobManager(hub), hub,
                     service_factory or default_service_factory)
    app.state.ctx = ctx

    from fastapi.responses import JSONResponse
    from .adapters.runtime import RuntimeError_

    @app.exception_handler(RuntimeError_)
    async def _domain_error(_req, exc: RuntimeError_):      # NotFound 404 / Busy 409 / MissingApiKey 400 / dữ liệu sai 400
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    auth = [Depends(require_token)]
    for r in (projects.router, pages.router, dialogues.router, translation.router, redraw.router, jobs.router,
              models_api.router, settings_api.router):
        app.include_router(r, dependencies=auth)
    app.add_api_websocket_route("/ws/projects/{project_id}/progress", jobs.progress_ws)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app
