"""Chạy server: python run.py   (Windows Server / Linux). Luôn 1 process để model AI chỉ có một bản trong RAM."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("webapp.main:create_app", factory=True, host=os.environ.get("MT_HOST", "127.0.0.1"),
                port=int(os.environ.get("MT_PORT", "8000")), workers=1, log_level="info")
