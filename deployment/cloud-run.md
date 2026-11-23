# Cloud Run Deployment & FastAPI Integration

The Node.js integration server (`src/server.ts`) proxies all `/api/*` and `/ws/projects/:id/progress` requests to an independently deployed Python FastAPI inference backend (`webapp.main:create_app`) when `FASTAPI_BACKEND_URL` is set to a valid `http://` or `https://` URL.

## 1. Environment Variables (Node.js / AI Studio)

- `FASTAPI_BACKEND_URL`: Public HTTPS URL of your Cloud Run FastAPI service (e.g. `https://manga-translator-api-xxxx.a.run.app`).
  - Must include the `https://` or `http://` scheme (do **not** put an API token here).
  - When unset or invalid, Node.js automatically runs in standalone mode (project/page/dialogue CRUD, ZIP import, JSON/TXT/PNG/JPG/PSD export, and `@google/genai` Gemini Vision translation/OCR/grammar/chat).
- `MT_API_TOKEN`: Shared Bearer token injected by the Node.js proxy when forwarding requests to FastAPI, and enforced on client requests to Node.js.
- `GEMINI_API_KEY`: Server-side Gemini API key.

## 2. Deploying the Python FastAPI Backend (`duyken07/Manga-translator-web`)

AI Studio runs on a Node.js container without PyTorch/YOLO/LaMa weights. Deploy the `backend/` folder from `duyken07/Manga-translator-web` as a separate Google Cloud Run service:

1. **Dockerfile** (in `backend/`):
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   ENV MT_HOST=0.0.0.0
   CMD ["sh", "-c", "uvicorn webapp.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy manga-translator-api \
     --source ./backend \
     --region asia-northeast1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --timeout 600 \
     --concurrency 20 \
     --min-instances 0 \
     --max-instances 1 \
     --set-env-vars "MT_API_TOKEN=YOUR_SHARED_TOKEN,GEMINI_API_KEY=YOUR_GEMINI_KEY,MT_DATA_DIR=/data,MODEL_DIR=/models"
   ```

   - Keep `--workers 1` and `--max-instances 1` (or mount Cloud Storage FUSE on `/data` and `/models` with SQLite/session affinity) so PyTorch models (`YOLO`, `koharu-yolo26s`, `manga-ocr-base`, `lama-manga`) share memory cleanly under `TORCH_INFER_LOCK`.
   - Note that Cloud Run allocates ephemeral disk by default; mount a persistent volume / Cloud Storage FUSE bucket at `/data` and `/models` to avoid re-downloading Hugging Face weights across cold starts.

3. **Connect AI Studio to Cloud Run**:
   - Copy the deployed Cloud Run URL (e.g. `https://manga-translator-api-xxxx.asia-northeast1.run.app`).
   - Set `FASTAPI_BACKEND_URL` to that URL and `MT_API_TOKEN` to the exact same shared token in AI Studio Secrets / Environment Variables.
   - Restart the app; the **Models** tab and `/api/runtime` will switch to `Cloud Run FastAPI Proxy` mode and unlock lazy Hugging Face weight downloads and LaMa/OpenCV redraw.
