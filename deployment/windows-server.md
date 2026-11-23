# Triển khai trên Windows Server (và Linux)

## Yêu cầu
Python 3.11+, ~6 GB RAM trống (torch + LaMa + Manga-OCR + YOLO), kết nối Internet **lần đầu** để tải model (hoặc chép sẵn file vào `MODEL_DIR`).

## Cài đặt
```powershell
cd manga-translator-web\backend
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt            # CPU: dùng wheel torch CPU của pytorch.org nếu muốn nhẹ hơn
copy ..\.env.example ..\.env               # chỉnh MT_DATA_DIR / MODEL_DIR sang ổ đĩa bền vững (vd D:\manga\data)
python run.py
```
Lần đầu chạy sinh token tại `data\api_token.txt` (hoặc đặt `MT_API_TOKEN`). Mọi request cần `Authorization: Bearer <token>`.
Model được tải tự động khi cần (hoặc `POST /api/models/download`); kiểm tra `GET /api/models/status`.

## Chạy như dịch vụ Windows (NSSM)
```powershell
nssm install MangaTranslator "D:\app\backend\.venv\Scripts\python.exe" "D:\app\backend\run.py"
nssm set MangaTranslator AppDirectory D:\app\backend
nssm set MangaTranslator AppEnvironmentExtra MT_DATA_DIR=D:\manga\data MODEL_DIR=D:\manga\models MT_HOST=127.0.0.1
nssm start MangaTranslator
```
Đặt IIS/Nginx/Caddy phía trước làm reverse proxy HTTPS (bắt buộc bật WebSocket cho `/ws/`). Không mở cổng 8000 ra Internet trực tiếp.

## Lưu ý quan trọng
- **Luôn 1 process** (`workers=1`): model torch dùng chung trong RAM và có khoá suy luận toàn cục; nhiều worker sẽ nhân đôi RAM.
- Bảo vệ thư mục `data\` bằng ACL (chứa `secret.key` mã hóa API key và `api_token.txt`); trên Windows `chmod 600` không có tác dụng.
- Sao lưu: `data\app.db` + `data\projects\` (+ `secret.key` để giải mã được API key). Di chuyển thư mục data → cache redraw tự tính lại.
- GPU: đặt `lama_device=cuda` trong Settings nếu cài torch CUDA.
