@echo off
echo Starting SecureID Production Server (FastAPI + WebRTC)...
python -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
