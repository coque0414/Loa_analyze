# Loa_analyze
uvicorn main:app --reload
python -m uvicorn main:app --reload

# (가상환경 진입 후)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

로 실행하세욤.

# Cloudflare Tunnel
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
cloudflared tunnel --url http://127.0.0.1:8000