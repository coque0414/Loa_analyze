from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager

from routers.api import api_router
from routers.pages import pages_router

# 👇 IntentClassifier 사전 로드를 위해 import
from services.intent_classifier import get_intent_classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작/종료 시 한 번만 실행되는 훅.
    여기서 무거운 모델들을 미리 로드해 두면,
    첫 요청이 느려지는 걸 방지할 수 있음.
    """
    # (선택) 임베딩 모델 먼저 로드
    # get_embedder()

    # ✅ IntentClassifier (내부에서 embedder까지 로드)
    get_intent_classifier()

    yield  # ← 여기까지가 startup, 이후부터는 앱이 요청 처리
    # 필요하면 여기서 종료 시 정리 작업(cleanup) 가능

app = FastAPI(title="LoA Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_methods=["GET", "POST", "OPTIONS"]
)

app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# 라우터 등록
app.include_router(api_router, prefix="/api")
app.include_router(pages_router)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)