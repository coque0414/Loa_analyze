# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from routers.api import api_router
from routers.pages import pages_router


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    print("[Startup] 서버 시작...")
    
    # IntentClassifier 초기화 (캐시 있으면 빠름)
    from services.intent_classifier import get_intent_classifier
    get_intent_classifier()
    
    print("[Startup] 준비 완료!")
    yield
    print("[Shutdown] 서버 종료")
 

app = FastAPI(title="LoA Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

app.include_router(api_router, prefix="/api")
app.include_router(pages_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)