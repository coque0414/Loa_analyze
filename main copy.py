from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    buttons = [
    {"label": "오늘의 소식", "href": "/news",      "img": "/static/img/news.png"},
    {"label": "그래프 보기", "href": "/graphs",    "img": "/static/img/graphs.png"},
    {"label": "기능 사용",   "href": "/features",  "img": "/static/img/features.png"},
    {"label": "테스트중",     "href": "/testing",   "img": "/static/img/testing.png"},
    ]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "buttons": buttons
    })

# 예시로 간단한 페이지 핸들러도 하나 추가
@app.get("/page1")
async def page1(request: Request):
    return templates.TemplateResponse("page1.html", {"request": request})

@app.get("/news")
async def news(request: Request):
    return templates.TemplateResponse("news.html", {"request": request})

@app.get("/graphs")
async def graphs(request: Request):
    return templates.TemplateResponse("graphs.html", {"request": request})

@app.get("/features")
async def features(request: Request):
    return templates.TemplateResponse("features.html", {"request": request})

@app.get("/testing")
async def testing(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return JSONResponse({"ok": True})

@app.get("/health")
async def health():
    return {"status": "ok"}