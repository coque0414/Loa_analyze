from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from typing import List, Dict

templates = Jinja2Templates(directory="templates")
pages_router = APIRouter()

@pages_router.get("/", response_class=None)
async def index(request: Request):
    buttons = [
        {"label": "오늘의 소식", "href": "/news",     "img": "/static/img/news.png"},
        {"label": "그래프 보기", "href": "/graphs",   "img": "/static/img/graphs.png"},
        {"label": "보석 그래프",  "href": "/jewelrys", "img": "/static/img/jewelrys.png"},
        {"label": "테스트중",  "href": "/testing",  "img": "/static/img/testing.png"},
    ]
    return templates.TemplateResponse("index.html", {"request": request, "buttons": buttons})

@pages_router.get("/graphs")
async def graphs(request: Request):
    # services.db.get_items() 사용 (존재하면 호출)
    try:
        from services.db import get_items
        items = await get_items()
    except Exception:
        items = []
    return templates.TemplateResponse("graphs.html", {"request": request, "items": items})

@pages_router.get("/news")
async def news(request: Request):
    """최근 커뮤니티 게시물 목록 페이지"""
    posts = []
    try:
        from services.db import posts_col
        if posts_col is not None:
            posts = await posts_col.find({}, {"_id": 0}).sort("date", -1).to_list(length=50)
    except Exception:
        posts = []
    return templates.TemplateResponse("news.html", {"request": request, "posts": posts})

@pages_router.get("/jewelrys")
async def jewelrys(request: Request):
    """보석 메타(이름·이미지)를 보여주는 페이지"""
    jewelry_items: List[Dict] = []
    try:
        from services.db import jewelry_col
        if jewelry_col is not None:
            pipeline = [
                {"$group": {"_id": "$item_code", "name": {"$first": "$name"}, "img": {"$first": "$image_url"}}},
                {"$sort": {"_id": 1}},
            ]
            cursor = jewelry_col.aggregate(pipeline)
            async for doc in cursor:
                jewelry_items.append({"code": doc["_id"], "name": doc.get("name"), "img": doc.get("img")})
    except Exception:
        jewelry_items = []
    return templates.TemplateResponse("jewelrys.html", {"request": request, "jewelry_items": jewelry_items})

@pages_router.get("/testing")
async def testing(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})