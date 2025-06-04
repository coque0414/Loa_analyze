import os
from datetime import date
from typing import Any, Dict, List

import motor.motor_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ──────────────────── 상수: 아이템 코드 목록
ITEM_CODES = [
    65201505, 65200805, 65203005, 65203305, 65203105, 65200605,
    65203905, 65201005, 65200505, 65202805, 65204105, 65203505, 65203705
]
# ──────────────────── FastAPI & 템플릿 ────────────────────
app = FastAPI(title="LoA Dashboard API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 배포 시 도메인 제한 권장
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ──────────────────── MongoDB (Motor) ────────────────────
MONGO_URI   = os.getenv("MONGODB_URI")
DB_NAME     = "lostark"
COLL_NAME   = "market_items"

client      = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
collection  = client[DB_NAME][COLL_NAME]

# ──────────────────── Helpers ────────────────────

def build_match_single(code: str, start: date | None, end: date | None) -> Dict[str, Any]:
    cond: Dict[str, Any] = {"item_code": code}
    if start:
        cond["date"] = {"$gte": start.isoformat()}
    if end:
        cond.setdefault("date", {})["$lte"] = end.isoformat()
    return cond


def build_match_multi(codes: List[str], start: date | None, end: date | None) -> Dict[str, Any]:
    cond: Dict[str, Any] = {"item_code": {"$in": codes}}
    if start:
        cond["date"] = {"$gte": start.isoformat()}
    if end:
        cond.setdefault("date", {})["$lte"] = end.isoformat()
    return cond

# ──────────────────── 공통 유틸 ────────────────────
async def get_items() -> List[Dict[str, Any]]:
    """ITEM_CODES → (code, name, img_url) 리스트 반환"""
    pipeline = [
        {"$match": {"item_code": {"$in": ITEM_CODES}}},
        {"$group": {"_id": "$item_code", "name": {"$first": "$name"}}},
        {"$sort": {"_id": 1}},
    ]
    cursor = collection.aggregate(pipeline)
    result = []
    async for doc in cursor:
        code = doc["_id"]
        name = doc.get("name") or "Item"
        # ❗️ 실제 아이템별 이미지 매핑이 확정되면 아래 로직을 교체하세요
        img_url = "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/use_9_25.png"
        result.append({"code": code, "name": name, "img": img_url})
    return result

# ──────────────────── API: 단일 아이템 ────────────────────
@app.get("/api/market", response_model=List[Dict[str, Any]])
async def api_market(
    code: int = Query(..., description="item_code (e.g. 65201005)"),
):
    """단일 아이템: 날짜별 평균가 · 판매량(trade_count)"""
    match = {"item_code": code}  # DB에 문자열로 저장된 경우를 위해 str 변환
    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": "$date",
            "avg_price": {"$avg": "$avg_price"},
            "trade_count": {"$sum": "$trade_count"},
        }},
        {"$sort": {"_id": 1}},
    ]
    cursor = collection.aggregate(pipeline)
    data = [
        {"date": d["_id"], "avg_price": d["avg_price"], "trade_count": d["trade_count"]}
        async for d in cursor
    ]
    if not data:
        raise HTTPException(status_code=404, detail="No data")
    return data

# ──────────────────── API: 다중 아이템 ────────────────────
@app.get("/api/markets", response_model=List[Dict[str, Any]])
async def api_markets():
    match = {"item_code": {"$in": ITEM_CODES}}
    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": {"date": "$date", "item_code": "$item_code"},
            "avg_price":   {"$avg": "$avg_price"},
            "trade_count": {"$sum": "$trade_count"},
        }},
        {"$sort": {"_id.date": 1, "_id.item_code": 1}},
    ]
    cursor = collection.aggregate(pipeline)
    data = [
        {
            "item_code": d["_id"]["item_code"],
            "date":      d["_id"]["date"],
            "avg_price": d["avg_price"],
            "trade_count": d["trade_count"],
        }
        async for d in cursor
    ]
    if not data:
        raise HTTPException(status_code=404, detail="No data")
    return data
@app.get("/api/markets", response_model=List[Dict[str, Any]])
async def api_markets(
    codes: str | None = Query(
        None,
        description="콤마(,)로 구분된 item_code 목록. 비우면 기본 ITEM_CODES 사용.",
    ),
    start: date | None = Query(None, description="YYYY-MM-DD"),
    end: date | None = Query(None, description="YYYY-MM-DD"),
):
    """다중 아이템: item_code × 날짜 조합별 평균가 · 거래량"""
    codes_list = (
        [c.strip() for c in codes.split(",") if c.strip()]
        if codes
        else [str(c) for c in ITEM_CODES]
    )

    match = build_match_multi(codes_list, start, end)
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": {"date": "$date", "item_code": "$item_code"},
                "avg_price": {"$avg": "$avg_price"},
                "trade_count": {"$trd": "$trade_count"},
            }
        },
        {"$sort": {"_id.date": 1, "_id.item_code": 1}},
    ]
    cursor = collection.aggregate(pipeline)
    data = [
        {
            "item_code": d["_id"]["item_code"],
            "date": d["_id"]["date"],
            "avg_price": d["avg_price"],
            "trade_count": d["trade_count"],
        }
        async for d in cursor
    ]
    if not data:
        raise HTTPException(status_code=404, detail="No data for given codes")
    return data

# ──────────────────── 페이지 라우팅 ────────────────────
@app.get("/")
async def index(request: Request):
    buttons = [
        {"label": "오늘의 소식", "href": "/news",     "img": "/static/img/news.png"},
        {"label": "그래프 보기", "href": "/graphs",   "img": "/static/img/graphs.png"},
        {"label": "기능 사용",  "href": "/features", "img": "/static/img/features.png"},
        {"label": "테스트중",  "href": "/testing",  "img": "/static/img/testing.png"},
    ]
    return templates.TemplateResponse("index.html", {"request": request, "buttons": buttons})

@app.get("/graphs")
async def graphs(request: Request):
    items = await get_items()
    return templates.TemplateResponse("graphs.html", {"request": request,"items": items})

@app.get("/news")
async def news(request: Request):
    return templates.TemplateResponse("news.html", {"request": request})

@app.get("/features")
async def features(request: Request):
    return templates.TemplateResponse("features.html", {"request": request})

@app.get("/testing")
async def testing(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)