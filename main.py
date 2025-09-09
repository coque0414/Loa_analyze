import os, textwrap
from datetime import date
from typing import Any, Dict, List
# from dotenv import load_dotenv

import motor.motor_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Body, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import motor.motor_asyncio
from openai import OpenAI

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 기존
# embedder = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")

embedder = None
def get_embedder():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        # 첫 필요 시점(요청 시)에 로딩
        embedder = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")
    return embedder

# load_dotenv()
# ──────────────────── 상수: 아이템 코드 목록
ITEM_CODES = [
    65201505, 65200805, 65203005, 65203305, 65203105, 65200605,
    65203905, 65201005, 65200505, 65202805, 65204105, 65203505, 65203705
]
JEWELRY_CODES = [65031100, 65031090, 65031080, 65031070, 65032100, 65032090, 65032080, 65032070]

# Lost Ark 공식 CDN 경로 패턴
CDN_BASE = "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/"
# 예: use_9_25.png  (실제 번호는 아이템 별 상이 → 임시 플레이스홀더)
PLACEHOLDER_IMG = CDN_BASE + "use_12_105.png"

# ──────────────────── FastAPI & 템플릿 ────────────────────
app = FastAPI(title="LoA Dashboard API")
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 배포 시 도메인 제한 권장
    allow_methods=["GET"],
    allow_headers=["*"],
)

# OpenAI client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────── MongoDB (Motor) ────────────────────
MONGO_URI   = os.getenv("MONGODB_URI")
DB_NAME     = "lostark"
MARKET_COLL  = "market_items"
JEWELRY_COLL = "jewelry_value"

client      = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
collection  = client[DB_NAME][MARKET_COLL]
jewelry_collection = client[DB_NAME][JEWELRY_COLL]

db           = client[DB_NAME]
market_col   = db[MARKET_COLL]
jewelry_col  = db[JEWELRY_COLL]

posts        = db["community_posts"]  # ← 이 줄을 꼭 추가하세요
summary_col  = db["daily_summary"]
predictions_col  = db["predict_graphs"]
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

async def get_jewelry_items() -> List[Dict[str, Any]]:
    """JEWELRY_CODES 리스트로부터 jewelry_value 컬렉션에서 메타 조회"""
    pipeline = [
        {"$match": {"item_code": {"$in": JEWELRY_CODES}}},
        {"$group": {"_id": "$item_code", "name": {"$first": "$name"}, "img": {"$first": "$image_url"}}},
        {"$sort": {"_id": 1}},
    ]
    items = []
    async for doc in jewelry_col.aggregate(pipeline):
        items.append({"code": doc["_id"], "name": doc["name"], "img": doc["img"]})
    return items

# ──────────────────── 공통 유틸 ────────────────────
async def get_items() -> List[Dict[str, Any]]:
    """ITEM_CODES → (code, name, img_url) 리스트 반환"""
    pipeline = [
        {"$match": {"item_code": {"$in": ITEM_CODES}}},
        {"$group": {"_id": "$item_code", "name": {"$first": "$name"}}},
        # {"$sort": {"_id": 1}},
    ]
    cursor = market_col.aggregate(pipeline)
    result = []
    async for doc in cursor:
        code = doc["_id"]
        name = doc.get("name") or "Item"
        # ❗️ 실제 아이템별 이미지 매핑이 확정되면 아래 로직을 교체하세요
        img_url = "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/use_9_25.png"
        result.append({"code": code, "name": name, "img": img_url})
    return result

# ──────────────────── 유틸: semantic_search ────────────────────
async def semantic_search(question: str, limit: int = 5):
    # 1) 질문 임베딩
    model = get_embedder()
    q_emb = embedder.encode(question, convert_to_numpy=True).reshape(1, -1)
    
    # 2) embedding 필드가 있는 문서만 불러오기
    docs = await posts.find(
        {"embedding": {"$exists": True}},
        {"title":1, "text":1, "embedding":1, "_id":0}
    ).to_list(length=10000)
    
    if not docs:
        print("[semantic_search] embedding 필드 문서 없음")
        return []
    
    # 3) numpy 행렬로 쌓고 코사인 유사도 계산
    embs = np.vstack([d["embedding"] for d in docs])      # shape (N, D)
    sims = cosine_similarity(q_emb, embs)[0]              # shape (N,)
    
    # 4) Top-K 문서 뽑기
    idxs = sims.argsort()[::-1][:limit]
    found = [docs[i] for i in idxs]
    print(f"[semantic_search] question={question!r} → found {len(found)} docs")
    return found

# ──────────────────── 유틸: rag_qa ────────────────────
async def rag_qa(question: str, k: int = 5) -> str:
    # 1) 상위 k개 문서 검색
    ctx_docs = await semantic_search(question, limit=k)
    if not ctx_docs:
        return "⚠️ 관련 문서를 찾지 못했습니다."

    # 2) 컨텍스트 요약 문자열 생성
    context = "\n".join(f"- {d['title']}: {d['text']}" for d in ctx_docs)

    # 3) ChatGPT 프롬프트 조합
    prompt = textwrap.dedent(f'''
    너는 로스트아크 커뮤니티 게시물만 참고해서 대답하는 어시스턴트야.
    다음 게시물들과 관련 없는 질문이 들어오면 반드시 이렇게 말해:
    "⚠️ 이 질문은 게시물 내용과 관련성이 낮아 답변할 수 없습니다."

    [게시물 요약]
    {context}

    [질문]
    {question}

    [답변 - 한국어로 정확하게 설명하되, 친절하게:]
    ''')

    # 4) OpenAI Chat Completion 호출 (gpt-4o 사용)
    chat_res = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return chat_res.choices[0].message.content.strip()

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
    cursor = market_col.aggregate(pipeline)
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
    cursor = market_col.aggregate(pipeline)
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
    cursor = market_col.aggregate(pipeline)
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

# ──────────────────── API: jewelry single ────────────────────
@app.get("/api/jewelry", response_model=List[Dict[str, Any]])
async def api_jewelry(
    code: int = Query(..., description="item_code (e.g. 65031100)"),
):
    """단일 보석 아이템: 날짜별 평균 가격(avg_value) · 거래 횟수(count)"""
    pipeline = [
        {"$match": {"item_code": code}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}},
            "avg_value": {"$avg": "$price"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}},
    ]
    data = []
    async for d in jewelry_col.aggregate(pipeline):
        data.append({"date": d["_id"], "avg_value": d["avg_value"], "count": d["count"]})
    if not data:
        raise HTTPException(status_code=404, detail="No jewelry data for given code")
    return data

# ─── RAG 챗봇 엔드포인트 ─────────────────

@app.post("/api/chat", response_model=Dict[str, str])
async def api_chat(body: Dict[str, Any] = Body(...)):
    question = body.get("question")
    if not question:
        raise HTTPException(400, "question 필수")

    answer = await rag_qa(question, k=5)
    return {"answer": answer}

# ──────────────────── 요약문 기능.. ────────────────────
@app.get("/api/daily_summary")
async def api_daily_summary(
    date: str = Query(..., description="예: 2025-06-01"),
    keyword: str = Query("유각")
):
    doc = await summary_col.find_one(
        {"date": date, "keyword": keyword},
        {"_id": 0, "summary": 1, "representatives": 1}
    )
    if not doc:
        # 204 No Content – 프론트에서 메시지로 처리
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # 대표 게시글은 title·url 두 가지만 돌려주면 충분
    reps = [
        {"title": r["title"], "url": r["url"]}
        for r in (doc.get("representatives") or [])[:3]
    ]
    return {"summary": doc["summary"], "representatives": reps}
# ──────────────────── API: 예측 시계열 ────────────────────
@app.get(
    "/api/predictions",
    response_model=List[Dict[str, Any]],
    summary="item_code에 대한 일주일 예측값 반환"
)
async def api_predictions(
    item_code: int = Query(...,alias="item_code", description="item_code")
):
    doc = await predictions_col.find_one(
        {"item_code": item_code},
        {"_id": 0,"item_code":1, "predictions": 1}
    )
    if not doc or not doc.get("predictions"):
        raise HTTPException(status_code=404, detail="Predictions not found")
    return doc["predictions"]

# ──────────────────── 페이지 라우팅 ────────────────────
@app.get("/")
async def index(request: Request):
    buttons = [
        {"label": "오늘의 소식", "href": "/news",     "img": "/static/img/news.png"},
        {"label": "그래프 보기", "href": "/graphs",   "img": "/static/img/graphs.png"},
        {"label": "보석 그래프",  "href": "/jewelrys", "img": "/static/img/jewelrys.png"},
        {"label": "테스트중",  "href": "/testing",  "img": "/static/img/testing.png"},
    ]
    return templates.TemplateResponse("index.html", {"request": request, "buttons": buttons})

@app.get("/graphs", response_class=HTMLResponse)
async def graphs(request: Request):
    items = await get_items()
    return templates.TemplateResponse("graphs.html", {"request": request,"items": items})

@app.get("/news", response_class=HTMLResponse)
async def news(request: Request):
    return templates.TemplateResponse("news.html", {"request": request})

@app.get("/jewelrys", response_class=HTMLResponse)
async def jewelryss(request: Request):
    items = await get_jewelry_items()
    return templates.TemplateResponse("jewelrys.html", {"request": request, "jewelry_items": items})

@app.get("/testing", response_class=HTMLResponse)
async def testing(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return JSONResponse({"ok": True})

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)