from fastapi import APIRouter, Query, HTTPException, Body, Response, Request
from typing import Any, Dict, List, Optional
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from fastapi.responses import StreamingResponse

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

from services.db import market_col, jewelry_col, summary_col, predictions_col, maps_col, market_snapshots_col
# from services.qa import rag_qa
from services.qa import answer_router   # ← rag_qa 대신 이걸 임포트
from services.utils import build_match_multi, build_match_single  # optional

from skills.island import is_island_intent, answer_island_calendar #로아 모험섬
from skills.market import answer_market_price, answer_market_compare, is_market_intent, resolve_glossary_pair #로아 마켓(거래소)
from pydantic import BaseModel

class ChatIn(BaseModel):
    q: Optional[str] = None          # 새 UI가 보내는 키
    question: Optional[str] = None   # 예전 UI/도구가 보내던 키

    def text(self) -> str:
        return (self.q or self.question or "").strip()

gfs_bucket = AsyncIOMotorGridFSBucket(maps_col.database)
api_router = APIRouter()

def _looks_like_compare(q: str) -> bool:
    return any(kw in q for kw in ["비교", ",", "vs", "VS", "와", "과", "랑", "하고"]) \
           or len(resolve_glossary_pair(q)) >= 2

@api_router.post("/api/chat")
async def api_chat(payload: ChatIn, request: Request):
    question = payload.q.strip()
    if is_market_intent(question):
        if _looks_like_compare(question):
            res = await answer_market_compare(question)
        else:
            res = await answer_market_price(question)
        return {
            "answer": res.get("answer",""),
            "type": res.get("type","text"),
            "answer_html": res.get("answer_html"),
            "items": res.get("items",[])
        }

@api_router.get("/market")
async def api_market(code: int = Query(...)):
    # 기존 로직을 그대로 여기로 옮기면 됨
    pipeline = [
        {"$match": {"item_code": code}},
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

@api_router.get("/maps/file/{file_id}")
async def maps_file(file_id: str):
    try:
        stream = await gfs_bucket.open_download_stream(ObjectId(file_id))
        data = await stream.read()
    except Exception:
        raise HTTPException(404, "file not found")
    # contentType 메타를 안 넣었을 수 있으니 기본 png로
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=31536000"})

@api_router.get("/maps/bydoc/{doc_id}")
async def maps_bydoc(doc_id: str):
    doc = await maps_col.find_one({"_id": ObjectId(doc_id)})
    if not doc or "image" not in doc or "gridfs_id" not in doc["image"]:
        raise HTTPException(404, "document or image not found")
    gid = doc["image"]["gridfs_id"]
    if isinstance(gid, dict) and "$oid" in gid:
        gid = gid["$oid"]
    return await maps_file(str(gid))

@api_router.post("/chat")
async def api_chat(payload: ChatIn, request: Request):
    question = payload.text()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
        
    # 모험섬 먼저 처리
    if is_island_intent(question):
        res = await answer_island_calendar(question)
        return {"answer": res["answer"], "type": "island", "answer_html": res.get("answer_html"), "items": res.get("items", [])}

    # 2) ✅ 마켓(거래소) 의도면 여기서 처리 (RAG보다 앞!)
    if is_market_intent(question):
        if "," in question:
            res = await answer_market_compare(question)
        else:
            res = await answer_market_price(question)
        return {
            "answer": res.get("answer",""),
            "type": res.get("type","text"),
            "answer_html": res.get("answer_html"),
            "items": res.get("items",[]),
            "chart_url": res.get("chart_url"),
        }

    # 3) 그 외는 RAG로 answer_router: 지도/텍스트 자동 분기
    res = await answer_router(question)

    # 텍스트 응답 (하위호환: answer 필드를 유지)
    if isinstance(res, dict) and res.get("type") == "text":
        return {"answer": res.get("text", ""), "type": "text", "raw": res}

    # 지도 응답
    if isinstance(res, dict) and res.get("type") == "map":
        world = (res.get("world") or "").strip()
        region = (res.get("region") or "").strip()

        # ✅ 파일 URL 우선(상대경로). 프런트가 같은 오리진으로 호출하므로 그대로 <img src>에 써도 됨
        img_rel = res.get("image_url") or res.get("image_url_bydoc") or ""

        link = (res.get("source") or {}).get("region_url") or img_rel

        # 원하는 톤의 문장
        line1 = f"{world}에 있는 {region}의 지도를 보여드리겠습니다!".strip()
        line2 = "이미지 클릭 시 원문 지도로 이동합니다."
        line3 = "해당 지역의 지도를 보여드렸습니다. 자세한 항목은 링크에서 확인해보세요!"

        answer_html = f"""
        <figure style="max-width:420px">
          <a href="{link}" target="_blank" rel="noopener">
            <img src="{img_rel}" alt="{world} {region} 지도"
                 style="width:100%;border-radius:10px;border:1px solid #e5e7eb" />
          </a>
          <figcaption style="font-size:12px;color:#6b7280;margin-top:6px">
            {world} / {region}
          </figcaption>
        </figure>
        <div style="font-size:13px;color:#4b5563;margin-top:6px">{line3}</div>
        """.strip()

        return {
            "answer": f"{line1}\n{line2}",
            "type": "map",
            "image_url": img_rel,      # 상대경로
            "answer_html": answer_html,
            "world": world, "region": region,
            "source": res.get("source"),
            "raw": res,
        }

    # 예외 케이스
    return {"answer": str(res)}

@api_router.get("/market/price")
async def api_market_price(q: str):
    return await answer_market_price(q)
    #http://localhost:8000/api/market/price?q=원한 유각 가격

#차트 라우트(마켓용으로 쓰는것)
@api_router.get("/charts/price")
async def charts_price(slugs: str, days: int = 7):
    import matplotlib.pyplot as plt
    KST = timezone(timedelta(hours=9))
    now = datetime.now(KST)
    start = now - timedelta(days=max(1, days))

    slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
    if not slug_list:
        raise HTTPException(400, "slugs required")

    lines = []
    for slug in slug_list:
        cur = market_snapshots_col.find(
            {"slug": slug, "ts": {"$gte": start}},
            {"_id":0,"ts":1,"price":1,"name":1}
        ).sort("ts", 1)
        xs, ys, name = [], [], None
        async for d in cur:  # ✅ Motor 커서면 async for
            if d.get("price") is None: 
                continue
            xs.append(d["ts"]); ys.append(float(d["price"]))
            if not name: name = d.get("name", slug)
        if xs:
            lines.append((name or slug, xs, ys))

    if not lines:
        raise HTTPException(404, "no data")

    fig, ax = plt.subplots(figsize=(6,3))
    for name, xs, ys in lines:
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("날짜"); ax.set_ylabel("가격(G)"); ax.legend(); fig.tight_layout()

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")




# 더 많은 엔드포인트는 같은 방식으로 분리

@api_router.get("/loa/islands") #테스트용 get 엔드포인트
async def api_islands(period: str = "week"):
    q = "이번주 골드 모험섬 알려줘" if period=="week" else "이번달 골드 모험섬 알려줘"
    return await answer_island_calendar(q)
    # http://localhost:8000/api/loa/islands?period=month 으로 들어가서 확인 가능합니다.

@api_router.get("/market/compare")
async def api_market_compare(q: str):
    # 예: /market/compare?q=원한 유각, 아드 유각
    return await answer_market_compare(q)
    #http://localhost:8000/api/market/compare?q=원한 유각, 아드 유각
