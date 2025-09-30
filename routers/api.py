from fastapi import APIRouter, Query, HTTPException, Body
from typing import Any, Dict, List
from datetime import date

from services.db import market_col, jewelry_col, summary_col, predictions_col
from services.qa import rag_qa
from services.utils import build_match_multi, build_match_single  # optional

api_router = APIRouter()

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

@api_router.post("/chat")
async def api_chat(body: Dict[str, Any] = Body(...)):
    question = body.get("question")
    if not question:
        raise HTTPException(400, "question 필수")
    answer = await rag_qa(question, k=5)
    return {"answer": answer}
# 더 많은 엔드포인트는 같은 방식으로 분리