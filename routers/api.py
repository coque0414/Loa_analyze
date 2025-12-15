# routers/api.py
from fastapi import APIRouter, Query, HTTPException, Response, Request
from typing import Any, Dict, Optional
from datetime import datetime, timedelta, timezone
from io import BytesIO
from fastapi.responses import StreamingResponse

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from pydantic import BaseModel

from services.db import market_col, maps_col, market_snapshots_col, _get_db
from services.qa import answer_router
from services.intent_classifier import (
    get_intent_classifier,
    log_intent_feedback,
)

from skills.island import answer_island_calendar
from skills.market import answer_market_price, answer_market_compare
from skills.youtube import answer_youtube_recommend

import logging
import re

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


class ChatIn(BaseModel):
    q: Optional[str] = None
    question: Optional[str] = None

    def text(self) -> str:
        return (self.q or self.question or "").strip()


# ✅ gfs_bucket을 동적으로 생성하는 함수
def _get_gfs_bucket():
    return AsyncIOMotorGridFSBucket(_get_db())

# ============================================================
# ✅ 하드코딩 답변 (특정 질문용)
# ============================================================
HARDCODED_ANSWERS: Dict[str, Dict[str, Any]] = {
    "로스트아크는 무슨 게임이에요": {
        "answer": """로스트아크(Lost Ark)는 스마일게이트 RPG가 개발하고 스마일게이트에서 퍼블리싱하는 대한민국의 대규모 다중 사용자 온라인 액션 롤플레잉 게임(MMORPG)입니다.
게임 특징으로 쿼터뷰 시점의 액션 전투, 레이드 컨텐츠가 중심 기반으로 이루어져있습니다.

🎭 로스트아크의 스토리는 크게 3부작으로 나뉘어 있습니다.
사슬 전쟁이 끝난지 500년 후, 혼돈으로 물들어가는 아크라시아를 구하기 위해 흩어졌던 7개의 아크를 모으는 여정이 주를 이루며, 모험가와 함께 아크라시아를 지키는 7인의 에스더와 아크라시아를 침공하는 페트라니아의 카제로스와 그의 휘하에 있는 여섯 군단장의 대립을 다룹니다.
여기까지 1부 스토리 내용입니다!

💫 해당 게임에는 플레이어블 클래스가 존재합니다! 클래스는 모험가가 선택하는 직업으로, 전사/마법사/무도가 등 큰 직업군 아래 세분화 전직이 존재합니다.
각 클래스는 고유의 스킬셋과 전투 스타일을 가지고 있습니다. 아래의 설명 참조하시면 더 쉽게 이해 가능합니다!""",
        "type": "intro",
        "answer_html": """
<div style="max-width:600px;">

  <!-- 🔗 클래스 소개 섹션 (배너 이미지 + 링크) -->
  <a href="https://lostark.game.onstove.com/Class" target="_blank" rel="noopener" style="display:block; text-decoration:none; margin-bottom:16px;">
    <div style="position:relative; border-radius:12px; overflow:hidden; box-shadow:0 4px 16px rgba(0,0,0,0.15);">
      <!-- 클래스 배너 이미지 -->
      <img src="https://cdn-lostark.game.onstove.com/uploadfiles/banner/36970e2fb3a341d7a354e27b59079ec7.jpg" 
           alt="로스트아크 클래스" 
           style="width:100%; height:140px; object-fit:cover; display:block;"
           onerror="this.src='https://cdn-lostark.game.onstove.com/uploadfiles/banner/ee44adb41ca542c392623a0181004ff3.jpg'"/>
      <!-- 오버레이 텍스트 -->
      <div style="position:absolute; bottom:0; left:0; right:0; background:linear-gradient(transparent, rgba(0,0,0,0.8)); padding:16px;">
        <div style="color:white; font-weight:700; font-size:16px; display:flex; align-items:center; gap:8px;">
          💫 클래스 소개 페이지
          <span style="font-size:18px;">→</span>
        </div>
        <div style="color:rgba(255,255,255,0.85); font-size:13px; margin-top:4px;">전사, 마법사, 무도가 등 다양한 클래스를 확인해보세요!</div>
      </div>
    </div>
  </a>
  


  <!-- 🎬 트레일러 영상 섹션 (더 크게) -->
  <div style="margin-bottom:8px;">
    <div style="font-weight:700; color:#1f2937; margin-bottom:12px; font-size:15px;">🎬 공식 트레일러</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
      <div style="position:relative; padding-bottom:56.25%; height:0; overflow:hidden; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.12);">
        <iframe src="https://www.youtube.com/embed/ee_1DV9BSTk" 
                style="position:absolute; top:0; left:0; width:100%; height:100%; border:none;"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen></iframe>
      </div>
      <div style="position:relative; padding-bottom:56.25%; height:0; overflow:hidden; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.12);">
        <iframe src="https://www.youtube.com/embed/pkkoLRFP1nk" 
                style="position:absolute; top:0; left:0; width:100%; height:100%; border:none;"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen></iframe>
      </div>
    </div>
  </div>

  <!-- ⚔️ 하단 CTA -->
  <div style="background:linear-gradient(135deg, #0b7266 0%, #059669 100%); color:white; padding:12px 16px; border-radius:10px; font-size:14px; text-align:center; margin-bottom:16px; box-shadow:0 2px 8px rgba(11,114,102,0.3);">
    ⚔️ 아크라시아 대륙에서 잃어버린 아크를 찾아 모험을 떠나보세요!
  </div>

</div>
""".strip()
    },
    "가디언나이트에 대해 설명해줘": {
        "answer": """가디언나이트는 거대한 할버드로 묵직하고 강력한 공격을 하는 근접 클래스입니다.
화신화 상태에서는 날개를 활용한 폭발적인 기동력과 강력한 기운으로 전장을 헤집으며 마음껏 적들을 유린합니다.
이 클래스에 대해 더 알아보고 싶다면 아래 영상을 참고해보세요!""",
        "type": "intro",
        "answer_html": """
<div style="max-width:600px;">
  <div style="margin-bottom:8px;">
    <div style="font-weight:700; color:#1f2937; margin-bottom:12px; font-size:15px;">🎬 가디언나이트 공식 트레일러</div>
    <div style="position:relative; padding-bottom:56.25%; height:0; overflow:hidden; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.12);">
      <iframe src="https://www.youtube.com/embed/hyR1jERwqRs" 
              style="position:absolute; top:0; left:0; width:100%; height:100%; border:none;"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowfullscreen></iframe>
    </div>
  </div>
</div>
""".strip()
    },
}


def _normalize_query(q: str) -> str:
    """질문 정규화: 소문자화, 공백/특수문자 제거"""
    q = q.lower().strip()
    q = re.sub(r"[?\.\!\s]+", "", q)
    return q


def _check_hardcoded_answer(question: str) -> Optional[Dict[str, Any]]:
    """
    하드코딩된 답변이 있는지 확인
    유사 질문도 매칭되도록 키워드 기반 매칭
    """
    q_norm = _normalize_query(question)
    
    # 정확 매칭 시도
    for key, answer_data in HARDCODED_ANSWERS.items():
        key_norm = _normalize_query(key)
        if key_norm in q_norm or q_norm in key_norm:
            return answer_data
    
    # 키워드 기반 매칭 (로스트아크 + 게임 + 뭐/무슨/어떤)
    if "로스트아크" in q_norm and ("게임" in q_norm or "뭐" in q_norm or "무슨" in q_norm or "어떤" in q_norm or "소개" in q_norm):
        return HARDCODED_ANSWERS.get("로스트아크는 무슨 게임이에요")
    
    # 가디언나이트 관련
    if "가디언나이트" in q_norm and ("설명" in q_norm or "뭐" in q_norm or "소개" in q_norm or "알려" in q_norm):
        return HARDCODED_ANSWERS.get("가디언나이트에 대해 설명해줘")
    
    return None

api_router = APIRouter()


# ============================================================
# 메인 채팅 엔드포인트
# ============================================================

@api_router.post("/chat")
async def chat_endpoint(payload: ChatIn, request: Request):
    question = payload.text()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    
    # ─────────────────────────────────────────────
    # 0) ✅ 하드코딩 답변 체크 (최우선)
    # ─────────────────────────────────────────────
    hardcoded = _check_hardcoded_answer(question)
    if hardcoded:
        logger.info(f"[Hardcoded] query='{question[:50]}' → 하드코딩 답변 반환")
        return {
            "answer": hardcoded.get("answer", ""),
            "type": hardcoded.get("type", "text"),
            "answer_html": hardcoded.get("answer_html"),
            "items": hardcoded.get("items", []),
        }
    
    # ─────────────────────────────────────────────
    # 1) 임베딩 기반 의도 분류
    # ─────────────────────────────────────────────
    classifier = get_intent_classifier()
    intent_result = classifier.classify(question)
    
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    scores = intent_result["scores"]
    
    logger.info(f"[Intent] query='{question[:50]}' → {intent} (conf={confidence:.3f})")
    
    # ─────────────────────────────────────────────
    # 2) 의도별 핸들러 라우팅
    # ─────────────────────────────────────────────
    actual_handler = intent
    success = True
    response_data = {}
    
    try:
        if intent == "island":
            response_data = await _handle_island(question)
        
        elif intent == "market_compare":
            response_data = await _handle_market_compare(question)
        
        elif intent == "market_price":
            response_data = await _handle_market_price(question)
        
        elif intent == "map":
            response_data = await _handle_map_or_fallback(question)
            actual_handler = response_data.get("_handler", "map")
        
        # ✅ YouTube 추천 라우팅 추가!
        elif intent == "youtube_recommend":
            response_data = await _handle_youtube_recommend(question)
        
        else:
            # unknown 또는 general_qa → RAG
            response_data = await _handle_general_qa(question)
            actual_handler = "general_qa"
    
    except Exception as e:
        logger.exception(f"Handler error for intent={intent}")
        success = False
        response_data = {
            "answer": "처리 중 오류가 발생했어요. 다시 시도해 주세요.",
            "type": "error",
        }
    
    # ─────────────────────────────────────────────
    # 3) 피드백 로깅 (비동기, 실패해도 응답에 영향 없음)
    # ─────────────────────────────────────────────
    try:
        await log_intent_feedback(
            query=question,
            classified_intent=intent,
            confidence=confidence,
            actual_handler=actual_handler,
            success=success,
            response_type=response_data.get("type"),
            extra={"scores": scores},
        )
    except Exception:
        pass  # 로깅 실패는 무시
    
    # 내부 필드 제거 후 반환
    response_data.pop("_handler", None)
    return response_data


# ============================================================
# 의도별 핸들러 함수들
# ============================================================

async def _handle_island(question: str) -> Dict[str, Any]:
    """모험섬 일정 처리"""
    res = await answer_island_calendar(question)
    return {
        "answer": res.get("answer", ""),
        "type": "island",
        "answer_html": res.get("answer_html"),
        "items": res.get("items", []),
    }


async def _handle_market_compare(question: str) -> Dict[str, Any]:
    """시세 비교 처리"""
    res = await answer_market_compare(question)
    return {
        "answer": res.get("answer", ""),
        "type": res.get("type", "price_compare"),
        "answer_html": res.get("answer_html"),
        "items": res.get("items", []),
    }


async def _handle_market_price(question: str) -> Dict[str, Any]:
    """단일 시세 조회 처리"""
    res = await answer_market_price(question)
    return {
        "answer": res.get("answer", ""),
        "type": res.get("type", "price"),
        "answer_html": res.get("answer_html"),
        "items": res.get("items", []),
        "chart_url": res.get("chart_url"),
    }


async def _handle_youtube_recommend(question: str) -> Dict[str, Any]:
    """YouTube 영상 추천 처리"""
    res = await answer_youtube_recommend(question)
    return {
        "answer": res.get("answer", ""),
        "type": "youtube",
        "answer_html": res.get("answer_html"),
        "videos": res.get("videos", []),
    }


async def _handle_map_or_fallback(question: str) -> Dict[str, Any]:
    """지도 처리 (실패 시 RAG로 폴백)"""
    res = await answer_router(question)
    
    if isinstance(res, dict) and res.get("type") == "map":
        world = (res.get("world") or "").strip()
        region = (res.get("region") or "").strip()
        img_rel = res.get("image_url") or res.get("image_url_bydoc") or ""
        link = (res.get("source") or {}).get("region_url") or img_rel
        
        line1 = f"{world}에 있는 {region}의 지도를 보여드리겠습니다!".strip()
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
            "answer": f"{line1}\n이미지 클릭 시 원문 지도로 이동합니다.",
            "type": "map",
            "image_url": img_rel,
            "answer_html": answer_html,
            "world": world,
            "region": region,
            "source": res.get("source"),
            "_handler": "map",
        }
    
    # 지도가 아니면 텍스트 RAG 결과
    return {
        "answer": res.get("text", "") if isinstance(res, dict) else str(res),
        "type": "text",
        "_handler": "general_qa",  # 폴백됨
    }


async def _handle_general_qa(question: str) -> Dict[str, Any]:
    """일반 QA (RAG) 처리"""
    res = await answer_router(question)
    
    if isinstance(res, dict):
        if res.get("type") == "map":
            # 예상외로 지도가 나온 경우
            return await _handle_map_or_fallback(question)
        return {
            "answer": res.get("text", ""),
            "type": "text",
        }
    
    return {
        "answer": str(res),
        "type": "text",
    }


# ============================================================
# 기타 API 엔드포인트들
# ============================================================

@api_router.get("/market")
async def api_market(code: int = Query(...)):
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
        bucket = _get_gfs_bucket()  # ✅ 동적으로 버킷 가져오기
        stream = await bucket.open_download_stream(ObjectId(file_id))
        data = await stream.read()
    except Exception as e:
        print(f"[ERROR] maps_file failed: {e}")
        raise HTTPException(404, "file not found")
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000"}
    )


@api_router.get("/maps/bydoc/{doc_id}")
async def maps_bydoc(doc_id: str):
    doc = await maps_col.find_one({"_id": ObjectId(doc_id)})
    if not doc or "image" not in doc or "gridfs_id" not in doc["image"]:
        raise HTTPException(404, "document or image not found")
    gid = doc["image"]["gridfs_id"]
    if isinstance(gid, dict) and "$oid" in gid:
        gid = gid["$oid"]
    return await maps_file(str(gid))


@api_router.get("/market/price")
async def api_market_price_get(q: str):
    return await answer_market_price(q)


@api_router.get("/market/compare")
async def api_market_compare_get(q: str):
    return await answer_market_compare(q)


@api_router.get("/youtube/recommend")
async def api_youtube_recommend_get(q: str):
    """YouTube 영상 추천 API (GET)"""
    return await answer_youtube_recommend(q)


@api_router.get("/charts/price")
async def charts_price(slugs: str, days: int = 7):
    import matplotlib.pyplot as plt
    
    now = datetime.now(KST)
    start = now - timedelta(days=max(1, days))

    slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
    if not slug_list:
        raise HTTPException(400, "slugs required")

    lines = []
    for slug in slug_list:
        cur = market_snapshots_col.find(
            {"slug": slug, "ts": {"$gte": start}},
            {"_id": 0, "ts": 1, "price": 1, "name": 1}
        ).sort("ts", 1)
        xs, ys, name = [], [], None
        async for d in cur:
            if d.get("price") is None:
                continue
            xs.append(d["ts"])
            ys.append(float(d["price"]))
            if not name:
                name = d.get("name", slug)
        if xs:
            lines.append((name or slug, xs, ys))

    if not lines:
        raise HTTPException(404, "no data")

    fig, ax = plt.subplots(figsize=(6, 3))
    for name, xs, ys in lines:
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("날짜")
    ax.set_ylabel("가격(G)")
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@api_router.get("/loa/islands")
async def api_islands(period: str = "week"):
    q = "이번주 골드 모험섬 알려줘" if period == "week" else "이번달 골드 모험섬 알려줘"
    return await answer_island_calendar(q)


# ============================================================
# 피드백 조회 (관리자용)
# ============================================================

@api_router.get("/admin/intent-feedback")
async def get_intent_feedback(days: int = 7, limit: int = 100):
    """낮은 confidence 쿼리들 조회 (INTENT_EXAMPLES 보강용)"""
    from services.intent_classifier import get_low_confidence_queries
    return await get_low_confidence_queries(days=days, limit=limit)