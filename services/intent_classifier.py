# services/intent_classifier.py
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import logging

from services.embedder import get_embedder
from services.db import db  # MongoDB 연결

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# 의도별 대표 문장들
INTENT_EXAMPLES: Dict[str, List[str]] = {
    "market_compare": [
        "원한 아드 가격 비교해줘",
        "두 각인서 뭐가 더 비싸?",
        "결대랑 아드 시세 차이",
        "원한 아드 둘 다 알려줘",
        "어떤 각인이 더 비싼지",
        "가격 차이가 얼마나 나?",
        "원한이랑 아드 뭐가 더 비싸?",
        "결대 아드 가격 차이",
        "두 개 비교해줘",
        "뭐가 더 싸?",
        "원한 vs 아드",
        "결투의 대가 원한 비교",
    ],
    "market_price": [
        "원한 각인서 가격",
        "아드레날린 시세 얼마야",
        "결투의 대가 최저가",
        "유물 각인서 가격 알려줘",
        "원한 유각 얼마",
        "시세 알려줘",
        "거래소 가격",
        "지금 얼마야",
        "각인서 가격 검색",
    ],
    "island": [
        "오늘 모험섬 뭐야",
        "골드 섬 언제 떠",
        "다음 모험섬 일정",
        "모험섬 알려줘",
        "오늘 섬 일정",
        "골드 모험섬",
        "모험섬 보상",
    ],
    "map": [
        "아르테미스 지도 보여줘",
        "레온하트 어디있어",
        "베른 북부 위치",
        "지도 보여줘",
        "맵 알려줘",
        "어디에 있어",
        "위치가 어디야",
    ],
    "general_qa": [
        "낙원 콘텐츠가 뭐야",
        "카던 입장 조건",
        "이번 패치 내용",
        "공지사항 알려줘",
        "어떻게 해야 해",
        "방법 알려줘",
        "뭐야 이게",
        "설명해줘",
    ],
}

# 피드백 저장용 컬렉션
intent_feedback_col = db["intent_feedback"]


class IntentClassifier:
    """임베딩 기반 의도 분류기"""
    
    def __init__(self):
        self.embedder = get_embedder()
        self._intent_embeddings: Dict[str, np.ndarray] = {}
        self._example_embeddings: Dict[str, np.ndarray] = {}  # 개별 예시 임베딩
        self._build_intent_vectors()
    
    def _build_intent_vectors(self):
        """각 의도별 대표 벡터 생성"""
        for intent, examples in INTENT_EXAMPLES.items():
            embs = self.embedder.encode(examples, convert_to_numpy=True)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            # 평균 벡터를 대표 벡터로
            self._intent_embeddings[intent] = embs.mean(axis=0)
            # 개별 예시도 저장 (더 정밀한 매칭용)
            self._example_embeddings[intent] = embs
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산"""
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_norm, b_norm))
    
    def classify(self, query: str, threshold: float = 0.55) -> Dict:
        """
        의도 분류 수행
        
        Returns:
            {
                "intent": "market_compare" | "market_price" | ... | "unknown",
                "confidence": 0.85,
                "scores": {"market_compare": 0.85, ...},
                "best_example_score": 0.92,  # 가장 유사한 예시와의 점수
            }
        """
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        if q_emb.ndim > 1:
            q_emb = q_emb[0]
        
        scores = {}
        best_example_scores = {}
        
        for intent, intent_emb in self._intent_embeddings.items():
            # 1) 평균 벡터와 비교
            avg_score = self._cosine_sim(q_emb, intent_emb)
            
            # 2) 개별 예시 중 최고 점수
            example_embs = self._example_embeddings[intent]
            example_sims = [self._cosine_sim(q_emb, ex) for ex in example_embs]
            max_example_score = max(example_sims)
            
            # 최종 점수: 평균과 최고 예시의 가중 조합
            scores[intent] = avg_score * 0.4 + max_example_score * 0.6
            best_example_scores[intent] = max_example_score
        
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        return {
            "intent": best_intent if confidence >= threshold else "unknown",
            "confidence": confidence,
            "scores": scores,
            "best_example_score": best_example_scores.get(best_intent, 0.0),
        }


# 싱글톤 인스턴스
_classifier: Optional[IntentClassifier] = None

def get_intent_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


async def log_intent_feedback(
    query: str,
    classified_intent: str,
    confidence: float,
    actual_handler: str,
    success: bool,
    response_type: str = None,
    extra: dict = None,
):
    """
    의도 분류 피드백 로깅 (낮은 confidence나 불일치 케이스 분석용)
    """
    doc = {
        "query": query,
        "classified_intent": classified_intent,
        "confidence": confidence,
        "actual_handler": actual_handler,
        "success": success,
        "response_type": response_type,
        "ts": datetime.now(KST),
        "needs_review": confidence < 0.65 or classified_intent != actual_handler,
    }
    if extra:
        doc["extra"] = extra
    
    try:
        await intent_feedback_col.insert_one(doc)
    except Exception as e:
        logger.warning(f"Failed to log intent feedback: {e}")


async def get_low_confidence_queries(days: int = 7, limit: int = 100) -> List[dict]:
    """
    리뷰가 필요한 (낮은 confidence) 쿼리들 조회
    → 이 데이터로 INTENT_EXAMPLES 보강 가능
    """
    cutoff = datetime.now(KST) - timedelta(days=days)
    cursor = intent_feedback_col.find(
        {"needs_review": True, "ts": {"$gte": cutoff}},
        {"_id": 0, "query": 1, "classified_intent": 1, "confidence": 1, "actual_handler": 1}
    ).sort("ts", -1).limit(limit)
    
    return await cursor.to_list(length=limit)