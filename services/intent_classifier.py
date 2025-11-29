# services/intent_classifier.py
import numpy as np
import time
import hashlib
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# ============================================================
# 의도별 대표 문장들
# ============================================================
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

# 캐시 디렉토리
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _compute_examples_hash() -> str:
    content = str(sorted(INTENT_EXAMPLES.items()))
    return hashlib.md5(content.encode()).hexdigest()[:12]


class IntentClassifier:
    """임베딩 기반 의도 분류기 (캐싱 지원)"""
    
    def __init__(self):
        self._ready = False
        self._embedder = None
        self._intent_embeddings: Dict[str, np.ndarray] = {}
        self._example_embeddings: Dict[str, np.ndarray] = {}
        self._initialize()
    
    def _initialize(self):
        if self._ready:
            return
        
        start = time.time()
        print("[IntentClassifier] 초기화 시작...")
        
        cache_file = CACHE_DIR / f"intent_emb_{_compute_examples_hash()}.npz"
        
        # 캐시 로드 시도
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                self._intent_embeddings = data["intent_emb"].item()
                self._example_embeddings = data["example_emb"].item()
                self._ready = True
                print(f"[IntentClassifier] 캐시 로드 완료 ({time.time()-start:.2f}s)")
                return
            except Exception as e:
                print(f"[IntentClassifier] 캐시 로드 실패: {e}")
        
        # 캐시 없으면 새로 계산
        from services.embedder import get_embedder
        self._embedder = get_embedder()
        self._build_intent_vectors()
        
        # 캐시 저장
        try:
            np.savez(
                cache_file,
                intent_emb=self._intent_embeddings,
                example_emb=self._example_embeddings
            )
            print(f"[IntentClassifier] 캐시 저장: {cache_file}")
        except Exception as e:
            print(f"[IntentClassifier] 캐시 저장 실패: {e}")
        
        self._ready = True
        print(f"[IntentClassifier] 초기화 완료 ({time.time()-start:.2f}s)")
    
    def _build_intent_vectors(self):
        for intent, examples in INTENT_EXAMPLES.items():
            embs = self._embedder.encode(examples, convert_to_numpy=True)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            self._intent_embeddings[intent] = embs.mean(axis=0)
            self._example_embeddings[intent] = embs
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_norm, b_norm))
    
    def classify(self, query: str, threshold: float = 0.55) -> Dict:
        # embedder 로드 (캐시에서 로드한 경우 없을 수 있음)
        if self._embedder is None:
            from services.embedder import get_embedder
            self._embedder = get_embedder()
        
        q_emb = self._embedder.encode(query, convert_to_numpy=True)
        if q_emb.ndim > 1:
            q_emb = q_emb[0]
        
        scores = {}
        best_example_scores = {}
        
        for intent, intent_emb in self._intent_embeddings.items():
            avg_score = self._cosine_sim(q_emb, intent_emb)
            example_embs = self._example_embeddings[intent]
            example_sims = [self._cosine_sim(q_emb, ex) for ex in example_embs]
            max_example_score = max(example_sims)
            
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


# ============================================================
# 싱글톤
# ============================================================
_classifier: Optional[IntentClassifier] = None
_init_lock = threading.Lock()


def get_intent_classifier() -> IntentClassifier:
    global _classifier
    
    if _classifier is not None:
        return _classifier
    
    with _init_lock:
        if _classifier is None:
            _classifier = IntentClassifier()
    
    return _classifier


# ============================================================
# 피드백 로깅 (동기 PyMongo 사용)
# ============================================================
def _get_sync_db():
    """동기 MongoDB 클라이언트 (피드백 로깅용)"""
    from pymongo import MongoClient
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME", "lostark")
    client = MongoClient(uri)
    return client[db_name]


async def log_intent_feedback(
    query: str,
    classified_intent: str,
    confidence: float,
    actual_handler: str,
    success: bool,
    response_type: str = None,
    extra: dict = None,
):
    """피드백 로깅 (별도 스레드에서 동기 방식)"""
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
    
    def _insert():
        try:
            db = _get_sync_db()
            db["intent_feedback"].insert_one(doc)
        except Exception as e:
            logger.warning(f"Failed to log intent feedback: {e}")
    
    thread = threading.Thread(target=_insert, daemon=True)
    thread.start()


async def get_low_confidence_queries(days: int = 7, limit: int = 100) -> List[dict]:
    """리뷰 필요 쿼리 조회 (동기 방식)"""
    try:
        db = _get_sync_db()
        cutoff = datetime.now(KST) - timedelta(days=days)
        cursor = db["intent_feedback"].find(
            {"needs_review": True, "ts": {"$gte": cutoff}},
            {"_id": 0, "query": 1, "classified_intent": 1, "confidence": 1, "actual_handler": 1}
        ).sort("ts", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        logger.warning(f"Failed to get low confidence queries: {e}")
        return []