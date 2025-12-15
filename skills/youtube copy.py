# skills/youtube.py
"""
YouTube 영상 추천 스킬

기능:
- 명시적 영상 추천 요청 처리 (시나리오 A)
- 세그먼트 병합 및 중복 제거
- 썸네일 + 타임스탬프 딥링크 제공
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import re
import numpy as np
import time

from services.db import youtube_col
from services.embedder import get_embedder

KST = timezone(timedelta(hours=9))

# ============================================================
# 설정 상수
# ============================================================
DEFAULT_LIMIT = 2  # 최대 추천 영상 수
SIMILARITY_THRESHOLD = 0.35  # 최소 유사도 임계값
KEYWORD_BOOST = 0.15  # 키워드 매칭 가중치


# 성능 최적화 설정 (Flex 티어용)
MAX_DOCUMENTS = 3000  # 로드할 최대 문서 수 (5000 → 3000으로 감소)
CACHE_TTL = 1800  # 캐시 유지 시간 30분 (더 긴 캐싱)
EARLY_STOP_THRESHOLD = 10  # 상위 N개만 상세 분석 (성능 향상)

# 임베딩 캐시 (전역)
_embedding_cache = {
    "embeddings": None,
    "docs": None,
    "loaded_at": 0,
}

# ============================================================
# 의도 감지
# ============================================================
def is_youtube_intent(query: str) -> bool:
    """YouTube 영상 추천 의도인지 판별"""
    q = (query or "").lower().replace(" ", "")
    keywords = (
        "영상", "유튜브", "동영상", "추천", "공략영상",
        "유튜버", "보여줘영상", "찾아줘영상", "youtube",
        "영상으로", "영상있", "영상알려", "영상추천"
    )
    return any(k in q for k in keywords)


# ============================================================
# 검색 유틸리티
# ============================================================
def _extract_video_id(doc_id: str) -> str:
    """
    _id에서 video_id 추출
    예: "video:uu6DSpiL8o0#seg1" → "uu6DSpiL8o0"
    """
    if not doc_id:
        return ""
    # "video:" 접두사 제거
    if doc_id.startswith("video:"):
        doc_id = doc_id[6:]
    # "#seg" 이후 제거
    if "#" in doc_id:
        doc_id = doc_id.split("#")[0]
    return doc_id


def _get_thumbnail_url(video_id: str, quality: str = "mqdefault") -> str:
    """
    YouTube 썸네일 URL 생성
    quality: default, mqdefault, hqdefault, sddefault, maxresdefault
    """
    if not video_id:
        return ""
    return f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"


def _format_timestamp(seconds: float) -> str:
    """초를 MM:SS 또는 HH:MM:SS 형식으로 변환"""
    if seconds is None or seconds < 0:
        return ""
    
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _tokenize_query(query: str) -> List[str]:
    """질문에서 키워드 토큰 추출"""
    # 의도어/불용어 제거
    stopwords = (
        "영상", "유튜브", "동영상", "추천", "알려줘", "보여줘",
        "해줘", "있어", "찾아줘", "공략", "어떻게", "뭐야"
    )
    q = query.lower()
    for sw in stopwords:
        q = q.replace(sw, " ")
    
    # 토큰 추출
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", q)
    return [t for t in tokens if len(t) >= 2]


# ============================================================
# 세그먼트 병합
# ============================================================
def merge_video_segments(segments: List[Dict]) -> List[Dict]:
    """
    동일 영상의 세그먼트들을 병합하여 영상 단위로 그룹화
    
    입력: 개별 세그먼트 리스트
    출력: 영상 단위로 병합된 리스트 (최고 점수 세그먼트 정보 유지)
    """
    if not segments:
        return []
    
    # video_id 기준 그룹화
    video_groups: Dict[str, List[Dict]] = defaultdict(list)
    
    for seg in segments:
        doc_id = str(seg.get("_id", ""))
        video_id = _extract_video_id(doc_id)
        if video_id:
            video_groups[video_id].append(seg)
    
    # 각 영상별로 최고 점수 세그먼트 선택 + 관련 세그먼트 정보 유지
    merged = []
    for video_id, segs in video_groups.items():
        # 점수순 정렬
        segs_sorted = sorted(segs, key=lambda x: x.get("score", 0), reverse=True)
        best_seg = segs_sorted[0]
        
        # 관련 세그먼트 인덱스 수집
        segment_indices = sorted(set(
            s.get("segment_idx") for s in segs if s.get("segment_idx") is not None
        ))
        
        merged.append({
            "video_id": video_id,
            "title": best_seg.get("title", ""),
            "channel_title": best_seg.get("channel_title", ""),
            "url": best_seg.get("url", ""),
            "segment_url": best_seg.get("segment_url", ""),
            "segment_start": best_seg.get("segment_start"),
            "segment_end": best_seg.get("segment_end"),
            "segment_text": best_seg.get("segment_text", ""),
            "published_at": best_seg.get("published_at"),
            "duration": best_seg.get("duration", ""),
            "tags": best_seg.get("tags", []),
            "score": best_seg.get("score", 0.0),
            "keyword_match": best_seg.get("keyword_match", False),
            "matched_segments": segment_indices,
            "thumbnail": _get_thumbnail_url(video_id),
        })
    
    # 점수순 정렬
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


# ============================================================
# YouTube 검색
# ============================================================
async def search_youtube_videos(
    query: str,
    limit: int = DEFAULT_LIMIT,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """
    YouTube 영상 검색 (Flex 티어 최적화)
    
    최적화 기법:
    1. 임베딩/문서 30분 캐싱 → 반복 검색 시 즉시 응답
    2. 최대 3000개 문서만 로드 → 메모리/시간 절약
    3. 최신 영상 우선 정렬 → 관련성 높은 결과
    4. 상위 50개만 상세 분석 → 조기 종료로 성능 향상
    5. numpy 연산 최적화 → 빠른 유사도 계산
    """
    embedder = get_embedder()
    
    # 질의 임베딩
    q_emb = embedder.encode(query, convert_to_numpy=True).astype(np.float32)
    if q_emb.ndim > 1:
        q_emb = q_emb[0]
    
    # 키워드 토큰
    query_tokens = _tokenize_query(query)
    
    # ✅ 캐시 확인 (30분 유효)
    now = time.time()
    if (_embedding_cache["embeddings"] is None or 
        (now - _embedding_cache["loaded_at"] > CACHE_TTL)):
        
        print(f"[INFO] 임베딩 캐시 갱신 중... (최대 {MAX_DOCUMENTS}개 문서)")
        
        try:
            # ✅ 최신 영상 우선, 최대 3000개만 로드
            docs = await youtube_col.find(
                {"embedding": {"$exists": True}},
                {
                    "_id": 1, "title": 1, "channel_title": 1,
                    "url": 1, "segment_url": 1, "segment_idx": 1,
                    "segment_start": 1, "segment_end": 1, "segment_text": 1,
                    "published_at": 1, "duration": 1, "tags": 1, "embedding": 1,
                }
            ).sort("published_at", -1).to_list(length=MAX_DOCUMENTS)
        except Exception as e:
            print(f"[ERROR] youtube_col search failed: {e}")
            return []
        
        if not docs:
            return []
        
        # 임베딩 추출
        valid_docs = []
        embeddings = []
        
        for doc in docs:
            emb = doc.get("embedding")
            if emb is not None:
                valid_docs.append(doc)
                embeddings.append(np.asarray(emb, dtype=np.float32))
        
        if not valid_docs:
            return []
        
        emb_matrix = np.vstack(embeddings).astype(np.float32)
        
        # ✅ 캐시 저장
        _embedding_cache.update({
            "embeddings": emb_matrix,
            "docs": valid_docs,
            "loaded_at": now
        })
        
        print(f"[INFO] 캐시 갱신 완료: {len(valid_docs)}개 문서")
    
    # 캐시에서 로드
    emb_matrix = _embedding_cache["embeddings"]
    cached_docs = _embedding_cache["docs"]
    
    # ✅ 최적화된 코사인 유사도 계산
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12)
    similarities = emb_norms @ q_norm
    
    # ✅ 조기 종료: 상위 N개만 선택 (전체 정렬 X)
    if len(similarities) > EARLY_STOP_THRESHOLD:
        top_indices = np.argpartition(similarities, -EARLY_STOP_THRESHOLD)[-EARLY_STOP_THRESHOLD:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    else:
        top_indices = np.argsort(similarities)[::-1]
    
    # 상위 문서만 복사
    docs = []
    for idx in top_indices:
        doc = dict(cached_docs[idx])  # 얕은 복사
        doc["score"] = float(similarities[idx])
        docs.append(doc)
    
    # ✅ 키워드 부스팅 (태그 강화)
    for doc in docs:
        title = (doc.get("title") or "").lower()
        text = (doc.get("segment_text") or "").lower()
        tags = [t.lower() for t in (doc.get("tags") or [])]
        
        match_count = 0
        tag_exact_match = False
        
        for token in query_tokens:
            if token in title:
                match_count += 2  # 제목 우선
            elif token in text:
                match_count += 1
            elif any(token in tag for tag in tags):
                match_count += 1.5  # 태그 가중치 증가
                if token in tags:  # 완전 일치
                    tag_exact_match = True
        
        if match_count > 0:
            boost = min(KEYWORD_BOOST * match_count, 0.3)
            if tag_exact_match:
                boost += 0.1  # 태그 정확 일치 보너스
            doc["score"] = min(1.0, doc.get("score", 0) + boost)
            doc["keyword_match"] = True
        else:
            doc["keyword_match"] = False
    
    # 임계값 필터링
    filtered = [d for d in docs if d.get("score", 0) >= score_threshold]
    
    if not filtered:
        return []
    
    # 세그먼트 병합
    merged = merge_video_segments(filtered)
    
    return merged[:limit]


# ============================================================
# 응답 생성
# ============================================================
def _build_video_card_html(video: Dict, show_timestamp: bool = True) -> str:
    """개별 영상 카드 HTML 생성"""
    title = video.get("title", "영상")
    channel = video.get("channel_title", "")
    thumbnail = video.get("thumbnail", "")
    score = video.get("score", 0)
    
    # 링크 결정: 타임스탬프 있으면 segment_url, 없으면 url
    segment_start = video.get("segment_start")
    segment_url = video.get("segment_url", "")
    base_url = video.get("url", "")
    
    # 타임스탬프 표시 여부 결정
    timestamp_html = ""
    link_url = base_url
    
    if show_timestamp and segment_start is not None and segment_url:
        link_url = segment_url
        ts_formatted = _format_timestamp(segment_start)
        if ts_formatted:
            timestamp_html = f'''
            <div style="font-size:12px;color:#dc2626;margin-top:4px">
                ⏱️ {ts_formatted} 부터 관련 내용
            </div>
            '''
    
    # 제목 길이 제한
    display_title = title[:50] + "..." if len(title) > 50 else title
    
    return f'''
    <div style="display:flex;gap:12px;padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;margin-bottom:10px">
        <a href="{link_url}" target="_blank" rel="noopener" style="flex-shrink:0">
            <img src="{thumbnail}" alt="썸네일" 
                 style="width:120px;height:68px;object-fit:cover;border-radius:8px;border:1px solid #e5e7eb"/>
        </a>
        <div style="flex:1;min-width:0">
            <a href="{link_url}" target="_blank" rel="noopener" 
               style="text-decoration:none;color:#111827;font-weight:600;font-size:14px;line-height:1.3;display:block">
                {display_title}
            </a>
            <div style="font-size:12px;color:#6b7280;margin-top:4px">
                {channel}
            </div>
            {timestamp_html}
        </div>
    </div>
    '''


def build_youtube_response(
    videos: List[Dict],
    query: str,
) -> Dict[str, Any]:
    """
    YouTube 추천 응답 생성
    
    Returns:
        {
            "type": "youtube",
            "answer": 텍스트 응답,
            "answer_html": HTML 카드 UI,
            "videos": 영상 메타데이터 리스트,
        }
    """
    if not videos:
        return {
            "type": "youtube",
            "answer": "죄송해요, 관련 영상을 찾지 못했어요. 다른 키워드로 검색해 보시겠어요?",
            "answer_html": None,
            "videos": [],
        }
    
    # 텍스트 응답
    if len(videos) == 1:
        answer_text = f"'{videos[0]['title'][:30]}...' 영상을 추천드려요!"
    else:
        answer_text = f"관련 영상 {len(videos)}개를 찾았어요. 확인해 보세요!"
    
    # HTML 카드 생성
    cards_html = "".join(_build_video_card_html(v) for v in videos)
    
    wrapper_html = f'''
    <div style="max-width:480px">
        <div style="font-size:14px;color:#374151;margin-bottom:12px">
            📺 추천 영상
        </div>
        {cards_html}
    </div>
    '''
    
    return {
        "type": "youtube",
        "answer": answer_text,
        "answer_html": wrapper_html.strip(),
        "videos": videos,
    }


# ============================================================
# 메인 핸들러
# ============================================================
async def answer_youtube_recommend(query: str) -> Dict[str, Any]:
    """
    YouTube 영상 추천 메인 핸들러
    """
    # 검색 실행
    videos = await search_youtube_videos(
        query=query,
        limit=DEFAULT_LIMIT,
        score_threshold=SIMILARITY_THRESHOLD,
    )
    
    # 응답 생성
    return build_youtube_response(videos, query)


# ============================================================
# 보조 함수: QA 보충 영상 (시나리오 B용 - 나중에 사용)
# ============================================================
async def get_supplementary_videos(
    query: str,
    limit: int = 1,
    score_threshold: float = 0.45,  # 보충용은 더 엄격하게
) -> List[Dict]:
    """
    QA 응답에 보충할 관련 영상 검색
    
    시나리오 B: 텍스트 답변 + 보충 영상
    - 관련 영상이 없으면 빈 리스트 반환
    - 점수가 낮으면 추천하지 않음
    """
    videos = await search_youtube_videos(
        query=query,
        limit=limit,
        score_threshold=score_threshold,
    )
    return videos


def build_supplementary_html(videos: List[Dict]) -> Optional[str]:
    """
    보충 영상 HTML 생성 (시나리오 B용)
    
    영상이 없으면 None 반환
    """
    if not videos:
        return None
    
    cards_html = "".join(_build_video_card_html(v) for v in videos)
    
    return f'''
    <div style="margin-top:16px;padding-top:12px;border-top:1px solid #e5e7eb">
        <div style="font-size:13px;color:#6b7280;margin-bottom:8px">
            📺 관련 영상도 참고해보세요
        </div>
        {cards_html}
    </div>
    '''.strip()