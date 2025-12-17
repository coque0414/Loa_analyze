# skills/youtube.py
"""
YouTube 영상 추천 스킬 - Title/Tags 기반 매칭 + 타임스탬프 추출

핵심 로직:
1. 1차 매칭: title과 tags로 영상 선택 (임베딩 사용 안 함)
2. 2차 타임스탬프: 매칭된 영상의 segment_text에서 키워드 관련 구간 찾기
3. 타임스탬프 못 찾으면 0:00부터 시작
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import re
import time

from services.db import youtube_col

KST = timezone(timedelta(hours=9))

# ============================================================
# 설정 상수
# ============================================================
DEFAULT_LIMIT = 2
TITLE_MATCH_THRESHOLD = 0.3  # title/tags 매칭 최소 점수
MAX_VIDEOS = 500  # 고유 영상 최대 개수
CACHE_TTL = 1800  # 30분

# 캐시 (영상별 그룹화된 데이터)
_video_cache = {
    "videos": None,  # {video_id: {"title", "tags", "channel_title", "url", "segments": [...]}}
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
# 유틸리티 함수
# ============================================================
def _extract_video_id(doc_id: str) -> str:
    """_id에서 video_id 추출"""
    if not doc_id:
        return ""
    if doc_id.startswith("video:"):
        doc_id = doc_id[6:]
    if "#" in doc_id:
        doc_id = doc_id.split("#")[0]
    return doc_id


def _get_thumbnail_url(video_id: str, quality: str = "mqdefault") -> str:
    """YouTube 썸네일 URL 생성"""
    if not video_id:
        return ""
    return f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"


def _format_timestamp(seconds: float) -> str:
    """초를 MM:SS 형식으로 변환"""
    if seconds is None or seconds < 0:
        return "0:00"
    
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _make_timestamp_url(base_url: str, seconds: float) -> str:
    """타임스탬프가 포함된 YouTube URL 생성"""
    if not base_url:
        return ""
    
    seconds = int(seconds) if seconds else 0
    
    # URL 파싱해서 기존 t 파라미터만 제거하고 새로 추가
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    
    try:
        parsed = urlparse(base_url)
        query_params = parse_qs(parsed.query)
        
        # 기존 t 파라미터 제거
        query_params.pop('t', None)
        
        # 새 타임스탬프 추가 (0이 아닐 때만)
        if seconds > 0:
            query_params['t'] = [str(seconds)]
        
        # 파라미터를 단일 값으로 변환 (parse_qs는 리스트로 반환)
        flat_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        # URL 재조립
        new_query = urlencode(flat_params)
        new_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return new_url
        
    except Exception as e:
        print(f"[YouTube] URL 파싱 에러: {e}, base_url={base_url}")
        # 폴백: 단순히 &t= 추가
        if seconds > 0:
            separator = "&" if "?" in base_url else "?"
            return f"{base_url}{separator}t={seconds}"
        return base_url


def _tokenize_query(query: str) -> List[str]:
    """
    질문에서 키워드 토큰 추출 (불용어 제거)
    """
    stopwords = (
        "영상", "유튜브", "동영상", "추천", "알려줘", "보여줘",
        "해줘", "있어", "찾아줘", "공략", "어떻게", "뭐야", "주세요",
        "좀", "해주세요", "볼래", "보고싶어", "관련", "대한"
    )
    q = query.lower()
    for sw in stopwords:
        q = q.replace(sw, " ")
    
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", q)
    return [t for t in tokens if len(t) >= 2]


# ============================================================
# 1차: Title + Tags 기반 영상 매칭
# ============================================================
def _calculate_title_tags_score(query_tokens: List[str], title: str, tags: List[str]) -> float:
    """
    title과 tags만으로 매칭 점수 계산 (0.0 ~ 1.0)
    
    - title 완전 일치: 10점
    - title 부분 일치: 7점
    - tags 완전 일치: 8점
    - tags 부분 일치: 5점
    """
    if not query_tokens:
        return 0.0
    
    title_lower = (title or "").lower()
    tags_lower = [t.lower() for t in (tags or [])]
    tags_joined = " ".join(tags_lower)
    
    total_score = 0.0
    max_possible = len(query_tokens) * 10.0  # 최대 점수
    
    for token in query_tokens:
        token_score = 0.0
        
        # 1. Title 매칭 (우선순위 최고)
        # 완전 일치 (단어 경계)
        if re.search(rf'(?:^|[^가-힣a-z0-9]){re.escape(token)}(?:[^가-힣a-z0-9]|$)', title_lower):
            token_score = 10.0
        # 부분 일치
        elif token in title_lower:
            token_score = 7.0
        
        # 2. Tags 매칭
        if token_score == 0:
            # 완전 일치
            if token in tags_lower:
                token_score = 8.0
            # 부분 일치
            elif token in tags_joined:
                token_score = 5.0
        
        total_score += token_score
    
    # 정규화
    return min(1.0, total_score / max_possible) if max_possible > 0 else 0.0


# ============================================================
# 2차: Segment Text에서 타임스탬프 찾기
# ============================================================
def _find_best_timestamp(query_tokens: List[str], segments: List[Dict]) -> Tuple[float, str]:
    """
    세그먼트들의 text에서 질문 키워드와 가장 관련 있는 구간의 타임스탬프 찾기
    
    Returns:
        (start_seconds, matched_text_preview)
        못 찾으면 (0, "")
    """
    if not query_tokens or not segments:
        return 0, ""
    
    best_score = 0
    best_start = 0
    best_text = ""
    
    for seg in segments:
        text = (seg.get("segment_text") or "").lower()
        start = seg.get("segment_start", 0) or 0
        
        if not text:
            continue
        
        # 토큰 매칭 점수 계산
        match_count = 0
        for token in query_tokens:
            if token in text:
                match_count += 1
        
        if match_count > 0:
            # 매칭된 토큰 수 / 전체 토큰 수 = 매칭률
            score = match_count / len(query_tokens)
            
            if score > best_score:
                best_score = score
                best_start = start
                # 미리보기 텍스트 (50자)
                best_text = text[:50] + "..." if len(text) > 50 else text
    
    return best_start, best_text


# ============================================================
# 캐시 로드: 영상별 그룹화
# ============================================================
async def _ensure_video_cache():
    """
    youtube_col에서 데이터를 로드하고 영상별로 그룹화
    """
    now = time.time()
    
    if (_video_cache["videos"] is not None and 
        (now - _video_cache["loaded_at"] < CACHE_TTL)):
        print(f"[YouTube] 캐시 히트 (영상 {len(_video_cache['videos'])}개)")
        return _video_cache["videos"]
    
    print("[YouTube] 캐시 갱신 시작...")
    
    try:
        from services.db import youtube_col
        print(f"[YouTube] youtube_col 임포트 성공: {youtube_col}")
        
        # 모든 세그먼트 로드
        docs = await youtube_col.find(
            {},
            {
                "_id": 1, "title": 1, "channel_title": 1,
                "url": 1, "tags": 1,
                "segment_idx": 1, "segment_start": 1, 
                "segment_end": 1, "segment_text": 1,
                "published_at": 1, "duration": 1,
            }
        ).sort("published_at", -1).to_list(length=10000)
        
        print(f"[YouTube] DB에서 {len(docs)}개 문서 로드됨")
        
    except Exception as e:
        print(f"[YouTube] ❌ DB 로드 에러: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    if not docs:
        print("[YouTube] ⚠️ DB에 문서가 없음!")
        return {}
    
    # 영상별 그룹화
    videos: Dict[str, Dict] = {}
    
    for doc in docs:
        doc_id = str(doc.get("_id", ""))
        video_id = _extract_video_id(doc_id)
        
        if not video_id:
            continue
        
        if video_id not in videos:
            videos[video_id] = {
                "video_id": video_id,
                "title": doc.get("title", ""),
                "channel_title": doc.get("channel_title", ""),
                "url": doc.get("url", ""),
                "tags": doc.get("tags", []),
                "published_at": doc.get("published_at"),
                "duration": doc.get("duration", ""),
                "segments": [],
            }
        
        # 세그먼트 추가
        videos[video_id]["segments"].append({
            "segment_idx": doc.get("segment_idx"),
            "segment_start": doc.get("segment_start"),
            "segment_end": doc.get("segment_end"),
            "segment_text": doc.get("segment_text", ""),
        })
    
    # 세그먼트 정렬
    for vid, vdata in videos.items():
        vdata["segments"].sort(key=lambda x: x.get("segment_start") or 0)
    
    _video_cache.update({
        "videos": videos,
        "loaded_at": now,
    })
    
    print(f"[YouTube] ✅ 캐시 완료: {len(videos)}개 영상 (세그먼트 총 {len(docs)}개)")
    
    # 샘플 출력
    if videos:
        sample = list(videos.values())[:3]
        for v in sample:
            print(f"  - {v['title'][:40]}... (tags={v['tags'][:3]})")
    
    return videos


# ============================================================
# 메인 검색 함수
# ============================================================
async def search_youtube_videos(
    query: str,
    limit: int = DEFAULT_LIMIT,
    score_threshold: float = TITLE_MATCH_THRESHOLD,
    debug: bool = False,
) -> List[Dict]:
    """
    YouTube 영상 검색 - Title/Tags 기반 매칭 + 타임스탬프 추출
    
    1. title과 tags로 영상 매칭 (임베딩 사용 안 함)
    2. 매칭된 영상의 segment_text에서 관련 타임스탬프 찾기
    3. 못 찾으면 0:00부터 시작
    """
    print(f"\n[YouTube] search_youtube_videos 시작")
    print(f"[YouTube] 파라미터: query='{query}', limit={limit}, threshold={score_threshold}")
    
    query_tokens = _tokenize_query(query)
    
    print(f"[YouTube] 추출된 토큰: {query_tokens}")
    
    if not query_tokens:
        print("[YouTube] ⚠️ 토큰이 없어서 검색 불가")
        return []
    
    # 캐시 로드
    print("[YouTube] 캐시 로드 중...")
    videos = await _ensure_video_cache()
    
    if not videos:
        print("[YouTube] ⚠️ 캐시된 영상이 없음")
        return []
    
    print(f"[YouTube] 캐시에서 {len(videos)}개 영상 로드됨")
    
    # 1차: Title + Tags 매칭
    scored_videos = []
    
    for video_id, vdata in videos.items():
        title = vdata.get("title", "")
        tags = vdata.get("tags", [])
        
        score = _calculate_title_tags_score(query_tokens, title, tags)
        
        if score >= score_threshold:
            scored_videos.append({
                **vdata,
                "match_score": score,
            })
    
    print(f"[YouTube] 1차 매칭 결과: {len(scored_videos)}개 (threshold={score_threshold})")
    
    if not scored_videos:
        # 임계값 낮춰서 재시도
        relaxed = score_threshold * 0.6
        print(f"[YouTube] 임계값 완화 재시도: {relaxed}")
        
        for video_id, vdata in videos.items():
            title = vdata.get("title", "")
            tags = vdata.get("tags", [])
            
            score = _calculate_title_tags_score(query_tokens, title, tags)
            
            if score >= relaxed:
                scored_videos.append({
                    **vdata,
                    "match_score": score,
                })
        
        print(f"[YouTube] 완화 후 결과: {len(scored_videos)}개")
    
    if not scored_videos:
        print("[YouTube] ⚠️ 매칭되는 영상 없음")
        return []
    
    # 점수순 정렬
    scored_videos.sort(key=lambda x: x["match_score"], reverse=True)
    
    # 상위 결과 출력
    print(f"\n[YouTube] 상위 매칭 결과:")
    for i, v in enumerate(scored_videos[:5], 1):
        print(f"  {i}. {v['title'][:50]}... (점수={v['match_score']:.3f})")
        print(f"     tags: {v.get('tags', [])[:5]}")
    
    # 2차: 타임스탬프 추출
    results = []
    
    for vdata in scored_videos[:limit * 2]:  # 여유있게 검토
        video_id = vdata["video_id"]
        segments = vdata.get("segments", [])
        
        # 세그먼트 text에서 관련 타임스탬프 찾기
        best_start, matched_text = _find_best_timestamp(query_tokens, segments)
        
        base_url = vdata.get("url", "")
        timestamp_url = _make_timestamp_url(base_url, best_start)
        
        results.append({
            "video_id": video_id,
            "title": vdata.get("title", ""),
            "channel_title": vdata.get("channel_title", ""),
            "url": base_url,
            "segment_url": timestamp_url,
            "segment_start": best_start,
            "segment_text_preview": matched_text,
            "thumbnail": _get_thumbnail_url(video_id),
            "tags": vdata.get("tags", []),
            "published_at": vdata.get("published_at"),
            "duration": vdata.get("duration", ""),
            "match_score": vdata["match_score"],
        })
        
        ts = _format_timestamp(best_start)
        print(f"[YouTube] ✓ 선택: {vdata['title'][:40]}... (시작={ts})")
        
        if len(results) >= limit:
            break
    
    print(f"\n[YouTube] 최종 결과: {len(results)}개 영상")
    return results


# ============================================================
# 응답 생성
# ============================================================
def _build_video_card_html(video: Dict) -> str:
    """개별 영상 카드 HTML"""
    title = video.get("title", "영상")
    channel = video.get("channel_title", "")
    thumbnail = video.get("thumbnail", "")
    
    segment_start = video.get("segment_start", 0)
    segment_url = video.get("segment_url", "")
    base_url = video.get("url", "")
    
    # 타임스탬프 표시
    link_url = segment_url if segment_url else base_url
    ts_formatted = _format_timestamp(segment_start)
    
    timestamp_html = ""
    if segment_start > 0:
        timestamp_html = f'''
        <div style="font-size:12px;color:#dc2626;margin-top:4px">
            ⏱️ {ts_formatted} 부터 관련 내용
        </div>
        '''
    else:
        timestamp_html = f'''
        <div style="font-size:12px;color:#6b7280;margin-top:4px">
            ▶️ 처음부터 시청
        </div>
        '''
    
    display_title = title[:50] + "..." if len(title) > 50 else title
    
    # 매칭 점수 뱃지
    match_score = video.get("match_score", 0)
    badge = ""
    if match_score >= 0.7:
        badge = '<span style="background:#10b981;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-left:4px">정확</span>'
    elif match_score >= 0.4:
        badge = '<span style="background:#f59e0b;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-left:4px">관련</span>'
    
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
                {channel} {badge}
            </div>
            {timestamp_html}
        </div>
    </div>
    '''


def build_youtube_response(videos: List[Dict], query: str) -> Dict[str, Any]:
    """YouTube 추천 응답 생성"""
    if not videos:
        return {
            "type": "youtube",
            "answer": "죄송해요, 관련 영상을 찾지 못했어요. 다른 키워드로 검색해 보시겠어요?",
            "answer_html": None,
            "videos": [],
        }
    
    if len(videos) == 1:
        answer_text = f"'{videos[0]['title'][:30]}...' 영상을 추천드려요!"
    else:
        answer_text = f"관련 영상 {len(videos)}개를 찾았어요. 확인해 보세요!"
    
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
    """YouTube 영상 추천 메인 핸들러"""
    print(f"\n{'='*50}")
    print(f"[YouTube] answer_youtube_recommend 호출됨")
    print(f"[YouTube] 입력 쿼리: {query}")
    print(f"{'='*50}")
    
    try:
        videos = await search_youtube_videos(
            query=query,
            limit=DEFAULT_LIMIT,
            debug=True  # 디버깅 활성화
        )
        print(f"[YouTube] 검색 결과: {len(videos)}개 영상 찾음")
        
        response = build_youtube_response(videos, query)
        print(f"[YouTube] 응답 생성 완료 (type={response.get('type')})")
        return response
        
    except Exception as e:
        print(f"[YouTube] ❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return {
            "type": "youtube",
            "answer": "영상 검색 중 오류가 발생했어요.",
            "answer_html": None,
            "videos": [],
        }


# ============================================================
# 보조 함수: QA 응답에 보충 영상
# ============================================================
async def get_supplementary_videos(
    query: str,
    limit: int = 1,
    score_threshold: float = 0.35
) -> List[Dict]:
    """QA 응답에 보충할 영상 검색"""
    videos = await search_youtube_videos(
        query=query,
        limit=limit,
        score_threshold=score_threshold,
        debug=False
    )
    return videos


def build_supplementary_html(videos: List[Dict]) -> Optional[str]:
    """보충 영상 HTML"""
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