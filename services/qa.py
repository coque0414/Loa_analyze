import textwrap
import numpy as np
import time
from openai import OpenAI
import os
import asyncio
import re
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
from collections import defaultdict

from rapidfuzz import fuzz, process
from bson import ObjectId

from services.db import posts_col, docs_col, maps_col, guide_col
from services.embedder import get_embedder
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# 설정 상수
# ============================================================================
DEFAULT_CACHE_TTL = 600  # 10분 (30초 → 10분으로 증가)
DEFAULT_SCORE_THRESHOLD = 0.28  # 0.35 → 0.28로 낮춤 (더 관대한 검색)
DEFAULT_K_DOCS = 10  # 8 → 10으로 증가 (더 많은 문서 검색)
MAX_CONTEXT_CHARS = 5000  # 4000 → 5000으로 증가
DOCS_WEIGHT = float(os.getenv("DOCS_WEIGHT", "1.20"))
GUIDE_WEIGHT = float(os.getenv("GUIDE_WEIGHT", "1.25"))  # 1.15 → 1.25 (가이드 더 우대)
KEYWORD_BOOST = 0.25  # 0.20 → 0.25 (키워드 매칭 더 강화)

# OpenAI 클라이언트 (재시도 로직 포함)
openai = OpenAI(
    api_key=os.getenv("openai_api_key"),
    timeout=30.0,  # 타임아웃 설정
    max_retries=2  # 재시도 횟수
)

# ============================================================================
# 캐시 구조
# ============================================================================
_emb_cache = {
    "embs": None,
    "docs": None,
    "loaded_at": 0,
    "doc_hash": None  # 문서 변경 감지용
}

_map_cache = {
    "worlds": None,
    "regions": None,
    "worlds_norm": None,
    "regions_norm": None,
    "loaded_at": 0,
}

# ============================================================================
# 정규화 및 유틸리티 함수
# ============================================================================

@lru_cache(maxsize=1000)
def _norm_kor(s: str) -> str:
    """한글/영문/숫자만 남기고 소문자화 (캐싱 적용)"""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", s)
    return s


@lru_cache(maxsize=500)
def _tokenize_korean(text: str) -> Tuple[str, ...]:
    """한글 토큰 추출 (캐싱으로 반복 호출 최적화)"""
    return tuple(re.findall(r"[가-힣A-Za-z0-9]+", text.lower()))


def _normalize_doc(raw: dict, source: str) -> dict:
    """문서 정규화 - 다양한 필드명 통합"""
    title = (
        raw.get("title") or 
        raw.get("name") or 
        raw.get("doc_name") or 
        str(raw.get("_id", ""))
    )
    
    # 본문 우선순위
    text = (
        raw.get("body") or 
        raw.get("text") or 
        raw.get("content") or 
        raw.get("doc_body") or 
        ""
    )
    
    emb = raw.get("embedding")
    doc_id = raw.get("_id") or raw.get("doc_id") or ""
    chunk_idx = raw.get("chunk_idx") if "chunk_idx" in raw else None
    content_type = raw.get("content_type")
    
    return {
        "title": title,
        "text": text,
        "body": text,
        "embedding": np.asarray(emb, dtype=np.float32) if emb is not None else None,
        "source": source,
        "doc_id": doc_id,
        "chunk_idx": chunk_idx,
        "content_type": content_type
    }


# ============================================================================
# 문서 청크 병합
# ============================================================================

def merge_chunked_docs(docs_list: List[Dict]) -> List[Dict]:
    """
    동일 문서의 청크들을 병합
    - doc_id 기반 그룹화 우선
    - title 패턴 매칭 보조
    """
    if not docs_list:
        return []

    groups = defaultdict(list)  # dict 대신 defaultdict 사용
    title_pat = re.compile(r'^(.*?)(?:\s*#\s*(\d+))\s*$', re.IGNORECASE)
    
    for d in docs_list:
        title = (d.get("title") or "").strip()
        src = d.get("source", "unknown")
        doc_id = str(d.get("doc_id") or d.get("_id") or "")
        
        # doc_id 기반 그룹화
        if doc_id and "#" in doc_id:
            parts = doc_id.split("#")
            base = parts[0]
            try:
                idx = int(parts[-1])
            except (ValueError, IndexError):
                idx = d.get("chunk_idx")
        else:
            base = doc_id if doc_id else title
            idx = d.get("chunk_idx")
            
            # title 패턴 보조
            if not doc_id:
                m = title_pat.match(title)
                if m:
                    base = m.group(1).strip()
                    try:
                        idx = int(m.group(2))
                    except (ValueError, TypeError):
                        pass
        
        key = (base, src)
        groups[key].append((idx, d))
    
    merged = []
    for (base, src), items in groups.items():
        if len(items) == 1:
            single = items[0][1]
            # text/body 보장
            single["text"] = single.get("text") or single.get("body") or ""
            single["body"] = single["text"]
            single["doc_id"] = single.get("doc_id") or base
            merged.append(single)
            continue
        
        # 청크 정렬
        sorted_items = sorted(
            items, 
            key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0)
        )
        
        # 본문 병합
        texts = [
            (it[1].get("text") or it[1].get("body") or 
             it[1].get("content") or it[1].get("doc_body") or "")
            for it in sorted_items
        ]
        merged_text = "\n\n".join(filter(None, texts))
        
        # 임베딩 평균
        embs = [it[1].get("embedding") for it in sorted_items 
                if it[1].get("embedding") is not None]
        emb = None
        if embs:
            try:
                emb = np.mean(np.vstack(embs), axis=0).astype(np.float32)
            except Exception:
                emb = embs[0]
        
        # score/keyword_match 집계
        scores = [float(it[1].get("score", 0.0)) for it in sorted_items 
                  if it[1].get("score") is not None]
        keyword_any = any(bool(it[1].get("keyword_match", False)) 
                         for it in sorted_items)
        
        merged.append({
            "title": base,
            "text": merged_text,
            "body": merged_text,
            "embedding": emb,
            "source": src,
            "doc_id": base,
            "score": max(scores) if scores else 0.0,
            "keyword_match": keyword_any
        })
    
    return merged


# ============================================================================
# 의미론적 검색
# ============================================================================

async def semantic_search(
    question: str,
    limit: int = DEFAULT_K_DOCS,
    max_docs: int = 10000,
    cache_ttl: int = DEFAULT_CACHE_TTL
) -> List[Dict]:
    """
    의미론적 문서 검색
    
    개선사항:
    - 캐시 TTL 증가 (30초 → 10분)
    - 키워드 매칭 로직 최적화
    - docs_col, guide_col 우대 가중치 적용
    - 3개 컬렉션 통합 검색 (posts, docs, guide)
    """
    embedder = get_embedder()
    q_emb = embedder.encode(question, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
    
    now = time.time()
    
    # 캐시 갱신 체크
    if _emb_cache["embs"] is None or (now - _emb_cache["loaded_at"] > cache_ttl):
        # posts_col 비활성화 (필요시 주석 해제)
        posts = []
        
        # docs_col 로드
        try:
            docs = await docs_col.find(
                {"embedding": {"$exists": True}},
                {
                    "title": 1, "text": 1, "body": 1, "content": 1,
                    "doc_body": 1, "body_text": 1, "embedding": 1, "_id": 1
                }
            ).to_list(length=max_docs)
        except Exception as e:
            print(f"[ERROR] docs_col fetch failed: {e}")
            docs = []
        
        # guide_col 로드
        try:
            guides = await guide_col.find(
                {"embedding": {"$exists": True}},
                {
                    "title": 1, "text": 1, "body": 1, "content": 1,
                    "doc_body": 1, "body_text": 1, "embedding": 1, "_id": 1
                }
            ).to_list(length=max_docs)
        except Exception as e:
            print(f"[ERROR] guide_col fetch failed: {e}")
            guides = []
        
        # 3개 컬렉션 통합
        combined = []
        # for p in posts:
        #     combined.append(_normalize_doc(p, source="posts"))
        for d in docs:
            combined.append(_normalize_doc(d, source="docs"))
        for g in guides:
            combined.append(_normalize_doc(g, source="guide"))
        
        if not combined:
            return []
        
        # 청크 병합
        merged = merge_chunked_docs(combined)
        merged_with_emb = [m for m in merged if m.get("embedding") is not None]
        
        if not merged_with_emb:
            return []
        
        embs = np.vstack([c["embedding"] for c in merged_with_emb]).astype(np.float32)
        
        # 캐시 업데이트
        _emb_cache.update({
            "embs": embs,
            "docs": merged_with_emb,
            "loaded_at": now
        })
    
    embs = _emb_cache["embs"]
    base_docs = _emb_cache["docs"]
    # ✅ 각 문서 dict를 얕은 복사해서 로컬용 리스트 생성
    docs = [dict(d) for d in base_docs]

    # 코사인 유사도 계산
    try:
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sims = (q_norm @ embs_norm.T)[0]
    except Exception:
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(q_emb, embs)[0]
    
    # 키워드 매칭 보정
    tokens = _tokenize_korean(question)
    
    for idx, doc in enumerate(docs):
        title = (doc.get("title") or "").lower()
        text = (doc.get("text") or "").lower()
        
        match_count = sum(1 for tok in tokens if tok and (tok in title or tok in text))
        
        if match_count > 0:
            sims[idx] = min(1.0, sims[idx] + KEYWORD_BOOST * min(match_count, 3))
            docs[idx]['keyword_match'] = True
        else:
            docs[idx]['keyword_match'] = False

    # 출처 가중치 적용도 docs에 대해 그대로 진행
    for idx, doc in enumerate(docs):
        source = doc.get("source")
        if source == "docs":
            sims[idx] = min(1.0, sims[idx] * DOCS_WEIGHT)
        elif source == "guide":
            sims[idx] = min(1.0, sims[idx] * GUIDE_WEIGHT)

    # 상위 N개 선택
    idxs = sims.argsort()[::-1][:limit]
    results = []
    for i in idxs:
        d = dict(docs[i])         # 여기서 한 번 더 dict() 써도 ok
        d["score"] = float(sims[i])
        results.append(d)

    return results


# ============================================================================
# 스니펫 추출 및 컨텍스트 압축
# ============================================================================

def get_snippets_from_docs(
    ctx_docs: List[Dict],
    question: str,
    per_doc_sentences: int = 5  # 3 → 5로 증가 (문서당 더 많은 문장)
) -> List[Dict]:
    """
    질문 관련 문장 추출 (최적화)
    
    개선:
    - 토큰화 캐싱
    - 그룹화 로직 개선 (defaultdict)
    """
    if not ctx_docs:
        return []
    
    q_tokens = set(_tokenize_korean(question))
    snippets = []
    
    for d in ctx_docs:
        text = (d.get("text") or d.get("body") or 
                d.get("content") or d.get("doc_body") or "")
        
        sents = [s.strip() for s in re.split(r'(?<=[\.\?!\n])\s+', text) if s.strip()]
        
        for s in sents:
            low = s.lower()
            match_count = sum(1 for t in q_tokens if t and (t in low))
            
            if match_count > 0:
                char_len = max(len(s), 1)
                sent_score = (match_count / char_len) * 100.0
                doc_score = float(d.get("score", 0.0))
                kw = bool(d.get("keyword_match", False))
                
                final_score = sent_score + (doc_score * 50.0) + (15.0 if kw else 0.0)
                
                snippets.append({
                    "text": s,
                    "title": d.get("title", ""),
                    "source": d.get("source", "unknown"),
                    "doc_score": doc_score,
                    "keyword_match": kw,
                    "sent_score": final_score
                })
        
        # 관련 문장이 없으면 문서 앞부분 추가
        doc_key = (d.get("title", ""), d.get("source", "unknown"))
        has_snippet = any(
            (s["title"], s["source"]) == doc_key for s in snippets
        )
        
        if not has_snippet and sents:
            preview = " ".join(sents[:per_doc_sentences])
            snippets.append({
                "text": preview,
                "title": d.get("title", ""),
                "source": d.get("source", "unknown"),
                "doc_score": float(d.get("score", 0.0)),
                "keyword_match": d.get("keyword_match", False),
                "sent_score": 5.0 + float(d.get("score", 0.0)) * 10.0
            })
    
    # 그룹화: defaultdict로 최적화
    grouped = defaultdict(list)
    for s in sorted(snippets, key=lambda x: x["sent_score"], reverse=True):
        key = (s["title"], s["source"])
        grouped[key].append(s)
    
    # 문서당 상위 N개만 유지
    final = []
    for items in grouped.values():
        final.extend(items[:per_doc_sentences])
    
    return sorted(final, key=lambda x: x["sent_score"], reverse=True)


def compress_context(snippets: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    컨텍스트 압축 (extractive greedy)
    """
    if not snippets:
        return ""
    
    out = []
    cur_len = 0
    
    for s in snippets:
        part = f'- "{s["text"]}"'
        part_len = len(part)
        
        if cur_len + part_len <= max_chars:
            out.append(part)
            cur_len += part_len + 1
        else:
            remaining = max_chars - cur_len - 1
            if remaining > 20:
                trunc = part[:remaining].rstrip()
                if trunc.count('"') % 2 == 1:
                    trunc = trunc.rstrip('"')
                out.append(trunc + "…(생략)")
            break
    
    return "\n".join(out)


# ============================================================================
# 지도 관련 함수
# ============================================================================

async def _ensure_gazetteer(ttl_sec: int = 600):
    """maps_col에서 world/region 이름 캐싱"""
    now = time.time()
    if _map_cache["worlds"] is None or (now - _map_cache["loaded_at"] > ttl_sec):
        try:
            worlds = await maps_col.distinct("world.name")
            regions = await maps_col.distinct("region.name")
        except Exception as e:
            print(f"[ERROR] gazetteer load failed: {e}")
            worlds, regions = [], []
        
        worlds = [w for w in worlds if isinstance(w, str) and w.strip()]
        regions = [r for r in regions if isinstance(r, str) and r.strip()]
        
        _map_cache.update({
            "worlds": worlds,
            "regions": regions,
            "worlds_norm": [_norm_kor(w) for w in worlds],
            "regions_norm": [_norm_kor(r) for r in regions],
            "loaded_at": now
        })


def _best_match_by_norm(
    q_norm: str,
    norm_list: List[str],
    cutoff: int = 85
) -> Optional[Tuple[int, float]]:
    """정규화된 후보에서 최적 매칭"""
    if not norm_list:
        return None
    res = process.extractOne(q_norm, norm_list, scorer=fuzz.partial_ratio, score_cutoff=cutoff)
    return (res[2], float(res[1])) if res else None


def _has_map_keyword(t: str) -> bool:
    """지도 관련 키워드 존재 여부"""
    t = (t or "").lower()
    keywords = (
        "지도", "맵", "map", "어디", "위치", "구역",
        "북부", "남부", "동부", "서부", "섬", "지역", "대륙", "마을", "성"
    )
    return any(k in t for k in keywords)


async def _extract_world_region_from_text(text: str) -> Tuple[Optional[str], Optional[str], Dict]:
    """자연어에서 (world, region) 추출"""
    await _ensure_gazetteer()
    qn = _norm_kor(text)
    
    bw = _best_match_by_norm(qn, _map_cache["worlds_norm"], cutoff=85)
    br = _best_match_by_norm(qn, _map_cache["regions_norm"], cutoff=85)
    
    w = _map_cache["worlds"][bw[0]] if bw else None
    r = _map_cache["regions"][br[0]] if br else None
    
    info = {
        "world_score": bw[1] if bw else 0.0,
        "region_score": br[1] if br else 0.0,
        "via": "fuzzy",
        "had_keyword": _has_map_keyword(text),
    }
    
    # 키워드가 있으면 더 낮은 임계값으로 재시도
    if not (w or r) and info["had_keyword"]:
        bw2 = _best_match_by_norm(qn, _map_cache["worlds_norm"], cutoff=75)
        br2 = _best_match_by_norm(qn, _map_cache["regions_norm"], cutoff=75)
        
        if bw2 and not w:
            w = _map_cache["worlds"][bw2[0]]
            info["world_score"] = max(info["world_score"], bw2[1])
        if br2 and not r:
            r = _map_cache["regions"][br2[0]]
            info["region_score"] = max(info["region_score"], br2[1])
    
    return w, r, info


async def _find_map_doc(world: Optional[str], region: Optional[str]) -> Optional[Dict]:
    """maps_col에서 지도 문서 검색"""
    if not (world or region):
        return None
    
    filt = {}
    if world:
        filt["world.name"] = world
    if region:
        filt["region.name"] = region
    
    doc = await maps_col.find_one(filt, sort=[("created_at", -1)])
    if doc:
        return doc
    
    # 정규식 보완
    rx = {}
    if world:
        rx["world.name"] = {"$regex": re.escape(world), "$options": "i"}
    if region:
        rx["region.name"] = {"$regex": re.escape(region), "$options": "i"}
    
    if rx:
        return await maps_col.find_one(rx, sort=[("created_at", -1)])
    
    return None


def _map_payload_from_doc(doc: Dict, confidence: float, why: Dict) -> Dict:
    """지도 문서 → API 응답 페이로드"""
    gid = doc.get("image", {}).get("gridfs_id")
    gid = str(gid) if not isinstance(gid, dict) else gid.get("$oid") or ""
    did = str(doc.get("_id", ""))
    
    return {
        "type": "map",
        "doc_id": did,
        "world": (doc.get("world") or {}).get("name"),
        "region": (doc.get("region") or {}).get("name"),
        "image_gridfs_id": gid,
        "image_url": f"/api/maps/file/{gid}",
        "image_url_bydoc": f"/api/maps/bydoc/{did}",
        "source": doc.get("source") or {},
        "confidence": float(confidence),
        "reason": why,
    }


async def maybe_answer_with_map(question: str) -> Optional[Dict]:
    """
    지도 의도 감지 및 응답
    
    오발화 방지:
    - 명시 키워드 없으면 높은 매칭 점수 요구
    """
    w, r, why = await _extract_world_region_from_text(question)
    
    # 오발화 방지
    if not why["had_keyword"]:
        if not (why["world_score"] >= 90 and why["region_score"] >= 88):
            return None
    
    doc = await _find_map_doc(w, r)
    if not doc:
        return None
    
    conf = max(why.get("world_score", 0.0), why.get("region_score", 0.0))
    return _map_payload_from_doc(doc, conf, why)


# ============================================================================
# RAG QA
# ============================================================================

async def rag_qa(
    question: str,
    k: int = DEFAULT_K_DOCS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> str:
    """
    RAG 기반 질의응답
    
    개선사항:
    - OpenAI API 재시도 로직 (클라이언트 설정)
    - 타임아웃 설정
    - 에러 핸들링 강화
    - 상세한 답변 유도 프롬프트
    """
    ctx_docs = await semantic_search(question, limit=k)
    
    if not ctx_docs:
        return "죄송해요. 지금 가지고 있는 문서로는 이 질문에 대한 근거를 찾지 못했어요."
    
    # 스니펫 추출 및 압축 (문서당 5문장, 최대 4000자)
    snippets = get_snippets_from_docs(ctx_docs, question, per_doc_sentences=5)
    compressed = compress_context(snippets, max_chars=MAX_CONTEXT_CHARS)
    
    # 서버 측 판정: 낮은 유사도면 거절
    top_score = float(ctx_docs[0].get("score", 0.0))
    kw_docs = sum(1 for d in ctx_docs if d.get("keyword_match"))
    
    if (top_score < score_threshold) or (top_score < 0.45 and kw_docs == 0):
        return (
            "지금 보유한 문서(공지/가이드 등)에서 확실한 근거를 찾지 못했어요. \n"
            "또 다른 질문이 있다면 언제든지 환영입니다!"
        )
    
    # ✅ 간결한 프롬프트
    system_prompt = textwrap.dedent('''
    너는 로스트아크 전문 어시스턴트야. 질문에 정확하고 간결하게 답변해줘.
    
    **핵심 원칙:**
    1. 제공된 문서 정보만 사용 (추측 절대 금지)
    2. 질문에 직접 답하는 내용만 작성
    3. 2-3문장으로 간결하게
    4. 핵심 정보만 전달
    
    **금지 사항:**
    - 문서에 없는 내용 추측 금지
    - 장황한 배경 설명 금지
    - 불필요한 부연 설명 금지
    - "문서에 따르면" 같은 메타 표현 금지
    ''')
    
    user_prompt = textwrap.dedent(f'''
    [참고 자료]
    {compressed}

    [사용자 질문]
    {question}

    [지시사항]
    위 참고 자료를 바탕으로 질문에 핵심만 3-4문장으로 간결하게 답변해줘.
    문서에 있는 정보만 사용하고, 질문에 직접 답하는 내용만 포함해줘.
    ''')
    
    # 디버그 로그
    print("=" * 40)
    print(f"[DEBUG] Top score: {top_score:.3f}, KW docs: {kw_docs}")
    print(f"[DEBUG] Context length: {len(compressed)} chars")
    print(f"[DEBUG] Context preview:\n{compressed[:500]}...")
    print("=" * 40)
    
    # OpenAI API 호출 (재시도/타임아웃 자동 처리)
    try:
        chat_res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # 0.0 → 0.N: 약간의 창의성 허용하여 설명을 더 자연스럽게
            max_tokens=400   # 충분한 답변 길이 보장
        )
        return chat_res.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"[ERROR] OpenAI API failed: {e}")
        return "죄송해요. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


# ============================================================================
# 라우터
# ============================================================================

async def answer_router(question: str) -> Dict:
    """
    질의 라우팅
    - 지도 의도 → 지도 응답
    - 일반 질의 → RAG 응답
    """
    # 1) 지도 시도
    try:
        mp = await maybe_answer_with_map(question)
        if mp:
            return mp
    except Exception as e:
        print(f"[WARN] Map pipeline failed: {e}")
    
    # 2) 텍스트 RAG
    txt = await rag_qa(question)
    return {"type": "text", "text": txt}