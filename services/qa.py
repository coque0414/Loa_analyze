import textwrap
import numpy as np
import time
from openai import OpenAI
import os
import asyncio
import re

from rapidfuzz import fuzz, process  # pip install rapidfuzz
from bson import ObjectId

from services.db import posts_col, docs_col, maps_col
from services.embedder import get_embedder
from dotenv import load_dotenv
load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory cache to avoid reloading embeddings every request.
# This makes short-lived repeated queries much faster. TTL is configurable.
_emb_cache = {"embs": None, "docs": None, "loaded_at": 0}

# 지도 이름(gazetteer) 캐시
_map_cache = {
    "worlds": None,       # ["아르테미스", ...]
    "regions": None,      # ["레온하트", ...]
    "worlds_norm": None,  # 정규화 문자열 리스트
    "regions_norm": None,
    "loaded_at": 0,
}



def _normalize_doc(raw: dict, source: str) -> dict:
    title = raw.get("title") or raw.get("name") or raw.get("doc_name") or str(raw.get("_id", ""))
    # DB에 따라 다양한 본문 필드가 존재하므로 우선순위로 통합해서 text와 body 둘 다 채움
    text = raw.get("body") or raw.get("text") or raw.get("content") or ""
    emb = raw.get("embedding")
    # 추가: 원본 id, chunk index, content_type 보존
    doc_id = raw.get("_id") or raw.get("doc_id") or ""
    chunk_idx = raw.get("chunk_idx") if ("chunk_idx" in raw) else None
    content_type = raw.get("content_type") or None
    return {
        "title": title,
        "text": text,
        "body": text,  # downstream에서 body로 접근하는 경우 대비
        "embedding": np.asarray(emb, dtype=np.float32) if emb is not None else None,
        "source": source,
        "doc_id": doc_id,
        "chunk_idx": chunk_idx,
        "content_type": content_type
    }


# 추가: 동일 문서의 청크들(title에 "#숫자"가 붙은 경우)을 합치는 유틸
def merge_chunked_docs(docs_list):
    """
    입력: docs_list: [{title, text, embedding, source, ...}, ...]
    - title이 "notice: ... #1" 같은 형태면 base title을 추출하여 동일 base+source 그룹을 병합.
    - 텍스트는 청크 index 순으로 결합(\n\n 구분). embedding은 가능한 경우 평균.
    - score는 그룹 내 최대, keyword_match는 any 합산.
    반환: 병합된 docs 리스트
    """
    if not docs_list:
        return []

    groups = {}
    # 기존 title 기반 패턴(보조) 유지
    title_pat = re.compile(r'^(.*?)(?:\s*#\s*(\d+))\s*$', flags=re.IGNORECASE)
    for d in docs_list:
        title = (d.get("title") or "").strip()
        src = d.get("source", "unknown")
        doc_id = str(d.get("doc_id") or d.get("_id") or "")
        # 우선: doc_id 기반 그룹화(_id에 '#숫자' 포함 여부 확인)
        if doc_id:
            if "#" in doc_id:
                parts = doc_id.split("#")
                base = parts[0]
                try:
                    idx = int(parts[-1])
                except Exception:
                    # fallback to chunk_idx field
                    idx = d.get("chunk_idx")
            else:
                base = doc_id
                idx = d.get("chunk_idx")
        else:
            # doc_id가 없으면 title 기반으로 시도 (이전 동작)
            m = title_pat.match(title)
            if m:
                base = m.group(1).strip()
                try:
                    idx = int(m.group(2))
                except Exception:
                    idx = d.get("chunk_idx")
            else:
                base = title
                idx = d.get("chunk_idx")
        key = (base, src)
        groups.setdefault(key, []).append((idx, d))

    merged = []
    for (base, src), items in groups.items():
        if len(items) == 1:
            # 단일 항목: text/body 보장
            single = items[0][1]
            if single.get("text") is None and single.get("body"):
                single["text"] = single.get("body")
            if single.get("body") is None and single.get("text"):
                single["body"] = single.get("text")
            # 보장: doc_id 존재하면 유지, 없으면 base 사용
            single["doc_id"] = single.get("doc_id") or base
            merged.append(single)
            continue
        # 정렬: chunk_idx 우선, 없으면 idx None->앞 배치 처리
        sorted_items = sorted(items, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
        # 본문 병합: 각 청크에서 가능한 본문 필드를 우선순위로 취함
        texts = []
        for it in sorted_items:
            chunk = it[1]
            t = chunk.get("text") or chunk.get("body") or chunk.get("content") or chunk.get("doc_body") or ""
            texts.append(t)
        merged_text = "\n\n".join(t for t in texts if t)
        # embedding 평균 (가능한 경우만)
        embs = [it[1].get("embedding") for it in sorted_items if it[1].get("embedding") is not None]
        emb = None
        if embs:
            try:
                emb = np.mean(np.vstack(embs), axis=0).astype(np.float32)
            except Exception:
                emb = embs[0]
        # score / keyword_match 집계
        scores = [float(it[1].get("score", 0.0)) for it in sorted_items if it[1].get("score") is not None]
        keyword_any = any(bool(it[1].get("keyword_match", False)) for it in sorted_items)
        # doc_id는 base로 설정 (원본 _id의 # 이전 부분)
        merged_doc = {
            "title": base,
            "text": merged_text,
            "body": merged_text,  # downstream 호환성 확보
            "embedding": emb,
            "source": src,
            "doc_id": base,
            "score": max(scores) if scores else 0.0,
            "keyword_match": keyword_any
        }
        merged.append(merged_doc)
    return merged


async def semantic_search(question: str, limit: int = 5, max_docs: int = 10000, cache_ttl: int = 30):
    """Search semantically across `posts_col` and `docs_col`.
    Returns: list of docs with added 'score' float field (descending by score).
    """
    embedder = get_embedder()
    q_emb = embedder.encode(question, convert_to_numpy=True).astype(np.float32).reshape(1, -1)

    now = time.time()
    # refresh cache when empty or stale
    if _emb_cache["embs"] is None or (now - _emb_cache["loaded_at"] > cache_ttl):
        # posts_col 사용을 일시 비활성화: posts를 빈 리스트로 설정
        posts = []
        try:
            # 본문이 body 등에 들어있는 경우를 위해 가능한 본문 필드를 모두 요청
            docs = await docs_col.find(
                {"embedding": {"$exists": True}},
                {
                    "title": 1,
                    "text": 1,
                    "body": 1,
                    "content": 1,
                    "doc_body": 1,
                    "body_text": 1,
                    "embedding": 1,
                    "_id": 1
                }
            ).to_list(length=max_docs)
        except Exception:
            docs = []

        combined = []
        # posts 처리 루프는 건너뜁니다 (임시 비활성화)
        # for p in posts:
        #     try:
        #         combined.append(_normalize_doc(p, source="posts"))
        #     except Exception:
        #         continue
        for d in docs:
            try:
                combined.append(_normalize_doc(d, source="docs"))
            except Exception:
                continue

        if not combined:
            return []

        # --- 변경: 동일 문서의 청크들을 병합해서 하나의 문서로 합침 ---
        merged = merge_chunked_docs(combined)
        # 임베딩이 없는 항목은 제외
        merged_with_emb = [m for m in merged if m.get("embedding") is not None]
        if not merged_with_emb:
            return []
        embs = np.vstack([c["embedding"] for c in merged_with_emb]).astype(np.float32)
        # 캐시에 병합된 docs 저장
        _emb_cache["embs"] = embs
        _emb_cache["docs"] = merged_with_emb
        _emb_cache["loaded_at"] = now

    embs = _emb_cache["embs"]
    docs = _emb_cache["docs"]

    # Normalize once and compute dot product for cosine similarity
    try:
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sims = (q_norm @ embs_norm.T)[0]
    except Exception:
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(q_emb, embs)[0]

    # --- 추가: 키워드 직접 매칭 보정 (예: "낙원" 같은 단어가 문서에 있으면 score 보정) ---
    try:
        # 한글 포함 토큰 인식으로 변경
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", question.lower())
        boost = 0.20  # 보정값(필요하면 0.15~0.35로 조정)
        for idx, doc in enumerate(docs):
            title = (doc.get("title") or "").lower()
            text = (doc.get("text") or "").lower()
            # 부분 문자열 매칭으로 완화하고, 일치 토큰 수를 셈
            match_count = sum(1 for tok in tokens if tok and (tok in title or tok in text))
            if match_count > 0:
                # match_count에 따라 가중치를 주되 과다 보정 방지
                sims[idx] = min(1.0, sims[idx] + boost * min(match_count, 3))
                docs[idx]['keyword_match'] = True
            else:
                docs[idx]['keyword_match'] = False
    except Exception:
        # 보정 실패해도 기본 sims는 유지
        pass
    # --- 보정 끝 ---

    # docs_col 우대 가중치 적용 (docs 출처 문서의 score를 더 높임)
    try:
        # 환경변수로 조정하려면: float(os.getenv("DOCS_WEIGHT", "1.20"))
        docs_weight = float(os.getenv("DOCS_WEIGHT", "1.20"))  # 기본 1.20 (20% 우대)
        for idx, doc in enumerate(docs):
            if doc.get("source") == "docs":
                sims[idx] = min(1.0, sims[idx] * docs_weight)
    except Exception:
        # 실패해도 검색은 계속 작동
        pass

    idxs = sims.argsort()[::-1][:limit]
    results = []
    for i in idxs:
        d = dict(docs[i])  # copy
        d["score"] = float(sims[i])
        results.append(d)
    return results


def _extract_relevant_excerpt(text: str, question: str, max_sentences: int = 3, max_chars: int = 800):
    """
    간단한 발췌 함수:
    - 문서를 마침표/물음표/느낌표로 분리한 문장들 중 질문 키워드가 포함된 문장들을 모음.
    - 관련 문장이 없으면 문서 앞부분(요약용)을 반환.
    - 결과는 최대 max_chars로 자름.
    """
    if not text:
        return ""
    # 문장 분리(단순 방식). 한국어 특성상 완전하지 않음 — 필요하면 형태소 분석기 권장.
    sents = re.split(r'(?<=[\.\?!\n])\s+', text)
    # tokens 인식 정규식 수정
    tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", question.lower()))
    matches = []
    for s in sents:
        low = s.lower()
        if any(tok in low for tok in tokens if tok):
            matches.append(s.strip())
            if len(matches) >= max_sentences:
                break
    if not matches:
        # 관련 문장이 없으면 앞 문장 몇 개를 사용
        matches = [s.strip() for s in sents[:max_sentences] if s.strip()]
    excerpt = " ".join(matches)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars].rstrip() + "…(생략)"
    return excerpt

# 추가: 질문 기반 문장 추출 (Question -> Relevant Sentences)
def get_snippets_from_docs(ctx_docs, question: str, per_doc_sentences: int = 3):
    """
    각 문서(text)에서 질문과 관련도가 높은 문장들을 추출하여
    스니펫 리스트를 반환.
    각 스니펫은 dict: {text, title, source, doc_score, keyword_match, sent_score}
    """
    if not ctx_docs:
        return []

    # 한글 포함 토큰 인식으로 변경
    q_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", question.lower()))
    snippets = []
    for d in ctx_docs:
        # text 필드가 비어있을 수 있으므로 body/content 등도 고려
        text = d.get("text") or d.get("body") or d.get("content") or d.get("doc_body") or ""
        # 문장 분리
        sents = [s.strip() for s in re.split(r'(?<=[\.\?!\n])\s+', text) if s.strip()]
        # 문장별 점수: (토큰 일치 개수 / 문장 길이 가중치) + 문서 유사도 가중치 + 키워드 부스터
        for s in sents:
            low = s.lower()
            # 단어 경계 대신 부분 문자열 매칭으로 완화 (한글/조합어 인식 개선)
            match_count = sum(1 for t in q_tokens if t and (t in low))
            # 부분 문자열 매칭 보정(긴 문장 불리 방지)
            char_len = max(len(s), 1)
            # 문장 기반 점수: 토큰 매치 비율
            sent_score = (match_count / char_len) * 100.0
            # 보조: 문서 전체 score와 키워드 여부 반영
            doc_score = float(d.get("score", 0.0))
            kw = bool(d.get("keyword_match", False))
            # 최종 점수 조합
            final_score = sent_score + (doc_score * 50.0) + (15.0 if kw else 0.0)
            if match_count > 0:
                snippets.append({
                    "text": s,
                    "title": d.get("title", ""),
                    "source": d.get("source", "unknown"),
                    "doc_score": doc_score,
                    "keyword_match": kw,
                    "sent_score": final_score
                })
        # 관련 문장이 전혀 없으면 문서 앞부분을 후보로 추가
        # (문서 단위로 이미 스니펫이 존재하는지 정확히 검사)
        doc_key_title = d.get("title", "")
        doc_key_source = d.get("source", "unknown")
        has_snippet_for_doc = any(s["title"] == doc_key_title and s["source"] == doc_key_source for s in snippets)
        if not has_snippet_for_doc:
            preview = " ".join(sents[:per_doc_sentences]) if sents else ""
            if preview:
                snippets.append({
                    "text": preview,
                    "title": d.get("title", ""),
                    "source": d.get("source", "unknown"),
                    "doc_score": float(d.get("score", 0.0)),
                    "keyword_match": d.get("keyword_match", False),
                    "sent_score": 5.0 + float(d.get("score", 0.0)) * 10.0
                })
    # 문장별 상위 N(=per_doc_sentences) 제한: 문서별로 너무 많은 문장 추가 방지
    # group by (title, source) and keep top per_doc_sentences per doc
    grouped = {}
    for s in sorted(snippets, key=lambda x: x["sent_score"], reverse=True):
        key = (s["title"], s["source"])
        grouped.setdefault(key, []).append(s)
    final = []
    for key, items in grouped.items():
        final.extend(items[:per_doc_sentences])
    # 정렬된 최종 리스트 반환
    return sorted(final, key=lambda x: x["sent_score"], reverse=True)

# 추가: 컨텍스트 압축 로직 (extractive greedy)
def compress_context(snippets, max_chars: int = 2000):
    """
    snippets: get_snippets_from_docs 반환값 리스트
    최대 max_chars 까지 중요도 순으로 스니펫을 합쳐 반환.
    초과 시 마지막 스니펫을 잘라서 표시하고 '…(생략)' 추가.
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
            # 남은 공간에 맞춰 잘라서 추가
            remaining = max_chars - cur_len - 1
            if remaining > 20:
                trunc = part[:remaining].rstrip()
                # 안전하게 따옴표 닫힘 보장
                if trunc.count('"') % 2 == 1:
                    trunc = trunc.rstrip('"')
                out.append(trunc + "…(생략)")
            # 더 이상 추가 불가
            break
    return "\n".join(out)

def _norm_kor(s: str) -> str:
    """한글/영문/숫자만 남기고 소문자, 공백/특수문자 제거"""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", s)
    return s

async def _ensure_gazetteer(ttl_sec: int = 600):
    """maps_col에서 world/region 이름 전체를 distinct로 가져와 캐시"""
    now = time.time()
    if _map_cache["worlds"] is None or (now - _map_cache["loaded_at"] > ttl_sec):
        try:
            worlds = await maps_col.distinct("world.name")
            regions = await maps_col.distinct("region.name")
        except Exception:
            worlds, regions = [], []

        worlds = [w for w in worlds if isinstance(w, str) and w.strip()]
        regions = [r for r in regions if isinstance(r, str) and r.strip()]

        _map_cache["worlds"] = worlds
        _map_cache["regions"] = regions
        _map_cache["worlds_norm"] = [_norm_kor(w) for w in worlds]
        _map_cache["regions_norm"] = [_norm_kor(r) for r in regions]
        _map_cache["loaded_at"] = now

def _best_match_by_norm(q_norm: str, norm_list: list[str], cutoff: int = 85):
    """정규화된 후보(norm_list)에서 q_norm과 부분일치 최고 점수 후보 반환 (idx, score) 또는 None"""
    if not norm_list:
        return None
    res = process.extractOne(q_norm, norm_list, scorer=fuzz.partial_ratio, score_cutoff=cutoff)
    # res: (match_string, score, index) 형태
    return (res[2], float(res[1])) if res else None

def _has_map_keyword(t: str) -> bool:
    t = (t or "").lower()
    # '지도'가 없어도 위치/구역을 암시하는 단서들
    kw = ("지도", "맵", "map", "어디", "위치", "구역", "북부", "남부", "동부", "서부", "섬", "지역", "대륙", "마을", "성")
    return any(k in t for k in kw)

async def _extract_world_region_from_text(text: str) -> tuple[str|None, str|None, dict]:
    """자연어에서 (world, region) 후보 추출 + 근거 점수"""
    await _ensure_gazetteer()
    qn = _norm_kor(text)

    # world/region 각각 베스트 후보 뽑기
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

    # 지도 키워드가 있으면 허들을 조금 낮춰서 재시도 (예: 75)
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

async def _find_map_doc(world: str|None, region: str|None):
    """docs_map에서 최신 1건 찾기 (정확매칭 → 느슨한 정규식 순)"""
    if not (world or region):
        return None
    filt = {}
    if world:  filt["world.name"]  = world
    if region: filt["region.name"] = region
    doc = await maps_col.find_one(filt, sort=[("created_at", -1)])
    if doc:
        return doc

    # 느슨한 정규식 보완
    rx = {}
    if world:  rx["world.name"]  = {"$regex": re.escape(world),  "$options": "i"}
    if region: rx["region.name"] = {"$regex": re.escape(region), "$options": "i"}
    if rx:
        return await maps_col.find_one(rx, sort=[("created_at", -1)])
    return None

def _map_payload_from_doc(doc: dict, confidence: float, why: dict) -> dict:
    gid = doc.get("image", {}).get("gridfs_id")
    gid = str(gid) if not isinstance(gid, dict) else gid.get("$oid") or ""
    did = str(doc.get("_id", ""))

    return {
        "type": "map",
        "doc_id": did,
        "world": (doc.get("world") or {}).get("name"),
        "region": (doc.get("region") or {}).get("name"),
        "image_gridfs_id": gid,

        # ✅ 파일 ID 경로(가장 안전)
        "image_url": f"/api/maps/file/{gid}",

        # 보조
        "image_url_bydoc": f"/api/maps/bydoc/{did}",

        "source": doc.get("source") or {},
        "confidence": float(confidence),
        "reason": why,
    }

async def maybe_answer_with_map(question: str) -> dict|None:
    """
    자연어가 지도 의도/정보로 해석되면 docs_map에서 찾아서 payload 반환.
    아니면 None 반환.
    """
    w, r, why = await _extract_world_region_from_text(question)

    # 오발화 방지: 명시 키워드가 없으면 비교적 높은 매칭일 때만 지도 응답
    if not why["had_keyword"]:
        if not (why["world_score"] >= 90 and why["region_score"] >= 88):
            return None

    doc = await _find_map_doc(w, r)
    if not doc:
        return None

    # confidence = world/region 스코어 중 큰 값 (0~100)
    conf = max(why.get("world_score", 0.0), why.get("region_score", 0.0))
    return _map_payload_from_doc(doc, conf, why)

async def rag_qa(question: str, k: int = 5, score_threshold: float = 0.35):
    """
    score_threshold: 최고 유사도가 이 값 미만이면 '관련성 낮음'으로 처리.
    """
    ctx_docs = await semantic_search(question, limit=k)
    if not ctx_docs:
        return "죄송해요. 지금 가지고 있는 문서로는 이 질문에 대한 근거를 찾지 못했어요."

    # 기존 발췌 대신 질문 기반 스니펫을 먼저 추출하고, 전체 컨텍스트를 압축
    snippets = get_snippets_from_docs(ctx_docs, question, per_doc_sentences=3)
    # 압축된 컨텍스트 (모델 입력용)
    compressed = compress_context(snippets, max_chars=2000)

    # 발췌 추출: 각 문서에서 질문과 관련된 문장만 뽑아 컨텍스트로 제공
    excerpts = []
    for s in snippets:
        excerpts.append({
            "title": s["title"],
            "source": s["source"],
            "score": s["doc_score"],
            "keyword_match": s["keyword_match"],
            "excerpt": s["text"]
        })

    # 디버깅/확인용 컨텍스트 문자열 (모델에 전달할 내용) -> compressed 사용
    context = compressed

    # 서버 측 판정: 최고 유사도 및 키워드 매칭 기반 거절
    top_score = float(ctx_docs[0].get("score", 0.0))
    kw_docs = sum(1 for d in ctx_docs if d.get("keyword_match"))
    if (top_score < score_threshold) or (top_score < 0.45 and kw_docs == 0):
        return (
            "지금 보유한 문서(공지/가이드 등)에서 확실한 근거를 찾지 못했어요. \n"
            "또 다른 질문이 있다면 언제든지 환영입니다!"
        )

    # 시스템 메시지: 발췌(인용문)을 반드시 그대로 인용하고 그 문장 근거로 상세히 설명하도록 명시
    system_prompt = textwrap.dedent('''
    너는 로스트아크 관련 문서를 바탕으로 간결하고 친절하게 한국어로 답하는 조력자다.
    - 제공된 '발췌' 안에서만 사실을 사용한다(추측/확대해석 금지).
    - 숫자/날짜/조건은 정확히 보존한다.
    - 필요하면 1~3개의 짧은 불릿으로 구조화한다.
    - 파일명/점수/출처 목록 등 메타데이터는 출력하지 않는다.
    ''')

    user_prompt = textwrap.dedent(f'''
    [근거 발췌]
    {context}

    [질문]
    {question}

    [요청]
    위 근거만 사용해 한국어로 간단·명료하고 친절하게 답하세요.
    필요하면 1~3개의 짧은 불릿으로 정리하세요.
    근거 밖 정보는 넣지 마세요.
    ''')

    # 디버그: 모델 호출 직전 전달되는 컨텍스트 일부를 로깅
    print("="*40)
    print("[DEBUG] context 전달 내용:\n", context[:1000])
    print("="*40)

    chat_res = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    return chat_res.choices[0].message.content.strip()

async def answer_router(question: str) -> dict:
    """
    - 지도 의도/매칭 성공 → {"type":"map", ...}
    - 그 외 → {"type":"text", "text": "..."}  (rag_qa 결과)
    """
    # 1) 지도 먼저 시도
    try:
        mp = await maybe_answer_with_map(question)
        if mp:
            return mp
    except Exception as e:
        # 지도 해석 실패는 조용히 무시하고 텍스트 RAG로
        print("[WARN] map pipeline failed:", e)

    # 2) 텍스트 RAG
    txt = await rag_qa(question)
    return {"type": "text", "text": txt}

    
# async def rag_qa(question: str, k: int = 5) -> str:
#     ctx_docs = await semantic_search(question, limit=k)
#     if not ctx_docs:
#         return "⚠️ 관련 문서를 찾지 못했습니다."
#     context = "\n".join(f"- {d['title']}: {d['text']}" for d in ctx_docs)
#     prompt = textwrap.dedent(f'''
#     너는 로스트아크 커뮤니티 게시물만 참고해서 대답하는 어시스턴트야.
#     다음 게시물들과 관련 없는 질문이 들어오면 반드시 이렇게 말해:
#     "⚠️ 이 질문은 게시물 내용과 관련성이 낮아 답변할 수 없습니다."

#     [게시물 요약]
#     {context}

#     [질문]
#     {question}

#     [답변 - 한국어로 정확하게 설명하되, 친절하게:]
#     ''')
#     chat_res = openai.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2
#     )
#     return chat_res.choices[0].message.content.strip()