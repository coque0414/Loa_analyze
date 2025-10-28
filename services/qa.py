import textwrap
import numpy as np
import time
from openai import OpenAI
import os
import asyncio
import re

from services.db import posts_col, docs_col
from services.embedder import get_embedder
from dotenv import load_dotenv
load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory cache to avoid reloading embeddings every request.
# This makes short-lived repeated queries much faster. TTL is configurable.
_emb_cache = {"embs": None, "docs": None, "loaded_at": 0}


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
        part = f'- {s["title"]} ({s["source"]}, score={s["doc_score"]:.3f}, kw={s["keyword_match"]}): "{s["text"]}"'
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

async def rag_qa(question: str, k: int = 5, score_threshold: float = 0.20) -> str:
    """
    score_threshold: 최고 유사도가 이 값 미만이면 '관련성 낮음'으로 처리.
    """
    ctx_docs = await semantic_search(question, limit=k)
    if not ctx_docs:
        return "⚠️ 관련 문서를 찾지 못했습니다."

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

    # 서버 측 판정: 최고 유사도가 낮으면 거부
    top_score = ctx_docs[0].get("score", 0.0)
    if (top_score < score_threshold):
        return "⚠️ 이 질문은 게시물 또는 문서 내용과 관련성이 낮아 답변할 수 없습니다."

    # 시스템 메시지: 발췌(인용문)을 반드시 그대로 인용하고 그 문장 근거로 상세히 설명하도록 명시
    system_prompt = textwrap.dedent('''
    너는 로스트아크 공식 공지와 커뮤니티 문서에 기반해 답변하는 어시스턴트야.
    반드시 다음을 지켜라:
    1) 아래에 제공된 각 문서의 '발췌(인용문)'을 그대로 따옴표로 인용하고, 그 인용문을 근거로 질문에 대해 구체적으로 설명하라.
    2) 인용문에 나온 구체적 내용(숫자/버전/날짜 등)을 그대로 재현하되, 불확실한 내용은 '※ 불확실'로 표기하라.
    3) 문서 본문 전체가 아닌 제공된 발췌만을 사용해 답변하라(추측 금지).
    4) 답변 마지막에 사용한 문서 출처(_id, title, source)를 반드시 표시하라.
    ''')

    user_prompt = textwrap.dedent(f'''
    [검색된 문서 발췌(인용) - 압축된 컨텍스트]
    {context}

    [질문]
    {question}

    [요청]
    위 발췌들만 참고해 한국어로 답변하고, 인용된 문장을 반드시 따옴표로 표시하세요. 마지막에 사용한 문서 출처를 목록으로 표시하세요.
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