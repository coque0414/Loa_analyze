from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from services.db import glossary_col, market_snapshots_col
from services.loa_api import loa_markets_options, loa_markets_items, loa_market_item_by_code

import re
from typing import List, Dict, Any
import asyncio
import inspect

import io, base64
import matplotlib
matplotlib.use("Agg")  # 서버에서 GUI 없이 그리기
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# ✅ 한글 폰트 우선순위(설치된 첫 폰트 자동 선택)
rcParams["font.family"] = [
    "Malgun Gothic",          # Windows
    "Apple SD Gothic Neo",    # macOS
    "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
    "DejaVu Sans"             # 최후 폴백
]
rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지
rcParams["figure.facecolor"] = "white"
rcParams["savefig.facecolor"] = "white"

USE_SNAPSHOTS = False  # ✅ 스냅샷 저장 안 함 모드
CATEGORY_CODE_CACHE: dict[str, Optional[int]] = {}

KST = timezone(timedelta(hours=9))

INTENT_STOPWORDS = {"가격", "시세", "비교", "보여줘", "알려줘", "유각", "각인서", "얼마", "해주세요", "해줘"}

def _clean(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())

def _strip_stop(q: str) -> str:
    s = _clean(q)
    for w in INTENT_STOPWORDS:
        s = s.replace(w, "")
    return s.strip()

def _simple_tokens(q: str) -> List[str]:
    # 구분자 기준으로 잘라서 후보 토큰
    s = re.sub(r"(와|과|랑|하고|및|그리고|vs|VS|/|,)", " ", q)
    toks = [t for t in s.split() if t]
    # 중복 제거(순서 유지)
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


async def _find_glossary_by_token(tok: str):
    nq = "".join(tok.strip().lower().split())
    return await _maybe_await(glossary_col.find_one({
        "$or": [
            {"aliases_norm": {"$in": [nq]}},                      # 배열 내부 완전일치
            {"aliases_norm": {"$elemMatch": {"$regex": nq}}},     # 부분일치
            {"term_norm": nq},
            {"name_ko": {"$regex": tok}},
        ]
    }))

async def _scan_mentions(q: str, limit: int = 2) -> List[Dict[str,Any]]:
    """
    붙여쓴 질의(예: '결대아드') 대응: engraving만 훑으며 alias가 포함되는 문서를 찾아냄.
    Motor면 async-for, PyMongo면 for 로 자동 처리.
    """
    qn = _clean(q)
    hits: List[Dict[str,Any]] = []
    cur = glossary_col.find(
        {"type": "engraving"},
        {"_id": 1, "aliases": 1, "slug": 1, "name_ko": 1}
    )
    try:
        # Motor 커서
        async for d in cur:
            als = d.get("aliases") or []
            if any(a in qn for a in als):
                if not any(h.get("slug") == d.get("slug") for h in hits):
                    hits.append(d)
            if len(hits) >= limit:
                break
    except TypeError:
        # PyMongo 커서
        for d in cur:
            als = d.get("aliases") or []
            if any(a in qn for a in als):
                if not any(h.get("slug") == d.get("slug") for h in hits):
                    hits.append(d)
            if len(hits) >= limit:
                break
    return hits

async def resolve_glossary_pair(q: str) -> List[Dict[str,Any]]:
    """
    질의에서 최대 2개 품목을 glossary 문서로 추출
    """
    base = _strip_stop(q)
    toks = _simple_tokens(base)
    out: List[Dict[str,Any]] = []

    for t in toks:
        g = await _find_glossary_by_token(t)   # ✅ await 추가
        if g and not any(x.get("slug") == g.get("slug") for x in out):
            out.append(g)
        if len(out) >= 2:
            break

    if len(out) < 2:
        # 붙여쓴 케이스 스캔
        hits = await _scan_mentions(base, limit=2)  # ✅ await 추가
        for g in hits:
            if not any(x.get("slug") == g.get("slug") for x in out):
                out.append(g)
            if len(out) >= 2:
                break
    return out[:2]



async def _maybe_await(value):
    # 값이 awaitable이면 await 해서 결과를 반환, 아니면 그대로 반환
    if inspect.isawaitable(value):
        return await value
    return value

def _extract_item_code(gl: dict) -> int | None:
    # glossary._id 에서 engraving:65203505 형태의 숫자 코드만 추출
    sid = str(gl.get("_id", ""))
    m = re.search(r":(\d{5,})$", sid)
    return int(m.group(1)) if m else None

def _parse_ts(s: str):
    # "YYYY-MM-DD" 또는 ISO datetime을 KST로 맞춰 파싱
    if not s:
        return datetime.now(KST)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=KST)
        return dt.astimezone(KST)
    except Exception:
        return datetime.now(KST)


def _bulk_upsert_stats_as_snapshots(slug: str, name: str, stats: list[dict]):
    """ /markets/items/{code} 의 Stats 배열을 market_snapshots에 적재 """
    if not stats:
        return
    docs = []
    for st in stats:
        ts = _parse_ts(st.get("Date"))
        price = st.get("AvgPrice")
        cnt = st.get("TradeCount")  # 있을 때만
        if price is None:
            continue
        doc = {
            "slug": slug,
            "name": name,
            "price": float(price),
            "ts": ts,
        }
        if cnt is not None:
            try:
                doc["count"] = int(cnt)
            except Exception:
                pass
        docs.append(doc)
    if docs:
        market_snapshots_col.insert_many(docs, ordered=False)

async def _render_price_chart_data_uri(slug: str, days: int = 14) -> str | None:
    """market_snapshots에서 최근 N일 가격/거래량을 읽어 matplotlib로 PNG를 그려 data URI 반환"""
    end = datetime.now(KST)
    start = end - timedelta(days=max(1, days))

    # PyMongo(동기)와 Motor(비동기) 모두 호환
    cur = market_snapshots_col.find(
        {"slug": slug, "ts": {"$gte": start}},
        {"_id": 0, "ts": 1, "price": 1, "count": 1}
    ).sort("ts", 1)

    xs, ys, vols = [], [], []
    try:
        async for d in cur:  # Motor 커서
            xs.append(d["ts"]); ys.append(float(d["price"])); vols.append(d.get("count"))
    except TypeError:
        for d in cur:       # PyMongo 커서
            xs.append(d["ts"]); ys.append(float(d["price"])); vols.append(d.get("count"))

    if not xs:
        return None

    fig, ax1 = plt.subplots(figsize=(6.2, 2.6), dpi=160)
    ax1.plot(xs, ys, marker="o", linewidth=1.8)
    ax1.set_ylabel("가격(G)")
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m.%d"))
    fig.autofmt_xdate(rotation=0)

    if any(v is not None for v in vols):
        ax2 = ax1.twinx()
        ax2.bar(xs, [v or 0 for v in vols], alpha=0.45)
        ax2.set_ylabel("거래량")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return data_uri


def _norm(s: str) -> str:
    return "".join((s or "").lower().split())

INTENT_STOPWORDS = [
    # 가격/시세 의도
    "가격", "시세", "얼마", "최저가", "현재가", "평균가", "가격비교", "비교",
    # 요청어/조사
    "알려줘", "보여줘", "주세요", "검색", "조회", "좀", "요", "은", "는", "이", "가", "의",
    # 플랫폼/잡음
    "거래소", "마켓"
]

def _strip_intent_words(q: str) -> str:
    t = q or ""
    t = re.sub(r"[,\.\?\!]", " ", t)          # 구두점 제거
    for w in INTENT_STOPWORDS:
        t = t.replace(w, " ")
    t = re.sub(r"\s{2,}", " ", t).strip()     # 중복 공백 정리
    return t



# 1) glossary에서 질의 해석
async def resolve_glossary(q: str):
    # 의도어 제거 후 매칭
    cleaned = _strip_intent_words(q)
    nq = _norm(cleaned)

    # 1) 완전 일치
    doc = await _maybe_await(glossary_col.find_one({
        "$or": [
            {"term_norm": nq},
            {"aliases_norm": {"$in": [nq]}},  # ✅ 배열 내부 탐색
            {"aliases_norm": {"$elemMatch": {"$regex": nq}}},  # 부분 일치까지 허용
            {"aliases": {"$regex": nq}},  # 혹시 aliases만 있는 경우도 대비
        ]
    }))
    if doc:
        return doc

    # # 2) 느슨한 contains (백업)
    # doc = await _maybe_await(glossary_col.find_one({
    #     "$or": [
    #         {"term_norm": nq},
    #         {"aliases_norm": nq},
    #         {"aliases_norm": {"$elemMatch": {"$regex": nq}}}
    #     ]
    # }))
    # return doc

def _extract_code_from_node(node: dict) -> Optional[int]:
    """노드(dict)에서 코드로 쓸만한 키를 찾아 int로 리턴"""
    for key in ("Code", "CategoryCode", "Id", "CategoryId"):
        if key in node:
            try:
                return int(node[key])
            except Exception:
                pass
    return None

def _find_code_in_tree(node, category_name: str) -> Optional[int]:
    """node가 list/dict 무엇이든 전부 재귀 순회하며 Name 매칭되는 코드 탐색"""
    if isinstance(node, dict):
        name = str(node.get("Name", ""))
        if category_name in name:
            code = _extract_code_from_node(node)
            if code is not None:
                return code
        # 하위 값들 재귀
        for _, v in node.items():
            got = _find_code_in_tree(v, category_name)
            if got is not None:
                return got
    elif isinstance(node, list):
        for item in node:
            got = _find_code_in_tree(item, category_name)
            if got is not None:
                return got
    return None

async def _category_code(category_name: str) -> Optional[int]:
    """로아 마켓 옵션에서 카테고리 코드 조회 (비동기 + 메모리 캐시)."""
    if not category_name:
        return None
    if category_name in CATEGORY_CODE_CACHE:
        return CATEGORY_CODE_CACHE[category_name]

    data = await loa_markets_options()  # 어떤 형태가 와도 OK(리스트/딕셔너리)
    code = _find_code_in_tree(data, category_name)
    CATEGORY_CODE_CACHE[category_name] = code
    return code

# 3) 마켓 조회 + 스냅샷 저장
def _record_snapshot(slug: str, name: str, price: float|int, extra: Dict[str,Any]|None=None):
    doc = {
        "slug": slug,
        "name": name,
        "price": float(price) if price is not None else None,
        "ts": datetime.now(KST),
    }
    if extra: doc.update({"extra": extra})
    market_snapshots_col.insert_one(doc)

async def fetch_market_price(gl):
    code_from_id = _extract_item_code(gl)
    if code_from_id:
        raw = await loa_market_item_by_code(code_from_id)
        data = raw[1] if isinstance(raw, list) and len(raw) >= 2 else (raw or {})
        name  = data.get("Name") or gl.get("term_ko")
        stats = data.get("Stats") or []
        # ⛔ 저장 안 함
        if USE_SNAPSHOTS:
            _bulk_upsert_stats_as_snapshots(gl["slug"], name, stats)
        # 기본 현재가(히스토리 마지막 AvgPrice)
        cur_price = stats[-1]["AvgPrice"] if stats else None
        # …(가능하면 /markets/items 로 CurrentMinPrice 보강하는 부분은 그대로)
        return {"slug": gl["slug"], "name": name, "price": cur_price, "stats": stats, "raw": data}
    # 폴백 분기도 동일하게 반환에 "stats" 추가 or None

def _render_price_chart_from_stats(stats, days=14, show_volume=True, width=3.8, height=1.6):
    end = datetime.now(KST); start = end - timedelta(days=max(1, days))
    xs, ys, vols = [], [], []
    for st in stats:
        ts = _parse_ts(st.get("Date"))
        if ts < start:  # 최근 N일만
            continue
        avg = st.get("AvgPrice"); cnt = st.get("TradeCount")
        if avg is None: 
            continue
        xs.append(ts); ys.append(float(avg)); vols.append(cnt)
    if not xs:
        return None

    fig, ax1 = plt.subplots(figsize=(width, height), dpi=160)
    ax1.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax1.plot(xs, ys, linewidth=1.05)
    for sp in ("top","right"): ax1.spines[sp].set_visible(False)
    ax1.spines["left"].set_linewidth(0.8); ax1.spines["bottom"].set_linewidth(0.8)
    ax1.tick_params(axis="both", labelsize=8, pad=2); ax1.margins(x=0.02)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m.%d"))

    if show_volume and any(v is not None for v in vols):
        ax2 = ax1.twinx()
        ax2.bar(xs, [v or 0 for v in vols], alpha=0.30, color="#6b7280", linewidth=0)
        ax2.set_yticks([]); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5); fig.subplots_adjust(bottom=0.24)
    plt.figtext(0.02, 0.02, "파란선: 평균가(AvgPrice), 회색 막대: 거래량", fontsize=8, color="#6b7280")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# 4) 질의 1건 가격 응답
async def answer_market_price(query: str) -> Dict[str, Any]:
    gl = await resolve_glossary(query)
    if not gl or not (gl.get("resolver") or {}).get("market"):
        return {"type": "text", "answer": "어떤 품목인지 못 알아들었어요. 예) '원한 유각 가격' 처럼 물어봐 주세요."}

    row = await fetch_market_price(gl)
    if not row or row.get("price") is None:
        return {"type": "text", "answer": f"'{gl.get('term_ko')}' 거래소 검색 결과가 없습니다."}

    name  = row["name"]
    slug  = row["slug"]
    price = row["price"]
    stats = row.get("stats") or []  # ← 히스토리는 API의 Stats 배열을 그대로 사용

    # ✅ DB 스냅샷 없이 바로 그리기(거래량 막대 포함 + 미니 사이즈)
    img_data_uri = _render_price_chart_from_stats(stats, days=14, show_volume=True, width=3.8, height=1.6)

    html = None
    if img_data_uri:
        html = f"""
        <div style="max-width:360px">
          <div style="font-size:14px;margin:4px 0 8px 0">
            거래소에서의 <b>{name}</b> 가격을 불러왔어요.
          </div>
          <img src="{img_data_uri}" alt="{name} 최근 시세"
               style="width:100%;height:auto;border:1px solid #e5e7eb;border-radius:8px"/>
        </div>
        """.strip()

    return {
        "type": "price",
        "answer": f"{name} 현재 최저가 {price:,.0f}G 입니다.",
        "answer_html": html,
        "item": row
    }




# 5) 비교 질의 (쉼표/스페이스로 다중 항목)
def _split_terms(q: str) -> List[str]:
    raw = q.replace("및", ",").replace("그리고", ",")
    parts = []
    for token in raw.replace("  "," ").split(","):
        token = token.strip()
        if not token: continue
        parts.extend([t for t in token.split(" ") if t])
    # 너무 잘게 쪼개졌다면, 쉼표 기준으로만 쓰고 싶다면 위 로직을 줄이세요.
    # 여기서는 "원한 유각, 아드 유각" 같은 포맷 권장.
    uniq, seen = [], set()
    for p in parts:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

async def answer_market_compare(query: str) -> Dict[str,Any]:
    pairs = await resolve_glossary_pair(query)  # ✅ await
    if len(pairs) < 2:
        return {"type":"text","answer":"두 개 품목을 알아듣지 못했어요. 예) '원한, 결투의 대가 비교' 처럼 물어봐 주세요."}

    g1, g2 = pairs[0], pairs[1]
    r1 = await fetch_market_price(g1)
    r2 = await fetch_market_price(g2)
    if not r1 or not r2:
        return {"type":"text","answer":"두 품목 중 하나의 시세를 찾지 못했어요."}

    name1, name2 = r1["name"], r2["name"]
    price1, price2 = r1.get("price"), r2.get("price")
    stats1, stats2 = r1.get("stats") or [], r2.get("stats") or []

    img1 = _render_price_chart_from_stats(stats1, days=14, show_volume=True, width=3.6, height=1.5)
    img2 = _render_price_chart_from_stats(stats2, days=14, show_volume=True, width=3.6, height=1.5)

    html = f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;max-width:760px">
      <div style="flex:1 1 360px;max-width:360px">
        <div style="font-size:14px;margin:4px 0 8px 0"><b>{name1}</b></div>
        <img src="{img1}" alt="{name1} 최근 시세" style="width:100%;height:auto;border:1px solid #e5e7eb;border-radius:8px"/>
      </div>
      <div style="flex:1 1 360px;max-width:360px">
        <div style="font-size:14px;margin:4px 0 8px 0"><b>{name2}</b></div>
        <img src="{img2}" alt="{name2} 최근 시세" style="width:100%;height:auto;border:1px solid #e5e7eb;border-radius:8px"/>
      </div>
    </div>
    """.strip()

    summary = f"{name1} {price1:,.0f}G vs {name2} {price2:,.0f}G"
    return {
        "type":"price_compare",
        "answer": f"{summary} — 나란히 비교해서 확인해 보세요.",
        "answer_html": html,
        "items": [r1, r2],
    }



# 의도 판별(간단): '가격', '시세', '비교'
def is_market_intent(q: str) -> bool:
    t = _norm(q)
    return any(k in t for k in ["가격","시세","거래소","비교","얼마","최저가"])
