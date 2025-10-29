# skills/island.py
from datetime import datetime, timedelta, timezone
from typing import Literal, Tuple, List, Dict
from services.loa_api import get_calendar

KST = timezone(timedelta(hours=9))
_cache = {"data": None, "ts": 0.0, "ttl": 1800.0}  # 30분 TTL

# ✅ 의도: '모험섬' 질의 전반(골드 유무 상관없이) 트리거
def is_island_intent(q: str) -> bool:
    t = (q or "").lower().replace(" ", "")
    # 오늘/내일/이번주/이번달 등의 기간 키워드 + 모험/섬
    keys = ("모험섬","모험","섬","island")
    return any(k in t for k in keys)

def _has_gold(rewards: list) -> Tuple[bool, str]:
    """리워드 목록에 골드가 있는지, 표시용 문자열"""
    if not rewards: 
        return (False, "")
    names = [(ri.get("Name") or "") for ri in rewards if isinstance(ri, dict)]
    joined = ", ".join([n for n in names if n])
    low = joined.lower()
    hit = ("gold" in low) or ("골드" in low)
    return (hit, joined)

def _period_from_query(q: str) -> Literal["today","tomorrow","week","month"]:
    t = (q or "").replace(" ", "").lower()
    if "내일" in t: return "tomorrow"
    if ("달" in t) or ("월" in t): return "month"
    if ("주" in t) or ("주간" in t): return "week"
    return "today"  # 기본: 오늘

def _period_range(period: Literal["today","tomorrow","week","month"]) -> Tuple[datetime, datetime]:
    now = datetime.now(KST)
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == "tomorrow":
        start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == "week":
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
    else:  # month
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nextm = start.replace(year=start.year+1, month=1) if start.month==12 else start.replace(month=start.month+1)
        end = nextm
    return start, end

async def _get_calendar_cached():
    import time
    now_s = time.time()
    if _cache["data"] and (now_s - _cache["ts"] < _cache["ttl"]):
        return _cache["data"]
    data = await get_calendar()
    _cache.update({"data": data, "ts": now_s})
    return data

async def collect_next_island_slot_today():
    data = await _get_calendar_cached()
    now = datetime.now(KST)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    # 1) 오늘의 모든 시간 모으기
    times = set()
    for ev in data:
        if "모험" not in (ev.get("CategoryName") or ""):
            continue
        for ts in (ev.get("StartTimes") or []):
            dt = _parse_ts_kst(ts)
            if day_start <= dt < day_end:
                times.add(dt)

    # 2) 지금 이후로 가장 가까운 1개 선택
    future = sorted(t for t in times if t >= now)
    if not future:
        return None  # 오늘 남은 일정 없음
    chosen = future[0]
    dt_key = chosen.strftime("%Y-%m-%dT%H:%M:%S")

    # 3) 그 시간대에 뜨는 섬만 모으고, 보상은 해당 시간만 필터링
    islands = []
    for ev in data:
        if "모험" not in (ev.get("CategoryName") or ""):
            continue
        starts = ev.get("StartTimes") or []
        if dt_key not in starts:
            continue

        name = ev.get("ContentsName") or ev.get("Title") or "모험섬"
        icon_island = ev.get("ContentsIcon")
        r_items, r_names, has_gold = _rewards_for_time(ev.get("RewardItems") or [], dt_key)

        islands.append({
            "name": name,
            "icon": icon_island,
            "rewards": ", ".join(r_names) if r_names else "",
            "reward_items": r_items,          # 아이콘/이름 포함
            "has_gold": has_gold
        })

    islands.sort(key=lambda it: (not it["has_gold"], it["name"]))  # 골드 우선
    return {
        "when": chosen,
        "when_iso": chosen.isoformat(),
        "when_human": chosen.strftime("%m/%d(%a) %H:%M"),
        "islands": islands,
        "has_any_gold": any(it["has_gold"] for it in islands)
    }


async def answer_island_calendar(query: str):
    slot = await collect_next_island_slot_today()
    if not slot:
        return {
            "type": "island",
            "answer": "오늘 남은 모험섬 일정이 없어요. 내일 다시 확인해 주세요!",
            "period": "today",
            "items": []
        }

    # 카드 UI (한 타임만)
    style = """
    <style>
      .is-wrap{max-width:560px}
      .is-card{margin:12px 0;padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#ffffffcc;backdrop-filter:saturate(140%) blur(2px)}
      .is-time{font-weight:700;margin-bottom:8px;font-size:14px;color:#111827}
      .is-row{margin:8px 0}
      .is-name{font-weight:600;color:#111827;margin-left:6px}
      .is-icons img{width:20px;height:20px;border-radius:4px;margin-right:6px;vertical-align:-3px}
      .is-rewards{margin-top:6px}
    </style>
    """.strip()

    rows = []
    for it in slot["islands"]:
        main_tag = _main_reward_tag(it["rewards"])
        # 보상 아이콘(공식 CDN) + 툴팁
        icons_html = "".join(
            f'<img src="{ri.get("icon")}" title="{ri.get("name","")}" alt="{ri.get("name","")}" />'
            for ri in (it.get("reward_items") or []) if ri.get("icon")
        )
        chip_list = "".join(
            _chip(n.strip(), "#f3f4f6", "#374151", "11px")
            for n in (it["rewards"].split(",") if it["rewards"] else []) if n.strip()
        )
        row = (
            f'<div class="is-row">'
            f'  {main_tag}<span class="is-name">{it["name"]}</span>'
            f'  <div class="is-icons" style="margin-top:6px">{icons_html}</div>'
            f'  <div class="is-rewards">{chip_list or _chip("보상 정보 없음", "#f3f4f6", "#6b7280", "11px")}</div>'
            f'</div>'
        )
        rows.append(row)

    html = f'<div class="is-wrap">{style}<div class="is-card"><div class="is-time">{slot["when_human"]}</div>{"".join(rows)}</div></div>'

    head = "오늘의 모험섬 다음 일정을 알려드릴게요!"
    tail = "3개의 섬이 등장하니, 시간과 보상을 확인하고 입장해 주세요~"
    return {
        "type": "island",
        "answer": f"{head}\n{tail}",
        "period": "today",
        "items": [{
            "when": slot["when_iso"],
            "islands": slot["islands"],
            "has_any_gold": slot["has_any_gold"]
        }],
        "answer_html": html
    }



# --- UI 헬퍼: 칩/태그 스타일 및 분류 -------------------------

def _chip(text: str, bg: str = "#e5e7eb", fg: str = "#111827", size: str = "12px") -> str:
    return (
        f'<span style="display:inline-block;padding:2px 6px;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:{size};line-height:1;'
        f'margin-right:4px;margin-top:2px;">{text}</span>'
    )

ICONS = {
    "골드": "/static/icons/gold.png",
    "카드": "/static/icons/card.png",
    "주화": "/static/icons/coin.png",
    "실링": "/static/icons/silling.png",
    "재료": "/static/icons/material.png",
}

def _main_reward_tag(reward_str: str) -> str:
    s = reward_str or ""
    if "골드" in s: key="골드"
    elif "카드" in s: key="카드"
    elif "주화" in s: key="주화"
    elif "실링" in s: key="실링"
    else: key="재료"
    return f'<img src="{ICONS[key]}" alt="{key}" style="width:20px;height:20px;border-radius:4px;vertical-align:-3px;margin-right:6px">'



# 파일 상단 유틸 아래에 추가/교체
def _normalize_period(q: str):
    """
    API가 오늘만 제공하므로, 내일/주/월 요청은 today로 폴백.
    return: (period: 'today', note: str|None)
    """
    t = (q or "").replace(" ", "").lower()
    if any(k in t for k in ("내일", "주", "주간", "달", "월")):
        return "today", "공식 API는 당일 일정만 제공하여, 오늘 일정으로 안내드릴게요."
    return "today", None

def _is_next_intent(q: str) -> bool:
    t = (q or "").replace(" ", "").lower()
    return ("다음" in t) or ("지금" in t) or ("곧" in t)

# ISO 문자열 → KST aware datetime (Z/offset/naive 모두 대응)
def _parse_ts_kst(s: str) -> datetime:
    s = s.strip()
    # 'T' 뒤쪽에 타임존 기호가 있으면(+/-/Z) tz-aware 처리
    tail = s.split("T", 1)[1] if "T" in s else s
    if "Z" in tail or "+" in tail or "-" in tail[2:]:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST)
    # 타임존 표기가 없으면 KST 기준으로 본다
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=KST)

# 특정 시간(dt_key)에 해당하는 보상만 걸러내기
def _rewards_for_time(reward_groups: list, dt_key: str):
    items = []
    for grp in (reward_groups or []):
        for it in (grp.get("Items") or []):
            sts = it.get("StartTimes")
            if not sts or dt_key in sts:  # StartTimes 없으면 상시 보상으로 간주
                items.append({
                    "name": it.get("Name",""),
                    "icon": it.get("Icon"),
                    "grade": it.get("Grade","")
                })
    names = [i["name"] for i in items if i["name"]]
    has_gold = any(("골드" in n) or ("gold" in n.lower()) for n in names)
    return items, names, has_gold
