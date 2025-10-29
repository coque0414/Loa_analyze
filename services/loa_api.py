# services/loa_api.py
import os, httpx
from dotenv import load_dotenv

load_dotenv()  # .env 로드

BASE = "https://developer-lostark.game.onstove.com"
# JWT = os.getenv("LOA_API_KEY")

def _get_key() -> str:
    k = os.getenv("LOA_API_KEY", "")
    # .env에서 따옴표/공백이 섞인 경우 제거
    k = k.strip().strip('"').strip("'")
    if not k:
        raise RuntimeError("LOA_API_KEY is not set")
    return k

def _headers():
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {_get_key()}",   # ← 대문자 Authorization / Bearer
    }

async def get_calendar():
    async with httpx.AsyncClient(base_url=BASE, headers=_headers(), timeout=20) as cli:
        r = await cli.get("/gamecontents/calendar")
        # 실패시 응답 바디까지 보이게
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"calendar HTTP {r.status_code}: {r.text[:300]}") from e
        return r.json()

#여서부터는 마켓api(즉 거래소)
def _jwt():
    k = os.getenv("LOA_API_KEY","").strip().strip('"').strip("'")
    if not k: raise RuntimeError("LOA_API_KEY missing")
    return k
    
def _hdr():
    return {"Accept":"application/json","Authorization":f"Bearer {_jwt()}"}

async def loa_markets_options():
    async with httpx.AsyncClient(base_url=BASE, headers=_hdr(), timeout=20) as cli:
        r = await cli.get("/markets/options"); r.raise_for_status(); return r.json()

async def loa_markets_items(item_name: str, category_code: int|None):
    payload = {"ItemName": item_name, "CategoryCode": category_code}
    async with httpx.AsyncClient(base_url=BASE, headers=_hdr(), timeout=20) as cli:
        r = await cli.post("/markets/items", json=payload); r.raise_for_status(); return r.json()

async def loa_market_item_by_code(item_code: int | str):
    """거래소 아이템 단건 상세 (시세 히스토리 포함)
    응답 예: [ {...option...}, { "Name": "...", "Stats": [ {"Date": "...", "AvgPrice": ...}, ... ] } ]
    """
    async with httpx.AsyncClient(base_url=BASE, headers=_hdr(), timeout=20) as cli:
        r = await cli.get(f"/markets/items/{item_code}")
        r.raise_for_status()
        return r.json()