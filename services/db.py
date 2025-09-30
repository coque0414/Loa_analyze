import os
from dotenv import load_dotenv, find_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import warnings

# .env 자동 탐색 및 로드 (프로젝트 루트나 상위 디렉터리의 .env를 찾아 로드)
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
else:
    # find_dotenv 못 찾으면 기본 동작(환경변수 또는 시스템 환경에서 읽기)
    load_dotenv()

# 환경변수 우선
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

# 설정 누락 시 경고
if ("MONGODB_URI" not in os.environ) and ("MONGO_URI" not in os.environ):
    warnings.warn("환경변수 MONGODB_URI 또는 MONGO_URI가 설정되어 있지 않습니다. 로컬 기본값으로 연결합니다.")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

market_col = db["market_items"]
jewelry_col = db["jewelry_value"]
summary_col = db["daily_summary"]
predictions_col = db["predict_graphs"]
posts_col = db["community_posts"]

async def get_items(limit: int | None = None):
    """
    market_col에서 (item_code, name, image_url) 목록을 가져옵니다.
    - 기본: 모든 아이템을 item_code 오름차순으로 반환
    - limit 지정 가능
    - DB가 연결되어 있지 않으면 빈 리스트 반환
    """
    if 'market_col' not in globals() or market_col is None:
        return []

    pipeline = [
        {"$group": {"_id": "$item_code", "name": {"$first": "$name"}, "img": {"$first": "$image_url"}}},
        {"$sort": {"_id": 1}},
    ]
    if limit is not None:
        pipeline.append({"$limit": int(limit)})

    items = []
    try:
        cursor = market_col.aggregate(pipeline)
        async for doc in cursor:
            img = doc.get("img") or "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/use_9_25.png"
            items.append({"code": doc["_id"], "name": doc.get("name") or "Item", "img": img})
    except Exception:
        # 안전하게 빈 리스트 반환 (로그가 필요하면 print/로거 추가)
        return []

    return items