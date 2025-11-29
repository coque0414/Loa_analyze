# services/db.py
import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import warnings

# .env 로드
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

if not MONGO_URI:
    warnings.warn("MONGODB_URI가 설정되어 있지 않습니다.")
if not DB_NAME:
    warnings.warn("DB_NAME이 설정되어 있지 않습니다.")


# ============================================================
# ✅ 핵심: event loop 체크 + 클라이언트 재생성
# ============================================================
_client = None
_client_loop = None  # 클라이언트가 생성된 event loop


def _get_client():
    global _client, _client_loop
    
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    
    # 클라이언트가 없거나, 다른 event loop에서 생성된 경우 재생성
    if _client is None or _client_loop is not current_loop:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
        _client = AsyncIOMotorClient(MONGO_URI)
        _client_loop = current_loop
    
    return _client


def _get_db():
    return _get_client()[DB_NAME]


# ============================================================
# ✅ 동적 컬렉션 접근 클래스
# ============================================================
class _DynamicCollection:
    def __init__(self, col_name: str):
        self._col_name = col_name
    
    @property
    def _col(self):
        return _get_db()[self._col_name]
    
    def find(self, *args, **kwargs):
        return self._col.find(*args, **kwargs)
    
    def find_one(self, *args, **kwargs):
        return self._col.find_one(*args, **kwargs)
    
    def insert_one(self, *args, **kwargs):
        return self._col.insert_one(*args, **kwargs)
    
    def insert_many(self, *args, **kwargs):
        return self._col.insert_many(*args, **kwargs)
    
    def update_one(self, *args, **kwargs):
        return self._col.update_one(*args, **kwargs)
    
    def update_many(self, *args, **kwargs):
        return self._col.update_many(*args, **kwargs)
    
    def delete_one(self, *args, **kwargs):
        return self._col.delete_one(*args, **kwargs)
    
    def delete_many(self, *args, **kwargs):
        return self._col.delete_many(*args, **kwargs)
    
    def aggregate(self, *args, **kwargs):
        return self._col.aggregate(*args, **kwargs)
    
    def count_documents(self, *args, **kwargs):
        return self._col.count_documents(*args, **kwargs)
    
    def distinct(self, *args, **kwargs):
        return self._col.distinct(*args, **kwargs)
    
    @property
    def database(self):
        return _get_db()


class _DynamicDB:
    def __getitem__(self, name: str):
        return _get_db()[name]
    
    def __getattr__(self, name: str):
        return getattr(_get_db(), name)


db = _DynamicDB()

market_col = _DynamicCollection("market_items")
jewelry_col = _DynamicCollection("jewelry_value")
summary_col = _DynamicCollection("daily_summary")
predictions_col = _DynamicCollection("predict_graphs")
posts_col = _DynamicCollection("community_posts")
docs_col = _DynamicCollection("docs_schema")
maps_col = _DynamicCollection("docs_map")
glossary_col = _DynamicCollection("docs_glossary")
guide_col = _DynamicCollection("docs_guide")
market_snapshots_col = _DynamicCollection("market_snapshots")


async def get_items(limit: int | None = None):
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
    except Exception as e:
        print(f"[ERROR] get_items failed: {e}")
        return []

    return items