import requests
import json
from loa_api_token import Token
from pymongo import MongoClient, ASCENDING

# 1) 사용자 설정 부분 ---------------------------------
API_KEY       = Token
MONGODB_URI   = "mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLoa"
DB_NAME       = "lostark"           # 사용할 데이터베이스명
COLLECTION    = "market_items"         # 사용할 콜렉션명
# ITEM_CODE     = 65203305
ITEM_CODES  = [65201505, 65200805, 65203005,65203305,65203105,65200605,65203905,65201005,65200505,65202805,
               65204105,65203505,65203705]  # ← 원하는 아이템 코드를 여기에 추가
# ---------------------------------------------------

# MongoDB 연결 및 인덱스 (최초 1회만)
client     = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION]
collection.create_index(
    [("item_code", ASCENDING), ("date", ASCENDING)],
    unique=True,
    name="idx_item_date"
)

# 아이템 코드별 순회
for item_code in ITEM_CODES:
    url = f"https://developer-lostark.game.onstove.com/markets/items/{item_code}"
    headers = {
        "accept": "application/json",
        "authorization": API_KEY
    }

    try:
        response = requests.get(url, headers=headers)
        raw = response.json()

        if not isinstance(raw, list) or len(raw) < 2:
            print(f"[{item_code}] 잘못된 응답 형식 또는 데이터 없음")
            continue

        data = raw[1]  # 실제 데이터
        for stat in data["Stats"]:
            doc = {
                "item_code":   item_code,
                "name":        data["Name"],
                "date":        stat["Date"],
                "avg_price":   stat["AvgPrice"],
                "trade_count": stat["TradeCount"],
            }
            collection.update_one(
                {"item_code": item_code, "date": stat["Date"]},
                {"$set": doc},
                upsert=True
            )
        print(f"[{item_code}] 저장 완료")

    except Exception as e:
        print(f"[{item_code}] 오류 발생: {e}")
print('저장 끝!')