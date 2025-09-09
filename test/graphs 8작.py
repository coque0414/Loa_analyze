from pymongo import MongoClient
from datetime import datetime

# 1) MongoDB 연결 설정
client = MongoClient("mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority")
db = client["lostark"]
collection = db["jewelry_value"]

# 2) 이미 파싱해 둔 JSON 데이터 예시
raw = {
    "success": True,
    "itemData": [
        {
            "date": "2025-03-29",
            "name": "8레벨 작열의 보석",
            "price": 289870
        },
        {
            "date": "2025-03-30",
            "name": "8레벨 작열의 보석",
            "price": 289870
        },
        {
            "date": "2025-03-31",
            "name": "8레벨 작열의 보석",
            "price": 288888
        },
        {
            "date": "2025-04-01",
            "name": "8레벨 작열의 보석",
            "price": 285500
        },
        {
            "date": "2025-04-02",
            "name": "8레벨 작열의 보석",
            "price": 284500
        },
        {
            "date": "2025-04-03",
            "name": "8레벨 작열의 보석",
            "price": 288000
        },
        {
            "date": "2025-04-04",
            "name": "8레벨 작열의 보석",
            "price": 289000
        },
        {
            "date": "2025-04-05",
            "name": "8레벨 작열의 보석",
            "price": 288000
        },
        {
            "date": "2025-04-06",
            "name": "8레벨 작열의 보석",
            "price": 286500
        },
        {
            "date": "2025-04-07",
            "name": "8레벨 작열의 보석",
            "price": 281000
        },
        {
            "date": "2025-04-08",
            "name": "8레벨 작열의 보석",
            "price": 274888
        },
        {
            "date": "2025-04-09",
            "name": "8레벨 작열의 보석",
            "price": 269900
        },
        {
            "date": "2025-04-10",
            "name": "8레벨 작열의 보석",
            "price": 269989
        },
        {
            "date": "2025-04-11",
            "name": "8레벨 작열의 보석",
            "price": 277000
        },
        {
            "date": "2025-04-12",
            "name": "8레벨 작열의 보석",
            "price": 272000
        },
        {
            "date": "2025-04-13",
            "name": "8레벨 작열의 보석",
            "price": 260000
        },
        {
            "date": "2025-04-14",
            "name": "8레벨 작열의 보석",
            "price": 262000
        },
        {
            "date": "2025-04-15",
            "name": "8레벨 작열의 보석",
            "price": 259888
        },
        {
            "date": "2025-04-16",
            "name": "8레벨 작열의 보석",
            "price": 264443
        },
        {
            "date": "2025-04-17",
            "name": "8레벨 작열의 보석",
            "price": 261333
        },
        {
            "date": "2025-04-18",
            "name": "8레벨 작열의 보석",
            "price": 262000
        },
        {
            "date": "2025-04-19",
            "name": "8레벨 작열의 보석",
            "price": 267000
        },
        {
            "date": "2025-04-20",
            "name": "8레벨 작열의 보석",
            "price": 266999
        },
        {
            "date": "2025-04-21",
            "name": "8레벨 작열의 보석",
            "price": 270000
        },
        {
            "date": "2025-04-22",
            "name": "8레벨 작열의 보석",
            "price": 272890
        },
        {
            "date": "2025-04-23",
            "name": "8레벨 작열의 보석",
            "price": 272997
        },
        {
            "date": "2025-04-24",
            "name": "8레벨 작열의 보석",
            "price": 275000
        },
        {
            "date": "2025-04-25",
            "name": "8레벨 작열의 보석",
            "price": 274899
        },
        {
            "date": "2025-04-26",
            "name": "8레벨 작열의 보석",
            "price": 288700
        },
        {
            "date": "2025-04-27",
            "name": "8레벨 작열의 보석",
            "price": 291000
        },
        {
            "date": "2025-04-28",
            "name": "8레벨 작열의 보석",
            "price": 285000
        },
        {
            "date": "2025-04-29",
            "name": "8레벨 작열의 보석",
            "price": 287499
        },
        {
            "date": "2025-04-30",
            "name": "8레벨 작열의 보석",
            "price": 284444
        },
        {
            "date": "2025-05-01",
            "name": "8레벨 작열의 보석",
            "price": 238000
        },
        {
            "date": "2025-05-02",
            "name": "8레벨 작열의 보석",
            "price": 248888
        },
        {
            "date": "2025-05-03",
            "name": "8레벨 작열의 보석",
            "price": 250000
        },
        {
            "date": "2025-05-04",
            "name": "8레벨 작열의 보석",
            "price": 242000
        },
        {
            "date": "2025-05-05",
            "name": "8레벨 작열의 보석",
            "price": 240000
        },
        {
            "date": "2025-05-06",
            "name": "8레벨 작열의 보석",
            "price": 234993
        },
        {
            "date": "2025-05-07",
            "name": "8레벨 작열의 보석",
            "price": 227900
        },
        {
            "date": "2025-05-08",
            "name": "8레벨 작열의 보석",
            "price": 234990
        },
        {
            "date": "2025-05-09",
            "name": "8레벨 작열의 보석",
            "price": 239500
        },
        {
            "date": "2025-05-10",
            "name": "8레벨 작열의 보석",
            "price": 249700
        },
        {
            "date": "2025-05-11",
            "name": "8레벨 작열의 보석",
            "price": 255000
        },
        {
            "date": "2025-05-12",
            "name": "8레벨 작열의 보석",
            "price": 255000
        },
        {
            "date": "2025-05-13",
            "name": "8레벨 작열의 보석",
            "price": 247000
        },
        {
            "date": "2025-05-14",
            "name": "8레벨 작열의 보석",
            "price": 241000
        },
        {
            "date": "2025-05-15",
            "name": "8레벨 작열의 보석",
            "price": 241999
        },
        {
            "date": "2025-05-16",
            "name": "8레벨 작열의 보석",
            "price": 241800
        },
        {
            "date": "2025-05-17",
            "name": "8레벨 작열의 보석",
            "price": 242000
        },
        {
            "date": "2025-05-18",
            "name": "8레벨 작열의 보석",
            "price": 239000
        },
        {
            "date": "2025-05-19",
            "name": "8레벨 작열의 보석",
            "price": 238690
        },
        {
            "date": "2025-05-20",
            "name": "8레벨 작열의 보석",
            "price": 235000
        },
        {
            "date": "2025-05-21",
            "name": "8레벨 작열의 보석",
            "price": 233000
        },
        {
            "date": "2025-05-22",
            "name": "8레벨 작열의 보석",
            "price": 236500
        },
        {
            "date": "2025-05-23",
            "name": "8레벨 작열의 보석",
            "price": 234987
        },
        {
            "date": "2025-05-24",
            "name": "8레벨 작열의 보석",
            "price": 231999
        },
        {
            "date": "2025-05-25",
            "name": "8레벨 작열의 보석",
            "price": 222222
        },
        {
            "date": "2025-05-26",
            "name": "8레벨 작열의 보석",
            "price": 219999
        },
        {
            "date": "2025-05-27",
            "name": "8레벨 작열의 보석",
            "price": 210000
        },
        {
            "date": "2025-05-28",
            "name": "8레벨 작열의 보석",
            "price": 203500
        },
        {
            "date": "2025-05-29",
            "name": "8레벨 작열의 보석",
            "price": 197000
        },
        {
            "date": "2025-05-30",
            "name": "8레벨 작열의 보석",
            "price": 187000
        },
        {
            "date": "2025-05-31",
            "name": "8레벨 작열의 보석",
            "price": 197555
        },
        {
            "date": "2025-06-01",
            "name": "8레벨 작열의 보석",
            "price": 207000
        },
        {
            "date": "2025-06-02",
            "name": "8레벨 작열의 보석",
            "price": 226000
        },
        {
            "date": "2025-06-03",
            "name": "8레벨 작열의 보석",
            "price": 203474
        },
        {
            "date": "2025-06-04",
            "name": "8레벨 작열의 보석",
            "price": 208000
        },
        {
            "date": "2025-06-05",
            "name": "8레벨 작열의 보석",
            "price": 218000
        },
        {
            "date": "2025-06-06",
            "name": "8레벨 작열의 보석",
            "price": 220000
        },
        {
            "date": "2025-06-07",
            "name": "8레벨 작열의 보석",
            "price": 218799
        },
        {
            "date": "2025-06-08",
            "name": "8레벨 작열의 보석",
            "price": 217000
        },
        {
            "date": "2025-06-09",
            "name": "8레벨 작열의 보석",
            "price": 216800
        },
        {
            "date": "2025-06-10",
            "name": "8레벨 작열의 보석",
            "price": 215000
        },
        {
            "date": "2025-06-10",
            "name": "8레벨 작열의 보석",
            "price": 216000
        }
    ]
}

# 3) 문서 리스트 생성
ITEM_CODE = 65032080
IMAGE_URL = "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/use_12_113.png"

docs = []
for item in raw["itemData"]:
    # 날짜 문자열을 datetime 객체로 변환(선택)
    date_obj = datetime.strptime(item["date"], "%Y-%m-%d")
    
    doc = {
        "date":       date_obj,         # 또는 그냥 item["date"] (문자열) 그대로 사용
        "name":       item["name"],
        "price":      item["price"],
        "item_code":  ITEM_CODE,
        "image_url":  IMAGE_URL
    }
    docs.append(doc)

for doc in docs:
    collection.update_one(
        {"item_code": doc["item_code"], "date": doc["date"]},
        {"$set": doc},
        upsert=True
    )