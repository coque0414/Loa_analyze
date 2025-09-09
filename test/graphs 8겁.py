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
            "name": "8레벨 겁화의 보석",
            "price": 290000
        },
        {
            "date": "2025-03-30",
            "name": "8레벨 겁화의 보석",
            "price": 290000
        },
        {
            "date": "2025-03-31",
            "name": "8레벨 겁화의 보석",
            "price": 293999
        },
        {
            "date": "2025-04-01",
            "name": "8레벨 겁화의 보석",
            "price": 285500
        },
        {
            "date": "2025-04-02",
            "name": "8레벨 겁화의 보석",
            "price": 287000
        },
        {
            "date": "2025-04-03",
            "name": "8레벨 겁화의 보석",
            "price": 283000
        },
        {
            "date": "2025-04-04",
            "name": "8레벨 겁화의 보석",
            "price": 289000
        },
        {
            "date": "2025-04-05",
            "name": "8레벨 겁화의 보석",
            "price": 288000
        },
        {
            "date": "2025-04-06",
            "name": "8레벨 겁화의 보석",
            "price": 283900
        },
        {
            "date": "2025-04-07",
            "name": "8레벨 겁화의 보석",
            "price": 280000
        },
        {
            "date": "2025-04-08",
            "name": "8레벨 겁화의 보석",
            "price": 272500
        },
        {
            "date": "2025-04-09",
            "name": "8레벨 겁화의 보석",
            "price": 268000
        },
        {
            "date": "2025-04-10",
            "name": "8레벨 겁화의 보석",
            "price": 269122
        },
        {
            "date": "2025-04-11",
            "name": "8레벨 겁화의 보석",
            "price": 277000
        },
        {
            "date": "2025-04-12",
            "name": "8레벨 겁화의 보석",
            "price": 271980
        },
        {
            "date": "2025-04-13",
            "name": "8레벨 겁화의 보석",
            "price": 260000
        },
        {
            "date": "2025-04-14",
            "name": "8레벨 겁화의 보석",
            "price": 258000
        },
        {
            "date": "2025-04-15",
            "name": "8레벨 겁화의 보석",
            "price": 257777
        },
        {
            "date": "2025-04-16",
            "name": "8레벨 겁화의 보석",
            "price": 260000
        },
        {
            "date": "2025-04-17",
            "name": "8레벨 겁화의 보석",
            "price": 261878
        },
        {
            "date": "2025-04-18",
            "name": "8레벨 겁화의 보석",
            "price": 261000
        },
        {
            "date": "2025-04-19",
            "name": "8레벨 겁화의 보석",
            "price": 267800
        },
        {
            "date": "2025-04-20",
            "name": "8레벨 겁화의 보석",
            "price": 265000
        },
        {
            "date": "2025-04-21",
            "name": "8레벨 겁화의 보석",
            "price": 268000
        },
        {
            "date": "2025-04-22",
            "name": "8레벨 겁화의 보석",
            "price": 270000
        },
        {
            "date": "2025-04-23",
            "name": "8레벨 겁화의 보석",
            "price": 272999
        },
        {
            "date": "2025-04-24",
            "name": "8레벨 겁화의 보석",
            "price": 273999
        },
        {
            "date": "2025-04-25",
            "name": "8레벨 겁화의 보석",
            "price": 274999
        },
        {
            "date": "2025-04-26",
            "name": "8레벨 겁화의 보석",
            "price": 288000
        },
        {
            "date": "2025-04-27",
            "name": "8레벨 겁화의 보석",
            "price": 289000
        },
        {
            "date": "2025-04-28",
            "name": "8레벨 겁화의 보석",
            "price": 290000
        },
        {
            "date": "2025-04-29",
            "name": "8레벨 겁화의 보석",
            "price": 287000
        },
        {
            "date": "2025-04-30",
            "name": "8레벨 겁화의 보석",
            "price": 284888
        },
        {
            "date": "2025-05-01",
            "name": "8레벨 겁화의 보석",
            "price": 238000
        },
        {
            "date": "2025-05-02",
            "name": "8레벨 겁화의 보석",
            "price": 247500
        },
        {
            "date": "2025-05-03",
            "name": "8레벨 겁화의 보석",
            "price": 250000
        },
        {
            "date": "2025-05-04",
            "name": "8레벨 겁화의 보석",
            "price": 242000
        },
        {
            "date": "2025-05-05",
            "name": "8레벨 겁화의 보석",
            "price": 239000
        },
        {
            "date": "2025-05-06",
            "name": "8레벨 겁화의 보석",
            "price": 231999
        },
        {
            "date": "2025-05-07",
            "name": "8레벨 겁화의 보석",
            "price": 226000
        },
        {
            "date": "2025-05-08",
            "name": "8레벨 겁화의 보석",
            "price": 235600
        },
        {
            "date": "2025-05-09",
            "name": "8레벨 겁화의 보석",
            "price": 235000
        },
        {
            "date": "2025-05-10",
            "name": "8레벨 겁화의 보석",
            "price": 248500
        },
        {
            "date": "2025-05-11",
            "name": "8레벨 겁화의 보석",
            "price": 254000
        },
        {
            "date": "2025-05-12",
            "name": "8레벨 겁화의 보석",
            "price": 253000
        },
        {
            "date": "2025-05-13",
            "name": "8레벨 겁화의 보석",
            "price": 248900
        },
        {
            "date": "2025-05-14",
            "name": "8레벨 겁화의 보석",
            "price": 240500
        },
        {
            "date": "2025-05-15",
            "name": "8레벨 겁화의 보석",
            "price": 239700
        },
        {
            "date": "2025-05-16",
            "name": "8레벨 겁화의 보석",
            "price": 238800
        },
        {
            "date": "2025-05-17",
            "name": "8레벨 겁화의 보석",
            "price": 244000
        },
        {
            "date": "2025-05-18",
            "name": "8레벨 겁화의 보석",
            "price": 239000
        },
        {
            "date": "2025-05-19",
            "name": "8레벨 겁화의 보석",
            "price": 238700
        },
        {
            "date": "2025-05-20",
            "name": "8레벨 겁화의 보석",
            "price": 235000
        },
        {
            "date": "2025-05-21",
            "name": "8레벨 겁화의 보석",
            "price": 233700
        },
        {
            "date": "2025-05-22",
            "name": "8레벨 겁화의 보석",
            "price": 233000
        },
        {
            "date": "2025-05-23",
            "name": "8레벨 겁화의 보석",
            "price": 231999
        },
        {
            "date": "2025-05-24",
            "name": "8레벨 겁화의 보석",
            "price": 231888
        },
        {
            "date": "2025-05-25",
            "name": "8레벨 겁화의 보석",
            "price": 219000
        },
        {
            "date": "2025-05-26",
            "name": "8레벨 겁화의 보석",
            "price": 218998
        },
        {
            "date": "2025-05-27",
            "name": "8레벨 겁화의 보석",
            "price": 211000
        },
        {
            "date": "2025-05-28",
            "name": "8레벨 겁화의 보석",
            "price": 202999
        },
        {
            "date": "2025-05-29",
            "name": "8레벨 겁화의 보석",
            "price": 196500
        },
        {
            "date": "2025-05-30",
            "name": "8레벨 겁화의 보석",
            "price": 187999
        },
        {
            "date": "2025-05-31",
            "name": "8레벨 겁화의 보석",
            "price": 196900
        },
        {
            "date": "2025-06-01",
            "name": "8레벨 겁화의 보석",
            "price": 204000
        },
        {
            "date": "2025-06-02",
            "name": "8레벨 겁화의 보석",
            "price": 222999
        },
        {
            "date": "2025-06-03",
            "name": "8레벨 겁화의 보석",
            "price": 200000
        },
        {
            "date": "2025-06-04",
            "name": "8레벨 겁화의 보석",
            "price": 205000
        },
        {
            "date": "2025-06-05",
            "name": "8레벨 겁화의 보석",
            "price": 215000
        },
        {
            "date": "2025-06-06",
            "name": "8레벨 겁화의 보석",
            "price": 218400
        },
        {
            "date": "2025-06-07",
            "name": "8레벨 겁화의 보석",
            "price": 217000
        },
        {
            "date": "2025-06-08",
            "name": "8레벨 겁화의 보석",
            "price": 215000
        },
        {
            "date": "2025-06-09",
            "name": "8레벨 겁화의 보석",
            "price": 216449
        },
        {
            "date": "2025-06-10",
            "name": "8레벨 겁화의 보석",
            "price": 214000
        },
        {
            "date": "2025-06-10",
            "name": "8레벨 겁화의 보석",
            "price": 213999
        }
    ]
}

# 3) 문서 리스트 생성
ITEM_CODE = 65031080
IMAGE_URL = "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/use_12_103.png"

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