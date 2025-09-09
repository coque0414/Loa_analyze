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
            "name": "7레벨 작열의 보석",
            "price": 101996
        },
        {
            "date": "2025-03-30",
            "name": "7레벨 작열의 보석",
            "price": 101996
        },
        {
            "date": "2025-03-31",
            "name": "7레벨 작열의 보석",
            "price": 99999
        },
        {
            "date": "2025-04-01",
            "name": "7레벨 작열의 보석",
            "price": 98000
        },
        {
            "date": "2025-04-02",
            "name": "7레벨 작열의 보석",
            "price": 99600
        },
        {
            "date": "2025-04-03",
            "name": "7레벨 작열의 보석",
            "price": 99999
        },
        {
            "date": "2025-04-04",
            "name": "7레벨 작열의 보석",
            "price": 104000
        },
        {
            "date": "2025-04-05",
            "name": "7레벨 작열의 보석",
            "price": 99999
        },
        {
            "date": "2025-04-06",
            "name": "7레벨 작열의 보석",
            "price": 99555
        },
        {
            "date": "2025-04-07",
            "name": "7레벨 작열의 보석",
            "price": 100900
        },
        {
            "date": "2025-04-08",
            "name": "7레벨 작열의 보석",
            "price": 97400
        },
        {
            "date": "2025-04-09",
            "name": "7레벨 작열의 보석",
            "price": 92399
        },
        {
            "date": "2025-04-10",
            "name": "7레벨 작열의 보석",
            "price": 95000
        },
        {
            "date": "2025-04-11",
            "name": "7레벨 작열의 보석",
            "price": 97389
        },
        {
            "date": "2025-04-12",
            "name": "7레벨 작열의 보석",
            "price": 97000
        },
        {
            "date": "2025-04-13",
            "name": "7레벨 작열의 보석",
            "price": 95000
        },
        {
            "date": "2025-04-14",
            "name": "7레벨 작열의 보석",
            "price": 95490
        },
        {
            "date": "2025-04-15",
            "name": "7레벨 작열의 보석",
            "price": 92999
        },
        {
            "date": "2025-04-16",
            "name": "7레벨 작열의 보석",
            "price": 89000
        },
        {
            "date": "2025-04-17",
            "name": "7레벨 작열의 보석",
            "price": 92000
        },
        {
            "date": "2025-04-18",
            "name": "7레벨 작열의 보석",
            "price": 92000
        },
        {
            "date": "2025-04-19",
            "name": "7레벨 작열의 보석",
            "price": 93500
        },
        {
            "date": "2025-04-20",
            "name": "7레벨 작열의 보석",
            "price": 92899
        },
        {
            "date": "2025-04-21",
            "name": "7레벨 작열의 보석",
            "price": 96800
        },
        {
            "date": "2025-04-22",
            "name": "7레벨 작열의 보석",
            "price": 93000
        },
        {
            "date": "2025-04-23",
            "name": "7레벨 작열의 보석",
            "price": 93000
        },
        {
            "date": "2025-04-24",
            "name": "7레벨 작열의 보석",
            "price": 95000
        },
        {
            "date": "2025-04-25",
            "name": "7레벨 작열의 보석",
            "price": 99000
        },
        {
            "date": "2025-04-26",
            "name": "7레벨 작열의 보석",
            "price": 100000
        },
        {
            "date": "2025-04-27",
            "name": "7레벨 작열의 보석",
            "price": 104000
        },
        {
            "date": "2025-04-28",
            "name": "7레벨 작열의 보석",
            "price": 99900
        },
        {
            "date": "2025-04-29",
            "name": "7레벨 작열의 보석",
            "price": 97777
        },
        {
            "date": "2025-04-30",
            "name": "7레벨 작열의 보석",
            "price": 97999
        },
        {
            "date": "2025-05-01",
            "name": "7레벨 작열의 보석",
            "price": 84999
        },
        {
            "date": "2025-05-02",
            "name": "7레벨 작열의 보석",
            "price": 88000
        },
        {
            "date": "2025-05-03",
            "name": "7레벨 작열의 보석",
            "price": 92500
        },
        {
            "date": "2025-05-04",
            "name": "7레벨 작열의 보석",
            "price": 87500
        },
        {
            "date": "2025-05-05",
            "name": "7레벨 작열의 보석",
            "price": 88777
        },
        {
            "date": "2025-05-06",
            "name": "7레벨 작열의 보석",
            "price": 82500
        },
        {
            "date": "2025-05-07",
            "name": "7레벨 작열의 보석",
            "price": 82900
        },
        {
            "date": "2025-05-08",
            "name": "7레벨 작열의 보석",
            "price": 83843
        },
        {
            "date": "2025-05-09",
            "name": "7레벨 작열의 보석",
            "price": 86000
        },
        {
            "date": "2025-05-10",
            "name": "7레벨 작열의 보석",
            "price": 90400
        },
        {
            "date": "2025-05-11",
            "name": "7레벨 작열의 보석",
            "price": 95500
        },
        {
            "date": "2025-05-12",
            "name": "7레벨 작열의 보석",
            "price": 91600
        },
        {
            "date": "2025-05-13",
            "name": "7레벨 작열의 보석",
            "price": 87900
        },
        {
            "date": "2025-05-14",
            "name": "7레벨 작열의 보석",
            "price": 84400
        },
        {
            "date": "2025-05-15",
            "name": "7레벨 작열의 보석",
            "price": 87600
        },
        {
            "date": "2025-05-16",
            "name": "7레벨 작열의 보석",
            "price": 89300
        },
        {
            "date": "2025-05-17",
            "name": "7레벨 작열의 보석",
            "price": 87400
        },
        {
            "date": "2025-05-18",
            "name": "7레벨 작열의 보석",
            "price": 87399
        },
        {
            "date": "2025-05-19",
            "name": "7레벨 작열의 보석",
            "price": 85500
        },
        {
            "date": "2025-05-20",
            "name": "7레벨 작열의 보석",
            "price": 84400
        },
        {
            "date": "2025-05-21",
            "name": "7레벨 작열의 보석",
            "price": 80400
        },
        {
            "date": "2025-05-22",
            "name": "7레벨 작열의 보석",
            "price": 81000
        },
        {
            "date": "2025-05-23",
            "name": "7레벨 작열의 보석",
            "price": 85000
        },
        {
            "date": "2025-05-24",
            "name": "7레벨 작열의 보석",
            "price": 85900
        },
        {
            "date": "2025-05-25",
            "name": "7레벨 작열의 보석",
            "price": 75499
        },
        {
            "date": "2025-05-26",
            "name": "7레벨 작열의 보석",
            "price": 75000
        },
        {
            "date": "2025-05-27",
            "name": "7레벨 작열의 보석",
            "price": 75400
        },
        {
            "date": "2025-05-28",
            "name": "7레벨 작열의 보석",
            "price": 71250
        },
        {
            "date": "2025-05-29",
            "name": "7레벨 작열의 보석",
            "price": 73000
        },
        {
            "date": "2025-05-30",
            "name": "7레벨 작열의 보석",
            "price": 69400
        },
        {
            "date": "2025-05-31",
            "name": "7레벨 작열의 보석",
            "price": 70000
        },
        {
            "date": "2025-06-01",
            "name": "7레벨 작열의 보석",
            "price": 74800
        },
        {
            "date": "2025-06-02",
            "name": "7레벨 작열의 보석",
            "price": 77000
        },
        {
            "date": "2025-06-03",
            "name": "7레벨 작열의 보석",
            "price": 73988
        },
        {
            "date": "2025-06-04",
            "name": "7레벨 작열의 보석",
            "price": 73555
        },
        {
            "date": "2025-06-05",
            "name": "7레벨 작열의 보석",
            "price": 74000
        },
        {
            "date": "2025-06-06",
            "name": "7레벨 작열의 보석",
            "price": 79000
        },
        {
            "date": "2025-06-07",
            "name": "7레벨 작열의 보석",
            "price": 82800
        },
        {
            "date": "2025-06-08",
            "name": "7레벨 작열의 보석",
            "price": 77777
        },
        {
            "date": "2025-06-09",
            "name": "7레벨 작열의 보석",
            "price": 76490
        },
        {
            "date": "2025-06-10",
            "name": "7레벨 작열의 보석",
            "price": 74400
        },
        {
            "date": "2025-06-10",
            "name": "7레벨 작열의 보석",
            "price": 72444
        }
    ]
}

# 3) 문서 리스트 생성
ITEM_CODE = 65032070
IMAGE_URL = "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/use_12_112.png"

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