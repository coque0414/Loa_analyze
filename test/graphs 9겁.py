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
            "name": "9레벨 겁화의 보석",
            "price": 897000
        },
        {
            "date": "2025-03-30",
            "name": "9레벨 겁화의 보석",
            "price": 897000
        },
        {
            "date": "2025-03-31",
            "name": "9레벨 겁화의 보석",
            "price": 874000
        },
        {
            "date": "2025-04-01",
            "name": "9레벨 겁화의 보석",
            "price": 875000
        },
        {
            "date": "2025-04-02",
            "name": "9레벨 겁화의 보석",
            "price": 873000
        },
        {
            "date": "2025-04-03",
            "name": "9레벨 겁화의 보석",
            "price": 874444
        },
        {
            "date": "2025-04-04",
            "name": "9레벨 겁화의 보석",
            "price": 870000
        },
        {
            "date": "2025-04-05",
            "name": "9레벨 겁화의 보석",
            "price": 860000
        },
        {
            "date": "2025-04-06",
            "name": "9레벨 겁화의 보석",
            "price": 873000
        },
        {
            "date": "2025-04-07",
            "name": "9레벨 겁화의 보석",
            "price": 855998
        },
        {
            "date": "2025-04-08",
            "name": "9레벨 겁화의 보석",
            "price": 825555
        },
        {
            "date": "2025-04-09",
            "name": "9레벨 겁화의 보석",
            "price": 818000
        },
        {
            "date": "2025-04-10",
            "name": "9레벨 겁화의 보석",
            "price": 817999
        },
        {
            "date": "2025-04-11",
            "name": "9레벨 겁화의 보석",
            "price": 810000
        },
        {
            "date": "2025-04-12",
            "name": "9레벨 겁화의 보석",
            "price": 810000
        },
        {
            "date": "2025-04-13",
            "name": "9레벨 겁화의 보석",
            "price": 810000
        },
        {
            "date": "2025-04-14",
            "name": "9레벨 겁화의 보석",
            "price": 788888
        },
        {
            "date": "2025-04-15",
            "name": "9레벨 겁화의 보석",
            "price": 760000
        },
        {
            "date": "2025-04-16",
            "name": "9레벨 겁화의 보석",
            "price": 789999
        },
        {
            "date": "2025-04-17",
            "name": "9레벨 겁화의 보석",
            "price": 785000
        },
        {
            "date": "2025-04-18",
            "name": "9레벨 겁화의 보석",
            "price": 788699
        },
        {
            "date": "2025-04-19",
            "name": "9레벨 겁화의 보석",
            "price": 806600
        },
        {
            "date": "2025-04-20",
            "name": "9레벨 겁화의 보석",
            "price": 803800
        },
        {
            "date": "2025-04-21",
            "name": "9레벨 겁화의 보석",
            "price": 815000
        },
        {
            "date": "2025-04-22",
            "name": "9레벨 겁화의 보석",
            "price": 820000
        },
        {
            "date": "2025-04-23",
            "name": "9레벨 겁화의 보석",
            "price": 835000
        },
        {
            "date": "2025-04-24",
            "name": "9레벨 겁화의 보석",
            "price": 824000
        },
        {
            "date": "2025-04-25",
            "name": "9레벨 겁화의 보석",
            "price": 825000
        },
        {
            "date": "2025-04-26",
            "name": "9레벨 겁화의 보석",
            "price": 879900
        },
        {
            "date": "2025-04-27",
            "name": "9레벨 겁화의 보석",
            "price": 877800
        },
        {
            "date": "2025-04-28",
            "name": "9레벨 겁화의 보석",
            "price": 869000
        },
        {
            "date": "2025-04-29",
            "name": "9레벨 겁화의 보석",
            "price": 868000
        },
        {
            "date": "2025-04-30",
            "name": "9레벨 겁화의 보석",
            "price": 870000
        },
        {
            "date": "2025-05-01",
            "name": "9레벨 겁화의 보석",
            "price": 719000
        },
        {
            "date": "2025-05-02",
            "name": "9레벨 겁화의 보석",
            "price": 720000
        },
        {
            "date": "2025-05-03",
            "name": "9레벨 겁화의 보석",
            "price": 754800
        },
        {
            "date": "2025-05-04",
            "name": "9레벨 겁화의 보석",
            "price": 722900
        },
        {
            "date": "2025-05-05",
            "name": "9레벨 겁화의 보석",
            "price": 720000
        },
        {
            "date": "2025-05-06",
            "name": "9레벨 겁화의 보석",
            "price": 700000
        },
        {
            "date": "2025-05-07",
            "name": "9레벨 겁화의 보석",
            "price": 675000
        },
        {
            "date": "2025-05-08",
            "name": "9레벨 겁화의 보석",
            "price": 700000
        },
        {
            "date": "2025-05-09",
            "name": "9레벨 겁화의 보석",
            "price": 733333
        },
        {
            "date": "2025-05-10",
            "name": "9레벨 겁화의 보석",
            "price": 755900
        },
        {
            "date": "2025-05-11",
            "name": "9레벨 겁화의 보석",
            "price": 770000
        },
        {
            "date": "2025-05-12",
            "name": "9레벨 겁화의 보석",
            "price": 750000
        },
        {
            "date": "2025-05-13",
            "name": "9레벨 겁화의 보석",
            "price": 757800
        },
        {
            "date": "2025-05-14",
            "name": "9레벨 겁화의 보석",
            "price": 725000
        },
        {
            "date": "2025-05-15",
            "name": "9레벨 겁화의 보석",
            "price": 725000
        },
        {
            "date": "2025-05-16",
            "name": "9레벨 겁화의 보석",
            "price": 705000
        },
        {
            "date": "2025-05-17",
            "name": "9레벨 겁화의 보석",
            "price": 738000
        },
        {
            "date": "2025-05-18",
            "name": "9레벨 겁화의 보석",
            "price": 728888
        },
        {
            "date": "2025-05-19",
            "name": "9레벨 겁화의 보석",
            "price": 719900
        },
        {
            "date": "2025-05-20",
            "name": "9레벨 겁화의 보석",
            "price": 729199
        },
        {
            "date": "2025-05-21",
            "name": "9레벨 겁화의 보석",
            "price": 703888
        },
        {
            "date": "2025-05-22",
            "name": "9레벨 겁화의 보석",
            "price": 688888
        },
        {
            "date": "2025-05-23",
            "name": "9레벨 겁화의 보석",
            "price": 699200
        },
        {
            "date": "2025-05-24",
            "name": "9레벨 겁화의 보석",
            "price": 690000
        },
        {
            "date": "2025-05-25",
            "name": "9레벨 겁화의 보석",
            "price": 657000
        },
        {
            "date": "2025-05-26",
            "name": "9레벨 겁화의 보석",
            "price": 657698
        },
        {
            "date": "2025-05-27",
            "name": "9레벨 겁화의 보석",
            "price": 617000
        },
        {
            "date": "2025-05-28",
            "name": "9레벨 겁화의 보석",
            "price": 602800
        },
        {
            "date": "2025-05-29",
            "name": "9레벨 겁화의 보석",
            "price": 588888
        },
        {
            "date": "2025-05-30",
            "name": "9레벨 겁화의 보석",
            "price": 555000
        },
        {
            "date": "2025-05-31",
            "name": "9레벨 겁화의 보석",
            "price": 597999
        },
        {
            "date": "2025-06-01",
            "name": "9레벨 겁화의 보석",
            "price": 622000
        },
        {
            "date": "2025-06-02",
            "name": "9레벨 겁화의 보석",
            "price": 674000
        },
        {
            "date": "2025-06-03",
            "name": "9레벨 겁화의 보석",
            "price": 609000
        },
        {
            "date": "2025-06-04",
            "name": "9레벨 겁화의 보석",
            "price": 610000
        },
        {
            "date": "2025-06-05",
            "name": "9레벨 겁화의 보석",
            "price": 645000
        },
        {
            "date": "2025-06-06",
            "name": "9레벨 겁화의 보석",
            "price": 655998
        },
        {
            "date": "2025-06-07",
            "name": "9레벨 겁화의 보석",
            "price": 664799
        },
        {
            "date": "2025-06-08",
            "name": "9레벨 겁화의 보석",
            "price": 654333
        },
        {
            "date": "2025-06-09",
            "name": "9레벨 겁화의 보석",
            "price": 650000
        },
        {
            "date": "2025-06-10",
            "name": "9레벨 겁화의 보석",
            "price": 647777
        },
        {
            "date": "2025-06-10",
            "name": "9레벨 겁화의 보석",
            "price": 654000
        }
    ]
}

# 3) 문서 리스트 생성
ITEM_CODE = 65031090
IMAGE_URL = "https://cdn-lostark.game.onstove.com/efui_iconatlas/use/use_12_104.png"

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