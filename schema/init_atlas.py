import json, os
from pymongo import MongoClient, ASCENDING, TEXT

MONGO_URI = os.environ.get("MONGO_URI")  # "mongodb+srv://user:pass@cluster/db?retryWrites=true&w=majority"
DB_NAME   = os.environ.get("DB_NAME", "loa_prod")

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return {"$jsonSchema": schema}

def create_collection_with_validator(db, name, schema):
    try:
        db.create_collection(name, validator=schema)
        print(f"Created collection: {name}")
    except Exception as e:
        if "already exists" in str(e):
            # 컬렉션 존재 시 validator 업데이트
            db.command('collMod', name, validator=schema)
            print(f"Updated validator: {name}")
        else:
            raise

def ensure_indexes(db):
    # glossary
    db.glossary.create_index([("type", ASCENDING)])
    db.glossary.create_index([("name_ko", ASCENDING)])
    db.glossary.create_index([("aliases", ASCENDING)])
    db.glossary.create_index([("updated_at", ASCENDING)])

    # docs: 하이브리드 검색 대비, 키워드+엔티티+시간
    db.docs.create_index([("source", ASCENDING), ("published_at", ASCENDING)])
    db.docs.create_index([("game_version", ASCENDING)])
    db.docs.create_index([("entity_refs", ASCENDING)])
    # 간단 텍스트 인덱스(초기): title, body
    db.docs.create_index([("title", TEXT), ("body", TEXT)], default_language="none")

    # recipes
    db.recipes.create_index([("valid_from", ASCENDING), ("valid_to", ASCENDING)])

    # prices_daily: item_id+date 복합키 성능
    db.prices_daily.create_index([("item_id", ASCENDING), ("date", ASCENDING)], unique=True)

def seed_minimum(db):
    # 필요시 간단 시드
    if db.glossary.count_documents({"_id": "item:65203305"}) == 0:
        db.glossary.insert_one({
            "_id": "item:65203305",
            "type": "item",
            "name_ko": "돌격대장 각인서",
            "aliases": ["돌대","돌장","돌격대장","돌격대장 각서"],
            "meta": {"tier": 3, "rarity": "전설", "tradable": True},
            "updated_at": "2025-09-23T00:00:00Z"
        })
        print("Seeded glossary: item:65203305")

if __name__ == "__main__":
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    glossary_schema = load_schema("./schema/glossary.schema.json")
    docs_schema     = load_schema("./schema/docs.schema.json")

    create_collection_with_validator(db, "glossary", glossary_schema)
    create_collection_with_validator(db, "docs", docs_schema)
    create_collection_with_validator(db, "recipes", {"$jsonSchema": {"bsonType":"object"}})
    create_collection_with_validator(db, "prices_daily", {"$jsonSchema": {"bsonType":"object"}})

    ensure_indexes(db)
    seed_minimum(db)
    print("Done.")
