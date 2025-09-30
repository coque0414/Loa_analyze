import os, re, hashlib, datetime as dt
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ====== 설정 ======
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME   = os.environ.get("DB_NAME")
COLLECTION_NAME = "docs_schema"

# 모델 준비 (한국어 멀티태스크 추천)
model = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")

# ====== 유틸 ======
def now_utc():
    return dt.datetime.utcnow().isoformat() + "Z"

def hash_body(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

def embed(text: str):
    return model.encode([text], normalize_embeddings=True)[0].tolist()

def chunk_text(text, max_len=600):
    """
    공지 본문을 청크 단위로 나눔 (기본 400~800자 사이 권장)
    여기서는 간단히 줄바꿈 기준으로 split 후 합치기
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) > max_len:
            chunks.append(buf.strip())
            buf = line
        else:
            buf += " " + line
    if buf: chunks.append(buf.strip())
    return chunks

# ====== Mongo ======
client = MongoClient(MONGO_URI)
db = client[DB_NAME][COLLECTION_NAME]

# ====== HTML 파서 ======
def parse_notice(url: str):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    # 제목
    title = soup.select_one("h4.news-title").get_text(strip=True)

    # 날짜 (페이지 구조에 따라 조정 필요)
    date_el = soup.select_one("span.date")
    published_at = None
    if date_el:
        # "2025.09.25 05:00" → ISO 형식으로 변환
        dtext = date_el.get_text(strip=True).replace(".", "-")
        published_at = dt.datetime.strptime(dtext, "%Y-%m-%d %H:%M").isoformat() + "Z"

    # 본문
    body_el = soup.select_one("div.fr-view")
    body_text = body_el.get_text("\n", strip=True) if body_el else ""

    return {
        "title": title,
        "published_at": published_at,
        "body_text": body_text
    }

# ====== 저장 프로세스 ======
def ingest_notice(url: str, game_version: str = None):
    data = parse_notice(url)
    title = data["title"]
    published_at = data["published_at"]
    body_text = data["body_text"]

    chunks = chunk_text(body_text)

    for idx, chunk in enumerate(chunks, start=1):
        doc_id = f"notice:{url.split('/')[-1]}#{idx}"
        doc = {
            "_id": doc_id,
            "source": "official",
            "content_type": "notice",
            "title": title,
            "body": chunk,
            "lang": "ko",
            "url": url,
            "author": "LOA_Official",
            "published_at": published_at,
            "game_version": game_version or published_at[:10],
            "entity_refs": [],   # 필요시 아이템/각인 매핑
            "tags": ["공지"],
            "chunk_idx": idx,
            "ingested_at": now_utc(),
            "hash": hash_body(chunk),
        }

        # === 임베딩 추가 ===
        doc["embedding"] = embed(chunk)
        doc["embedding_model"] = "BM-K/KoSimCSE-roberta-multitask"
        doc["embedding_at"] = now_utc()

        # MongoDB에 upsert
        db.docs.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
        print(f"[OK] Upserted {doc_id}")

# ====== 실행 예시 ======
if __name__ == "__main__":
    url = "https://lostark.game.onstove.com/News/Notice/Views/13247"
    ingest_notice(url)
