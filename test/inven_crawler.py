import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# 환경 변수 로딩 (.env에 MONGO_URI 포함)
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB 접속
client = MongoClient(MONGO_URI)
db = client["loa_project"]
collection = db["inven_posts"]

# 인벤 게시판 URL (예: 로스트아크 공략 게시판)
BASE_URL = "https://www.inven.co.kr/board/lostark/4821"

def crawl_page(page_num):
    """
    인벤 게시판에서 특정 페이지 크롤링
    """
    url = f"{BASE_URL}?p={page_num}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, "html.parser")
    rows = soup.select("div.board_list tbody tr")

    posts = []

    for row in rows:
        title_tag = row.select_one("td.tit a.subject-link")
        date_tag = row.select_one("td.date")
        if not title_tag or not date_tag:
            continue

        title = title_tag.get_text(strip=True)
        href = "https:" + title_tag["href"]
        post_id = href.split("l=")[-1]
        date_str = date_tag.get_text(strip=True)

        try:
            written_at = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except:
            written_at = datetime.now()  # 실패 시 현재시간 대입

        posts.append({
            "post_id": post_id,
            "title": title,
            "url": href,
            "written_at": written_at,
            "crawled_at": datetime.utcnow(),
            "source": "inven"
        })

    return posts

def run_inven_crawler(start_page=1, end_page=5):
    total_count = 0

    for page in range(start_page, end_page + 1):
        print(f"크롤링 중: 페이지 {page}")
        posts = crawl_page(page)
        if posts:
            for post in posts:
                if not collection.find_one({"post_id": post["post_id"]}):
                    collection.insert_one(post)
                    total_count += 1
        time.sleep(1)  # 인벤 서버에 부담 안 주게

    print(f"총 {total_count}개의 게시글 저장 완료.")

if __name__ == "__main__":
    run_inven_crawler(start_page=1, end_page=10)