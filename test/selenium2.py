from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo.mongo_client import MongoClient
import time
import re
from datetime import datetime

# ✔️ MongoDB 연결
uri = "mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLoa"
client = MongoClient(uri)

db = client["lostark"]
collection = db["inven_posts"]

# ✔️ 크롬 드라이버 설정
chrome_driver_path = "test/chromedriver-win64/chromedriver.exe"
options = Options()
# options.add_argument("--headless")  # 백그라운드 실행 시 사용
options.add_argument("--start-maximized")

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.set_window_size(1920, 1080)

base_url = "https://www.inven.co.kr/board/lostark/6271"
driver.get(base_url)
time.sleep(2)

# ✔️ 게시판 목록 순회
for i in range(7, 17):  # 테스트로 10개만
    try:
        selector = f"#new-board > form > div > table > tbody > tr:nth-child({i})"
        tr = driver.find_element(By.CSS_SELECTOR, selector)
        tds = tr.find_elements(By.TAG_NAME, "td")

        post_id = int(tds[0].text.strip())
        title = tds[1].text.strip()
        link = tds[1].find_element(By.CSS_SELECTOR, "a.subject-link").get_attribute("href")
        author = tds[2].text.strip()
        # created_raw = tds[3].text.strip()
        # created_at = datetime.strptime(created_raw, "%Y.%m.%d").strftime("%Y-%m-%d")
        views = int(tds[4].text.strip().replace(",", ""))
        likes = int(tds[5].text.strip())

        print(post_id, title, link, author, views, likes)

    except:
        pass

    #     # ✔️ 게시물 본문 진입
    #     driver.get(link)
    #     time.sleep(1)
    #     try:
    #         content = driver.find_element(By.CSS_SELECTOR, "div.article").text
    #     except:
    #         content = "(본문 없음)"

    #     # ✔️ 댓글 크롤링
    #     comments = []
    #     try:
    #         comment_blocks = driver.find_elements(By.CSS_SELECTOR, "div.comment-list > div.comment")
    #         for c in comment_blocks:
    #             try:
    #                 c_author = c.find_element(By.CLASS_NAME, "nick").text.strip()
    #                 c_text = c.find_element(By.CLASS_NAME, "text").text.strip()
    #                 c_date_raw = c.find_element(By.CLASS_NAME, "date").text.strip()
    #                 c_date = re.search(r"\d{4}-\d{2}-\d{2}", c_date_raw)
    #                 c_date = c_date.group() if c_date else datetime.today().strftime("%Y-%m-%d")
    #                 comments.append({
    #                     "author": c_author,
    #                     "text": c_text,
    #                     "created_at": c_date
    #                 })
    #             except:
    #                 continue
    #     except:
    #         pass

    #     # ✔️ MongoDB 저장
    #     document = {
    #         "post_id": post_id,
    #         "title": title,
    #         "author": author,
    #         "created_at": created_at,
    #         "views": views,
    #         "likes": likes,
    #         "content": content,
    #         "comments": comments,
    #         "url": link,
    #         "scraped_at": datetime.now().isoformat()
    #     }

    #     collection.insert_one(document)
    #     print(f"✅ 저장됨: {title}")

    #     # ✔️ 다시 목록으로
    #     driver.back()
    #     time.sleep(1)

    # except Exception as e:
    #     print(f"❌ 실패: {i}번째 tr. 에러: {e}")
    #     continue

driver.quit()