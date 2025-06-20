from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
import time
from datetime import datetime

# --- MongoDB 설정 ---
MONGODB_URI = "mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLoa"
DB_NAME = "lostark"
COLLECTION_NAME = "community_posts"
client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]
collection.create_index([("url", 1)], unique=True)

# --- Selenium 설정 ---
options = webdriver.ChromeOptions()
# options.add_argument('--headless')

# options.add_argument('--disable-gpu')                    # GPU 비활성화 (윈도우 헤드리스 환경에서 권장)
# options.add_argument('--no-sandbox')                     # 샌드박스 모드 비활성화 (리눅스 환경에서 흔히 사용)
# options.add_argument('--disable-dev-shm-usage')          # /dev/shm 파티션 부족 문제 방지
# options.add_argument('--ignore-certificate-errors')      # SSL 인증서 오류 무시
# options.add_argument('--ignore-ssl-errors')              # SSL 오류 무시
options.add_argument("--window-size=1920,1080")          # 윈도우 크기 지정 (일부 사이트가 해상도 체크를 할 수도 있음)
# options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                      "AppleWebKit/537.36 (KHTML, like Gecko) "
#                      "Chrome/114.0.5735.199 Safari/537.36")
# options.add_argument('--enable-unsafe-swiftshader')

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# 시작 URL
start_url = (
    "https://www.inven.co.kr/board/lostark/6271?query=list&p=1&sterm=&name=subjcont&keyword=보석"
)
driver.get(start_url)

# 크롤링 기준: 오늘로부터 며칠 전까지 허용
MAX_DAYS_OLD = 8  # 1일 전까지 허용

# 키워드 매핑
item_keywords = {
    "보석" : [65021100, 65032100, 65022100, 65031100, 65031090, 65021090, 65032090, 65022090, 65031080, 65032080,65031070,65032070]
}

# 게시물 목록 로딩 함수
def wait_for_all_rows():
    return wait.until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "#new-board form div table tbody tr")
        )
    )

while True:
    try:
        rows = wait_for_all_rows()
    except TimeoutException:
        print("게시물 목록 로드 실패.")
        break

    # 첫 페이지 판단
    try:
        page1_btn = driver.find_element(By.CSS_SELECTOR, "#paging li:nth-child(2) a")
        is_first = "현재 선택된 페이지" in (page1_btn.get_attribute("title") or "")
    except NoSuchElementException:
        is_first = False

    start_idx = 8 if is_first else 0
    posts = driver.find_elements(
        By.CSS_SELECTOR,
        "#new-board form div table tbody tr td.tit div div a"
    )[start_idx:]

    main_handle = driver.current_window_handle
    stop_crawling = False

    for post in posts:
        href = post.get_attribute("href")
        driver.execute_script("window.open(arguments[0]);", href)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(1)

        # 게시물 날짜 확인 (절대 날짜로 일수 비교)
        try:
            date_str = driver.find_element(
                By.CSS_SELECTOR,
                "#tbArticle > div.articleHead.hC_silver1 > div > div.articleDate"
            ).text.strip()
            post_date = datetime.strptime(date_str, '%Y.%m.%d %H:%M')
            days_old = (datetime.now() - post_date).days
            if days_old > MAX_DAYS_OLD:
                stop_crawling = True
        except Exception:
            stop_crawling = False

        if stop_crawling:
            driver.close()
            driver.switch_to.window(main_handle)
            break

        # 게시물 데이터 추출
        try:
            title = driver.find_element(
                By.CSS_SELECTOR,
                "#tbArticle > div.articleMain > div.articleSubject > div.articleTitle"
            ).text.strip()
        except:
            title = None
        try:
            author = driver.find_element(
                By.CSS_SELECTOR,
                "#tbArticle > div.articleHead.hC_silver1 > div > div.articleWriter > span"
            ).text.strip()
        except:
            author = None
        try:
            content = driver.find_element(By.ID, "powerbbsContent").text.replace("\n", " ").strip()
        except:
            content = None

        img_urls = [img.get_attribute("src") for img in driver.find_elements(By.CSS_SELECTOR, "#powerbbsContent img")]

        # 키워드 인덱스 매핑
        item_index = []
        ts = ((title or "") + (content or "")).replace(" ", "")
        for keyword, codes in item_keywords.items():
            if keyword.replace(" ", "") in ts:
                item_index.extend(codes if isinstance(codes, list) else [codes])
        item_index = list(set(item_index))

        # MongoDB에 upsert 저장
        doc = {
            "title": title,
            "url": href,
            "author": author,
            "date": date_str,
            "text": content,
            "images": img_urls,
            "keyword": "보석",
            "item_index": item_index
        }
        collection.update_one({"url": href}, {"$set": doc}, upsert=True)

        driver.close()
        driver.switch_to.window(main_handle)
        time.sleep(0.5)

    if stop_crawling:
        print("기준 날짜 이전 게시물 발견. 크롤링 종료.")
        break

    # 페이지 네비게이션 처리
    try:
        current_btn = driver.find_element(
            By.CSS_SELECTOR,
            "#paging li a[title='현재 선택된 페이지입니다.']"
        )
        next_li = current_btn.find_element(By.XPATH, "../following-sibling::li[1]")
        next_a = next_li.find_element(By.TAG_NAME, "a")
        if "disabled" in next_a.get_attribute("class"):
            more_btn = driver.find_element(By.CSS_SELECTOR, "#new-board > div.board-bottom > div > button")
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#new-board > div.board-bottom > div > button")))
            more_btn.click()
            wait_for_all_rows()
            time.sleep(1)
            continue
        else:
            next_a.click()
            wait_for_all_rows()
            time.sleep(1)
            continue
    except (NoSuchElementException, TimeoutException):
        print("크롤링 완료.")
        break

# 종료 및 자원 해제
driver.quit()
