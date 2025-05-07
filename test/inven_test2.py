from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

# ✔️ ChromeDriver 경로 설정
chrome_driver_path = "test\chromedriver-win64\chromedriver.exe"  # 예: "C:/chromedriver.exe"

# ✔️ Chrome 옵션 설정
options = Options()
# options.add_argument("--headless")  # 창 숨기고 싶을 때
options.add_argument("--start-maximized")

# ✔️ 드라이버 실행
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.set_window_size(1920, 1080)  # 창 크기 설정

# ✔️ 수집할 페이지 수 (자유게시판)
base_url = "https://www.inven.co.kr/board/lostark/6271"
page_range = range(1, 3)  # 1~2페이지 예시

# ✔️ 데이터 저장용 리스트
all_data = []

for page in page_range:
    print(f"[페이지 {page}] 수집 중...")
    driver.get(f"{base_url}?p={page}")
    time.sleep(2)

    posts = driver.find_elements(By.CSS_SELECTOR, "div.board-list > table > tbody > tr")

    for post in posts:
        try:
            link = post.find_element(By.CSS_SELECTOR, "td.tit > a").get_attribute("href")
            title = post.find_element(By.CSS_SELECTOR, "td.tit > a").text.strip()
        except:
            continue  # 공지사항 등 스킵

        # 게시글 상세 페이지로 이동
        driver.get(link)
        time.sleep(1)

        try:
            writer = driver.find_element(By.CSS_SELECTOR, ".article-meta .user").text.strip()
            date = driver.find_element(By.CSS_SELECTOR, ".article-meta .date").text.strip()
            content = driver.find_element(By.CSS_SELECTOR, ".articleBody").text.strip()
        except:
            continue

        # 댓글 수집
        comments = []
        try:
            comment_elements = driver.find_elements(By.CSS_SELECTOR, ".comment .text")
            for c in comment_elements:
                comments.append(c.text.strip())
        except:
            pass

        # 데이터 저장
        all_data.append({
            "제목": title,
            "작성자": writer,
            "작성일": date,
            "내용": content,
            "댓글 수": len(comments),
            "댓글 목록": comments,
            "링크": link
        })

# ✔️ CSV로 저장
df = pd.DataFrame(all_data)
df.to_csv("inven_lostark_posts.csv", index=False, encoding="utf-8-sig")

print("✅ 크롤링 완료! 'inven_lostark_posts.csv' 파일로 저장됐습니다.")

# 드라이버 종료
driver.quit()
