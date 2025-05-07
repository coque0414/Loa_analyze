from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

# ChromeDriver 경로 설정
chrome_driver_path = "test\chromedriver-win64\chromedriver.exe"  # 예: "C:/chromedriver.exe"

# 브라우저 옵션 설정 (창 안 뜨게 하고 싶으면 headless=True)
options = Options()
# options.add_argument("--headless")  # 창 숨기기 옵션 (필요 시 주석 제거)
options.add_argument("--start-maximized")

# 드라이버 실행
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.set_window_size(1920, 1080)  # 창 크기 설정

# 자유게시판 URL
url = "https://www.inven.co.kr/board/lostark/6271"

# 사이트 열기
driver.get(url)
time.sleep(2)  # 로딩 대기

# 게시글 리스트 가져오기 (한 페이지당 기본 20개)
posts = driver.find_elements(By.CSS_SELECTOR, "div.board-list > table > tbody > tr")

for post in posts:
    try:
        # 제목
        title = post.find_element(By.CSS_SELECTOR, "td.tit > a").text.strip()
        # 링크
        link = post.find_element(By.CSS_SELECTOR, "td.tit > a").get_attribute("href")
        # 작성자
        writer = post.find_element(By.CSS_SELECTOR, "td.user").text.strip()
        print(f"제목: {title}")
        print(f"링크: {link}")
        print(f"작성자: {writer}")
        print("=" * 40)
    except:
        # 공지사항 등은 제외
        continue

driver.quit()
