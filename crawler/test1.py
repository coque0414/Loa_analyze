from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1) Chrome 옵션 설정 (헤드리스 모드)
opts = Options()
opts.add_argument('--headless')
opts.add_argument('--disable-gpu')
opts.add_argument(
    'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/91.0.4472.124 Safari/537.36'
)

# 2) ChromeDriver 실행 (chromedriver 경로를 실제 경로로 바꿔주세요)
service = Service('/path/to/chromedriver')
driver = webdriver.Chrome(service=service, options=opts)

try:
    # 3) 페이지 로드
    url = 'https://www.inven.co.kr/board/lostark/6271?p=1'
    driver.get(url)

    # 4) 테이블 로딩될 때까지 최대 10초 대기
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.board_list tbody tr')))

    # 5) 5~30번째 tr 요소 추출
    rows = driver.find_elements(By.CSS_SELECTOR, 'table.board_list tbody tr')[4:30]

    # 6) 각각의 셀에서 데이터 꺼내기
    for row in rows:
        num   = row.find_element(By.CSS_SELECTOR, 'td.num span').text
        title = row.find_element(By.CSS_SELECTOR, 'td.tit a.subject-link').text
        date  = row.find_element(By.CSS_SELECTOR, 'td.date').text
        view  = row.find_element(By.CSS_SELECTOR, 'td.view').text
        reco  = row.find_element(By.CSS_SELECTOR, 'td.reco').text
        print(f"{num}\t{date}\t{view}\t{reco}\t{title}")

finally:
    driver.quit()