#셀레니움으로 어떻게 가져올 수 있는지 보게해주는 테스트입니다.

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from tabulate import tabulate
import time
import csv

# ✔️ ChromeDriver 경로 설정
chrome_driver_path = "test/chromedriver-win64/chromedriver.exe"  # 예: "C:/chromedriver.exe"

# ✔️ Chrome 옵션 설정
options = Options()
# options.add_argument("--headless")  # 창 숨기고 싶을 때
options.add_argument("--start-maximized")

# ✔️ 드라이버 실행
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.set_window_size(1920, 1080)  # 창 크기 설정


base_url = "https://www.inven.co.kr/board/lostark/6271"
driver.get(base_url)
time.sleep(3)  # 로딩 대기

table_data = []
# 4. tr 5 ~ 104 순회
for i in range(5, 105):
    try:
        selector = f"#new-board > form > div > table > tbody > tr:nth-child({i})"
        tr = driver.find_element(By.CSS_SELECTOR, selector)
        tds = tr.find_elements(By.TAG_NAME, "td")
        
        row = [td.text.strip() for td in tds]
        if row:  # 비어있지 않은 경우만
            table_data.append(row)
    except Exception as e:
        print(f"{i-4}. tr:nth-child({i}) 요소를 찾을 수 없습니다.")

driver.quit()

# 5. 표 출력
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

csv_filename = "inven_board.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerows(table_data)

print(f"\n✅ CSV 저장 완료: {csv_filename}")