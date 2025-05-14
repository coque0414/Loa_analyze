from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

# --- 설정 ---
options = Options()
options.add_argument("window-size=1280,720")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)
url = "https://www.inven.co.kr/board/lostark/5708/139317"
driver.get(url)

time.sleep(3)

try:
    # 댓글 리스트 수집
    comment_list = driver.find_elements(By.CSS_SELECTOR, "#powerbbsCmt2 div.cmtWrap > ul > li[id^='cmt']")

    if not comment_list:
        print("💬 댓글이 없습니다.")
    else:
        print(f"💬 총 {len(comment_list)}개의 댓글을 찾았습니다:\n")
        for idx, comment in enumerate(comment_list, 1):
            try:
                author = comment.find_element(By.CSS_SELECTOR, ".cmt_info").text.strip()
                content = comment.find_element(By.CSS_SELECTOR, ".cmt_text").text.strip()
                print(f"[{idx}] 작성자: {author}")
                print(f"    내용: {content}\n")
            except Exception as e:
                print(f"[{idx}] 댓글 파싱 실패: {e}")
except Exception as e:
    print("[!] 댓글을 가져오는 중 오류가 발생했습니다.")
    print("에러 메시지:", e)
finally:
    driver.quit()
    
# try:
#     # 댓글 div 찾기
#     comment_section = driver.find_element(By.CSS_SELECTOR, "#powerbbsCmt2 > div.cmtWrap")
#     print("✅ 댓글 컨테이너 내용:\n")
#     print(comment_section.text)
# except Exception as e:
#     print("[!] 댓글 컨테이너를 찾을 수 없습니다.")
#     print("에러 메시지:", e)
# driver.quit()