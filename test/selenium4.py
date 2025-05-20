from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import csv

# --- 설정 ---
options = Options()
options.add_argument("window-size=1280,720")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)
# base_url = "https://www.inven.co.kr/search/lostark/article/유각/1?sort=recency"

results = []

max_pages = 3 #어디 페이지까지 긁어올지 정하는부분
name_item = "유각"

for page in range(1, max_pages + 1):
    base_url = f"https://www.inven.co.kr/search/lostark/article/{name_item}/{page}?sort=recency"
    print(f"[+] {page} 페이지 크롤링 중...")

    # 메인 페이지 접속 및 게시물 수집
    driver.get(base_url)
    time.sleep(2)

    posts = driver.find_elements(By.CSS_SELECTOR, "#lostarkBody li > h1 > a")[:20]

    for post in posts:
        title = post.text
        href = post.get_attribute("href")

        # 새 탭 열기
        driver.execute_script("window.open(arguments[0]);", href)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(2)

        # 작성자, 작성일
        try:
            author = driver.find_element(
                By.CSS_SELECTOR,
                "#tbArticle > div.articleHead.hC_silver1 > div > div.articleWriter > span"
            ).text
        except:
            author = None

        try:
            date = driver.find_element(
                By.CSS_SELECTOR,
                "#tbArticle > div.articleHead.hC_silver1 > div > div.articleDate"
            ).text
        except:
            date = None

        # 본문 내용 블록
        try:
            content_block = driver.find_element(By.ID, "powerbbsContent")
        except:
            print(f"[!] 분문이 없는 게시물: {title} → 패스")
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            continue

        # 본문 텍스트 추출
        #text_elements = content_block.find_elements(By.XPATH, ".//p | .//div | .//span")
        #text_parts = [el.text.strip() for el in text_elements if el.text.strip()]
        #text_content = "\n".join(text_parts)
        text_content = content_block.text.strip()

        # 이미지 파일명 추출
        imgs = content_block.find_elements(By.TAG_NAME, "img")
        img_names = []
        for idx, img in enumerate(imgs, start=1):
            src = img.get_attribute("src")
            if not src:
                continue
            ext = os.path.splitext(src)[1].split("?")[0] or ".jpg"
            filename = f"{title[:10].strip().replace(' ', '_')}_{idx}{ext}"
            img_names.append(filename)

        # 결과 저장
        results.append({
            "title": title,
            "url": href,
            "author": author,
            "date": date,
            "text": text_content,
            "images": img_names
        })

        # 탭 닫고 메인 창으로 복귀
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(0.5)

driver.quit()

# # 출력
# for i, post in enumerate(results, 1):
#     print(f"\n=== [{i}] {post['title']} ===")
#     print(f"URL      : {post['url']}")
#     print(f"작성자    : {post['author']}")
#     print(f"작성일    : {post['date']}")
#     print("분문:")
#     print(post['text'])

#     if post['images']:
#         print("이미지 파일명:")
#         for img in post['images']:
#             print(f" - {img}")

with open("inven_posts.csv", mode="w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=["title", "url", "author", "date", "text", "images"])
    writer.writeheader()
    for post in results:
        writer.writerow({
            "title": post["title"],
            "url": post["url"],
            "author": post["author"],
            "date": post["date"],
            "text": post["text"].replace("\n", " "),
            "images": "; ".join(post["images"]) if post["images"] else "",
            "keyword" : name_item
        })
print("저장완료티비")