from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os

# --- 설정 ---
options = Options()
options.add_argument("window-size=1280,720")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)
base_url = "https://www.inven.co.kr/search/lostark/article/유각/1?sort=recency"

results = []

# 메인 페이지 접속 및 게시물 수집
driver.get(base_url)
time.sleep(2)

posts = driver.find_elements(By.CSS_SELECTOR, "#lostarkBody li > h1 > a")[:3]

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


    # # 댓글 수집용 리스트 응 안해~
    # comments = []

    # try:
    #     # 댓글 컨테이너
    #     comment_container = WebDriverWait(driver, 5).until(
    #         EC.presence_of_element_located((By.ID, "pwbbsCmt"))
    #     )

    #     comment_items = comment_container.find_elements(By.CSS_SELECTOR, "li[id^='cmt']")

    #     current_comment = None

    #     for item in comment_items:
    #         class_list = item.get_attribute("class").split()
    #         is_reply = "reply" in class_list

    #         try:
    #             author = item.find_element(By.CSS_SELECTOR, "strong > span.nickname").text.strip()
    #         except:
    #             author = "작성자 없음"

    #         try:
    #             text = item.find_element(By.CSS_SELECTOR, "div.comment").text.strip()
    #         except:
    #             text = ""

    #         if not is_reply:
    #             # 일반 댓글
    #             current_comment = {
    #                 "author": author,
    #                 "comment": text,
    #                 "replies": []
    #             }
    #             comments.append(current_comment)
    #         else:
    #             # 대댓글
    #             if current_comment:
    #                 current_comment["replies"].append({
    #                     "author": author,
    #                     "comment": text
    #                 })

    # except:
    #     comments = []

    # 결과 저장
    results.append({
        "title": title,
        "url": href,
        "author": author,
        "date": date,
        "text": text_content,
        "images": img_names,
        # "comments": comments
    })

    # 탭 닫고 메인 창으로 복귀
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(0.5)

driver.quit()

# 출력
for i, post in enumerate(results, 1):
    print(f"\n=== [{i}] {post['title']} ===")
    print(f"URL      : {post['url']}")
    print(f"작성자    : {post['author']}")
    print(f"작성일    : {post['date']}")
    print("분문:")
    print(post['text'])

    if post['images']:
        print("이미지 파일명:")
        for img in post['images']:
            print(f" - {img}")

    # if post['comments']:
    #     print("댓글:")
    #     for cm in post['comments']:
    #         print(f"  ▶ {cm['author']}: {cm['comment']}")
    #         for rep in cm.get("replies", []):
    #             print(f"     ↳ {rep['author']}: {rep['comment']}")
    # else:
    #     print("댓글 없음")
