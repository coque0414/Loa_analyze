import requests
from bs4 import BeautifulSoup
import time
import json

# 기본 설정
BASE_LIST_URL = 'https://www.inven.co.kr/board/lostark/7000'
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/91.0.4472.124 Safari/537.36')
}


def get_post_list(page: int) -> list:
    """
    게시판 리스트 페이지에서 글 목록(제목, 링크, 작성자, 날짜)을 반환합니다.
    """
    params = {
        'sort': 'regdate',  # 등록일 순으로 정렬
        'page': page
    }
    resp = requests.get(BASE_LIST_URL, headers=HEADERS, params=params)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    posts = []
    for row in soup.select('table.board_list tbody tr'):
        link_tag = row.select_one('td.subject a')
        if not link_tag:
            continue
        title = link_tag.get_text(strip=True)
        url = link_tag['href']
        author = row.select_one('td.name').get_text(strip=True)
        date = row.select_one('td.time').get_text(strip=True)

        posts.append({
            'title': title,
            'url': url,
            'author': author,
            'date': date
        })
    return posts


def get_post_content(relative_url: str) -> str:
    """
    게시글 상세 페이지에서 본문 내용을 텍스트로 반환합니다.
    """
    full_url = 'https://www.inven.co.kr' + relative_url
    resp = requests.get(full_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    content_div = soup.select_one('div.bd')  # Inven 게시판 본문 컨테이너
    return content_div.get_text(strip=True) if content_div else ''


def main():
    all_posts = []
    # 원하는 페이지 범위를 지정
    for page in range(1, 6):  # 1페이지부터 5페이지까지 크롤링
        post_list = get_post_list(page)
        for post in post_list:
            content = get_post_content(post['url'])
            post['content'] = content
            all_posts.append(post)
            time.sleep(1)  # 서버 부담을 줄이기 위한 딜레이

    # JSON 파일로 저장 (필요 시 주석 해제)
    # with open('lostark_inven_posts.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_posts, f, ensure_ascii=False, indent=2)

    # 사람이 보기 좋은 형태로 콘솔에 출력
    print(json.dumps(all_posts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()