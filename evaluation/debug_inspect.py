"""
crawl_docs.py의 상세 페이지 선택자(h4.news-title / span.date / div.fr-view)가
지금 사이트 구조와 맞는지 확인하기 위한 1회성 진단 스크립트.

실행:
    python evaluation/debug_inspect.py https://lostark.game.onstove.com/News/Notice/Views/13522

콘솔에 출력되는 내용을 그대로 복사해서 알려주면 그걸 보고
crawl_docs.py의 실제 선택자를 맞게 고친다.
전체 HTML은 evaluation/data/_debug_page.html 로도 저장되니,
콘솔 출력만으로 부족하면 그 파일을 열어서 직접 확인해도 된다.
"""

import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# 제목/날짜/본문일 가능성이 있는 class/id 키워드 후보
KEYWORDS = ["tit", "date", "cont", "body", "view", "article", "news", "detail"]


def short(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def describe(el) -> str:
    tag = el.name
    cls = " ".join(el.get("class", []))
    id_ = el.get("id", "")
    attr = f'class="{cls}"' if cls else (f'id="{id_}"' if id_ else "")
    text = short(el.get_text(strip=True))
    return f"<{tag} {attr}>  텍스트: {text}"


def main():
    if len(sys.argv) < 2:
        print("사용법: python evaluation/debug_inspect.py <상세페이지 URL>")
        sys.exit(1)

    url = sys.argv[1]

    res = requests.get(url, headers=HEADERS, timeout=15)
    res.raise_for_status()

    out_path = Path(__file__).resolve().parent / "data" / "_debug_page.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(res.text, encoding="utf-8")

    soup = BeautifulSoup(res.text, "html.parser")

    print(f"URL: {url}")
    print(f"응답 길이: {len(res.text)}자 (전체 HTML은 {out_path} 에 저장됨)")
    print()

    print("===== <title> 태그 =====")
    if soup.title:
        print(soup.title.get_text(strip=True))
    print()

    print("===== h1~h4 태그 (제목 후보) =====")
    for tag in ["h1", "h2", "h3", "h4"]:
        for el in soup.find_all(tag):
            print(describe(el))
    print()

    print("===== class/id에 키워드가 들어간 요소 (제목/날짜/본문 후보) =====")
    seen = set()
    for el in soup.find_all(True):
        cls = " ".join(el.get("class", [])).lower()
        id_ = (el.get("id") or "").lower()
        haystack = cls + " " + id_

        if not haystack.strip():
            continue

        if any(k in haystack for k in KEYWORDS):
            key = (el.name, cls, id_)
            if key in seen:
                continue
            seen.add(key)
            print(describe(el))

    print()
    print("===== 현재 crawl_docs.py 선택자로 직접 조회한 결과 =====")
    title_el = soup.select_one("h4.news-title")
    date_el = soup.select_one("span.date")
    body_el = soup.select_one("div.fr-view")

    print("h4.news-title  ->", describe(title_el) if title_el else "(없음)")
    print("span.date      ->", describe(date_el) if date_el else "(없음)")
    print("div.fr-view    ->", describe(body_el) if body_el else "(없음)")


if __name__ == "__main__":
    main()
