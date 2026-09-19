"""
로스트아크 공식 사이트의 공지사항(Notice) / GM노트(GMNote) 목록을 크롤링해서
evaluation/data/raw_docs.jsonl 에 저장한다.

기존 schema/docs_pipeline.py의 상세페이지 파서(parse_notice)를 그대로 재사용한다.
- 제목: h4.news-title
- 날짜: span.date  ("%Y.%m.%d %H:%M" 형식 가정)
- 본문: div.fr-view

목록 페이지는 이 환경(샌드박스)에서 직접 접속해 구조를 확인할 수 없었기 때문에
pageIndex 쿼리 파라미터 + 정적 HTML 내 정규식 링크 추출 방식으로 우선 작성했다.
로컬에서 실행했을 때 "게시글 링크를 찾지 못했습니다" 경고가 뜨면, 목록 페이지가
JS로 렌더링되는 SPA라는 뜻이므로 --list-url-template 옵션으로 실제 목록 API URL을
넘겨주거나, 브라우저 개발자도구 Network 탭에서 확인한 JSON API 엔드포인트를
알려주면 그에 맞게 수정한다.
"""

import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://lostark.game.onstove.com"

# category 키 -> 실제 URL 경로 세그먼트
CATEGORY_PATHS = {
    "notice": "Notice",
    "gmnote": "GMNote",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "raw_docs.jsonl"


def today_str():
    return dt.datetime.now().date().isoformat()


def build_view_url(category: str, article_id: str) -> str:
    path = CATEGORY_PATHS[category]
    return f"{BASE_URL}/News/{path}/Views/{article_id}"


def discover_article_urls(
    category: str,
    max_pages: int,
    delay: float,
    list_url_template: str | None,
):
    """
    목록 페이지에서 상세글(Views) URL을 모은다.
    list_url_template 이 주어지면 그 템플릿을 사용하고, 아니면
    기본 목록 URL에 ?pageIndex=N 을 붙여서 시도한다.
    """
    path = CATEGORY_PATHS[category]
    view_pattern = re.compile(rf"/News/{path}/Views/(\d+)")

    urls = []
    seen_ids = set()

    for page in range(1, max_pages + 1):
        if list_url_template:
            url = list_url_template.format(category=path, page=page)
        else:
            url = f"{BASE_URL}/News/{path}/List?pageIndex={page}"

        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
        except requests.RequestException as e:
            print(f"[FAIL] {category} {page}페이지 요청 실패: {e}")
            break

        matches = view_pattern.findall(res.text)

        new_ids = [m for m in matches if m not in seen_ids]

        if not new_ids:
            print(
                f"[WARN] {category} {page}페이지: 새로운 게시글 링크를 찾지 못했습니다. "
                f"(목록이 JS로 렌더링되는 페이지일 수 있습니다 — "
                f"이 경우 --list-url-template 옵션이 필요합니다)"
            )
            break

        for article_id in new_ids:
            seen_ids.add(article_id)
            urls.append(build_view_url(category, article_id))

        print(f"[OK] {category} {page}페이지: {len(new_ids)}개 링크 수집")

        time.sleep(delay)

    return urls


def parse_article(url: str):
    """schema/docs_pipeline.py의 parse_notice()와 동일한 파싱 로직."""

    res = requests.get(url, headers=HEADERS, timeout=15)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    title_el = soup.select_one("h4.news-title")
    title = title_el.get_text(strip=True) if title_el else ""

    published_at = None
    date_el = soup.select_one("span.date")
    if date_el:
        dtext = date_el.get_text(strip=True).replace(".", "-")
        try:
            published_at = (
                dt.datetime.strptime(dtext, "%Y-%m-%d %H:%M").isoformat() + "Z"
            )
        except ValueError:
            published_at = None

    body_el = soup.select_one("div.fr-view")
    body_text = body_el.get_text("\n", strip=True) if body_el else ""

    return {
        "title": title,
        "published_at": published_at,
        "body_text": body_text,
    }


def load_existing(path: Path):
    if not path.exists():
        return {}

    existing = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                existing[doc["doc_id"]] = doc
    return existing


def crawl(category: str, max_pages: int, limit: int, delay: float, list_url_template):
    urls = discover_article_urls(category, max_pages, delay, list_url_template)

    if not urls:
        print(f"[FAIL] {category}: 수집된 게시글 링크가 없습니다.")
        return []

    urls = urls[:limit]

    docs = []

    for i, url in enumerate(urls, start=1):
        article_id = url.rstrip("/").split("/")[-1]
        doc_id = f"{category}_{article_id}"

        try:
            parsed = parse_article(url)
        except requests.RequestException as e:
            print(f"[SKIP] {url}: 요청 실패 ({e})")
            continue

        if not parsed["title"] or not parsed["body_text"]:
            print(f"[SKIP] {url}: 제목/본문을 찾지 못했습니다 (선택자 확인 필요)")
            continue

        docs.append(
            {
                "doc_id": doc_id,
                "title": parsed["title"],
                "source": "lostark_official",
                "category": category,
                "url": url,
                "published_at": parsed["published_at"],
                "crawled_at": today_str(),
                "text": parsed["body_text"],
            }
        )

        print(f"[{i}/{len(urls)}] 수집 완료: {doc_id} - {parsed['title']}")

        time.sleep(delay)

    return docs


def main():
    parser = argparse.ArgumentParser(description="로스트아크 공식 문서 크롤러")
    parser.add_argument(
        "--category",
        choices=["notice", "gmnote", "all"],
        default="all",
        help="수집할 카테고리 (기본: all)",
    )
    parser.add_argument("--max-pages", type=int, default=3, help="목록 페이지 수")
    parser.add_argument("--limit", type=int, default=20, help="카테고리당 최대 수집 개수")
    parser.add_argument("--delay", type=float, default=1.0, help="요청 사이 대기 시간(초)")
    parser.add_argument(
        "--list-url-template",
        default=None,
        help=(
            "목록 페이지 URL 템플릿. {category}, {page} 를 사용할 수 있다. "
            "예: 'https://lostark.game.onstove.com/News/{category}/List?page={page}'"
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="기존 raw_docs.jsonl에 이어서 저장 (doc_id 기준 덮어쓰기 병합)",
    )
    args = parser.parse_args()

    categories = (
        ["notice", "gmnote"] if args.category == "all" else [args.category]
    )

    DATA_DIR.mkdir(exist_ok=True)

    all_docs = load_existing(OUTPUT_PATH) if args.append else {}

    for category in categories:
        docs = crawl(
            category,
            args.max_pages,
            args.limit,
            args.delay,
            args.list_url_template,
        )
        for doc in docs:
            all_docs[doc["doc_id"]] = doc

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in all_docs.values():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n총 {len(all_docs)}개 문서를 {OUTPUT_PATH} 에 저장했습니다.")


if __name__ == "__main__":
    main()
