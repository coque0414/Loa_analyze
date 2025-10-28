import textwrap
import numpy as np
import time
from openai import OpenAI
import os
import asyncio
import re

from services.db import posts_col, docs_col
from services.embedder import get_embedder
from dotenv import load_dotenv
load_dotenv()

# qa.py의 semantic_search 함수 임포트
from services.qa import semantic_search

# .env 파일 로드
load_dotenv()

async def main():
    # 테스트할 질문과 제한 개수 설정
    question = "낙원 콘텐츠"
    limit = 5

    # semantic_search 함수 호출
    ctx_docs = await semantic_search(question, limit=limit)

    # 검색된 문서 수 출력
    print(f"검색된 문서 수: {len(ctx_docs)}")

    # 각 문서의 ID, 제목, 텍스트 일부 출력
    for d in ctx_docs:
        print(d["_id"], d.get("title"), d.get("text")[:150])

# 비동기 함수 실행
if __name__ == "__main__":
    asyncio.run(main())