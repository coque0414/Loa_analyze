import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "data" / "raw_docs.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "corpus.jsonl"


def chunk_text(text: str, max_len: int = 600, overlap: int = 0):
    """
    문서를 약 600자 단위로 나눈다.
    기존 졸업 프로젝트의 chunking 방식과 비슷하게
    줄바꿈 단위로 합쳐서 자른다.

    overlap > 0이면, 새 chunk를 시작할 때 직전 chunk의 마지막
    overlap자를 앞에 이어붙인다. 정답이 chunk 경계 근처에 걸쳐
    있어서 놓치는 경우를 줄이기 위한 옵션이다 (기본값 0 = 기존과 동일).
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chunks = []
    buffer = ""

    for line in lines:

        if len(buffer) + len(line) > max_len:

            if buffer:
                chunks.append(buffer.strip())

                if overlap > 0:
                    buffer = buffer[-overlap:] + " " + line
                else:
                    buffer = line

            else:
                buffer = line

        else:
            buffer += " " + line

    if buffer:
        chunks.append(buffer.strip())

    return chunks


def main():

    parser = argparse.ArgumentParser(description="raw_docs.jsonl을 chunk 단위 corpus로 변환")
    parser.add_argument(
        "--raw",
        default=str(RAW_PATH),
        help="입력 raw_docs.jsonl 경로"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="출력 corpus.jsonl 경로"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=600,
        help="chunk 최대 길이(자, 기본 600)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="chunk 간 겹치는 길이(자, 기본 0 = 겹침 없음)"
    )
    args = parser.parse_args()

    corpus = []

    with open(args.raw, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            doc = json.loads(line)

            chunks = chunk_text(
                doc["text"],
                max_len=args.max_len,
                overlap=args.overlap
            )

            for index, chunk in enumerate(chunks):

                corpus.append({
                    "doc_id": doc["doc_id"],
                    "chunk_id": f'{doc["doc_id"]}#{index + 1}',
                    "title": doc["title"],
                    "source": doc.get("source", "unknown"),
                    "text": chunk
                })

    with open(args.output, "w", encoding="utf-8") as f:

        for item in corpus:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"총 {len(corpus)}개의 chunk 생성 완료 -> {args.output}")


if __name__ == "__main__":
    main()
