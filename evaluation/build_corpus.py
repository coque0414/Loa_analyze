import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "data" / "raw_docs.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "corpus.jsonl"


def chunk_text(text: str, max_len: int = 600):
    """
    문서를 약 600자 단위로 나눈다.
    기존 졸업 프로젝트의 chunking 방식과 비슷하게
    줄바꿈 단위로 합쳐서 자른다.
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

            buffer = line

        else:
            buffer += " " + line

    if buffer:
        chunks.append(buffer.strip())

    return chunks


def main():

    corpus = []

    with open(RAW_PATH, "r", encoding="utf-8") as f:

        for line in f:

            doc = json.loads(line)

            chunks = chunk_text(doc["text"])

            for index, chunk in enumerate(chunks):

                corpus.append({
                    "doc_id": doc["doc_id"],
                    "chunk_id": f'{doc["doc_id"]}#{index + 1}',
                    "title": doc["title"],
                    "source": doc.get("source", "unknown"),
                    "text": chunk
                })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        for item in corpus:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"총 {len(corpus)}개의 chunk 생성 완료")


if __name__ == "__main__":
    main()
