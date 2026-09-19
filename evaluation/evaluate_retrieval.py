import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


# Loa_analyze 프로젝트 루트를 Python path에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))


from services.embedder import get_embedder


BASE_DIR = Path(__file__).resolve().parent

CORPUS_PATH = BASE_DIR / "data" / "corpus.jsonl"
QUESTION_PATH = BASE_DIR / "data" / "questions.jsonl"

RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)


def load_jsonl(path):

    items = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                items.append(json.loads(line))

    return items


def tokenize(text):
    """
    매우 간단한 keyword 검색용 tokenizer
    """

    return re.findall(
        r"[가-힣A-Za-z0-9]+",
        text.lower()
    )


def keyword_scores(query, corpus):

    query_tokens = tokenize(query)

    scores = []

    for doc in corpus:

        text = (
            doc["title"] + " " + doc["text"]
        ).lower()

        score = sum(
            1
            for token in query_tokens
            if token in text
        )

        scores.append(score)

    return np.array(scores, dtype=np.float32)


def cosine_scores(query_embedding, document_embeddings):

    query_embedding = query_embedding.reshape(-1)

    query_embedding = (
        query_embedding
        / (np.linalg.norm(query_embedding) + 1e-9)
    )

    document_embeddings = (
        document_embeddings
        /
        (
            np.linalg.norm(
                document_embeddings,
                axis=1,
                keepdims=True
            )
            + 1e-9
        )
    )

    return document_embeddings @ query_embedding


def rank_indices(scores):

    return np.argsort(scores)[::-1]


def is_relevant(item, question):
    """
    question에 relevant_chunk_ids(청크 단위 정답, 실제 크롤링 데이터용)가 있으면
    그것으로 채점하고, 없으면 예전 형식인 relevant_doc_id(문서 단위 정답,
    sample_data 스모크 테스트용)로 채점한다.
    """

    if "relevant_chunk_ids" in question:
        return item["chunk_id"] in question["relevant_chunk_ids"]

    return item["doc_id"] == question["relevant_doc_id"]


def relevant_label(question):

    if "relevant_chunk_ids" in question:
        return ", ".join(question["relevant_chunk_ids"])

    return question["relevant_doc_id"]


def hit_at_k(ranked_indices, corpus, question, k):

    top_k = ranked_indices[:k]

    for index in top_k:

        if is_relevant(corpus[index], question):
            return 1

    return 0


def reciprocal_rank(
    ranked_indices,
    corpus,
    question
):

    for rank, index in enumerate(ranked_indices, start=1):

        if is_relevant(corpus[index], question):
            return 1 / rank

    return 0


def main():

    corpus = load_jsonl(CORPUS_PATH)
    questions = load_jsonl(QUESTION_PATH)

    print("문서 Chunk 수:", len(corpus))
    print("평가 질문 수:", len(questions))

    # 기존 프로젝트의 KoSimCSE 임베더 사용
    embedder = get_embedder()

    print("문서 임베딩 생성 중...")

    corpus_texts = [
        doc["title"] + " " + doc["text"]
        for doc in corpus
    ]

    document_embeddings = embedder.encode(
        corpus_texts,
        convert_to_numpy=True
    )

    methods = {
        "keyword": {
            "hit1": [],
            "hit3": [],
            "mrr": []
        },

        "semantic": {
            "hit1": [],
            "hit3": [],
            "mrr": []
        },

        "hybrid": {
            "hit1": [],
            "hit3": [],
            "mrr": []
        },
    }

    details = []

    for q in questions:

        question_text = q["question"]

        # -------------------------
        # Keyword
        # -------------------------

        kw_scores = keyword_scores(
            question_text,
            corpus
        )

        kw_rank = rank_indices(kw_scores)

        # -------------------------
        # Semantic
        # -------------------------

        query_embedding = embedder.encode(
            question_text,
            convert_to_numpy=True
        )

        sem_scores = cosine_scores(
            query_embedding,
            document_embeddings
        )

        sem_rank = rank_indices(sem_scores)

        # -------------------------
        # Hybrid
        # -------------------------

        # 현재 프로젝트의 keyword boost 개념을
        # 단순화해서 재현
        keyword_boost = np.minimum(
            kw_scores,
            3
        ) * 0.25

        hybrid_scores = (
            sem_scores
            + keyword_boost
        )

        hybrid_rank = rank_indices(
            hybrid_scores
        )

        rankings = {
            "keyword": kw_rank,
            "semantic": sem_rank,
            "hybrid": hybrid_rank
        }

        for method, ranking in rankings.items():

            methods[method]["hit1"].append(
                hit_at_k(
                    ranking,
                    corpus,
                    q,
                    1
                )
            )

            methods[method]["hit3"].append(
                hit_at_k(
                    ranking,
                    corpus,
                    q,
                    3
                )
            )

            methods[method]["mrr"].append(
                reciprocal_rank(
                    ranking,
                    corpus,
                    q
                )
            )

        hybrid_top3 = [
            corpus[i]["chunk_id"]
            for i in hybrid_rank[:3]
        ]

        details.append({
            "question": question_text,
            "type": q.get("type", ""),
            "relevant": relevant_label(q),
            "hybrid_top1": hybrid_top3[0],
            "hybrid_top3": ", ".join(hybrid_top3),
            "hit3": hit_at_k(
                hybrid_rank,
                corpus,
                q,
                3
            )
        })

    summary = {}

    print("\n===== Retrieval Evaluation =====")

    for method, values in methods.items():

        hit1 = np.mean(values["hit1"])
        hit3 = np.mean(values["hit3"])
        mrr = np.mean(values["mrr"])

        summary[method] = {
            "hit@1": round(float(hit1), 4),
            "hit@3": round(float(hit3), 4),
            "mrr": round(float(mrr), 4)
        }

        print(
            f"{method:10s} "
            f"Hit@1={hit1:.3f} "
            f"Hit@3={hit3:.3f} "
            f"MRR={mrr:.3f}"
        )

    # 질문 유형별(general/paraphrase/disambiguation) hybrid Hit@3
    # -> 어떤 유형의 질문에서 특히 잘/못 찾는지 확인용
    by_type = {}
    for d in details:
        qtype = d["type"] or "unknown"
        by_type.setdefault(qtype, []).append(d["hit3"])

    if any(k != "unknown" for k in by_type):
        print("\n===== 질문 유형별 Hybrid Hit@3 =====")
        for qtype, hits in by_type.items():
            hit3 = np.mean(hits)
            summary.setdefault("hybrid_by_type", {})[qtype] = {
                "count": len(hits),
                "hit@3": round(float(hit3), 4),
            }
            print(f"{qtype:15s} Hit@3={hit3:.3f} (n={len(hits)})")

    # summary 저장

    with open(
        RESULT_DIR / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    # 질문별 결과 저장

    with open(
        RESULT_DIR / "retrieval_details.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "type",
                "relevant",
                "hybrid_top1",
                "hybrid_top3",
                "hit3"
            ]
        )

        writer.writeheader()
        writer.writerows(details)


if __name__ == "__main__":
    main()
