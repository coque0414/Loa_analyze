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


def hit_at_k(ranked_indices, corpus, relevant_doc_id, k):

    top_k = ranked_indices[:k]

    for index in top_k:

        if corpus[index]["doc_id"] == relevant_doc_id:
            return 1

    return 0


def reciprocal_rank(
    ranked_indices,
    corpus,
    relevant_doc_id
):

    for rank, index in enumerate(ranked_indices, start=1):

        if corpus[index]["doc_id"] == relevant_doc_id:
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

        question = q["question"]
        relevant_doc_id = q["relevant_doc_id"]

        # -------------------------
        # Keyword
        # -------------------------

        kw_scores = keyword_scores(
            question,
            corpus
        )

        kw_rank = rank_indices(kw_scores)

        # -------------------------
        # Semantic
        # -------------------------

        query_embedding = embedder.encode(
            question,
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
                    relevant_doc_id,
                    1
                )
            )

            methods[method]["hit3"].append(
                hit_at_k(
                    ranking,
                    corpus,
                    relevant_doc_id,
                    3
                )
            )

            methods[method]["mrr"].append(
                reciprocal_rank(
                    ranking,
                    corpus,
                    relevant_doc_id
                )
            )

        hybrid_top3 = [
            corpus[i]["doc_id"]
            for i in hybrid_rank[:3]
        ]

        details.append({
            "question": question,
            "relevant_doc": relevant_doc_id,
            "hybrid_top1": hybrid_top3[0],
            "hybrid_top3": ", ".join(hybrid_top3),
            "hit3": hit_at_k(
                hybrid_rank,
                corpus,
                relevant_doc_id,
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
                "relevant_doc",
                "hybrid_top1",
                "hybrid_top3",
                "hit3"
            ]
        )

        writer.writeheader()
        writer.writerows(details)


if __name__ == "__main__":
    main()
