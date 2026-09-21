"""
evaluate_retrieval.py의 hybrid 방식에서 keyword boost 가중치를 바꿔가며
Hit@1 / Hit@3 / MRR이 어떻게 달라지는지 비교한다.

evaluate_retrieval.py를 그대로 재사용하기 때문에, 문서 임베딩과 질문 임베딩은
한 번만 계산하고 그 결과에 여러 가중치를 적용해 랭킹만 다시 계산한다
(가중치마다 임베딩을 새로 만들 필요가 없다).

실행:
    python evaluation/build_corpus.py   (raw_docs.jsonl이 바뀌었다면)
    python evaluation/sweep_keyword_boost.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from evaluate_retrieval import (  # noqa: E402
    CORPUS_PATH,
    QUESTION_PATH,
    RESULT_DIR,
    cosine_scores,
    get_embedder,
    hit_at_k,
    keyword_scores,
    load_jsonl,
    rank_indices,
    reciprocal_rank,
)

# 비교해볼 keyword boost 가중치들
# (evaluate_retrieval.py 기본값인 0.25도 포함해서 기준선과 바로 비교할 수 있게 한다)
BOOST_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def main():

    corpus = load_jsonl(CORPUS_PATH)
    questions = load_jsonl(QUESTION_PATH)

    print("문서 Chunk 수:", len(corpus))
    print("평가 질문 수:", len(questions))

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

    print("질문별 keyword/semantic 점수 사전 계산 중...")

    # 질문마다 keyword 점수와 semantic 점수를 한 번만 계산해서 재사용한다.
    precomputed = []

    for q in questions:

        kw_scores = keyword_scores(q["question"], corpus)

        query_embedding = embedder.encode(
            q["question"],
            convert_to_numpy=True
        )

        sem_scores = cosine_scores(query_embedding, document_embeddings)

        precomputed.append((q, kw_scores, sem_scores))

    print()
    print(f"{'boost':>6s} {'Hit@1':>8s} {'Hit@3':>8s} {'MRR':>8s}")

    rows = []

    for weight in BOOST_WEIGHTS:

        hit1s, hit3s, mrrs = [], [], []

        for q, kw_scores, sem_scores in precomputed:

            keyword_boost = np.minimum(kw_scores, 3) * weight
            hybrid_scores = sem_scores + keyword_boost

            ranking = rank_indices(hybrid_scores)

            hit1s.append(hit_at_k(ranking, corpus, q, 1))
            hit3s.append(hit_at_k(ranking, corpus, q, 3))
            mrrs.append(reciprocal_rank(ranking, corpus, q))

        hit1 = float(np.mean(hit1s))
        hit3 = float(np.mean(hit3s))
        mrr = float(np.mean(mrrs))

        rows.append({
            "keyword_boost": weight,
            "hit@1": round(hit1, 4),
            "hit@3": round(hit3, 4),
            "mrr": round(mrr, 4),
        })

        print(f"{weight:6.2f} {hit1:8.3f} {hit3:8.3f} {mrr:8.3f}")

    RESULT_DIR.mkdir(exist_ok=True)

    out_path = RESULT_DIR / "keyword_boost_sweep.csv"

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["keyword_boost", "hit@1", "hit@3", "mrr"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\n결과를 {out_path} 에 저장했습니다.")


if __name__ == "__main__":
    main()
