"""
Final Test Set(evaluation/data/final_test.jsonl, 50문항) 평가.

validation(70문항)에서 이미 확정한 설정을 그대로 재사용한다 — 여기서는
boost/chunk/overlap/top-k 중 아무것도 다시 튜닝하지 않는다:
    - KoSimCSE 임베딩
    - keyword boost = evaluate_retrieval.KEYWORD_BOOST_WEIGHT (validation에서 0.05로 확정)
    - chunk 600자 / overlap 없음 (validation에서 overlap=100 실험이 역효과로 기각됨)
    - Top-3

50문항 중 40개(general/paraphrase/disambiguation)는 정답 chunk가 있는
"답변 가능" 질문이라 Keyword/Semantic/Hybrid의 Hit@1·Hit@3·MRR을 그대로 측정한다.
나머지 10개(no_answer)는 corpus 안에 정답이 없는 질문이라 Hit@K로 채점할 수
없다 — 여기서는 아직 "근거 없음 판정"을 내리지 않고, hybrid 기준 top-1/top-3
유사도 점수만 기록한다. 이 점수 분포를 보고 다음 단계(근거 없음 거절 threshold
결정)를 진행한다.

실행:
    python evaluation/evaluate_final_test.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from evaluate_retrieval import (  # noqa: E402
    CORPUS_PATH,
    KEYWORD_BOOST_WEIGHT,
    RESULT_DIR,
    cosine_scores,
    get_embedder,
    hit_at_k,
    keyword_scores,
    load_jsonl,
    rank_indices,
    reciprocal_rank,
)

FINAL_TEST_PATH = BASE_DIR / "data" / "final_test.jsonl"


def main():

    corpus = load_jsonl(CORPUS_PATH)
    all_questions = load_jsonl(FINAL_TEST_PATH)

    answerable = [q for q in all_questions if q["type"] != "no_answer"]
    no_answer = [q for q in all_questions if q["type"] == "no_answer"]

    print("문서 Chunk 수:", len(corpus))
    print("Final Test 문항 수:", len(all_questions),
          f"(답변 가능 {len(answerable)} / 무응답 {len(no_answer)})")
    print("적용 설정: keyword_boost =", KEYWORD_BOOST_WEIGHT, "(validation에서 확정, 재조정 없음)")

    embedder = get_embedder()

    print("문서 임베딩 생성 중...")

    corpus_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
    document_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True)

    # -------------------------------------------------
    # 1) 답변 가능 40문항: Keyword / Semantic / Hybrid 비교
    # -------------------------------------------------

    methods = {
        "keyword": {"hit1": [], "hit3": [], "mrr": []},
        "semantic": {"hit1": [], "hit3": [], "mrr": []},
        "hybrid": {"hit1": [], "hit3": [], "mrr": []},
    }

    by_type = {}
    details = []

    for q in answerable:

        kw_scores = keyword_scores(q["question"], corpus)
        kw_rank = rank_indices(kw_scores)

        query_embedding = embedder.encode(q["question"], convert_to_numpy=True)
        sem_scores = cosine_scores(query_embedding, document_embeddings)
        sem_rank = rank_indices(sem_scores)

        keyword_boost = np.minimum(kw_scores, 3) * KEYWORD_BOOST_WEIGHT
        hybrid_scores = sem_scores + keyword_boost
        hybrid_rank = rank_indices(hybrid_scores)

        rankings = {"keyword": kw_rank, "semantic": sem_rank, "hybrid": hybrid_rank}

        for method, ranking in rankings.items():
            methods[method]["hit1"].append(hit_at_k(ranking, corpus, q, 1))
            methods[method]["hit3"].append(hit_at_k(ranking, corpus, q, 3))
            methods[method]["mrr"].append(reciprocal_rank(ranking, corpus, q))

        hybrid_hit3 = hit_at_k(hybrid_rank, corpus, q, 3)

        qtype = q["type"]
        by_type.setdefault(qtype, []).append(hybrid_hit3)

        hybrid_top3 = [corpus[i]["chunk_id"] for i in hybrid_rank[:3]]

        details.append({
            "question": q["question"],
            "type": qtype,
            "relevant": ", ".join(q["relevant_chunk_ids"]),
            "hybrid_top1": hybrid_top3[0],
            "hybrid_top1_score": round(float(hybrid_scores[hybrid_rank[0]]), 4),
            "hybrid_top3": ", ".join(hybrid_top3),
            "hit3": hybrid_hit3,
        })

    summary = {}

    print("\n===== Final Test: 답변 가능 40문항 Retrieval 성능 =====")

    for method, values in methods.items():
        hit1 = float(np.mean(values["hit1"]))
        hit3 = float(np.mean(values["hit3"]))
        mrr = float(np.mean(values["mrr"]))
        summary[method] = {"hit@1": round(hit1, 4), "hit@3": round(hit3, 4), "mrr": round(mrr, 4)}
        print(f"{method:10s} Hit@1={hit1:.3f} Hit@3={hit3:.3f} MRR={mrr:.3f}")

    print("\n===== 유형별 Hybrid Hit@3 =====")
    summary["hybrid_by_type"] = {}
    for qtype, hits in by_type.items():
        hit3 = float(np.mean(hits))
        summary["hybrid_by_type"][qtype] = {"count": len(hits), "hit@3": round(hit3, 4)}
        print(f"{qtype:15s} Hit@3={hit3:.3f} (n={len(hits)})")

    # -------------------------------------------------
    # 2) 무응답 10문항: 아직 판정하지 않고 점수만 기록
    # -------------------------------------------------

    no_answer_rows = []

    print("\n===== Final Test: 무응답(no_answer) 10문항 — top score만 기록 =====")

    for q in no_answer:

        kw_scores = keyword_scores(q["question"], corpus)

        query_embedding = embedder.encode(q["question"], convert_to_numpy=True)
        sem_scores = cosine_scores(query_embedding, document_embeddings)

        keyword_boost = np.minimum(kw_scores, 3) * KEYWORD_BOOST_WEIGHT
        hybrid_scores = sem_scores + keyword_boost
        hybrid_rank = rank_indices(hybrid_scores)

        top1_idx = hybrid_rank[0]

        no_answer_rows.append({
            "question": q["question"],
            "hybrid_top1_chunk": corpus[top1_idx]["chunk_id"],
            "hybrid_top1_score": round(float(hybrid_scores[top1_idx]), 4),
            "hybrid_top3_scores": ", ".join(
                f"{corpus[i]['chunk_id']}:{hybrid_scores[i]:.4f}" for i in hybrid_rank[:3]
            ),
        })

        print(f"- {q['question']}")
        print(f"    top1 = {corpus[top1_idx]['chunk_id']} (score={hybrid_scores[top1_idx]:.4f})")

    # -------------------------------------------------
    # 저장
    # -------------------------------------------------

    RESULT_DIR.mkdir(exist_ok=True)

    with open(RESULT_DIR / "final_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(RESULT_DIR / "final_test_details.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "question", "type", "relevant",
                "hybrid_top1", "hybrid_top1_score", "hybrid_top3", "hit3",
            ]
        )
        writer.writeheader()
        writer.writerows(details)

    with open(RESULT_DIR / "final_test_no_answer_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "hybrid_top1_chunk", "hybrid_top1_score", "hybrid_top3_scores"]
        )
        writer.writeheader()
        writer.writerows(no_answer_rows)

    print(f"\n결과 저장 완료:")
    print(f"  {RESULT_DIR / 'final_test_summary.json'}")
    print(f"  {RESULT_DIR / 'final_test_details.csv'}")
    print(f"  {RESULT_DIR / 'final_test_no_answer_scores.csv'}")


if __name__ == "__main__":
    main()
