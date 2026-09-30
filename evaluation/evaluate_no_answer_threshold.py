"""
근거 없음(no-answer) 거절 threshold를 validation 데이터로 정한다.

절대 최종 test set(final_test.jsonl)의 no_answer 10문항으로 threshold를
정하면 안 된다 — 그러면 같은 데이터로 기준을 정하고 검증까지 하는
순환 논리가 된다. 그래서 여기서는:

  1. validation 70문항(정답 있음)의 hybrid top-1 점수 분포
  2. validation_no_answer.jsonl 10문항(정답 없음, final_test의 10개와
     겹치지 않는 새 주제로 구성 — 카지노/검투사/요일던전 등 raw_docs.jsonl에
     실제로 없는 것을 grep으로 확인한 것들)의 hybrid top-1 점수 분포

두 분포를 비교해서 threshold 후보를 제안한다. 이후 이 threshold를
final_test.jsonl의 no_answer 10문항(evaluate_final_test.py가 이미 만든
results/final_test_no_answer_scores.csv)에 적용해 "실제로 몇 개나 정상
거절했는가"를 채점하는 건 별도 스크립트/단계다 (여기서는 threshold만 정한다).

services/qa.py의 기존 거절 로직(절대 하한 0.28 + 키워드 무매칭 시 상향
하한 0.45)과는 점수 스케일 자체가 다르다 (qa.py는 문서 단위 병합 +
토큰당 +0.25 키워드 보정 + 출처별 1.2~1.25배 가중치를 쓴다). 구조(절대
하한 + 키워드 무매칭 시 상향 하한)는 참고하되 숫자는 이 프로젝트의
hybrid_scores 스케일(KEYWORD_BOOST_WEIGHT=0.05)에 맞게 새로 정한다.

실행:
    python evaluation/evaluate_no_answer_threshold.py
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from evaluate_retrieval import (  # noqa: E402
    CORPUS_PATH,
    KEYWORD_BOOST_WEIGHT,
    QUESTION_PATH,
    RESULT_DIR,
    cosine_scores,
    get_embedder,
    keyword_scores,
    load_jsonl,
    rank_indices,
)

NO_ANSWER_PATH = BASE_DIR / "data" / "validation_no_answer.jsonl"


def top1_hybrid_score(question_text, corpus, document_embeddings, embedder):

    kw_scores = keyword_scores(question_text, corpus)

    query_embedding = embedder.encode(question_text, convert_to_numpy=True)
    sem_scores = cosine_scores(query_embedding, document_embeddings)

    keyword_boost = np.minimum(kw_scores, 3) * KEYWORD_BOOST_WEIGHT
    hybrid_scores = sem_scores + keyword_boost

    ranking = rank_indices(hybrid_scores)
    top1_idx = ranking[0]

    return float(hybrid_scores[top1_idx]), corpus[top1_idx]["chunk_id"]


def describe(values):
    values = sorted(values)
    return {
        "n": len(values),
        "min": round(values[0], 4),
        "max": round(values[-1], 4),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def main():

    corpus = load_jsonl(CORPUS_PATH)
    answerable = load_jsonl(QUESTION_PATH)
    no_answer = load_jsonl(NO_ANSWER_PATH)

    print("문서 Chunk 수:", len(corpus))
    print("validation 답변 가능 문항:", len(answerable))
    print("validation 무응답 문항:", len(no_answer))
    print("keyword_boost:", KEYWORD_BOOST_WEIGHT, "(고정, 재조정 없음)")

    embedder = get_embedder()

    corpus_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
    document_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True)

    answerable_rows = []
    for q in answerable:
        score, chunk_id = top1_hybrid_score(q["question"], corpus, document_embeddings, embedder)
        answerable_rows.append({
            "id": q["id"], "question": q["question"],
            "top1_chunk": chunk_id, "top1_score": round(score, 4),
        })

    no_answer_rows = []
    for q in no_answer:
        score, chunk_id = top1_hybrid_score(q["question"], corpus, document_embeddings, embedder)
        no_answer_rows.append({
            "id": q["id"], "question": q["question"],
            "top1_chunk": chunk_id, "top1_score": round(score, 4),
        })

    answerable_scores = [r["top1_score"] for r in answerable_rows]
    no_answer_scores = [r["top1_score"] for r in no_answer_rows]

    answerable_stats = describe(answerable_scores)
    no_answer_stats = describe(no_answer_scores)

    print("\n===== 답변 가능(70문항) top-1 점수 분포 =====")
    print(answerable_stats)

    print("\n===== 무응답(10문항) top-1 점수 분포 =====")
    print(no_answer_stats)

    overlap = no_answer_stats["max"] >= answerable_stats["min"]

    print("\n===== 겹침 여부 =====")
    if overlap:
        print(
            f"겹침 있음: 무응답 최고점({no_answer_stats['max']}) >= "
            f"답변가능 최저점({answerable_stats['min']}) "
            f"-> 단순 고정 threshold로는 완벽히 분리되지 않음"
        )
    else:
        print(
            f"완전히 분리됨: 무응답 최고점({no_answer_stats['max']}) < "
            f"답변가능 최저점({answerable_stats['min']})"
        )

    # 후보 threshold: 두 분포 사이 중간값 (완전 분리가 아니면 참고용일 뿐)
    candidate = round((no_answer_stats["max"] + answerable_stats["min"]) / 2, 4)
    print(f"\n후보 threshold (두 분포 경계 중간값): {candidate}")
    print("이 값으로 각 그룹을 나눠봤을 때:")

    fp = sum(1 for s in no_answer_scores if s >= candidate)  # 근거 없는데 통과(오탐)
    fn = sum(1 for s in answerable_scores if s < candidate)  # 근거 있는데 거절(누락)
    print(f"  - 무응답 중 threshold를 넘겨버리는 경우(오탐): {fp}/{len(no_answer_scores)}")
    print(f"  - 답변 가능 중 threshold 밑으로 떨어지는 경우(불필요한 거절): {fn}/{len(answerable_scores)}")

    RESULT_DIR.mkdir(exist_ok=True)

    with open(RESULT_DIR / "no_answer_threshold_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "keyword_boost_weight": KEYWORD_BOOST_WEIGHT,
            "answerable_stats": answerable_stats,
            "no_answer_stats": no_answer_stats,
            "distributions_overlap": overlap,
            "candidate_threshold": candidate,
            "candidate_threshold_false_positive_on_no_answer": fp,
            "candidate_threshold_false_negative_on_answerable": fn,
        }, f, ensure_ascii=False, indent=2)

    with open(RESULT_DIR / "validation_answerable_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "top1_chunk", "top1_score"])
        writer.writeheader()
        writer.writerows(answerable_rows)

    with open(RESULT_DIR / "validation_no_answer_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "top1_chunk", "top1_score"])
        writer.writeheader()
        writer.writerows(no_answer_rows)

    print(f"\n결과 저장 완료:")
    print(f"  {RESULT_DIR / 'no_answer_threshold_analysis.json'}")
    print(f"  {RESULT_DIR / 'validation_answerable_scores.csv'}")
    print(f"  {RESULT_DIR / 'validation_no_answer_scores.csv'}")


if __name__ == "__main__":
    main()
