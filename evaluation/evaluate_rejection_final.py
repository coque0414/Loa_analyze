"""
근거 없음 거절 threshold를 최종 동결하고, Final Test Set(50문항)에 적용해
"몇 개나 정상적으로 거절/수용했는가"를 측정한다.

NO_ANSWER_THRESHOLD = 0.60 은 validation 데이터(questions.jsonl 70문항 +
validation_no_answer.jsonl 10문항)로만 정했다. evaluate_no_answer_threshold.py가
제안한 중간값(0.6139)보다 이 값이 더 낫다 — 같은 validation 오탐률
(no-answer 10개 중 1개 통과)에서 오차(정답인데 거절되는 답변 가능 질문 수)가
4/70이 아니라 3/70으로 더 적다:

    threshold   validation FP(무응답 통과)   validation FN(정답인데 거절)
    0.55        4/10                         1/70
    0.60        1/10                         3/70   <- 채택
    0.6139      1/10                         4/70
    0.65        1/10                         11/70
    0.70        0/10                         18/70

0.70처럼 더 엄격한 값은 무응답 오탐을 0으로 만들지만 정답 있는 질문의
18/70(26%)을 잘못 거절하게 돼서 실사용성이 떨어진다. 0.60이 두 오류의
합리적인 절충점이다.

이 threshold는 여기서 확정하고 다시 튜닝하지 않는다. final_test.jsonl
(validation과 chunk 100% 분리된 50문항)에 그대로 적용해 최종 성능만 잰다.

실행:
    python evaluation/evaluate_rejection_final.py
"""

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
    keyword_scores,
    load_jsonl,
    rank_indices,
)

FINAL_TEST_PATH = BASE_DIR / "data" / "final_test.jsonl"

# validation에서 확정, 여기서부터는 재조정하지 않는다.
NO_ANSWER_THRESHOLD = 0.60


def main():

    corpus = load_jsonl(CORPUS_PATH)
    questions = load_jsonl(FINAL_TEST_PATH)

    answerable = [q for q in questions if q["type"] != "no_answer"]
    no_answer = [q for q in questions if q["type"] == "no_answer"]

    print("문서 Chunk 수:", len(corpus))
    print("Final Test:", len(answerable), "답변 가능 /", len(no_answer), "무응답")
    print("적용 threshold:", NO_ANSWER_THRESHOLD, "(validation에서 확정, 재조정 없음)")
    print("keyword_boost:", KEYWORD_BOOST_WEIGHT)

    embedder = get_embedder()
    corpus_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
    document_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True)

    def top1_score(question_text):
        kw_scores = keyword_scores(question_text, corpus)
        query_embedding = embedder.encode(question_text, convert_to_numpy=True)
        sem_scores = cosine_scores(query_embedding, document_embeddings)
        keyword_boost = np.minimum(kw_scores, 3) * KEYWORD_BOOST_WEIGHT
        hybrid_scores = sem_scores + keyword_boost
        ranking = rank_indices(hybrid_scores)
        return float(hybrid_scores[ranking[0]])

    # 답변 가능 40문항: threshold를 넘겨서 "정상적으로 답변 시도"해야 성공
    accepted = 0
    wrongly_rejected = []
    for q in answerable:
        score = top1_score(q["question"])
        if score >= NO_ANSWER_THRESHOLD:
            accepted += 1
        else:
            wrongly_rejected.append({"question": q["question"], "score": round(score, 4)})

    # 무응답 10문항: threshold 밑으로 떨어져서 "정상적으로 거절"해야 성공
    rejected = 0
    wrongly_accepted = []
    for q in no_answer:
        score = top1_score(q["question"])
        if score < NO_ANSWER_THRESHOLD:
            rejected += 1
        else:
            wrongly_accepted.append({"question": q["question"], "score": round(score, 4)})

    print("\n===== Final Test: 근거 없음 거절 최종 성능 =====")
    print(f"답변 가능 40문항 중 정상 수용: {accepted}/{len(answerable)}")
    print(f"무응답 10문항 중 정상 거절:   {rejected}/{len(no_answer)}")

    if wrongly_rejected:
        print("\n[오차] 정답이 있는데 거절된 질문:")
        for w in wrongly_rejected:
            print(f"  - {w['question']} (score={w['score']})")

    if wrongly_accepted:
        print("\n[한계] 근거가 없는데 통과된 질문 (여전히 hallucination 위험):")
        for w in wrongly_accepted:
            print(f"  - {w['question']} (score={w['score']})")

    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / "final_rejection_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "threshold": NO_ANSWER_THRESHOLD,
            "answerable_accepted": accepted,
            "answerable_total": len(answerable),
            "no_answer_rejected": rejected,
            "no_answer_total": len(no_answer),
            "wrongly_rejected": wrongly_rejected,
            "wrongly_accepted": wrongly_accepted,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {RESULT_DIR / 'final_rejection_result.json'}")


if __name__ == "__main__":
    main()
