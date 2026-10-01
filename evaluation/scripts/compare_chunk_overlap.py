"""
실패 분석(retrieval_details.csv)에서 "청크 경계" 문제로 분류된 6개 질문을 포함해,
overlap=0(기존) vs overlap=100 청킹을 같은 조건에서 비교한다.

주의: overlap을 주면 chunk 번호(chunk_id)가 밀리기 때문에 questions.jsonl의
relevant_chunk_ids(overlap=0 기준)를 그대로 재사용할 수 없다. 그래서 정답 판정은
chunk_id 문자열 비교 대신, baseline(overlap=0) corpus에서 정답 chunk의 "중간
60자 스니펫"을 뽑아 그 스니펫을 포함하는 chunk를 정답으로 보는 방식을 쓴다.
(overlap을 얼마를 주든 원문 중간 부분은 어느 한 chunk에 온전히 포함되어 있을
확률이 매우 높다는 성질을 이용한다.)

실행:
    python evaluation/compare_chunk_overlap.py
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from build_corpus import RAW_PATH, chunk_text  # noqa: E402
from evaluate_retrieval import (  # noqa: E402
    KEYWORD_BOOST_WEIGHT,
    QUESTION_PATH,
    RESULT_DIR,
    cosine_scores,
    get_embedder,
    keyword_scores,
    load_jsonl,
    rank_indices,
)

# 실패 분석에서 "청크 경계"로 분류한 6개 질문 id
# (볼다이크 입장레벨/쿠르잔 남부 입장레벨/혼돈의 상아탑 인원수/
#  가르가디스 버그/카제로스 4막 하드-paraphrase/볼다이크-paraphrase)
BOUNDARY_FAILURE_IDS = {"q021", "q024", "q029", "q032", "q045", "q049"}

ANCHOR_LEN = 60


def build_corpus(raw_docs, max_len, overlap):

    corpus = []

    for doc in raw_docs:

        chunks = chunk_text(doc["text"], max_len=max_len, overlap=overlap)

        for index, chunk in enumerate(chunks):

            corpus.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f'{doc["doc_id"]}#{index + 1}',
                "title": doc["title"],
                "source": doc.get("source", "unknown"),
                "text": chunk
            })

    return corpus


def anchor_snippet(text: str) -> str:

    text = text.strip()

    if len(text) <= ANCHOR_LEN:
        return text

    mid = len(text) // 2
    half = ANCHOR_LEN // 2

    return text[mid - half: mid + half]


def evaluate_corpus(corpus, questions, anchors_by_qid, embedder):
    """
    anchors_by_qid: {question_id: [anchor_snippet, ...]} (baseline에서 추출한 정답 스니펫)
    반환: (hit3_by_qid, overall_hit3)
    """

    corpus_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
    document_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True)

    hit3_by_qid = {}

    for q in questions:

        anchors = anchors_by_qid[q["id"]]

        kw_scores = keyword_scores(q["question"], corpus)

        query_embedding = embedder.encode(q["question"], convert_to_numpy=True)
        sem_scores = cosine_scores(query_embedding, document_embeddings)

        keyword_boost = np.minimum(kw_scores, 3) * KEYWORD_BOOST_WEIGHT
        hybrid_scores = sem_scores + keyword_boost

        ranking = rank_indices(hybrid_scores)
        top3 = ranking[:3]

        hit = 0
        for idx in top3:
            candidate_text = corpus[idx]["text"]
            if any(anchor and anchor in candidate_text for anchor in anchors):
                hit = 1
                break

        hit3_by_qid[q["id"]] = hit

    overall_hit3 = float(np.mean(list(hit3_by_qid.values())))

    return hit3_by_qid, overall_hit3


def main():

    raw_docs = load_jsonl(RAW_PATH)
    questions = load_jsonl(QUESTION_PATH)

    print("원본 문서 수:", len(raw_docs))
    print("평가 질문 수:", len(questions))

    # baseline(overlap=0) corpus를 만들어 정답 chunk의 anchor 스니펫을 추출
    baseline_corpus = build_corpus(raw_docs, max_len=600, overlap=0)
    baseline_by_id = {c["chunk_id"]: c for c in baseline_corpus}

    anchors_by_qid = {}
    for q in questions:
        anchors_by_qid[q["id"]] = [
            anchor_snippet(baseline_by_id[cid]["text"])
            for cid in q["relevant_chunk_ids"]
            if cid in baseline_by_id
        ]

    embedder = get_embedder()

    print("\n[1/2] overlap=0 (baseline) 평가 중...")
    hit3_baseline, overall_baseline = evaluate_corpus(
        baseline_corpus, questions, anchors_by_qid, embedder
    )

    overlap_corpus = build_corpus(raw_docs, max_len=600, overlap=100)

    print(f"[2/2] overlap=100 평가 중... (chunk 수: {len(overlap_corpus)}, baseline: {len(baseline_corpus)})")
    hit3_overlap, overall_overlap = evaluate_corpus(
        overlap_corpus, questions, anchors_by_qid, embedder
    )

    print("\n===== 전체 Hit@3 비교 =====")
    print(f"overlap=0   : {overall_baseline:.3f}")
    print(f"overlap=100 : {overall_overlap:.3f}")

    print("\n===== '청크 경계' 실패 6문항 개별 비교 =====")
    print(f"{'id':6s} {'overlap=0':>10s} {'overlap=100':>12s}")
    fixed = 0
    for qid in sorted(BOUNDARY_FAILURE_IDS):
        b = hit3_baseline.get(qid, -1)
        o = hit3_overlap.get(qid, -1)
        marker = ""
        if b == 0 and o == 1:
            marker = "  <- 개선됨"
            fixed += 1
        print(f"{qid:6s} {b:10d} {o:12d}{marker}")

    print(f"\n6문항 중 {fixed}개가 overlap=100에서 새로 hit3=1이 됨")

    RESULT_DIR.mkdir(exist_ok=True)
    out_path = RESULT_DIR / "chunk_overlap_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": {"overlap_0": overall_baseline, "overlap_100": overall_overlap},
            "boundary_failure_questions": {
                qid: {"overlap_0": hit3_baseline.get(qid), "overlap_100": hit3_overlap.get(qid)}
                for qid in sorted(BOUNDARY_FAILURE_IDS)
            },
            "per_question": {
                q["id"]: {"overlap_0": hit3_baseline[q["id"]], "overlap_100": hit3_overlap[q["id"]]}
                for q in questions
            },
        }, f, ensure_ascii=False, indent=2)

    print(f"\n결과를 {out_path} 에 저장했습니다.")


if __name__ == "__main__":
    main()
