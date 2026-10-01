"""
Generation 평가 (8단계): Final Test 중 25문항(무응답 10개 전부 + 답변 가능
15개 표본)에 대해 실제로 답변을 만들어보고, 사람이 직접 3가지 기준으로
채점할 수 있게 CSV로 저장한다.

  - 정답 핵심정보 포함 여부
  - 제공 문서 밖의 추측 여부
  - 답 없는 질문에 대한 적절한 거절 여부

RAGAS 같은 프레임워크는 쓰지 않는다 — 기준을 직접 정의하고 사람이 확인하는
쪽이 지금 단계에 맞다(계획서 8단계 그대로).

이 스크립트는 services/qa.py의 rag_qa()를 그대로 재사용하지 않는다. qa.py는
MongoDB(docs_col/guide_col)에서 라이브 앱 데이터를 읽어오도록 짜여 있어서
이 evaluation/ 파이프라인의 corpus.jsonl과 연결되어 있지 않다. 대신 여기서는
qa.py와 동일한 판정 구조(step 7에서 확정한 hybrid retrieval + threshold=0.60)와
동일한 시스템/유저 프롬프트, 동일한 거절 문구를 그대로 재사용해서 "실제
프로덕션 코드가 이 corpus로 동작했다면 어떻게 답했을지"에 최대한 가깝게
재현한다.

정답/근거 여부는 채점하지 않는다(그건 이미 evaluate_final_test.py /
evaluate_rejection_final.py가 했다) — 여기서는 오직 LLM이 실제로 만들어낸
답변 텍스트 자체의 품질만 사람이 보고 판단한다.

실행 전:
    - OPENAI_API_KEY 또는 openai_api_key 환경변수(.env)에 유효한 키가 있어야 함
    - pip install openai (requirements.txt에 이미 포함)

실행:
    python evaluation/evaluate_generation.py
"""

import csv
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # evaluation/
sys.path.insert(0, str(SCRIPT_DIR))

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
from evaluate_rejection_final import NO_ANSWER_THRESHOLD  # noqa: E402

FINAL_TEST_PATH = BASE_DIR / "data" / "final_test.jsonl"

REJECTION_MESSAGE = (
    "지금 보유한 문서(공지/가이드 등)에서 확실한 근거를 찾지 못했어요. \n"
    "또 다른 질문이 있다면 언제든지 환영입니다!"
)

# services/qa.py의 rag_qa()와 동일한 프롬프트를 그대로 재사용한다.
SYSTEM_PROMPT = textwrap.dedent('''
너는 로스트아크 전문 어시스턴트야. 질문에 정확하고 간결하게 답변해줘.

**핵심 원칙:**
1. 제공된 문서 정보만 사용 (추측 절대 금지)
2. 질문에 직접 답하는 내용만 작성
3. 2-3문장으로 간결하게
4. 핵심 정보만 전달

**금지 사항:**
- 문서에 없는 내용 추측 금지
- 장황한 배경 설명 금지
- 불필요한 부연 설명 금지
- "문서에 따르면" 같은 메타 표현 금지
''')


def build_user_prompt(context: str, question: str) -> str:
    return textwrap.dedent(f'''
    [참고 자료]
    {context}

    [사용자 질문]
    {question}

    [지시사항]
    위 참고 자료를 바탕으로 질문에 핵심만 3-4문장으로 간결하게 답변해줘.
    문서에 있는 정보만 사용하고, 질문에 직접 답하는 내용만 포함해줘.
    ''')


def select_sample(all_questions):

    no_answer = [q for q in all_questions if q["type"] == "no_answer"]

    sample = list(no_answer)  # 10개 전부

    for qtype in ("general", "paraphrase", "disambiguation"):
        picked = [q for q in all_questions if q["type"] == qtype][:5]
        sample.extend(picked)

    return sample  # 10 + 5*3 = 25


def main():

    load_dotenv()

    api_key = os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "[ERROR] openai_api_key(또는 OPENAI_API_KEY) 환경변수가 없습니다. "
            ".env에 키를 설정한 뒤 다시 실행하세요."
        )
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)

    corpus = load_jsonl(CORPUS_PATH)
    all_questions = load_jsonl(FINAL_TEST_PATH)
    sample = select_sample(all_questions)

    print("문서 Chunk 수:", len(corpus))
    print("Generation 평가 표본:", len(sample), "개 (무응답 10 + 유형별 5)")
    print("적용 threshold:", NO_ANSWER_THRESHOLD, "/ keyword_boost:", KEYWORD_BOOST_WEIGHT)

    embedder = get_embedder()
    corpus_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
    document_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True)

    rows = []

    for i, q in enumerate(sample, start=1):

        kw_scores = keyword_scores(q["question"], corpus)
        query_embedding = embedder.encode(q["question"], convert_to_numpy=True)
        sem_scores = cosine_scores(query_embedding, document_embeddings)
        keyword_boost = np.minimum(kw_scores, 3) * KEYWORD_BOOST_WEIGHT
        hybrid_scores = sem_scores + keyword_boost
        ranking = rank_indices(hybrid_scores)

        top1_score = float(hybrid_scores[ranking[0]])
        top3_idx = ranking[:3]
        top3_chunks = [corpus[idx]["chunk_id"] for idx in top3_idx]

        if top1_score < NO_ANSWER_THRESHOLD:
            decision = "rejected"
            answer = REJECTION_MESSAGE
        else:
            decision = "accepted"
            context = "\n".join(f'- "{corpus[idx]["text"]}"' for idx in top3_idx)
            try:
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(context, q["question"])},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )
                answer = res.choices[0].message.content.strip()
            except Exception as e:
                answer = f"[ERROR] OpenAI 호출 실패: {e}"

        print(f"[{i}/{len(sample)}] ({q['type']}, {decision}, score={top1_score:.3f}) {q['question']}")

        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "relevant_chunk_ids": ", ".join(q.get("relevant_chunk_ids", [])),
            "top1_score": round(top1_score, 4),
            "decision": decision,
            "retrieved_top3": ", ".join(top3_chunks),
            "generated_answer": answer,
            "핵심정보_포함(1/0)": "",
            "문서밖_추측(1/0)": "",
            "거절_적절성(해당시,1/0)": "",
            "비고": "",
        })

    RESULT_DIR.mkdir(exist_ok=True)
    out_path = RESULT_DIR / "generation_eval.csv"

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n결과 저장: {out_path}")
    print("핵심정보_포함 / 문서밖_추측 / 거절_적절성 컬럼은 직접 채워서 검토하면 됩니다.")


if __name__ == "__main__":
    main()
