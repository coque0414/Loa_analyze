"""
evaluate_generation.py가 만든 results/generation_eval.csv의 25개 답변을
직접 읽고 채점한 결과를 코드로 고정한다.

이 스크립트는 LLM-as-a-judge나 RAGAS를 쓰지 않는다 — 계획서 8단계 그대로,
세 가지 기준을 사람이 직접 읽고 판정한다:

  - 핵심정보_포함:   relevant_chunk_ids의 핵심 사실이 답변에 들어있는가
                     (no_answer 질문은 해당 없음 = "")
  - 문서밖_추측:     문서에 없는 내용을 사실처럼 만들어냈는가 (1=추측함/나쁨, 0=안함/좋음)
  - 거절_적절성:     "근거 없음" 판정(하드 거절이든, LLM이 스스로 답변을 거부한 경우든)이
                     타당했는가 (1=적절, 0=부적절 — 즉 답이 있는데 거절했거나,
                     거절했어야 하는데 답을 꾸며냄)

판정 근거(비고)도 같이 남긴다 — 특히 두 가지 핵심 발견:

  1. f042, f049: threshold(0.60)를 통과해 LLM까지 넘어갔지만, 시스템 프롬프트의
     "추측 절대 금지" 지시 덕분에 LLM이 스스로 "정보 없음"이라고 답함.
     retrieval score threshold가 1차 방어선이고 생성 프롬프트가 2차 방어선으로
     작동한 사례.
  2. f047: 유일한 진짜 실패. 존재하지 않는 "랭킹전"에 대한 질문에, 문서 안에 실제로
     있는 다른 개념("The FIRST 이벤트")을 가져와 마치 정답인 것처럼 답변함.
     개별 사실은 문서 안에 있지만 질문 자체를 다른 것으로 바꿔치기한 hallucination.
  3. f016, f019, f032: 전부 실제 정답이 존재하는데 threshold 미달로 오거절됨 —
     final_test 전체에서 반복 확인된 "validation→final test 일반화 격차"가
     생성 단계에서도 그대로 나타난 것.

실행:
    python evaluation/scripts/grade_generation_eval.py
    (results/generation_eval.csv가 먼저 있어야 한다 — evaluate_generation.py 실행 결과)
"""

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # evaluation/
RESULT_DIR = BASE_DIR / "results"

IN_PATH = RESULT_DIR / "generation_eval.csv"
OUT_PATH = RESULT_DIR / "generation_eval_graded.csv"

# id -> (핵심정보_포함, 문서밖_추측, 거절_적절성, 비고)
# 핵심정보_포함은 no_answer 질문에는 해당 없음("")
GRADES = {
    # ---- no_answer 10개 ----
    "f041": ("", 0, 1, "임계값 미달로 하드 거절 - 적절"),
    "f042": ("", 0, 1, "threshold는 통과했지만 LLM이 '5막 정보 없음'이라고 스스로 정정 - 생성 프롬프트가 2차 방어선으로 작동"),
    "f043": ("", 0, 1, "임계값 미달로 하드 거절 - 적절"),
    "f044": ("", 0, 1, "임계값 미달로 하드 거절 - 적절"),
    "f045": ("", 0, 1, "임계값 미달로 하드 거절 - 적절"),
    "f046": ("", 0, 1, "임계값 미달로 하드 거절 - 적절"),
    "f047": ("", 1, 0, "유일한 진짜 실패: 존재하지 않는 '랭킹전'에 실제 존재하는 'The FIRST 이벤트' 내용을 가져와 정답인 것처럼 답변 - 개별 사실은 문서 안에 있지만 질문을 바꿔치기함"),
    "f048": ("", 0, 1, "임계값 미달로 하드 거절 - 적절"),
    "f049": ("", 0, 1, "threshold는 통과했지만 LLM이 '2027 신규 직업 정보 없음'이라고 스스로 정정"),
    "f050": ("", 0, 1, "임계값 미달로 하드 거절 - 적절"),
    # ---- 답변 가능 15개 ----
    "f001": (1, 0, "", "핵심정보 정확"),
    "f002": (1, 0, "", "핵심정보 정확"),
    "f003": (1, 0, "", "핵심정보 정확"),
    "f004": (0, 0, "", "검색이 틀린 chunk를 가져왔지만(hit3=0), LLM은 없는 정보(6주)를 추측하지 않고 '명시되어 있지 않다'고 정직하게 답함 - 추측은 없었으나 핵심정보 누락"),
    "f005": (1, 0, "", "핵심정보 정확"),
    "f016": (0, 0, 0, "오거절: 실제 정답(gmnote_1225#3)이 top3에 있었는데 threshold 미달로 거절됨"),
    "f017": (1, 0, "", "핵심정보 정확"),
    "f018": (1, 0, "", "핵심정보 정확, 노말/하드 차이까지 정확히 비교"),
    "f019": (0, 0, 0, "오거절: f004와 동일 사실(gmnote_1225#5)인데 이번엔 하드 거절까지 됨"),
    "f020": (1, 0, "", "핵심정보 정확"),
    "f031": (1, 0, "", "핵심정보 정확 - 유사 가격(7,000/15,000/19,800) 중 정확히 구분"),
    "f032": (0, 0, 0, "오거절: 실제 정답(15,000)이 존재하는데 거절됨 (25문항 중 최저 점수 0.4157)"),
    "f033": (1, 0, "", "핵심정보 정확"),
    "f034": (1, 0, "", "핵심정보 정확 - 40시간/100시간 보상 구분 성공"),
    "f035": (1, 0, "", "핵심정보 정확"),
}


def main():

    if not IN_PATH.exists():
        print(f"[ERROR] {IN_PATH} 가 없습니다. 먼저 evaluate_generation.py를 실행하세요.")
        sys.exit(1)

    with open(IN_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    missing_grades = [r["id"] for r in rows if r["id"] not in GRADES]
    if missing_grades:
        print(f"[WARN] 채점표에 없는 id: {missing_grades} (빈 값으로 둠)")

    for row in rows:
        grade = GRADES.get(row["id"])
        if grade:
            info, guess, reject_ok, note = grade
            row["핵심정보_포함(1/0)"] = info
            row["문서밖_추측(1/0)"] = guess
            row["거절_적절성(해당시,1/0)"] = reject_ok
            row["비고"] = note

    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- 요약 집계 ----
    no_answer_rows = [r for r in rows if r["type"] == "no_answer"]
    answerable_rows = [r for r in rows if r["type"] != "no_answer"]

    hallucinations = sum(
        1 for r in rows
        if GRADES.get(r["id"]) and GRADES[r["id"]][1] == 1
    )

    no_answer_handled_safely = sum(
        1 for r in no_answer_rows
        if GRADES.get(r["id"]) and GRADES[r["id"]][2] == 1
    )

    wrongly_rejected = sum(
        1 for r in answerable_rows
        if GRADES.get(r["id"]) and GRADES[r["id"]][2] == 0
    )

    correct_answerable = sum(
        1 for r in answerable_rows
        if GRADES.get(r["id"]) and GRADES[r["id"]][0] == 1
    )

    print("===== Generation 평가 요약 (25문항 수동 채점) =====")
    print(f"Hallucination(문서 밖 추측) 발생: {hallucinations}/25")
    print(f"무응답 질문 중 안전하게 처리됨(하드 거절 + LLM 자체 정정): {no_answer_handled_safely}/{len(no_answer_rows)}")
    print(f"답변 가능 질문 중 핵심정보 정확히 포함: {correct_answerable}/{len(answerable_rows)}")
    print(f"답변 가능 질문 중 오거절(threshold 문제): {wrongly_rejected}/{len(answerable_rows)}")
    print(f"\n결과 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
