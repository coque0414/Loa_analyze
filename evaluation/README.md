# RAG Retrieval 평가

졸업 프로젝트 당시 운영하던 MongoDB 기반 RAG 파이프라인은 프로젝트 종료 후
비용 문제로 DB가 삭제되어 더 이상 존재하지 않는다. 이 폴더는 당시의 검색
파이프라인(KoSimCSE 임베딩 + 키워드 보정 Hybrid Search)을 로스트아크 공식
사이트의 실제 공지사항/GM노트로 재현하고, Retrieval 성능과 "근거 없음" 거절
성능을 데이터로 검증한 기록이다.

## 데이터 흐름

```
공식 문서 24건 (lostark.game.onstove.com Notice/GMNote, 원문 URL·수집일 포함)
  -> 600자 단위 chunking (overlap 없음)
  -> 214개 chunk
  -> Validation 질문 70개 (튜닝용)
  -> Final Test 질문 50개 (검증용, validation과 chunk 100% 분리)
```

- `data/raw_docs.jsonl` — 크롤링한 원문 24건. 각 문서에 `url`, `crawled_at` 보존.
- `data/corpus.jsonl` — `build_corpus.py`가 생성하는 600자 chunk (gitignore 대상,
  `raw_docs.jsonl`에서 언제든 재생성 가능).
- `data/questions.jsonl` — **Validation 70문항**. 일반 33 / paraphrase 21 /
  disambiguation 16. Keyword boost, chunk 설정 등 모든 튜닝은 이 70문항으로만 했다.
- `data/final_test.jsonl` — **Final Test 50문항**. 일반 15 / paraphrase 15 /
  disambiguation 10 / 무응답(no_answer) 10. validation이 사용한 chunk는 단
  하나도 겹치지 않도록 구성했고, 이 셋을 보고는 설정을 다시 바꾸지 않았다.
- `data/validation_no_answer.jsonl` — 거절 threshold를 정하기 위한 validation
  전용 무응답 10문항 (final_test의 10개와 별개 주제).
- `sample_data/` — 실제 크롤링 이전에 썼던 합성 20문서/45문항 스모크 테스트
  데이터. 참고용으로 남겨둠 (더 이상 사용하지 않음).

## 폴더 구조

```
evaluation/
├─ data/      입력 데이터 (원문, 질문셋)
├─ scripts/   평가 파이프라인 스크립트
└─ results/   스크립트 실행 결과 (커밋됨 — 재실행 없이 결과 확인 가능)
```

## 평가 방법 및 재현 순서

모든 스크립트는 기존 프로젝트의 `services/embedder.py`(KoSimCSE-roberta-multitask)를
그대로 재사용한다.

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| 1. Chunking | `scripts/build_corpus.py` | `data/corpus.jsonl` |
| 2. Validation 기본 성능 | `scripts/evaluate_retrieval.py` | `results/summary.json`, `results/retrieval_details.csv` |
| 3. Keyword boost 튜닝 | `scripts/sweep_keyword_boost.py` | `results/keyword_boost_sweep.csv` |
| 4. Chunk overlap 실험 | `scripts/compare_chunk_overlap.py` | `results/chunk_overlap_comparison.json` |
| 5. Final Test 성능 | `scripts/evaluate_final_test.py` | `results/final_test_summary.json`, `results/final_test_details.csv` |
| 6. 거절 threshold 결정 | `scripts/evaluate_no_answer_threshold.py` | `results/no_answer_threshold_analysis.json`, `results/validation_*_scores.csv` |
| 7. 거절 최종 검증 | `scripts/evaluate_rejection_final.py` | `results/final_rejection_result.json` |
| 8. Generation 평가 | `scripts/evaluate_generation.py` → `scripts/grade_generation_eval.py` | `results/generation_eval.csv`, `results/generation_eval_graded.csv` |

재현하려면:
```bash
pip install torch transformers numpy beautifulsoup4 requests openai python-dotenv
python evaluation/scripts/crawl_docs.py          # (선택) 원문 재수집
python evaluation/scripts/build_corpus.py
python evaluation/scripts/evaluate_retrieval.py
python evaluation/scripts/sweep_keyword_boost.py
python evaluation/scripts/compare_chunk_overlap.py
python evaluation/scripts/evaluate_final_test.py
python evaluation/scripts/evaluate_no_answer_threshold.py
python evaluation/scripts/evaluate_rejection_final.py
python evaluation/scripts/evaluate_generation.py  # .env에 openai_api_key 필요
python evaluation/scripts/grade_generation_eval.py
```

---

## 결과

### 1) Validation (70문항) — 튜닝 과정, 최종 성능 아님

| 방식 | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| Keyword | 0.500 | 0.800 | 0.663 |
| Semantic | 0.357 | 0.586 | 0.514 |
| **Hybrid (boost=0.05)** | **0.543** | **0.800** | **0.685** |

**Keyword boost**: 0(순수 semantic)에서 0.05로 올리자 Hit@3가 58.6% → 80.0%로
가장 크게 뛰었고, 그 이상(0.10~0.30)은 평평하거나 오히려 소폭 하락했다.
"많이 섞을수록 좋다"가 아니라 "적당히만 섞으면 충분하다"는 결론이라
**boost=0.05로 동결**했다.

**Chunk overlap**: 실패 70문항 중 14건을 분석해 `비슷한 문서끼리 경쟁`(8건,
예: 카제로스 레이드 여러 막의 "아이템 레벨 XXXX 이상" 템플릿 문장이 거의
동일해 숫자로만 구분해야 함) vs `청크 경계`(6건, 같은 문서 안의 인접 chunk
혼동)로 분류했다. 경계 문제를 해결하기 위해 overlap=100을 실험했지만
**Hit@3가 81.4% → 72.9%로 오히려 악화**(겨냥한 경계 문제 6건 중 1건만
개선되고, 멀쩡하던 9개 질문이 새로 깨짐) — 이 데이터셋의 근본 문제는
경계가 아니라 "경쟁"이었다는 뜻이라 **overlap 없이 그대로 동결**했다.

**최종 동결 설정**: KoSimCSE 임베딩 + keyword boost 0.05 + chunk 600자/overlap
없음 + Top-3.

### 2) Final Test (50문항: 답변 가능 40 + 무응답 10) — 실제 최종 성능

Validation에서 확정한 설정을 그대로 적용, 여기서는 더 이상 튜닝하지 않았다.

| 방식 | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| Keyword | 0.500 | 0.750 | 0.656 |
| Semantic | 0.200 | 0.475 | 0.356 |
| **Hybrid** | **0.400** | **0.725** | **0.575** |

**핵심 발견**: Hybrid가 Keyword를 확실히 이기지 못한다(오히려 근소하게 낮음).
Validation·Final Test 두 독립된 질문셋에서 동일하게 재현된 결과라 우연이
아니다. 이 도메인(반복적인 템플릿 구조의 패치노트)에서는 Keyword 검색이
예상보다 강력했고, 범용 Semantic 임베딩은 거의 동일한 문장에서 숫자·고유
명사 하나만 다른 chunk를 구분하는 데 약했다. Hybrid는 Semantic 단독보다는
크게 개선하지만 Keyword 대비 뚜렷한 우위는 없다 — "Hybrid가 항상 이긴다"는
식으로 포장하지 않고 있는 그대로 보고한다.

### 3) 근거 없음(No-answer) 거절

Threshold는 validation 데이터로만 정했다(70 답변 가능 + validation 전용
무응답 10문항). 후보 비교:

| threshold | 무응답 오탐 | 정답인데 거절 |
|---|---|---|
| 0.55 | 4/10 | 1/70 |
| **0.60 (채택)** | **1/10** | **3/70** |
| 0.6139 (수식상 중간값) | 1/10 | 4/70 |
| 0.70 | 0/10 | 18/70 |

0.60은 0.70처럼 무응답을 완전히 걸러내진 못하지만(실제 존재하는 어휘와
유사한 질문, 예: "카제로스 레이드 5막은 언제 나와?"는 여전히 통과), 정답
오거절을 훨씬 적게 유지한다.

**Final Test 적용 결과(threshold=0.60 그대로, 재조정 없음)**:
- 답변 가능 40문항 중 **30개(75%) 정상 수용**
- 무응답 10문항 중 **7개(70%) 정상 거절**

**한계**: validation에서는 threshold=0.60이 정답을 겨우 4.3%(3/70)만
잘못 거절할 것으로 예측됐지만, final test에서는 25%(10/40)가 잘못
거절됐다. 두 질문셋의 점수 분포 자체가 달랐기 때문이다(validation은
"아이템 레벨 XXXX 이상"처럼 키워드 밀도가 매우 높은 질문 위주였고, final
test는 상대적으로 서술형 질문이 많았다 — median 0.753 vs 0.682). **고정
threshold 방식은 질문의 구체성·키워드 밀도에 따라 민감하게 반응할 수
있다는 실제 한계**로, 다음 개선 방향(상대적 스코어 margin 등)의 근거가 된다.

### 4) Generation (25문항 수동 채점, `grade_generation_eval.py`)

RAGAS 등 자동 평가 프레임워크 대신 세 기준(핵심정보 포함 / 문서 밖 추측 /
거절 적절성)을 직접 정의해 사람이 채점했다.

- **Hallucination(사실 날조): 25문항 중 1건(4%)**. 그마저도 "없는 사실을
  지어낸" 게 아니라, 존재하지 않는 "랭킹전"이라는 질문에 실제 문서에 있는
  다른 개념("The FIRST 이벤트")을 가져와 마치 정답인 것처럼 답한, **질문
  자체를 바꿔치기한 패턴**이었다.
- **레이어드 디펜스 확인**: threshold(1차 방어선)를 통과해 LLM까지 넘어간
  애매한 무응답 질문 3개 중 2개는 시스템 프롬프트의 "추측 절대 금지" 지시
  (2차 방어선) 덕분에 LLM이 스스로 "정보 없음"이라 답하며 걸러졌다. 결과적으로
  무응답 10문항 중 **9개(90%)가 최종적으로 사용자에게 거짓 답을 주지 않았다.**
  threshold 하나만으로는 70%였던 것이, 생성 단계의 안전장치까지 더하면 90%로
  올라간다는 뜻이다.
- **검색이 틀려도 지어내지 않음**: retrieval이 틀린 chunk를 가져온 경우에도
  LLM은 없는 정보를 추측하지 않고 "문서에 명시되어 있지 않다"고 정직하게
  답했다(`results/generation_eval_graded.csv`의 f004).
- 답변 가능 15문항 중 3개(20%)는 threshold 오거절로 답을 아예 못 받았다 —
  위에서 확인한 일반화 격차가 생성 단계에서도 동일하게 나타남.

## 알려진 한계 (정직하게 기록)

1. **"비슷한 문서끼리 경쟁" 문제는 미해결**이다. Keyword boost를 더 올려도
   (0.10~0.30) 개선되지 않았고, overlap도 역효과였다. 카제로스 레이드처럼
   여러 단계(서막~종막+익스트림)가 거의 동일한 서술 패턴을 공유하는 도메인
   에서는 KoSimCSE 같은 범용 임베딩 + 가벼운 키워드 보정만으로는 근본적인
   한계가 있다.
2. **고정 threshold의 질문셋 간 일반화 격차**(4%→25% 오거절)는 해결하지
   않고 한계로만 기록했다. Validation 수치를 그대로 믿고 배포하면 실제
   오거절률이 예상보다 훨씬 높을 수 있다는 뜻이다.
3. Final Test 50문항 중 동일한 24개 문서에서 뽑은 "미사용 chunk"로
   구성했다 — 완전히 새로운 문서로 만든 셋은 아니다(크롤링 재수집 없이
   같은 corpus 안에서 분리).
