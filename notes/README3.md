# Day2 작업 기록 — 도메인 조사 (2026-06-17)

오늘은 해커톤 진행 과정 2단계 **"도메인 조사 및 관련 논문/공식 탐색"** 을 수행했다.
목표는 스트레스 지수와 우리 feature들의 관계를 문헌에서 확인하고, 기존 컬럼만으로 만들 수 있는
파생지표(feature engineering 힌트)를 정리하는 것이다. 외부 데이터는 모델에 직접 쓰지 않고,
아이디어/가설 수립 참고용으로만 사용한다.

## 조사 방식

- 다중 출처 웹 검색(5개 각도) → 출처 22개 수집 → 주장 99개 추출
  → 상위 25개 적대적 검증(주장당 3표, 2표 이상 반박 시 폐기) → 21개 확정 / 4개 폐기
  → 중복 병합 후 최종 7개 finding 정리
- 검증은 비용 상한 때문에 상위 25개만 수행했으므로, **검증 안 된 주장 = 거짓이 아니라 "이번엔 미확인"** 이다.
  (특히 흡연·골밀도·학력·가족력은 검증 슬롯에 거의 못 올라가, EDA로 직접 상관을 확인해야 한다.)

## 변수별 스트레스 연관성 (검증 결과)

| 변수 | 스트레스와의 관계 | 근거 강도 |
| --- | --- | --- |
| `systolic/diastolic_blood_pressure` | 양(+). 번아웃↔전고혈압 OR 1.85, 심혈관질환 OR 1.21 | 높음 |
| `glucose`, `cholesterol`, BMI | 알로스타틱 부하(ALI) 구성요소(대사·지질·체형 축) | 높음 |
| `mean_working` | 양(+), 용량-반응 관계. 길수록 비선형 악화 | 높음 |
| `activity` | 음(−). 코르티솔 SMD -0.37, 수면 개선 SMD -0.30 | 중간 |
| `sleep_pattern` | 근로→번아웃 경로를 부분 매개(25~73%) | 중간 |
| `bone_density` | 직접 근거 미확인 (글루코코르티코이드성 골손실 가설은 미검증) | 낮음 |
| `smoke_status`, `edu_level`, 가족력/병력 | 이번 검증에서 결론 미확보 → EDA로 직접 확인 필요 | 미확인 |

## 핵심 개념 — Allostatic Load Index (ALI)

- **정의**: 만성 스트레스가 여러 신체 시스템에 남긴 누적 마모량을 하나의 숫자로 정량화한 지수.
  스트레스를 직접 못 재므로, 여러 생체지표의 위험 정도를 합산해 간접 측정한다.
- **계산(binarize-and-sum, 검증됨)**:
  1. 각 지표마다 위험 임계치를 정하고, 위험값이면 1점(HDL처럼 낮을수록 나쁜 건 반대 채점)
  2. 전부 합산 → 0~N 점수, 보통 >=4면 high allostatic load
  3. 임계치는 고정 임상값보다 **표본 상위 quartile** 사용이 원조 방식
- **우리 데이터 한계**: 면역(CRP, IL-6)·신경내분비(코르티솔) 마커가 없고, cholesterol이 HDL/LDL로 분리되지 않으며,
  glucose만 있고 HbA1c는 없다. → **완전한 ALI 재현 불가, 심혈관·대사 축만의 "부분 ALI"만 정당화됨.**
- **주의**: "스트레스↑ = 코르티솔↑"는 항상 참이 아니다. 고부담군은 오히려 저코르티솔(둔화) 프로파일을 보이기도 한다.
  → ALI는 확정 정답이 아니라 **검증할 가설**이다.

## Feature Engineering 후보 (오늘 도출)

### 바로 쓸 수 있는 파생 feature

1. **BMI** = `weight / (height_m)^2` — 체형 축
2. **MAP(평균동맥압)** = `DBP + 1/3 * (SBP - DBP)` — 대사증후군 예측력이 SBP/DBP/PP 단독보다 높음
3. **Pulse Pressure** = `SBP - DBP`
4. **부분 ALI (cardiometabolic surrogate, 0~5)** — SBP, DBP, glucose, cholesterol, BMI를
   각각 train 상위 quartile 임계치 기준으로 이진화 후 합산
5. **근로시간 다단계 binning** — WHO/ILO cut-point: `35-40 / 41-48 / 49-54 / >=55h`,
   고강도 구간(~59/73/84h)에서 번아웃 odds 2→3→4배 비선형 가속 → 스플라인/다항 변환도 후보
6. **상호작용 feature** — `activity x sleep_pattern`, `sleep_pattern x mean_working`

### 적용 원칙

- 원본 컬럼은 그대로 두고 파생 feature는 **추가**만 한다(대체 X).
- 부분 ALI는 단일 형태로 확정하지 말고 변형(이진 합산 / z-score 합산 / 개별 플래그)을 후보로 두고 CV로 비교.
- 모든 임계치/통계는 **train에서만 fit**, test에는 transform만 적용(누수 금지).
- 채택 여부는 **CV MAE ablation**(넣은 모델 vs 뺀 모델)으로 판정. 도움 안 되면 버린다.

## 피해야 할 것 (적대적 검증에서 반박됨)

- 우리 컬럼만으로 만든 **대사증후군 플래그**(1-2 반박) — HDL 분리가 없어 정의 부정확
- **BMI>=30 비만 플래그**를 대사증후군에 묶는 것(0-3 반박)
- 단순 z-score 합산을 "ALI의 정식 정의"로 정당화하는 것(1-2 반박) → 예측용 변형으로는 시도 가능하나 임상 근거는 아님
- ALI↔근로능력 연결(0-3 반박)

## 열린 질문 (다음에 결정)

1. binarize 임계치를 고정 임상값으로 할지 train quartile로 할지 — 어느 쪽이 CV MAE가 좋은가
2. 이 대회 target이 생리적 스트레스(코르티솔/ALI)형인지 자기보고 perceived stress형인지
   (ALI가 잘 들으면 생리지표형 target이라는 단서)
3. `bone_density`, `smoke_status`, `edu_level`, 가족력/병력의 인코딩 방식과 스트레스 연관성
4. `mean_working`의 최적 함수형 — 다단계 bin vs 비선형/스플라인 vs sleep 상호작용

## 주요 참고 출처

- Juster 2011 (PMID 21129851) — ALI와 만성 스트레스/번아웃
- Seeman/MacArthur (PMC2841407) — ALI 정식 계산법(binarize-and-sum)
- Frontiers in Psychiatry 2024 (PMC10909938, n=26,916) — 번아웃↔혈압/심혈관질환
- Tsai 2015 (PMID 25575802) — MAP의 대사증후군 예측력
- Kivimaki 2015 Lancet (n=528,908) — 근로시간↔뇌졸중 용량-반응
- WHO/ILO Pega 2020 (PMC7339147) — >=55h 근로↔허혈성 심질환
- Lin 2021 (J Occup Health) — 근로시간↔번아웃 비선형 가속, 수면 매개
- De Nys 2022 — 신체활동↔코르티솔/수면

## 오늘의 상태 업데이트

| Day | 작업 | 단계 | 상태 |
| --- | --- | --- | --- |
| Day2 | 도메인 조사 및 관련 논문/공식 탐색 | EDA | 완료 |
| Day2 | 평가산식과 검증 전략 설계 | EDA | 시작 전 |
| Day2 | 모델 후보 선정 | EDA | 시작 전 |

## 다음 작업

1. 노트북에 FE 셀 작성 — BMI, MAP, pulse pressure, 부분 ALI(변형 포함), 근로시간 binning (train fit → test transform)
2. MAE 기준 validation 전략 수립(KFold 또는 target bin StratifiedKFold)
3. `EXP001` DummyRegressor로 기준점 확보
4. FE feature ablation을 CV MAE로 비교, Notion 실험 로그에 기록
