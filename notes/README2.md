# DACON 스트레스 지수 해커톤 노트

## 목표

이 해커톤은 단순히 MAE 점수를 올리는 것만이 목적이 아니다. 개인적인 주요 목표는 테이블 데이터 모델링의 전체 워크플로우를 다시 배우고 연습하는 것이다:

1. 주제와 피처 이해하기
2. 관련 도메인 지식이나 논문 찾아보기
3. 적합한 사전학습 모델 또는 ML 모델 고민하기
4. 노트북에 데이터 불러오기
5. 데이터 확인 및 전처리 방법 결정하기
6. 학습, 검증, 예측, 제출
7. 무엇을 배웠는지, 어떤 부분이 약했는지, 그 약점을 어떻게 보완했는지 정리하기

마지막에는 Codex에게 다음을 요약해 달라고 요청한다:

- 무엇을 공부했는지
- 어떤 부분이 약했는지
- 그 약점을 어떻게 해결하거나 공부했는지
- 모델링 결정과 그 이유
- 다음 학습 로드맵

## 기억해야 할 대회 규칙

대회: `[초격차] AI 헬스케어 4기 해커톤: 스트레스 지수 예측 AI 해커톤`

평가지표:

- MAE: `mean(abs(y_true - y_pred))`
- 낮을수록 좋음
- 퍼블릭 리더보드는 테스트 데이터의 100%를 사용

중요한 규칙:

- 외부 데이터 사용 불가
- 사전학습 모델(pre-trained models) 사용 가능
- 일일 제출 제한: 3회
- 테스트 데이터는 모델 학습이나 전처리 fit에 사용하면 안 됨
- 규칙에서 언급하는 데이터 누수(data leakage) 예시:
  - 라벨 인코딩이나 원핫 인코딩에 테스트 데이터를 사용하는 경우
  - 스케일링에 테스트 데이터를 사용하는 경우
  - `pd.get_dummies()`를 테스트 데이터에 직접 적용해서 테스트의 카테고리를 학습하게 되는 경우
  - 결측치 대체(imputation)에 테스트 데이터의 통계량을 사용하는 경우
  - 그 외 모델 학습에 테스트 데이터가 어떤 식으로든 사용되는 경우

실전 규칙:

- 전처리는 `train`에만 fit 한다
- `test`는 `train`에서 학습한 객체나 통계량을 이용해 적용/변환한다

## 데이터 위치

프로젝트 루트:

```text
stress-score-hackathon/
```

데이터 폴더:

```text
data/
```

파일:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

노트북:

```text
stress_prediction.ipynb
```

## 환경

가상환경을 생성했다:

```text
.venv/
```

VS Code나 Jupyter에서 이 환경을 사용한다.

필요하다면 노트북 커널을 설치/등록한다:

```powershell
.venv\Scripts\activate
python -m ipykernel install --user --name hackathon --display-name "Python (hackathon)"
```

노트북에서는 다음을 선택한다:

```text
Python (hackathon)
```

기본 데이터 로딩 코드:

```python
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submission = pd.read_csv("data/sample_submission.csv")
```

상대경로가 안 먹히면 현재 작업 디렉토리를 확인한다:

```python
import os
os.getcwd()
```

## 데이터 개요

Shape:

```text
train: (3000, 18)
test: (3000, 17)
sample_submission: (3000, 2)
```

제출 파일 컬럼:

```text
ID, stress_score
```

Train/test 컬럼 차이:

- `train`에만 있는 컬럼: `stress_score`
- `test`에만 있는 컬럼: 없음
- 즉 `test`는 타깃을 제외하면 `train`과 동일한 피처 컬럼을 가짐

## 타깃

타깃 컬럼:

```text
stress_score
```

타깃 요약:

```text
min: 0.00
mean: about 0.48213
median: 0.48
max: 1.00
unique values: 101
```

타깃이 0~1 범위의 스트레스 점수인 회귀 문제이다.

유용한 코드:

```python
train["stress_score"].describe()
```

기억할 것:

```python
train["stress_score"].mean()
train["stress_score"].median()
```

괄호를 꼭 붙여야 한다. `()`가 없으면 `mean`, `median`은 계산된 값이 아니라 메서드 객체 자체를 가리킨다.

## 피처 그룹

ID 컬럼:

```text
ID
```

수치형 피처:

```text
age
height
weight
cholesterol
systolic_blood_pressure
diastolic_blood_pressure
glucose
bone_density
mean_working
```

범주형 피처:

```text
gender
activity
smoke_status
medical_history
family_medical_history
sleep_pattern
edu_level
```

타깃:

```text
stress_score
```

## 피처 의미 추정

| 컬럼 | 의미 | 타입 |
|---|---|---|
| `ID` | 샘플 식별자 | ID |
| `gender` | 성별 | 범주형 |
| `age` | 나이 | 수치형 |
| `height` | 키 | 수치형 |
| `weight` | 몸무게 | 수치형 |
| `cholesterol` | 콜레스테롤 수치 | 수치형 |
| `systolic_blood_pressure` | 수축기 혈압 | 수치형 |
| `diastolic_blood_pressure` | 이완기 혈압 | 수치형 |
| `glucose` | 혈당 | 수치형 |
| `bone_density` | 골밀도 | 수치형 |
| `activity` | 활동 수준 | 범주형 |
| `smoke_status` | 흡연 상태 | 범주형 |
| `medical_history` | 개인 병력 | 범주형 |
| `family_medical_history` | 가족 병력 | 범주형 |
| `sleep_pattern` | 수면 패턴 | 범주형 |
| `edu_level` | 교육 수준 | 범주형 |
| `mean_working` | 평균 근무 시간(추정, 시간 단위일 가능성) | 수치형 |
| `stress_score` | 스트레스 점수 | 타깃 |

## 범주형 값

`gender`

```text
F, M
```

`activity`

```text
intense, light, moderate
```

`smoke_status`

```text
current-smoker, ex-smoker, non-smoker
```

`medical_history`

```text
diabetes, heart disease, high blood pressure
```

`family_medical_history`

```text
diabetes, heart disease, high blood pressure
```

`sleep_pattern`

```text
normal, oversleeping, sleep difficulty
```

`edu_level`

```text
bachelors degree, graduate degree, high school diploma
```

## 결측치

결측치가 있는 컬럼:

| 컬럼 | Train 결측 개수 | Train 결측 비율 | Test 결측 개수 | Test 결측 비율 |
|---|---:|---:|---:|---:|
| `medical_history` | 1289 | 42.97% | 1309 | 43.63% |
| `family_medical_history` | 1486 | 49.53% | 1416 | 47.20% |
| `edu_level` | 607 | 20.23% | 647 | 21.57% |
| `mean_working` | 1032 | 34.40% | 1008 | 33.60% |

현재 생각:

- `medical_history`, `family_medical_history`
  - 결측이 "해당 없음(none)"을 의미할 수도 있고, "기록 안 됨"을 의미할 수도 있음
  - 우선은 `None` 같은 별도 카테고리로 채우는 것이 합리적인 첫 전략

- `edu_level`
  - 결측은 아마도 알 수 없음/미기록을 의미
  - 특정 교육 수준이라고 임의로 가정하지 말 것
  - 우선은 `Unknown`으로 채우는 것이 합리적인 첫 전략

- `mean_working`
  - 결측 처리가 좀 더 복잡함
  - 무직, 은퇴, 학생, 혹은 단순 미기록을 의미할 수 있음
  - 전체 평균/중앙값으로 채우는 것을 유일한 아이디어로 맹신하지 말 것
  - 항상 먼저 결측 indicator부터 만들 것

제안하는 첫 결측 피처:

```python
train["mean_working_missing"] = train["mean_working"].isna().astype(int)
test["mean_working_missing"] = test["mean_working"].isna().astype(int)
```

중요:

- `test`를 채울 때는 `train`에서만 계산한 값/통계량을 사용할 것
- 예시:

```python
median_working = train["mean_working"].median()
train["mean_working"] = train["mean_working"].fillna(median_working)
test["mean_working"] = test["mean_working"].fillna(median_working)
```

## Mean Working 결측치 가설

현재 논의 중인 것은 `mean_working` 결측치가 여러 요인에 좌우될 수 있다는 것이다:

1. 나이
   - 10대는 학생일 수 있음
   - 고령층은 은퇴했을 수 있음

2. 개인 병력 / 가족 병력
   - 건강 관련 요인이 근무 상태에 영향을 줄 수 있음
   - 다만 이는 더 강한 가정이므로 반드시 검증이 필요함

3. 교육 수준
   - 학력이 높을수록 근무 중일 확률이 높을 수 있음
   - 학력이 높은데 근무 시간이 결측이라면, 0보다는 중앙값 대체가 더 합리적일 수 있음

중요한 모델링 태도:

- 이것들은 사실이 아니라 가설임
- 우선 단순한 베이스라인부터 시작할 것
- 그 다음 규칙 기반 대체(rule-based imputation)를 교차검증 MAE로 비교할 것

가능한 실험:

| 실험 | 아이디어 |
|---|---|
| A | 결측 indicator + train 중앙값 대체 |
| B | 나이 기반 규칙: 어린/나이 든 경우 결측값을 0으로, 그 외는 train 중앙값으로 |
| C | 나이 + 병력 + 교육 수준 규칙 |
| D | 연령대별 중앙값 대체 |

핵심 교훈:

> 결측치 처리는 단순히 빈칸을 채우는 것이 아니다. 왜 값이 없는지에 대한 가설이며, 그 가설은 검증되어야 한다.

## 유용한 코드 스니펫

결측 컬럼 확인:

```python
missing_columns_train = train.columns[train.isnull().sum() > 0]
missing_columns_train
```

해당 컬럼만 확인:

```python
train[missing_columns_train].info()
```

이 코드는 `train`에 컬럼을 추가하지 않는다. 해당 컬럼들을 선택해서 보여줄 뿐이다.

결측 개수 확인:

```python
train[missing_columns_train].isnull().sum()
```

train/test 컬럼 차이 확인:

```python
set(train.columns) - set(test.columns)
set(test.columns) - set(train.columns)
```

피처 그룹 나누기:

```python
id_col = "ID"
target_col = "stress_score"

feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

numeric_cols = train[feature_cols].select_dtypes(include="number").columns.tolist()
categorical_cols = train[feature_cols].select_dtypes(exclude="number").columns.tolist()

numeric_cols, categorical_cols
```

## 다음 단계 제안

1. EDA 계속 진행
   - 타깃 분포
   - 수치형 피처 분포
   - 범주형 피처 개수
   - 피처와 `stress_score`의 관계

2. 도메인 지식 학습
   - 스트레스 점수와 건강/생활습관 변수의 관계
   - 혈압, 혈당, 수면, 흡연, 활동, 근무 시간
   - 아이디어 참고용으로만 사용, 외부 데이터로 활용 금지

3. 단순 베이스라인 구축
   - 단순한 전처리로 시작
   - 교차검증 사용
   - 평가지표: MAE

4. 전처리 가설 비교
   - 단순 중앙값/최빈값/Unknown 전략
   - 규칙 기반 `mean_working` 전략
   - CatBoost/LightGBM 사용 시 모델 자체의 결측 처리 기능 활용

5. Notion 로그 기록
   - 진행 로그
   - 실험 로그
   - 학습 이슈 로그

## Notion 진행 로그 템플릿

| 날짜 | 단계 | 한 일 | 배운 것 | 막힌 점/헷갈리는 점 | 다음 액션 |
|---|---|---|---|---|---|
| 2026-06-16 | 피처 이해 | train/test 구조, 타깃, 피처 타입, 결측 컬럼 확인 | 이건 `stress_score`(0~1)를 예측하는 회귀 문제. test는 타깃을 제외하면 train과 동일한 피처를 가짐. | 결측치를 어떻게 해석할지 결정 필요 | EDA 계속 진행하고 결측치 처리 전략 비교 |

## 현재 상태

완료:

- 대회 규칙 확인
- 개인 학습 목표 정리
- 데이터 경로 확인
- 가상환경 생성
- 노트북에서 데이터 수동 로딩
- 타깃 확인
- 피처 그룹 확인
- train/test 컬럼 차이 확인
- 결측 컬럼 확인
- 결측치 처리에 대한 초기 아이디어 논의

아직 안 한 것:

- 전체 EDA 시각화
- 도메인/논문 검색
- 베이스라인 모델
- 검증 전략
- 전처리 실험
- 제출 파일 생성
