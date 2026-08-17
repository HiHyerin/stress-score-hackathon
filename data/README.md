# Data

대회 원본 데이터(`train.csv`, `test.csv`, `sample_submission.csv`)는 저장소에 포함하지 않았습니다.
DACON [`[초격차] AI 헬스케어 4기 해커톤: 스트레스 지수 예측 AI 해커톤`](https://dacon.io) 페이지에서
로그인 후 내려받아 이 폴더(`data/`)에 넣으면 나머지 코드가 그대로 동작합니다.

## 스키마

- `train.csv`: 3,000 rows × 18 cols (feature 17 + `stress_score`)
- `test.csv`: 3,000 rows × 17 cols
- `sample_submission.csv`: `ID`, `stress_score`

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `ID` | ID | 샘플 식별자 |
| `gender` | 범주형 | 성별 (F/M) |
| `age` | 수치형 | 나이 |
| `height` | 수치형 | 신장 |
| `weight` | 수치형 | 체중 |
| `cholesterol` | 수치형 | 콜레스테롤 수치 |
| `systolic_blood_pressure` | 수치형 | 수축기 혈압 |
| `diastolic_blood_pressure` | 수치형 | 이완기 혈압 |
| `glucose` | 수치형 | 혈당 |
| `bone_density` | 수치형 | 골밀도 |
| `activity` | 범주형 | 활동 수준 (intense / light / moderate) |
| `smoke_status` | 범주형 | 흡연 상태 (current-smoker / ex-smoker / non-smoker) |
| `medical_history` | 범주형 | 개인 병력 (diabetes / heart disease / high blood pressure), 결측 42~44% |
| `family_medical_history` | 범주형 | 가족력, 결측 47~50% |
| `sleep_pattern` | 범주형 | 수면 패턴 (normal / oversleeping / sleep difficulty) |
| `edu_level` | 범주형 | 학력, 결측 20~22% |
| `mean_working` | 수치형 | 평균 근로시간, 결측 34% |
| `stress_score` | target | 0.00~1.00, **0.01 단위 이산값 (101 unique)** |

## 평가

- 지표: MAE (Mean Absolute Error)
- 규칙: 외부 데이터 학습 사용 불가, test 통계/분포를 전처리·인코딩·구간화에 사용 불가 (data leakage)

자세한 EDA 과정은 [`../notes/`](../notes/)를 참고하세요.
