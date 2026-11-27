# 승민에게 전달하는 최종 Forward — `forward_final`

이 브랜치는 **선정된 MLP의 5개 초기화 시드 앙상블**을 역설계에서 바로 불러 쓰도록 정리한 버전입니다. **가중치가 브랜치에 포함돼 있어 기존 Release를 받을 필요가 없습니다.** 과거 `sm_mlp`나 `unified-forward/forward_selection.json` 대신 아래 새 진입점을 사용하세요.

## 1. 받기와 환경 설치

```sh
git fetch origin
git switch --track origin/forward_final
python -m venv .venv-forward
```

Windows PowerShell:

```powershell
.venv-forward\Scripts\python.exe -m pip install -r integrated/requirements-forward-inference.txt
.venv-forward\Scripts\python.exe integrated/cli.py predict-forward-ensemble artifacts/integrated/forward-mlp-ensemble/inference_example_input.csv predictions.csv
```

macOS/Linux:

```sh
.venv-forward/bin/python -m pip install -r integrated/requirements-forward-inference.txt
.venv-forward/bin/python integrated/cli.py predict-forward-ensemble artifacts/integrated/forward-mlp-ensemble/inference_example_input.csv predictions.csv
```

Python 3.12에서 검증했습니다. 추론에는 원본 ANSYS ZIP·곡선 파일·CAD 라이브러리·Optuna·XGBoost가 필요하지 않습니다. 패키지는 실제 학습/저장 환경과 같은 버전으로 고정했습니다. 공유된 신뢰하는 체크포인트만 로드하세요.

예제 출력 정답은 `artifacts/integrated/forward-mlp-ensemble/inference_example_output.csv`입니다. 예제 입력은 Train 치수 3행으로, ANSYS 정답이 아닌 **같은 모델의 예상 출력**을 확인하는 용도입니다. 사용자 입력 파일을 넣으면 동일한 방식으로 계산합니다.

## 2. 모델과 입력

- 가중치: `artifacts/integrated/forward-mlp-ensemble/bundle.pkl`
- 진입점/해시: 같은 폴더의 `forward_selection.json`, `SHA256SUMS.txt`
- 구성: 분류 / 각도·변위 / 장벽 / 응력, 네 독립 작업 × 5개 네트워크 = 총 20개
- 멤버 시드: 42, 43, 44, 45, 46. 모든 멤버가 전체 유효 Train을 사용했으며 초기화만 다릅니다.

입력 CSV의 필수 열은 순서와 무관하게 `t_h,L_h,t_a,L_a,L,t_f,c,rho`입니다. `geom_id`는 선택 사항입니다. 길이는 mm, `rho`는 무차원 비율입니다. 치수 정규화와 응답 복원은 모델 안에서 하므로 직접 표준화하지 마세요.

## 3. 역설계 코드에서 호출

저장소 루트에서 다음 Python 예제를 실행할 수 있습니다. 실제 코드에서는 `repo`를 본인이 clone한 경로로 설정하세요. 모델 폴더 경로는 상대 위치에서 자동으로 결정됩니다.

```python
from pathlib import Path
import sys
import pandas as pd

repo = Path.cwd()  # 저장소 루트; 외부 프로그램이면 clone 경로를 명시
sys.path.insert(0, str(repo / "integrated"))
from forward_six.ensemble import load_ensemble, predict

# 반복 탐색을 시작하기 전에 한 번만 로드
bundle = load_ensemble(repo / "artifacts/integrated/forward-mlp-ensemble")

# 후보별 8개 치수를 담은 DataFrame; 여러 후보를 묶어서 예측 가능
candidates = pd.read_csv("candidates.csv")
result = predict(bundle, candidates)
result.to_csv("forward_predictions.csv", index=False)
```

반환 열의 의미:

| 열 | 의미 / 단위 |
|---|---|
| `predicted_class` | `bi>bi`, `bi>mono`, `mono>mono` 중 평균 확률이 가장 큰 조합 |
| `p_bi_mono` | partial 쌍안정 → full 단안정의 평균 예측 확률 |
| `probability_bi>bi`, `probability_bi>mono`, `probability_mono>mono` | 세 조합의 평균 확률 |
| `angle_deg` | partial 두 번째 안정점에서 위쪽 암의 초기 대비 회전각, ° |
| `deformation_mm` | partial 두 번째 에너지 골짜기의 적용 변위, mm |
| `log10_barrier` | partial 역방향 에너지 장벽의 log10 값; 원래 단위 mJ |
| `barrier_mJ` | `10 ** log10_barrier`, mJ |
| `hinge_transition_max_MPa` | partial 초기부터 두 번째 에너지 골짜기까지의 힌지 최대 응력, MPa |
| `*_member_sd` | 5개 멤버 예측 사이의 표본 표준편차. 검증된 신뢰구간/오차 한도가 아님 |

분류는 멤버 확률을 균등 평균하고 argmax를 취합니다. 각도·변위·응력은 예측값을 균등 평균합니다. 장벽은 **log10 예측값을 평균한 후 mJ로 복원**합니다. 회귀값은 partial이 쌍안정인 데이터에서 학습했으므로 다른 상태에서의 값을 검증된 물리량으로 해석하지 마세요.

형상 제약, 원하는 각도·변위·장벽의 허용오차, 전환 확률 하한, 응력 상한은 **역설계 쪽에서 따로 적용**해야 합니다. 이 Forward는 그 기준을 임의로 설정하거나 자동 필터링하지 않습니다. `predicted_class == 'bi>mono'`와 높은 전환 확률 기준 통과는 다른 조건입니다. API는 DataFrame/NumPy 추론용이며, 생성모델에 역전파하는 Tensor 연산 어댑터는 별도 구현이 필요합니다.

## 4. 선정 과정과 최종 Test 결과

여섯 후보를 같은 형상 분할, 같은 Train 5-fold 및 작업별 50회 TPE 예산으로 비교했습니다. 기본 비교는 예측 평균 없는 단일 모델입니다. 이후 MLP를 선정하고, 사용자 지시로 5개 시드 앙상블을 고정했습니다. Test를 보고 멤버·가중치·학습 횟수를 고르지 않았습니다.

| 지표 | 단일 MLP | 최종 5개 앙상블 |
|---|---:|---:|
| 안정성 F1 | 0.9792 | 0.9792 |
| 각도 MAE (°) | 1.3135 | 1.3393 |
| 변위 MAE (mm) | 0.1771 | 0.1755 |
| log10 장벽 MAE | 0.02856 | 0.02990 |
| 장벽 MAE (mJ) | 0.00885 | 0.00948 |
| 힌지 응력 MAE (MPa) | 0.1477 | 0.1387 |

분류는 Train 854 / Test 105, 회귀는 각 Train 836 / Test 143입니다. 분류 양성인 실제 bi>mono는 Test 49개이며, 앙상블은 47개 검출, 오검출 0개입니다. 전체 이진 정답은 103/105입니다. 앙상블이 모든 출력에서 단일 모델보다 우수한 것은 아닙니다. 기존 프로젝트에서 확인한 고정 Test이며 새 외부/ANSYS 검증은 아닙니다.

모델 폴더에 RMSE·R²·Precision·Recall·Balanced Accuracy, 형상별 예측, 멤버별 성능과 학습 기록도 포함합니다. `forward-refinement/`에는 50회 전체 튜닝 추이, `forward-final50/`에는 여섯 후보 최종 비교, `forward-gp-improvement/`에는 GP 추가 진단 요약을 보존합니다. GP 분류 추가 파일럿 16개 설정·80개 fold에서 개선이 없어 새 50회 튜닝은 하지 않았습니다.

## 5. 역설계 학습에 참고할 데이터

`artifacts/integrated/forward-grouped/data/geometry_targets.csv`에는 같은 형상 ID, 8개 치수, Train/Test 역할, Train fold, 안정성 조합과 네 회귀 물리량이 있습니다. 안정각·변위·장벽·응력은 유효한 부분만 존재합니다. 사용 목적에 맞는 유효값 마스크를 적용하고 **기존 `role`/`fold`를 유지**하세요. partial 쌍안정이라고 full 단안정이 보장되는 것은 아니므로 조합 라벨도 확인해야 합니다.

기존 `integrated/inverse/shared.py`는 과거 Forward를 읽는 구현입니다. 본 최종 모델로 자동 전환하지 않았으므로, 새 역설계에서는 위 API로 연결하세요. 이 브랜치에서 역설계 최종 모델을 선정하거나 새 역설계 실험을 수행하지 않았습니다.

## 6. 기록·무결성·재학습

`artifacts/integrated/forward-final-manifest.json`은 배포에 포함한 파일의 SHA-256입니다. 최종 `bundle.pkl`에는 20개 멤버 전체가 있으므로 개별 pickle을 추가로 받을 필요가 없습니다. `members/`에는 멤버별 학습 조건과 해시를 기록한 `fit.json`이 있습니다.

원본 학습 프로토콜의 절대 경로는 생성 당시의 이력이며 새 컴퓨터에서 추론에 사용하지 않습니다. 배포 묶음의 해시 목록과 상대 경로가 새 checkout의 무결성 기준입니다. 전체 원본 곡선, SQLite study, 과거 단일 가중치와 수집 ZIP은 이번 추론 전달 묶음에서 제외했습니다. **예측에 필요한 가중치는 모두 포함**돼 있습니다. 재학습·전체 과거 실험 재현은 별도 실험 경로와 해당 입력 자료를 준비해야 합니다.

세부 구현은 [forward_six/ENSEMBLE.md](forward_six/ENSEMBLE.md), 추론은 `forward_six/ensemble.py`, 학습은 `forward_six/train_ensemble.py`, 검증은 `forward_six/audit_ensemble.py`를 참고하세요.
