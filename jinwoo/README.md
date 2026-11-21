# 진우: Forward · CVAE 역설계 코드

기존 팀 코드와 별도로 수행한 전처리·Forward·역설계 구현입니다. 루트의 `src/cabd/`와 데이터를 변경하지 않고 이 폴더에 실행 스크립트를 정리했습니다. 데이터 스키마와 평가 기준은 루트 파이프라인과 다르므로 구분해서 사용합니다.

이 브랜치는 **코드와 실행 안내만** 포함합니다. 학습 CSV, 원본 해석 ZIP, 모델 가중치, 목표 CSV 및 실험 결과는 로컬 파일을 사용하거나 아래 명령으로 생성합니다.

## 구성

- `scripts/preprocess_forward.py`: 원본 수집 결과와 치수표 연결, 곡선 정리.
- `scripts/preprocess_energy_targets.py`: 에너지 기반 안정성·안정각·변위·장벽 추출.
- `scripts/preprocess_hinge_stress.py`: 두 번째 안정점까지의 힌지 최대응력 추출.
- `scripts/merge_forward_dataset.py`: 기존 결과와 추가 수집 결과 통합.
- `scripts/benchmark_scalar_forward.py`: 모델별 교차검증, 선정, 학습, 시험 평가.
- `scripts/inverse_cvae.py`: CVAE 학습·후보 생성, 조건부 GMM 비교 함수.
- `scripts/evaluate_inverse_v2.py`: 후보의 Forward 예측 및 형상·응력 필터 평가.
- `scripts/make_joint_inverse_goals.py`: 관측된 응답 관계를 고려한 목표 생성.
- `scripts/geometry_check.py`, `scripts/param_geom_v3.py`: 공유받은 공식 v3 형상 정의와 제약 검사.
- `scripts/train_forward.py`, `scripts/train_energy.py`: 과거 힘/에너지 곡선 모델 비교용 코드. 최종 스칼라 모델은 `benchmark_scalar_forward.py`입니다.

## 입력·출력

형상 변수 순서는 `t_h, L_h, t_a, L_a, L, t_f, c, rho`입니다. Forward 입력에는 젖음 상태(partial=0, full=1)를 더합니다. 출력은 안정성 분류, 안정각, 두 번째 안정점 변위, 역방향 에너지 장벽, 힌지 최대응력입니다.

안정성 모델은 상태별 쌍안정 확률을 출력합니다. partial 확률 >0.5 및 full 확률 ≤0.5이면 전환 가능으로 판정합니다. `unknown`은 판정 보류이며 단안정으로 처리하지 않습니다. 회귀는 partial 쌍안정이며 해당 정답이 있는 표본에 적용합니다.

Forward는 XGBoost, Gaussian Process, MLP, Extra Trees, RBF SVM, MLP+XGBoost 평균을 비교합니다. TR 내부 형상 기준 5-fold CV에서 전환 조합 F1과 회귀 MAE로 모델을 선정하고, 전체 TR로 다시 학습한 뒤 고정 TE에서 평가합니다. 같은 형상을 학습과 검증 사이에 나누지 않습니다.

CVAE의 정답은 기존 형상 치수 8개이며 조건은 그 형상의 실제 ANSYS 응답과 전환 여부입니다. 손실은 치수 복원 오차 + KL입니다. Forward는 CVAE 학습 손실에 들어가지 않고 생성 후 선별에 사용됩니다. 잠재변수는 2차원이며 3개 연속 조건과 전환 여부를 주는 경우 인코더는 12→32→32→4, 디코더는 6→32→32→8입니다. 연속 조건 두 개일 때 입력은 각각 11과 5입니다.

## 환경과 데이터 배치

아래 명령은 **`cd jinwoo` 후** 실행하는 macOS/Linux 예시입니다. Windows는 가상환경의 `Scripts/python.exe`를 사용하세요. 루트 `pip install -e .[ml]` 환경과 별도로 관리합니다.

```sh
python3.13 -m venv .venv-forward
.venv-forward/bin/python -m pip install -r requirements-forward.txt
python3.11 -m venv .venv-inverse
.venv-inverse/bin/python -m pip install -r requirements-inverse.txt
```

macOS에서 XGBoost OpenMP 로딩이 실패하면 OpenMP 런타임이 필요합니다. Forward는 requirements 파일의 라이브러리 버전을 사용합니다. CVAE 학습 환경은 Python 3.11, NumPy 1.26.4, PyTorch 2.2.2이며 형상 검사에는 Matplotlib를 사용합니다.

로컬 프로젝트의 `output/통합 전처리 데이터 v4/`에서 `scalar_forward_targets.csv`와 `jobs.csv`를 `data/`로 복사합니다. 루트 `data/CABD_dataset.csv`는 스키마가 다르므로 직접 대체하지 않습니다. 학습 CSV와 원본 ZIP은 Git에 포함하지 않습니다.

## Forward 학습

```sh
.venv-forward/bin/python scripts/benchmark_scalar_forward.py \
  data/scalar_forward_targets.csv runs/forward
```

출력 폴더는 새 경로여야 합니다. 교차검증 CSV, 시험 예측 CSV, 지표 JSON, 모델 5개를 저장합니다. 각도 평가의 허용오차는 스크립트별로 다르므로 `metrics.json`과 `TOLERANCES`를 확인하세요. 교차검증과 평균 앙상블은 별개의 절차입니다.

## CVAE 학습

```sh
# 안정각 + 변위 + 장벽 + 전환 여부
.venv-inverse/bin/python scripts/inverse_cvae.py fit \
  data/scalar_forward_targets.csv data/jobs.csv runs/forward runs/cvae-three \
  --condition-counts 3

# 안정각 + 변위 + 전환 여부
.venv-inverse/bin/python scripts/inverse_cvae.py fit \
  data/scalar_forward_targets.csv data/jobs.csv runs/forward runs/cvae-angle-displacement \
  --condition-counts 2

# 안정각 + 장벽 + 전환 여부
.venv-inverse/bin/python scripts/inverse_cvae.py fit \
  data/scalar_forward_targets.csv data/jobs.csv runs/forward runs/cvae-angle-barrier \
  --angle-barrier
```

각 실행은 새 출력 폴더를 사용합니다. `fit`의 Forward 경로는 공통 인터페이스의 인자이며 학습 손실에는 사용하지 않습니다. 장벽 변환은 세 물성 학습 명령에 `--barrier-transform log` 또는 `--barrier-transform log1p --barrier-scale 0.05`를 추가합니다.

## 목표 생성과 후보 평가

```sh
.venv-forward/bin/python scripts/make_joint_inverse_goals.py interpolation \
  data/scalar_forward_targets.csv runs/goals.csv

.venv-forward/bin/python scripts/evaluate_inverse_v2.py predict \
  --version local --cvae-only \
  data/scalar_forward_targets.csv runs/goals.csv \
  runs/forward runs/cvae-three runs/cvae-three

.venv-inverse/bin/python scripts/evaluate_inverse_v2.py geometry \
  --version local --cvae-only \
  data/scalar_forward_targets.csv runs/goals.csv \
  runs/forward runs/cvae-three runs/cvae-three
```

목표 생성은 `heldout`, `interpolation`, `angle-first`를 지원합니다.

`interpolation`은 가까운 실제 응답을 공통 가중치로 보간하고 기존 한 응답이 세 허용오차를 동시에 만족하는지 확인합니다. 개별 물성의 관측 범위 안이라는 사실만으로 임의 조합의 물리적 가능성이 보장되지는 않습니다.

각도·변위 모델의 평가는 두 단계 모두 `--angle-displacement --cvae-only`를 추가하고 출력 경로를 해당 학습 폴더로 바꿉니다. 각도·장벽 모델은 두 단계 모두 `--angle-barrier --cvae-only`를 사용합니다. `previous` 인자는 `--cvae-only`일 때 사용하지 않습니다.

세 물성 모델의 과거/현재 CVAE와 조건부 GMM을 함께 비교하려면 `--cvae-only`를 빼고 `previous` 경로에 비교할 기존 세 물성 CVAE 폴더를 지정하세요. 두 물성 평가는 `--cvae-only`를 사용합니다.

공통 필터는 공식 형상 제약, partial 쌍안정/full 단안정, 안정각 ±3°, 평가 대상일 때 변위 ±0.5 mm와 장벽 ±0.05 mJ, 예측 힌지 응력 ≤6 MPa입니다. 응력은 저장된 첫 해석 단계부터 두 번째 안정점까지 세 힌지에서 관측된 최대 von Mises 값입니다. 이는 설계 선별 기준이며 ANSYS 재해석으로 확인합니다.

`goal_results.csv`는 목표·시드별 통과 건수, `metrics.json`은 집계, `forward_screened_candidates.csv`는 통과 목표별 대표 형상 치수입니다. 시드별 목표 발견률, 세 시드를 합친 고유 목표 발견률, 개별 후보 통과율의 분모는 서로 다르므로 구분합니다. 후보 수와 시드는 스크립트의 `BUDGET`, `SEEDS`를 확인하세요.

## 원본 전처리

`merge_forward_dataset.py --help`에서 원본 ZIP 폴더, 초기 jobs.csv, 새 출력 폴더, 추가 수집 ZIP과 `--job-kits` 인자 순서를 확인하세요. 원본 파일은 수정하지 않습니다. `param_geom_v3.py`는 import해서 사용하며, 파일을 직접 실행하는 과거 STEP 비교 진단은 별도 STEP/stepparse가 필요합니다.
