# 진우·승민 통합 파이프라인

이 폴더가 `integrated` 브랜치의 공통 실험 진입점입니다. 기존 `src/cabd/`와 `jinwoo/`는 원본 모델 구조와 과거 실험으로 보존합니다.

**Forward는 공통 Train CV 기준으로 승민 MLP 앙상블을 선정했습니다. 역설계는 비교 실험을 마쳤지만 단일 최종 모델은 아직 미확정입니다.** `configs/model_status.json`의 `inverse.selected_model`은 `null`입니다. 현재 통합 모델로 새로운 ANSYS 검증을 했다는 의미도 아닙니다.

코드와 실행 설명만 이 브랜치에 추가합니다. 새 원본 ZIP·학습 CSV·가중치·목표 치수·후보·대량 결과는 Git에 포함하지 않습니다. 로컬 실험 기본 위치는 Git에서 제외된 `artifacts/integrated/`입니다.

## 무엇을 통합했나

| 항목 | 공통 규칙 |
|---|---|
| 정답 | 에너지의 첫 내부 봉우리와 그 이후 첫 골짜기로 안정점·역방향 장벽 추출 |
| 데이터 품질 | 원본 `ok`, 유효 곡선, 15 mm 완료만 사용. 불명확한 라벨은 unknown 유지 |
| 안정각 | 두 번째 에너지 골짜기 단계에서 측정점으로 계산한 초기 대비 위쪽 암 회전각 |
| 장벽 | 학습·생성 조건은 log10, 원래 단위 평가 때 `10**prediction`으로 mJ 복원 |
| 응력 | 이번 공통 학습·선별에서는 제외 |
| 분류 평가 | bi→mono를 양성으로 한 F1 주지표, Precision·Recall·Balanced Accuracy 함께 보고 |
| 회귀 평가 | 출력별 MAE 주지표, RMSE·R² 및 mJ 복원 장벽 오차 함께 보고 |
| 분할·선정 | 형상 Train/Test 유지, 공통 Train 5-fold, 설정 3개 비교, CV 선정 후 전체 Train 재학습 |

독립 회귀는 출력별 OOF MAE로 설정을 고릅니다. 공동 출력 MLP는 fold의 **학습 부분** 표준편차로 나눈 MAE를 세 출력에 같은 가중치로 평균합니다. Test는 설정 선택에 사용하지 않습니다.

## 구성

- `cli.py`: 한 곳에서 실행하는 명령 인터페이스
- `runtime_paths.py`: 저장소 기준 코드 경로와 외부 로컬 결과 경로
- `preprocessing/`: 진우 에너지 정답·측정점 추출 어댑터
- `forward/`: 진우 6종 + 승민 5종 모델군의 공통 학습·평가·예측
- `inverse/`: 두 CVAE, 조건부 GMM, DE, SLSQP 및 목표 유형·장벽 판정 비교
- `configs/inverse_initial.json`: 최초 로그 허용오차 비교의 시작 규칙
- `configs/model_status.json`: 현재 Forward 선정·역설계 미확정 상태와 최신 평가 기준
- `tests/`: 경로·정답·단위·판정 등 통합 코드 검증
- `VALIDATION.md`: 실제 원본 재추출·저장 모델 예측·생성 후보의 재현 확인 기록
- `RESULTS.md`: 평가 지표·분모·실험 조건별 주요 결과

기존 `cabd.geometry.params`, `cabd.models.mlp`, `cabd.models.cvae`, `cabd.models.rom`을 재사용합니다. 원본 모델을 새로 복제해 서로 다른 정의가 생기지 않도록 저장소의 `src/`를 직접 참조합니다.

## 환경

저장소 루트에서 Python 3.12의 별도 환경을 사용합니다. 루트 `pyproject.toml`은 기존 승민 실행 환경을 유지하고 있으므로 이번 통합 재현에서는 아래 파일로 환경을 만듭니다. CAD/ANSYS 패키지 설치는 ML 비교에 필요하지 않습니다.

```sh
python3.12 -m venv .venv-integrated
# macOS/Linux
.venv-integrated/bin/python -m pip install -r integrated/requirements.txt
# Windows: .venv-integrated\Scripts\python.exe를 사용
```

이 문서의 `python`은 위 환경의 Python을 뜻합니다. `requirements-lock.txt`는 실제 검증 환경 전체 목록, `requirements.txt`는 직접 사용하는 ML 패키지입니다. macOS에서 XGBoost/LightGBM의 OpenMP 로딩 오류가 나면 해당 컴퓨터의 OpenMP 런타임을 설정해야 합니다. 개인 컴퓨터의 임시 라이브러리 경로를 코드에 고정하지 않았습니다.

## 데이터 준비

기존 루트 `data/CABD_dataset.csv`는 힘 부호 판정 기반의 다른 스키마입니다. 이를 공통 정답 CSV 대신 넣지 않습니다. 로컬에 원본 결과 ZIP과 통합 치수표 `jobs.csv`를 준비합니다. 치수표에는 `geom_id,role,t_h,L_h,t_a,L_a,L,t_f,c,rho`가 필요합니다. `role`은 기존 `train/test`를 유지합니다.

```sh
python integrated/cli.py --runs artifacts/integrated prepare \
  --raw-roots /path/to/collection-results \
  --jobs /path/to/joined/jobs.csv \
  --case-index /path/to/previous/case_index.csv \
  --reference-targets /path/to/previous/scalar_forward_targets.csv
python integrated/cli.py forward-data
```

`--case-index`는 이전 실험에 사용한 ZIP 이름만 선택하는 옵션입니다. 재현에는 이를 권장합니다. 생략하면 지정한 경로의 수집 ZIP을 모두 읽으므로 대상 PC1–PC9 자료만 둡니다. 재해석·검증 결과와 같은 ID의 중복이 있으면 오류로 중단합니다. `--reference-targets`는 선택 사항이며, 제공하면 재추출한 정답이 기존 캐시와 일치하는지 검사합니다. symlink 권한 없이 Windows에서도 읽도록 구현했습니다.

이미 공통 전처리 CSV가 있다면 `unified-forward/data/`에 `jobs.csv`, `scalar_targets.csv`, 기존 `folds.csv`를 배치합니다. ROM+GP 재학습에는 `completed_force_curves.json`도 필요합니다. 원래 role과 fold를 바꾸지 않습니다. 이번 동결 실험의 분류 Train/Test는 854/105형상, 회귀는 836/143형상입니다.

결과 경로는 `--runs` 또는 `CABD_INTEGRATED_RUNS`로 바꿀 수 있습니다. `--runs`는 하위 명령 **앞**에 씁니다. 경로는 저장소 밖이어도 됩니다. 로그의 후보 참조는 결과 폴더 기준 상대 경로로 저장합니다.

## Forward 실행

```sh
python integrated/cli.py train-forward --group all
python integrated/cli.py report-forward
python integrated/cli.py audit --scope forward
python integrated/cli.py predict dimensions.csv predictions.csv --model sm_mlp --seed 0
```

진우: MLP·XGBoost·GP·Extra Trees·RBF SVM·MLP+XGBoost. 승민: MLP 앙상블·ARD GP·ROM+GP·Referee·GP 없는 Referee. 설정 3개를 공통 fold에서 고른 후 학습 시드 0–4로 반복합니다. 5-fold, 신경망 5개 앙상블, 학습 시드 5회 반복은 다른 개념입니다.

선정 Forward는 분류와 공동 회귀 각각 5개 신경망입니다. 역설계는 그중 학습 시드 0의 분류·회귀 앙상블을 고정합니다. 현재 역설계 어댑터는 기록된 `sm_mlp` 구성을 재현하므로 다른 Forward가 선정되는 새 실험에서는 별도 어댑터 검토가 필요합니다.

## 역설계 비교 — 최종 모델 선정 아님

```sh
python integrated/cli.py inverse
python integrated/cli.py audit --scope inverse
```

첫 비교는 실제 Test 전환 응답 49개×역설계 시드 3개, 목표당 최대 4,096 Forward 평가입니다. Train 788개(전환 394, bi→bi 394)를 사용합니다. 내부 630/158개 분할에서 목표 20개로 설정을 고른 후 전체 Train으로 재학습합니다. 현재 동결 데이터 재현을 위한 수량 검사이므로 새 데이터셋에 몰래 맞춰 바꾸지 않습니다.

- 최초 기준: 각도 ±1.7°, 변위 ±0.23 mm, log10 장벽 ±0.16
- 공통 필터: 예측 전환 확률 ≥0.9, 공식 형상 제약, 응력 제외
- CVAE 가중치 손실에는 Forward가 없습니다. 설정 검증·생성 후 선별에는 사용합니다.
- DE/SLSQP는 별도 역설계 신경망 학습 없이 같은 고정 Forward를 최적화합니다.
- 대표 후보는 통과 후보 중 허용오차 정규화 오차의 최댓값이 가장 작은 후보입니다.

`forward_witness_diagnostic.json`은 재실행한 모델에서 실제로 계산합니다. 과거 22/35/17개의 진단 수치를 새 학습 결과에 하드코딩하지 않습니다.

## 목표 유형과 허용오차 분석

```sh
# 실제 Test49 + 보간100 + 각도별 조건부100 + 희소/경계100 + 독립100
# 시드 0, 각도 ±2°, 변위 ±0.25 mm, 원래 장벽 ±0.03 mJ로 새 탐색
python integrated/cli.py target-strategies

# 같은 후보에서 장벽 판정만 제외. 모델 입력·가중치·탐색은 그대로
python integrated/cli.py without-barrier

# 추가 기록: 최초 로그 기준의 소폭 완화, 원래 단위 장벽 오차 분석
python integrated/cli.py relaxed
python integrated/cli.py review-barrier
```

장벽 판정 제외는 두 조건 CVAE 재학습이 아닙니다. 입력 장벽은 유지하고 같은 후보를 재판정합니다. 원래 세 변수 공간에서 선정한 희소 목표가 장벽을 제외한 두 변수 공간에서는 익숙한 목표가 될 수 있어 거리 통계도 별도로 제공합니다.

목표 발견률은 후보를 하나 이상 찾은 비율, 개별 생성 후보 통과율은 필터 전 전체 생성 후보 중 통과 비율입니다. 최적화 탐색점의 비율은 생성 후보 비율과 같은 의미로 비교하지 않습니다. 모두 대리모델 기준이며, 이번 통합의 새 ANSYS 성공률은 아닙니다.

## 재개·검증·보관

```sh
python -m unittest discover -s integrated/tests -v
python integrated/cli.py --help
```

완료된 설정·모델·목표 파일은 재실행 때 건너뜁니다. 설정·데이터·알고리즘을 바꾸면 새 결과 폴더를 사용해야 합니다. macOS에서 긴 실행을 잠자기 없이 유지하려면 `caffeinate -is python integrated/cli.py ...`로 실행할 수 있습니다. 다른 OS의 전원 관리는 해당 컴퓨터에서 설정합니다.

모든 개인 경로·데이터·가중치는 로컬에 둡니다. 과거 원본 코드와 현재 통합 규칙을 혼용하지 않도록 실행 진입점은 `integrated/cli.py`, 현재 모델 상태는 `configs/model_status.json`을 확인합니다.
