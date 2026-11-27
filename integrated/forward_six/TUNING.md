# 6개 후보 TPE 탐색 — 공통 20회 비교

각 후보마다 분류 20회·회귀 20회, 총 12개 study × 20회 = 240개 COMPLETE trial을 비교합니다. 각 trial은 같은 Train 내부 5-fold로 평가하므로 총 1,200개 fold 평가입니다. 실행 시간에 따라 15회 한도를 임시 적용했으나, 사용자가 원래 20회까지 계속하도록 명시하여 저장된 지점부터 재개했습니다. 분류와 회귀는 별도 탐색합니다. Test는 설정 선택에 사용하지 않고, 전체 Train 최종 재학습도 이 단계에서는 실행하지 않습니다.

## 목적함수와 무작위성

- 분류: 모든 검증 fold의 예측을 합친 OOF `bi>mono` F1 최대화.
- 회귀: fold 학습 부분의 출력 표준편차로 나눈 각도·변위·log10 장벽 MAE를 같은 가중치로 평균한 뒤, 5개 fold 점수를 평균하여 최소화. 실제 단위의 출력별 pooled OOF MAE/RMSE/R²도 별도 저장합니다.
- GP/XGBoost는 회귀 trial 하나의 설정을 출력별 회귀기 3개에 공통 적용합니다. MLP는 공동 회귀 신경망 1개입니다.
- 학습 seed=42, TPE seed=42, 기존 공통 fold 고정. 기본 모델/ROM 모델 쌍의 탐색 범위 동일. 여러 학습 모델 예측의 평균 없음.
- Optuna 4.6.0 TPE, `n_startup_trials=10`, pruning 없음. 첫 10회는 기존 기본 설정 1개와 무작위 탐색 9개, 이후 10회는 앞선 결과에 따른 TPE 제안입니다. trial은 COMPLETE만 비교 예산에 포함하며 운영 실패는 이력을 보존하고 동일 설정을 재시도합니다.

## 탐색 변수

| 계열 | 변수 |
|---|---|
| MLP | 너비 16/32/64/128/256, 깊이 1–4, 학습률 log[1e-4,1e-2], 가중치 감쇠 0 또는 log[1e-7,1e-2], SiLU/Tanh |
| XGBoost | 깊이 2–6, min_child_weight log[1,20], 학습률 log[.01,.2], 트리 수 100–1500, subsample·colsample_bytree [.6,1], L2 log[1e-3,30], L1 0 또는 log[1e-6,10] |
| GP | Matérn nu 1.5/2.5, 길이 척도·신호 크기 초기값 log[.1,10], 회귀 노이즈 초기값 log[1e-6,.1] |

MLP는 Adam/full-batch, 최대 1200 epochs·점검 간격 20·patience 160을 고정합니다. 내부 Train holdout에서 epoch 수를 선택하고 해당 fold의 전체 학습 부분으로 신경망 하나를 새로 학습합니다. GP는 ARD를 유지하며 커널 연속 파라미터를 내부 L-BFGS-B(최대 60회, random restart 0)로 조정합니다. GP의 TPE 초기값은 최종 커널 파라미터 고정값이 아닙니다. 최종 커널·내부 최적화 진단과 경고를 fold별 기록합니다. ROM 계수는 각 fold 학습 형상만으로 보정한 5개 공통 세트를 공유하며 TPE 변수가 아닙니다.

## 실행·이어하기

```sh
python integrated/cli.py --runs /path/to/output tune-forward-six --total-trials 20 --workers 3
# 계산 없이 현재 완료된 study 요약
python integrated/cli.py --runs /path/to/output tune-forward-six --report-only
```

`--total-trials`는 추가 횟수가 아니라 study별 **누적 완료 목표**입니다. SQLite study, TPE sampler RNG, trial 설정, fold별 예측·평가를 저장합니다. 중단된 RUNNING trial은 같은 설정·이미 완료한 fold를 재사용하여 이어갑니다. 코드/데이터/프로토콜 해시가 달라지면 같은 결과 폴더로 재개하지 못하도록 검사합니다.

결과 루트의 `forward-six/tuning/`:

- `protocol.json`: 탐색 공간·지표·seed·데이터 및 코드 해시.
- `rom/`: fold별 ROM 계수와 학습 형상 기록.
- `studies/<candidate>/<classification|regression>/study.sqlite3`, `sampler.pkl`.
- `trialXXXX/`: 설정, 5개 검증 예측 NPZ, fold 진단 JSON, pooled 결과.
- `best.json`, `trials.csv`: study별 현재 최고 기록과 전체 탐색 기록.
- `summary.csv`, `batch-complete.json`: 현재 완료된 study 요약과 배치 완료 기록.

소요 시간은 같은 컴퓨터의 동시 작업 영향을 받습니다. 동일 trial 수는 같은 계산 시간/개별 모델 fit 횟수를 뜻하지 않습니다. 첫 20회 결과를 전역 최적 또는 수렴 완료로 해석하지 않습니다.

## 발표용 기록과 해석

모든 후보에 동일한 20회 예산·같은 5-fold·같은 seed·같은 TPE 절차를 적용합니다. 기본 모델과 ROM 모델 쌍의 초기 10개 설정은 동일하며, 이후 TPE 제안은 각 모델의 관측 성능에 따라 달라집니다. 중단·재개 및 운영 실패는 기록하되 성공한 trial만 예산에 포함합니다.

`tuning_report.py --trials 20`은 `comparison_20trials.csv`, `search_history_20trials.csv`, `presentation_milestones_20trials.csv`, `selected_configs_20trials.json`, `audit_20trials.json`, `search_progress_20trials.png`를 생성합니다. 각 시도와 누적 최고 점수, 초기/적응 탐색 구분, 마지막 5회 개선량, 계산 시간, GP 내부 최적화 진단을 보존합니다. 곡선의 정체 경향을 관찰할 수 있지만 20회만으로 모든 모델의 수렴이나 전역 최적을 주장하지 않습니다. 모델의 최종 비교에는 독립 Test 평가가 추가로 필요합니다.

## 1차 20회 결과 — Train 내부 OOF

| 후보 | F1 ↑ | 각도 MAE (°) ↓ | 변위 MAE (mm) ↓ | log10 장벽 MAE ↓ | 장벽 MAE (mJ) ↓ |
|---|---:|---:|---:|---:|---:|
| MLP | 0.9618 | 1.2821 | 0.1653 | 0.0353 | 0.0155 |
| XGBoost | 0.9120 | 1.7164 | 0.2176 | 0.1229 | 0.0722 |
| GP | 0.9532 | 1.3788 | 0.1797 | 0.0596 | 0.0310 |
| ROM + MLP | 0.9580 | 1.2868 | 0.1731 | 0.0391 | 0.0128 |
| ROM + XGBoost | 0.9328 | 1.5020 | 0.1887 | 0.0893 | 0.0552 |
| ROM + GP | 0.9545 | 1.3502 | 0.1762 | 0.0471 | 0.0253 |

각 모델의 분류와 회귀는 별도 설정을 선택했습니다. 회귀 설정은 세 출력의 정규화 MAE를 함께 최소화했으며, 각 출력별로 별도 최고 trial을 고른 표가 아닙니다. 분류 854개·회귀 836개의 OOF 예측으로 평가했습니다. 이는 설정 선택에 사용한 CV 결과이며 독립 Test 성능이 아닙니다. 작은 점수 차이를 통계적으로 유의한 차이라고 주장하지 않습니다.

마지막 5회에서도 MLP 분류, ROM+XGBoost 분류, XGBoost 회귀가 개선됐습니다. GP 내부 최적화 일부도 상한에 도달했습니다. 발표에서는 “같은 20회 기회에서 얻은 최고 성능과 후반 개선량을 비교했다”고 설명하세요. 자세한 검증과 제한은 [VALIDATION.md](VALIDATION.md)에 기록했습니다.
