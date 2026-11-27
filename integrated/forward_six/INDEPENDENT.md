# 각도·변위 단독 회귀 비교

2026-11-27 사용자 요청으로 여섯 후보의 angle과 displacement를 각각 20회 TPE 탐색합니다. 공동 회귀의 기존 소스·study를 보존하기 위해 single_models.py / single_tune.py / single_report.py를 별도로 사용합니다. shared grouped helpers는 재사용하지만 기존 파일은 수정하지 않습니다.

공통 조건: 기존 형상 Train/Test와 Train 내부 5-fold, seed 42, 동일 계열의 탐색 범위, 기본 설정 1회+무작위 9회+TPE 10회, pruning 없음, 예측 평균 앙상블 없음. 총 12개 study×20회=240개 COMPLETE trial, 1,200개 fold 평가입니다. 각도 단독은 ° MAE, 변위 단독은 mm MAE를 fold별 계산 후 평균해 최소화합니다. 개별 응답 정규화·MLP 초기 epoch 선택·GP 내부 커널 최적화·XGBoost 표준화는 직전 grouped 구조와 같습니다.

Train 내 단독/공동 정답이 있는 형상 집합이 정확히 같은지 검사하며 현재 각 836개입니다. Test는 설정 선택과 점수 계산에 사용하지 않습니다. 분류·장벽·응력 결과는 그대로 유지합니다. 여섯 후보/두 출력에 대해 동등한 20회 기회를 주지만, 두 단독 study의 총 탐색 기회는 공동 study 하나의 20회보다 많습니다. 후보를 좁힌 후 추가 튜닝을 이어갈 수 있도록 SQLite·sampler RNG·fold 예측·설정을 보존합니다. 이번 배치는 20회 비교 단계입니다.

실행·재개:

```sh
python integrated/cli.py --runs /path/to/output tune-forward-independent --total-trials 20 --workers 6
CABD_INTEGRATED_RUNS=/path/to/output python integrated/forward_six/single_report.py --trials 20
```

결과는 forward-independent/tuning/ 아래에 저장합니다. protocol.json은 데이터/코드 해시와 탐색 정책을 포함하며, 변경 시 같은 study 재개를 거부합니다. ROM은 기존 fold Train 보정 캐시를 재사용합니다. single_models의 MLP 평균 출력은 1개(학습용 로그분산 1개 별도), GP/XGBoost도 1개 응답만 학습합니다. 다른 정답을 바꿔도 해당 단독 모델 예측이 바뀌지 않음을 검증했습니다.

비교 보고서:

- joint_vs_independent_20trials.csv / PNG: 두 출력 공동 회귀에서 고른 설정과 각 출력 단독 최적 설정의 동일 OOF 형상 비교. MAE/RMSE/R²와 오차 감소율 포함.
- fold_comparison_20trials.csv: 선택된 설정의 fold별 오차와 우열.
- matched_configs_first10.csv / matched_configs_first10_summary.csv: 동일한 초기 하이퍼파라미터 10개에서 구조별 출력 오차 비교. 설정 동일을 assert 합니다.
- joint_target_reselected_* 열: 저장된 공동 20회 결과를 각 출력 MAE로 재선택한 보조 비교. 각도/변위에서 다른 설정이 선택되면 실제 배포에는 공동 모델 2개가 필요할 수 있으므로, 하나의 공동 모델이 두 수치를 동시에 달성한다고 주장하지 않습니다.
- original_3output_mae 열: 최초 각도·변위·장벽 공동 회귀의 참고 성능.
- search_history / presentation_milestones / search_progress: 전체 시도·누적 최고 점수·후반 5회 개선량과 계산 시간.
- audit_20trials.json: 형상/정답/fold/목적함수/해시 검증, GP 내부 최적화와 MLP epoch 상한 진단.

판단은 각도 우선, 변위 보조로 비교하며 작은 차이를 유의한 개선으로 단정하지 않습니다. fold별 우열도 살펴봅니다. 선택용 CV 결과이므로 최종 Train 재학습과 독립 Test 평가가 추가로 필요합니다. 20회 정체가 전역 최적·완전 수렴을 보장하지 않습니다. 신규 단위/재개 검증을 포함해 23개 테스트가 통과했습니다.

## 후보 선택 후 이어하기

20회 비교를 완료한 뒤 선택된 후보만 누적 횟수를 늘릴 수 있습니다. 아래는 사용 예시이며 이번 배치가 자동으로 실행하는 추가 튜닝이 아닙니다.

```sh
python integrated/cli.py --runs /path/to/output continue-forward-independent --models mlp --tasks angle --total-trials 40 --workers 1
```

single_continue.py가 기존 protocol·코드/데이터 해시를 검사하고 선택 study의 저장 sampler에서 이어갑니다. --total-trials는 추가 횟수가 아니라 누적 완료 목표입니다. 원래 20회 comparison CSV/PNG는 유지하고 selected-extension-complete.json에 후속 단계를 기록합니다. 연장된 study의 best.json을 다른 후보의 20회 best.json과 동등 예산 결과로 섞지 마세요. 코드/탐색 공간을 바꾸려면 별도 실험이 필요합니다.

## 20회 비교 완료

240개 COMPLETE trial, 1,200개 fold 예측의 데이터·코드 해시, 형상 ID·정답·목적함수 일치를 검사했습니다. 비교 양쪽의 OOF 형상이 동일하며 초기 10개 설정도 정확히 일치합니다. 운영 실패 0건, 23개 테스트 통과, 비교/탐색 PNG 시각 확인 완료입니다. 실행 배치 시간은 2026-11-27 00:17:44~00:51:30 KST, 약 33.8분입니다.

| 후보 | 공동 각도 MAE (°) | 단독 각도 MAE (°) | 공동 변위 MAE (mm) | 단독 변위 MAE (mm) |
|---|---:|---:|---:|---:|
| MLP | 1.3267 | 1.3464 | 0.1730 | 0.1775 |
| XGBoost | 1.7492 | 1.7121 | 0.2146 | 0.2148 |
| GP | 1.3802 | 1.3788 | 0.1793 | 0.1797 |
| ROM + MLP | 1.3555 | 1.3744 | 0.1819 | 0.1804 |
| ROM + XGBoost | 1.4891 | 1.4792 | 0.1882 | 0.1874 |
| ROM + GP | 1.3538 | 1.3646 | 0.1753 | 0.1751 |

현재 두 구조 비교에서는 MLP 공동 회귀가 각도·변위 모두 전체 최저 MAE입니다. MLP 단독 회귀는 공동 대비 각도 MAE 약 1.48%, 변위 MAE 약 2.63% 증가했습니다. 공동 회귀가 두 출력 모두 5개 중 4개 fold에서 우세했습니다. 초기 동일 설정 10개에서도 두 출력 모두 공동 회귀가 10/10 우세했습니다. 이는 한 seed·해당 탐색 범위에서의 관찰이며 통계적 유의성이나 일반적 우월성을 증명하지 않습니다. GP/XGBoost의 공유 방식은 MLP와 다릅니다.

추가 튜닝 후보는 MLP 계열을 우선 추천합니다. 단, 최초 3출력 공동 MLP는 각도 1.282142°, 변위 0.165306 mm로 이번 2출력 공동보다도 오차가 작았습니다. 따라서 공동/독립 선택의 최종 판단에서 최초 3출력 구조를 참고 대조군으로 유지하는 것이 타당합니다. 분류·장벽·응력 가중치는 바꾸지 않았고, 새 전체 Train 재학습·Test 평가·Inverse 교체·GitHub push도 수행하지 않았습니다.

GP 내부 최적화 400개 중 118개에서 성공 수렴이 보고되지 않았고, 선택 설정의 20개 중 13개도 이에 해당합니다. MLP fold 49개가 epoch 상한 1,200을 선택했으며 선택 설정의 fold에서는 0개입니다. 세부 경고/진단은 기록을 보존했습니다. 완전 수렴이나 전역 최적이라고 주장하지 않습니다.

공동 후보도 저장된 탐색에서 이어갈 수 있습니다. 아래 예시는 준비된 후속 실행 경로이며 이번 20회 비교 뒤 자동 실행하지 않았습니다.

```sh
python integrated/cli.py --runs /path/to/output continue-forward-grouped --models mlp --tasks angle_displacement --total-trials 40 --workers 1
```

이번 결과는 설정 선택에 사용한 Train 내부 OOF 836개 평가입니다. 최종 모델 선정과 성능 주장은 후속 튜닝 후 독립 Test로 확인해야 합니다.
