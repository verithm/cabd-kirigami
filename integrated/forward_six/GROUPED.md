# 네 작업으로 분리한 Forward 구조

기존 models.py와 20회 TPE 기록을 보존하고 grouped.py에 별도 구현했습니다. 세 회귀 그룹의 20회 TPE 튜닝과 감사·비교 보고서를 완료했습니다. 전체 Train 최종 재학습과 Test 평가는 아직 수행하지 않았습니다.

- classification: 기존 계열별 안정성 분류 구조 유지.
- angle_displacement: 안정각(°)·두 번째 안정점 변위(mm). MLP는 공통 은닉층과 평균 2개/로그분산 2개, GP는 공유 커널의 다중 응답, XGBoost는 one_output_per_tree로 출력별 트리를 묶습니다.
- barrier: log10 역방향 장벽 독립 회귀.
- stress: partial 초기부터 두 번째 안정점까지 힌지 최대응력(MPa) 독립 회귀.

각 작업의 가중치·설정·정규화·결측 정답 판정은 독립입니다. 현재 회귀 세 그룹은 각각 Train 836개/Test 143개, 분류는 Train 854개/Test 105개이며 기존 형상 분할·fold를 유지합니다. partial 쌍안정 회귀에 full 결과가 반드시 필요하지는 않습니다.

응력은 기존 hinge_stress.csv의 initial_to_second_max_MPa입니다. 현재 scalar 정답과 archive·curve_file·mesh_rule·peak_set·valley_set·8개 치수를 비교했고 979개 partial 쌍안정 기록이 전부 일치했습니다. 불일치/누락은 정답으로 사용하지 않습니다. 응력은 MPa이며 로그 변환하지 않습니다.

공동 학습의 의미는 모델마다 다릅니다. MLP는 두 출력의 오차가 공통 은닉층에 전달됩니다. GP는 커널을 공유하며 결합 우도를 최적화하지만 출력 간 교차 공분산을 학습하지 않습니다. XGBoost는 그룹 설정/평가를 묶지만 출력별 트리 사이에서 표현을 공유하지 않습니다. 단순히 두 출력을 반환한다고 상관관계를 직접 학습한다고 주장하지 않습니다. 물리 관계도 자동으로 강제되지 않습니다.

각도·변위의 높은 상관관계는 공동 회귀를 검토할 근거입니다. 효과 검증은 동일 데이터·fold·튜닝 예산에서 독립 회귀와 비교해야 합니다. 각도·변위는 정규화 MAE 평균, 장벽은 log10 MAE, 응력은 MPa MAE로 별도 튜닝할 수 있습니다. 이전 3출력 회귀 study를 새 구조의 튜닝으로 이어 붙이지 않습니다. 기존 분류 구조/데이터는 동일합니다.

실행:

```sh
python integrated/cli.py --runs /path/to/output forward-grouped prepare --stress /path/to/hinge_stress.csv
python integrated/cli.py --runs /path/to/output forward-grouped smoke
```

API는 fit_group(name, training, group, config, ...)와 predict_group(model, rows)입니다. config는 해당 작업의 flat 설정입니다. Test를 fit에 전달하면 거부합니다. ROM 보정에는 해당 학습 형상만 사용합니다. 예측 평균 앙상블은 없습니다.

검증: Train 형상 48개의 소규모 실행으로 여섯 후보×네 작업의 학습·예측·저장/재로딩 일치 24건 통과, 단위/통합 테스트 19개 통과. 실행 확인용 가중치는 최종 모델이 아닙니다. 결과 루트 forward-grouped/의 protocol.json, data/data_audit.json, smoke/validation.json에 구조·출처 해시·검증 기록을 보존합니다.

## 20회 TPE 비교 실행

2026-11-26 사용자 지시에 따라 `tune-forward-grouped --total-trials 20 --workers 6`을 실행합니다. 분류는 기존 20회 결과를 유지하고, 여섯 후보의 세 회귀 그룹을 각각 20회 탐색합니다. 총 360개 COMPLETE trial, 1,800개 fold 평가입니다. 초기 10회는 기본 설정 1개+무작위 9개, 이후 10회는 Optuna TPE입니다. 각 작업은 같은 형상 fold와 seed 42를 사용합니다. 기본/ROM 모델 쌍의 탐색 범위와 초기 설정은 동일합니다.

각도·변위는 fold Train 표준편차로 나눈 MAE 두 개의 평균, 장벽은 log10 MAE, 응력은 MPa MAE를 5개 fold에서 평균하여 최소화합니다. 설정 선택 이후 출력별 pooled OOF MAE/RMSE/R²도 별도로 보고합니다. 기존 분류 결과는 변경하지 않습니다. Test 평가나 전체 Train 최종 재학습은 이 튜닝 단계에 포함하지 않습니다.

`forward-grouped/tuning/`에 프로토콜과 데이터/코드 SHA-256, SQLite, sampler RNG, trial 설정, fold별 정답·예측·진단·계산 시간, 최고 설정을 저장합니다. 코드/데이터가 달라지면 같은 study 재개를 거부합니다. ROM은 기존과 동일한 5개 fold Train 보정 캐시를 재사용하고 held-out 겹침이 없음을 검사합니다.

실행·재개:

```sh
python integrated/cli.py --runs /path/to/output tune-forward-grouped --total-trials 20 --workers 6
CABD_INTEGRATED_RUNS=/path/to/output python integrated/forward_six/grouped_report.py --trials 20
```

보고서는 results_20trials.csv, old_vs_grouped_20trials.csv, search_history_20trials.csv, presentation_milestones_20trials.csv, selected_configs_20trials.json, audit_20trials.json, search_progress_20trials.png, old_vs_grouped_20trials.png입니다. old_vs 비교는 저장된 예측의 형상 ID가 정확히 같은지 검사합니다.

해석 제한: 이전 회귀는 후보별 세 출력 공동 설정 20회, 새 회귀는 그룹별 20회입니다. 새 GP 쌍은 공유 커널, XGBoost 응답은 Train 표준화를 적용합니다. 따라서 결과는 전체 학습 프로토콜의 비교이며, 응력 추가만의 인과 효과나 공동 회귀만의 효과를 분리하는 실험은 아닙니다. 같은 횟수는 같은 계산 시간이나 전역 최적·완전 수렴을 뜻하지 않습니다. 작업 간 독립 가중치 때문에 응력 학습 손실은 다른 작업의 가중치를 수정하지 않습니다.

## 20회 완료 결과

총 18개 study 모두 20회 완료: 360 COMPLETE trial, 1,800개 fold 검증 예측을 원본에서 재계산했습니다. 데이터·코드 해시, 형상 ID·정답·fold·목적함수 일치, 이전/새 OOF 형상 배열 동일, 기본/ROM 초기 설정 동일, 5개 ROM 보정의 held-out 겹침 0을 확인했습니다. 운영 실패 0건입니다. 실행 검증 20개 통과, PNG 두 개 시각 확인 완료입니다.

| 후보 | 각도 MAE (°) | 변위 MAE (mm) | log10 장벽 MAE | 응력 MAE (MPa) |
|---|---:|---:|---:|---:|
| MLP | 1.3267 | 0.1730 | 0.0281 | 0.1826 |
| XGBoost | 1.7492 | 0.2146 | 0.1309 | 0.3499 |
| GP | 1.3802 | 0.1793 | 0.0596 | 0.2252 |
| ROM + MLP | 1.3555 | 0.1819 | 0.0308 | 0.1941 |
| ROM + XGBoost | 1.4891 | 0.1882 | 0.0899 | 0.3481 |
| ROM + GP | 1.3538 | 0.1753 | 0.0454 | 0.2181 |

모든 수치는 설정 선택에 사용한 Train 내부 OOF 결과(회귀 836개)입니다. 같은 MAE 기준에서 새 여섯 후보 중 MLP가 네 회귀 출력 모두 가장 작았습니다. MLP의 log10 장벽 MAE는 이전 0.035277에서 0.028091로 약 20.4% 감소했고, 원래 단위 장벽 MAE는 0.015493에서 0.009646 mJ로 약 37.7% 감소했습니다. 반면 각도 MAE는 1.282142에서 1.326724로 약 3.5%, 변위 MAE는 0.165306에서 0.172975로 약 4.6% 증가했습니다. 출력 분리가 모든 항목을 개선하지는 않았습니다.

응력은 MLP의 MAE가 0.182647 MPa로 가장 작지만, ROM+GP의 RMSE 0.323864 MPa와 R² 0.924486이 가장 좋았습니다. RMSE는 큰 오차의 영향을 더 받으므로 MAE 순위와 다를 수 있습니다. 로그 장벽의 개선이 원래 단위 장벽의 모든 지표 개선을 보장하지도 않습니다. 예를 들어 ROM+GP의 log10 MAE는 줄었지만 원래 단위 장벽 RMSE는 커졌습니다. 전체 지표는 결과 CSV에 보존했습니다.

GP 내부 최적화 600개 중 142개는 성공 수렴이 보고되지 않았고, 선택된 설정의 30개 중 9개도 해당합니다. MLP 전체 fold 학습 175개, 선택된 설정의 fold 7개가 1,200 epochs를 선택했습니다. 세부 진단은 fold JSON에 보존했습니다. 따라서 모두 수렴했다고 주장하지 않습니다. 마지막 5회 개선량과 전체 점수는 presentation_milestones_20trials.csv 및 search_history_20trials.csv에 있습니다.

실행 배치 시간은 2026-11-26 22:22:13~23:36:12 KST, 약 74분입니다. 같은 머신에서 6개 worker로 동시 실행했으며, 개별 fold 시간 합은 독립 실행 속도 비교가 아닙니다. 전체 Train 재학습·Test 평가·기존 Forward/Inverse 교체·GitHub push는 수행하지 않았습니다.
