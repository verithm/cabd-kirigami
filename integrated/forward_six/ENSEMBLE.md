# 최종 Forward: 선정 MLP의 5개 시드 평균 앙상블

2026-11-27 사용자는 여섯 후보의 공통 50회 튜닝 및 전체 Train/Test 비교 후, 선정된 MLP를 앙상블해 최종 Forward를 만들도록 요청했다. 기존 단일 모델 비교는 보존하고 `forward-mlp-ensemble/`에 최종 앙상블을 저장한다.

## 고정 구성

- 네 독립 작업: 안정성 분류 / 안정각·변위 공동 회귀 / log10 장벽 회귀 / partial 전환 구간 힌지 최대 응력 회귀.
- 작업별 5개 네트워크, 총 20개. 시드는 42/43/44/45/46이며 bootstrap 없이 모든 유효 Train 표본을 각각 학습한다.
- 하이퍼파라미터와 epoch는 `forward-final50/protocol.json`에 고정된 MLP 설정을 그대로 사용한다. epoch는 분류 40, 각도·변위 100, 장벽 1120, 응력 880이다. Test로 시드·epoch·멤버 수·가중치를 선택하지 않는다.
- 분류: 3개 안정성 조합에 대한 멤버 확률을 균등 평균한 뒤 argmax로 판정한다. hard label 다수결이나 새 threshold 튜닝이 아니다.
- 회귀: 예측 평균값을 각 출력 단위에서 균등 평균한다. 장벽은 학습된 log10 스케일에서 평균한 뒤 `10**mean_log`로 mJ를 복원한다. 개별 mJ 값의 산술평균과는 다르다.
- 각 모델은 새로 초기화한다. 시드 42의 네 네트워크가 기존 단일 최종 MLP와 정확히 동일한 가중치를 재현하는지 검증한다. 작업별 5개 멤버의 가중치 해시가 서로 다른지도 확인한다.

분류 Train 854/Test 105, 각 회귀 Train 836/Test 143이라는 기존 유효 표본 조건을 유지한다. 원래 Test는 프로젝트에서 이미 평가했던 고정 분할이다. 이번 비교를 새로운 외부 데이터 검증으로 주장하지 않는다.

## 예측 사용

모델 폴더에는 모든 작업과 멤버를 담은 `bundle.pkl`과 진입점 `forward_selection.json`이 있다. 이 두 파일로 사용한다. 신뢰하는 로컬 체크포인트만 불러온다.

```sh
python integrated/cli.py --runs /path/to/output predict-forward-ensemble geometry.csv predictions.csv
```

입력 CSV는 `t_h,L_h,t_a,L_a,L,t_f,c,rho`를 포함해야 한다. `geom_id`는 선택 사항이다. 형상 제약 검사는 역설계 후보 필터의 별도 단계다.

Python API는 `forward_six.ensemble.load_ensemble(folder)` 및 `predict(bundle, dataframe)`이다. 반환값에는 안정성 조합, 클래스 확률, `p_bi_mono`, 각도, 변위, log10 장벽, mJ 장벽, 힌지 응력이 포함된다. 기존 Inverse 코드의 가중치 참조는 이 작업에서 자동 교체하지 않았다.

`*_member_sd`는 5개 멤버 예측 평균값 사이의 표본 표준편차다. 교정된 신뢰구간, 실제 오차 상한, 파괴 확률이 아니다. 회귀 정답은 partial 쌍안정에서 정의되며 다른 상태에서 반환된 회귀값을 검증된 물리량으로 취급하지 않는다.

## 학습 및 검증

```sh
python integrated/cli.py --runs /path/to/output train-forward-ensemble --workers 4
python integrated/forward_six/audit_ensemble.py
```

동일 코드·데이터·선택 결과 SHA의 작업만 재개할 수 있다. 변경된 프로토콜이면 기존 결과 폴더 재사용을 거부한다. 모든 멤버 학습 및 저장/재로딩 검증 후에만 Test 평가를 수행한다. 단일 모델 및 멤버별 평가도 기록하지만 Test 점수로 멤버를 제외하거나 가중치를 바꾸지 않는다.

감사는 20개 멤버에서 Test 예측을 다시 계산해 soft voting·회귀 평균·장벽 복원·모든 지표를 독립적으로 검증한다. 기존 24개 study가 50회에 그대로 있고, 이전 최종 Train/Test 산출물의 SHA가 바뀌지 않았는지 확인한다.

주요 결과는 `single_vs_ensemble_test.csv`, `improvements.csv`, `member_test_metrics.csv`, `test_metrics.json`, `single_vs_ensemble.png`에 저장한다. 가중치는 `members/<task>/seed<seed>/model.pkl`에도 개별 보존한다. 확정된 구성에서 결과가 일부 악화돼도 이를 숨기거나 Test에 맞춰 재선택하지 않는다. 앙상블이 모든 출력에서 단일 모델을 능가한다고 가정하지 않는다.
