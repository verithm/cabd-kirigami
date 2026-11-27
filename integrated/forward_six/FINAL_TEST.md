# 50회 튜닝 확정 후 전체 Train 재학습과 고정 Test 평가

사용자의 2026-11-27 지시에 따라 여섯 후보의 네 작업을 최종 평가한다. 튜닝은 정확히 50 COMPLETE trial에서 끝내고 재개하지 않는다. 기존 학습·전처리·예측 규칙과 Train/Test 형상 분할은 유지한다.

## 고정 절차

1. `forward-refinement/selected_configs_50trials.json`의 각 후보·작업별 설정과 CV 목적함수를 고정한다. 결과 루트는 `forward-final50/`이다.
2. MLP의 최종 epoch는 선택된 설정의 5개 fold에서 Train 내부 조기 종료로 정했던 epoch의 중앙값을 사용한다. 최종 모델은 새로 초기화하고 모든 유효 Train 표본으로 이 횟수만 학습한다. Test 또는 별도 최종 validation으로 학습 횟수를 선택하지 않는다.
3. 전체 Train 형상 1,495개 중 분류 정답 유효 854개, 각도·변위/장벽/응력 정답 유효 각 836개를 작업별로 사용한다. 모든 유효 Train 표본을 학습에 포함한다. 모델의 전처리 및 응답 표준화는 Train에서만 적합한다.
4. ROM은 전체 Train의 완료 힘 곡선만 이용해 한 번 보정한다. 보정값은 세 ROM 후보의 작업들에 공통 사용하며 Test 곡선은 사용하지 않는다.
5. 24개 작업의 새 모델과 가중치를 저장하고 Train 일부 표본에서 재로딩 전후 예측의 완전 일치를 검증한다. 모든 재학습이 완료된 후에만 Test 예측을 시작한다.
6. 고정 Test 형상 227개 중 분류 유효 105개, 각 회귀 유효 143개에서 평가한다. 분류 주지표는 bi>mono를 양성으로 한 F1이며 Precision, Recall, Balanced Accuracy, 정확도, 혼동행렬을 함께 보존한다. 3개 안정성 조합의 혼동행렬과 지표도 별도 기록한다. 회귀는 각 출력별 MAE, RMSE, R²를 보고하고 장벽은 log10 및 원래 mJ 단위를 모두 기록한다.
7. 기존 CV에서 선정된 모델과 Test에서 관찰된 최고 모델을 구분해 기록한다. Test 결과를 이용한 재튜닝·설정 재선택은 하지 않는다. 배포 모델/Inverse 어댑터는 자동 교체하지 않는다.

MLP는 작업당 단일 네트워크, GP는 원래 구조, XGBoost는 내부 boosting을 사용한다. 별도 모델의 예측 평균 앙상블은 사용하지 않는다. GP 내부 커널 최적화 상한은 튜닝 때와 동일하며 실패/상한 경고는 그대로 저장한다.

## 실행

`CABD_INTEGRATED_RUNS`를 데이터 루트에 맞추고 기존 수치 라이브러리 환경에서 실행한다.

```sh
python integrated/forward_six/final_test.py --workers 4
python integrated/forward_six/audit_final_test.py
```

첫 실행에서 코드·데이터·선택 결과 SHA와 프로토콜을 고정한다. 동일 프로토콜 재실행은 완료 체크포인트를 재사용할 수 있지만 새 탐색은 하지 않는다. 프로토콜이나 코드가 달라지면 동일 출력 폴더 재사용을 거부한다. 체크포인트는 신뢰하는 로컬 pickle 파일만 불러온다.

평가 감사는 저장된 예측 CSV에서 지표를 독립적으로 다시 계산하고, 실제 Test 정답·형상 ID·학습 모집단·MLP 횟수·가중치 및 원본 파일 SHA·50회 고정 여부를 확인한다. 셸 출력 `run.log`는 학습 프로세스가 마지막 줄을 출력한 후 사후 감사에서 최종 해시를 고정한다.

## 산출물

- `protocol.json`: 설정·학습 횟수·데이터 ID·파일 SHA
- `models/<candidate>/<task>/model.pkl`, `fit.json`: 모델 및 학습 시간·경고·모집단
- `predictions/<candidate>/<task>.csv`: 형상별 실제값·예측값
- `test_comparison.csv`, `test_regression_metrics.csv`, `test_metrics.json`: 평가 지표
- `cv_vs_test.csv`, `test_comparison.png`: CV/Test 차이와 비교 그림
- `audit.json`, `complete.json`, `artifact_manifest.json`: 검증·완료·무결성 기록

고정 Test는 같은 프로젝트의 과거 실험에서 이미 보고된 분할이므로, 완전히 처음 공개되는 외부 검증 데이터라고 주장하지 않는다. 새로운 ANSYS 형상 검증은 별도 단계다.
