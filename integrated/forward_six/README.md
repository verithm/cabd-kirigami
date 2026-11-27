# Forward 후보 6종 — 예측 평균 앙상블 제외

현재 상태: **모델 구현·실행 검증 완료, TPE 탐색 완료: 후보별 분류·회귀 각각 20회 비교**. 상세 기준과 재개 명령은 [TUNING.md](TUNING.md)에 있습니다. 전체 Train 최종 학습·새 Test 평가·역설계 Forward 교체는 이 단계에 포함하지 않습니다. 이전 `sm_mlp` 앙상블과 기존 결과를 보존하며, 새 파일은 별도 `forward-six/` 아래에 저장합니다.

| 후보 | 출처 및 구조 | 회귀 입력 | 분류 입력 |
|---|---|---|---|
| `mlp` | 승민 MLP, 분류 신경망 1개 + 공동 회귀 신경망 1개 | 치수 8개 | 치수 8개 |
| `xgboost` | 진우 XGBoost, 상태 공통 이진 분류기 + 출력별 회귀기 | 치수 8개 | 치수 8개 + 상태 코드 |
| `gp` | 승민 ARD Matérn 5/2 GP, 2단계 분류 + 출력별 회귀 | 치수 8개 | 치수 8개 |
| `rom_mlp` | 같은 단일 MLP에 ROM 정보 추가 | 치수 8개 + ROM 3개 | 치수 8개 + ROM 3개 |
| `rom_xgboost` | 같은 XGBoost에 ROM 정보 추가 | 치수 8개 + ROM 3개 | 치수 8개 + 상태별 ROM 최소 힘 + 상태 코드 |
| `rom_gp` | 같은 GP에 ROM 정보 추가 | 치수 8개 + ROM 3개 | 치수 8개 + 상태별 ROM 최소 힘 |

ROM 3개는 partial의 각도·두 번째 안정점 변위·log10 장벽입니다. 기존 ROM의 `logE >= -4.2` 입력 하한 및 힘 부호 기반 특징 추출은 유지합니다. **ANSYS 정답은 공통 에너지 봉우리·골짜기 추출 결과**를 사용합니다. ROM의 특징 추출이 ANSYS 정답 추출을 대체하지 않습니다.

## 무엇을 유지하고 바꿨나

- MLP: 승민의 SiLU/Tanh 은닉층, full-batch Adam, 3-class cross-entropy, 회귀 3개 평균+3개 log-variance 및 Gaussian NLL 유지. 5개 신경망 평균을 제거하고 작업별 신경망 1개만 저장·예측. 출력되는 회귀 예측은 단일 신경망의 평균 head입니다. ROM 변형은 입력 차원을 늘렸습니다.
- GP: 승민의 partial-bi 분류와 partial-bi일 때 full-mono 분류, 출력별 독립 GP 유지. 커널의 주변우도 최적화는 통상적인 GP fitting이며 여러 모델의 예측을 평균하지 않습니다.
- XGBoost: 진우의 치수+partial/full 상태 코드 이진 분류기와 출력별 회귀 유지. XGBoost 내부 트리 부스팅은 유지합니다. 별도로 학습한 XGBoost들을 평균하지 않습니다.
- 분류의 내부 구조와 공동/독립 회귀 차이는 원본 계열을 유지한 의도적인 차이입니다. 전체 비교는 공통 `bi>mono` F1 및 출력별 MAE/RMSE/R²로 합니다. MLP 후보의 설정 선택은 세 회귀 출력의 학습 SD 정규화 MAE를 사용하도록 후속 HPO에서 연결할 수 있습니다.
- 모든 후보가 동일한 labelled 분류 형상 및 partial-bi 회귀 형상을 사용합니다. 각 후보는 명시적인 config와 seed로 한 번 fit됩니다. 여러 seed 결과를 별도 평가할 수 있지만 예측을 합치지 않습니다.

ROM 입력을 추가해도 원래 치수 8개의 스케일은 기본 모델과 같게 유지합니다. MLP는 [-1,1], GP는 [0,1]이며, 추가 ROM 열만 해당 학습 부분집합에서 구한 평균·표준편차로 정규화합니다. XGBoost는 치수·ROM 입력 모두 원래 값을 사용합니다.

## ROM 보정과 누출 방지

`fit_candidate()`는 **호출자가 전달한 Train 부분집합만** 허용합니다. CV에서는 fold의 학습 부분만 전달합니다. ROM은 그 부분집합에 해당하는 완료 힘 곡선만으로 보정합니다. Test 행을 전달하면 오류를 냅니다. ROM 캐시에는 치수·형상 ID·곡선 파일 SHA-256·보정 버전을 반영하므로 서로 다른 fold나 데이터의 계수가 섞이지 않습니다. 동일 fold의 ROM 계수는 세 ROM 모델이 공유합니다. 신경망 epoch 선택용 내부 holdout은 외부 CV 검증 fold와 별개입니다.

ROM 보정에는 `unified-forward/data/completed_force_curves.json`이 추가로 필요합니다. 기존 작은 CSV 배포 묶음에는 포함되지 않았으므로, scalar CSV만으로 ROM을 재학습할 수 없습니다. 실행 시 파일이 없으면 명확한 오류를 냅니다.

## 실행

저장소 루트에서 통합 환경의 Python을 사용합니다. `--runs`에는 기존 공통 데이터를 포함하는 상위 결과 경로를 지정합니다. 상대 경로는 예시이며 사용자별 절대 경로를 코드에 고정하지 않습니다.

```sh
# 구축된 후보 및 잠정 기본 설정 확인: 학습하지 않음
python integrated/cli.py --runs /path/to/output forward-six describe

# 소규모 Train 형상으로 6개 후보의 fit/predict/save/reload 실행 확인
python integrated/cli.py --runs /path/to/output forward-six smoke

# 향후 튜닝 도구에서 사용할 API (현재 자동 탐색은 없음)
# from forward_six.models import fit_candidate, predict_candidate, default_config

# 명시적으로 실행할 때만 전체 Train의 단일 설정 학습
python integrated/cli.py --runs /path/to/output forward-six train --models mlp --seed 0

# 명시적으로 실행할 때만 설정 1개의 공통 5-fold 평가 (Test 미사용)
python integrated/cli.py --runs /path/to/output forward-six cv --models rom_gp --config /path/to/config.json

python integrated/cli.py --runs /path/to/output forward-six predict \
  --checkpoint /path/to/output/forward-six/models/mlp/seed0.joblib \
  --input /path/to/geometries.csv --output /path/to/predictions.csv
```

`--config`는 `classification`, `regression` 섹션의 설정을 각각 받습니다. 이름을 잘못 지정하면 오류를 냅니다. 명시한 값만 잠정 기본값을 덮어씁니다. MLP의 `epochs`를 직접 주면 내부 epoch 선택을 생략하고 그 횟수로 학습합니다. 그 외에는 내부 Train holdout으로 epoch 수를 고른 뒤 전체 전달 Train으로 단일 신경망을 새로 학습합니다.

`smoke`는 클래스당 최대 16개 Train 형상, 축소된 모델/4 epochs/제한된 ROM 보정만 사용합니다. 결과는 `forward-six/smoke/validation.json`에 저장합니다. **여기서 성공은 실행·직렬화 검증 성공이며 예측 정확도나 Test 성능이 아닙니다.**

## 다음 단계

잠정 기본값은 기존 config 0을 바탕으로 모델을 실행 가능하게 만든 값이며 최적 설정이 아닙니다. 사용자와 합의한 첫 탐색 예산은 후보별 분류 20회·회귀 20회이며, 20회 이후 연장은 별도로 결정합니다. 기존 평가 결과를 이 단일 모델 구성의 성능으로 재사용하지 않습니다.

새 요청의 네 작업 구조(분류 / 각도·변위 / 장벽 / 응력)는 [GROUPED.md](GROUPED.md)에 별도로 기록했습니다. 새 구조의 세 회귀 그룹은 각각 20회 TPE 튜닝·기존 결과 비교·감사를 완료했습니다. 전체 Train 재학습과 새 Test 평가 전입니다.

각도·변위 단독 모델 20회 비교와 후속 후보별 재개는 [INDEPENDENT.md](INDEPENDENT.md)에 기록했습니다. 현재 공동/단독 비교에서는 MLP 공동 회귀를 추가 튜닝 후보로 권장하며, 최초 3출력 MLP도 참고 대조군으로 보존합니다.
