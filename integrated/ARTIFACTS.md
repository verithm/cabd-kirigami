# 통합 결과 파일 받기

## 설치

1. `integrated` 브랜치의 최신 커밋을 받습니다. 작은 CSV·설정은 `artifacts/integrated/`에 이미 포함되어 있습니다.
2. [Release](https://github.com/verithm/cabd-kirigami/releases/tag/integrated-artifacts-2026-11-24)에서 `CABD-integrated-weights-traces.zip`을 받습니다.
3. ZIP을 **저장소 최상위**에 풉니다. ZIP 안에 `artifacts/integrated/` 경로가 이미 있으므로 이 폴더 안에서 다시 풀지 않습니다.
4. [README.md](README.md)의 Python 환경·실행 방법을 따릅니다. `requirements-lock.txt`는 실제 검증 환경입니다.
5. Release의 `manifest.json`으로 파일 SHA-256을 확인할 수 있습니다. CSV·설정 ZIP도 복구·직접 전달용으로 제공합니다.

## 포함 범위

브랜치: `unified-forward/data/{jobs,scalar_targets,folds}.csv`, `unified-forward/forward_selection.json`, `unified-inverse/data/` 전체, `inverse-target-strategies/goals/`, `summary.csv`, `goal_results.csv`, `representatives.csv`, `protocol.json`. 재현을 위해 Forward·Inverse의 `protocol.json`도 포함합니다. 모두 `artifacts/integrated/` 기준 경로입니다.

Release ZIP: `unified-forward/models/sm_mlp/`, `unified-forward/models/sm_gp/`, `unified-inverse/models/conditional_gmm/`, `inverse-target-strategies/runs/*/slsqp/` 전체.

## 실험 해석과 새 결합 실험

- 최종 선정 Forward는 `sm_mlp`입니다. `sm_gp`는 요청에 따라 공유하는 비교 모델이며 선정 모델이 바뀐 것은 아닙니다.
- 기존 기록은 GMM과 SLSQP를 각각 평가한 결과입니다. **GMM 초기값 + SLSQP 결합법의 성능을 이미 검증한 결과는 아닙니다.** 새 결합 실험은 별도 결과 폴더에 저장해 기존 `runs/`를 덮어쓰지 마세요.
- 목표 유형은 실제 Test 49개와 나머지 네 유형 각 100개, 총 449개입니다. 목표 비교 기준은 안정각 ±2°, 변위 ±0.25 mm, 장벽 원래 스케일 ±0.03 mJ, bi→mono 예측 확률 ≥0.9 및 공식 형상 제약입니다. 응력은 제외합니다.
- 평가는 고정 Forward 기반이며 새 ANSYS 검증 결과가 아닙니다. 역설계 최종 모델 선정은 미확정입니다.
- 기존 joblib 모델을 불러올 때 통합 CLI 또는 통합 코드와 같은 import 경로를 사용하세요. 모델 클래스는 `integrated/forward/` 및 기존 `src/cabd/`에 있습니다.

## 추가: 진우 CVAE 가중치

Release의 `CABD-integrated-jw-cvae.zip`을 저장소 최상위에 풀면 다음 파일이 배치됩니다.

- `artifacts/integrated/unified-inverse/models/jw_cvae/seed0.pt`, `seed1.pt`, `seed2.pt`, `complete.json`
- `artifacts/integrated/unified-inverse/tuning/jw_cvae/selection.json`

코드에서 사용하는 정확한 식별자는 `jw_cvae`입니다(`jw_cave`가 아님). `complete.json`과 `selection.json`은 브랜치에도 포함합니다. 가중치는 기존에 학습한 동일 파일이며 재학습하지 않았습니다.

SHA-256은 [추가 manifest](jw-cvae-artifact-manifest.json) 및 Release의 `SHA256SUMS-jw-cvae.txt`에 있습니다. 최초 `artifact-manifest.json`은 원래 배포 묶음의 목록으로 유지합니다. 새 추가 묶음은 별도 manifest로 확인하세요. 선택 기록의 성능은 설정 선택용 검증 결과이며 최종 Test/ANSYS 성능을 의미하지 않습니다.
