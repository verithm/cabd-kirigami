# 데이터

## 설계 변수

단면 형상 파라미터화 v3(`src/cabd/geometry/params.py`). 원점: x = 0은 팁 힌지 바깥면, z = 0은 대칭면.

| 기호 | 의미 | 범위 | 단위 | 기준 형상 |
|---|---|---|---|---|
| `t_h` | 팁 힌지 ligament 두께 | 0.1 – 0.4 | mm | 0.3 |
| `L_h` | 팁 힌지 길이(= 슬롯 폭) | 2.0 – 4.0 | mm | 3.0 |
| `t_a` | 꼭짓점 힌지 ligament 두께 | 0.1 – 0.4 | mm | 0.3 |
| `L_a` | 꼭짓점 힌지 길이 | 2.0 – 4.0 | mm | 3.0 |
| `L` | 팔 길이: x = 0 → 꼭짓점 바깥면 | 5.0 – 8.0 | mm | 7.0 |
| `t_f` | 플랜지 두께 | 0.8 – 1.8 | mm | 1.2 |
| `c` | 꼭짓점 바깥면 ↔ 뒷벽 안쪽면 간격 | 2.0 – 4.0 | mm | 3.0 |
| `rho` | 공통 필렛 비율(필렛 반지름 = rho × 힌지 길이 / 2) | 0.3 – 0.8 | – | 0.533 |

**고정값:** 밑변 반길이 Z_o = 12.7 mm(밑변 25.4 mm), 뒷벽 두께 `t_w` = 3 mm, 압출 깊이 6 mm.

**유도량:** 팔 안쪽 모서리 각 α는 tan α = (Z_o − t_f − L_h − t_h − L_a/2) / (L − t_a)로 정해집니다. 그 밖의 유도 치수는 `params.derived()`.

**형상 제약(`params.check_constraints`):** 0 < α < 90°, 직선 ligament 길이 > 0, 자기 교차·물체 겹침 없음, 최소 두께 ≥ 0.1 mm(가정: 출력 가능한 최소값), 추정 메시 노드 ≤ 128,000(Ansys Student 한도).

## 출력

| 출력 | 정의 | 단위 | 학습 |
|---|---|---|---|
| 상태별 클래스 | 반력 F(u) 부호로 판정: `bi`, `mono`, `neg_end`, `unknown` | – | 3-class 라벨로 묶음 |
| θ₂ | u₂에서 위쪽 팔 A→B 벡터의 회전각 | ° | 그대로 |
| u₂ | 힘이 다시 0을 지나는 변위(두 번째 안정점, substep 사이 보간) | mm | 그대로 |
| ΔE_r | 역방향 장벽 U(u₁) − U(u₂), u₁ = 힘이 처음 0 아래로 내려가는 변위 | mJ | log₁₀ |

라벨 규칙:

| 상태 클래스 | 조건 | 학습에서 |
|---|---|---|
| `bi` | 힘이 0 아래로 내려갔다 다시 0을 지나 올라옴 | bi |
| `neg_end` | 곡선 끝까지 0 아래 | bi(회귀에서는 제외) |
| `mono` | 0 아래로 안 내려가고 곡선이 12 mm 이상 진행 | mono |
| `unknown` | 0 아래로 안 내려가고 12 mm 전에 끝남 | 제외 |

3-class = (살짝 젖음) > (완전 젖음): `bi>bi`, `bi>mono`, `mono>mono`. 회귀는 살짝 젖음 클래스가 `bi`인 형상만 씁니다. 결정 이유는 [DECISIONS D4, D5](DECISIONS.md#d4-안정성-라벨).

## 데이터셋 파일 (`data/`)

| 파일 | 내용 | 행 |
|---|---|---|
| `CABD_dataset.csv` | 기본 학습 512개(`train_base`), 시험 256개(`test`), 자리만 있는 행(`adaptive` 192, `inverse_validation` 40) | 1,000 |
| `CABD_dataset_ext.csv` | 추가 학습 `train_ext1` 512개, `train_ext2` 512개 | 1,024 |
| `CABD_test_results.csv` | 시험 256개(learning curve 평가용) | 256 |
| `CABD_dataset_columns.csv` | 열 설명(아래 요약) | – |
| `geometry_table.csv` | 형상별 치수·유도량·해석 상태 표(과거 시점 스냅샷, [D23](DECISIONS.md#d23-파일-사건)) | – |
| `folds.csv` | 512개의 고정 5-fold | 512 |
| `queries.csv` | inverse 질의(dev, final; `ansys40` = Ansys 검증 40개) | – |
| `curves/curves_base.json.gz` | 기본 학습·시험 형상의 힘-변위 곡선 | – |
| `curves/curves_ext.json.gz` | 추가 학습 형상의 곡선(ROM 맞춤용) | – |
| `curves/curves_test.json.gz` | 시험 256개 곡선(결과 뷰어 형식) | – |
| `sampling/params_*.csv`, `params_meta.json` | Sobol 표본 표와 seed·범위·필터 기록 | – |
| `kit_check/` | 키트 검증 그림용 개발 실행 곡선 3개 | – |

### 학습·시험 구성

| 세트 | 개수 | 회귀(살짝 젖음 bi) | 3-class 라벨 (bi>bi / bi>mono / mono>mono / 없음) |
|---|---|---|---|
| 학습 512 (`train_base`) | 512 | 408 | 237 / 210 / 29 / 36 |
| 학습 1,536 (+ext1 +ext2) | 1,536 | 1,217 | 698 / 638 / 83 / 117 |
| 시험 (`test`) | 256 | 207 | 117 / 112 / 12 / 15 |

### 열 요약 (`CABD_dataset_columns.csv`)

- 공통: `geom_id`(TR = 학습, TE = 시험), `group`, `sobol_seed`, 설계 변수 8개.
- 상태마다(`partial_*`, `full_*`):
  - `run_status`(not_run / finished / failed), `converged`(15 mm 도달 여부), `class`
  - `F_max_N`, `u1_mm`, `F_min_N`, `u2_mm`, `dE_f_mJ`(정방향 장벽 U(u₁) − U(0)), `dE_r_mJ`, `theta2_deg`, `u_end_mm`
  - `mesh_rule`(값을 낸 해석의 힌지 메시), `nodes`

## 수집 규칙 (Ansys)

자세한 사용법은 `ansys_kit/README.md`(키트 실행자용, 한국어).

- **해석:** 형상마다 2번.
  - 살짝 젖음: 프레임 E = 369.5 MPa, 힌지 E = 12.7 MPa.
  - 완전 젖음: 전체 E = 12.7 MPa.
- **경계조건:** 뒷벽 바깥면 고정, 꼭짓점 힌지 안쪽 모서리 +15 mm 변위 제어, bonded 접촉, large deflection.
- **풀이:** 자동 시간 스텝(초기 40, 최소 20, 최대 160 substep), 직접법(sparse) 솔버.
- **메시:** 전체 0.6 mm, 힌지 3개 0.1 mm hex(깊이 방향 8층). 메시 품질 검사에 실패하면 더 굵은 설정으로 자동 전환합니다. 수렴하지 않으면 힌지 0.2 mm, 4층으로 한 번 더 해석합니다.
- **수렴 미완료:** 15 mm 전에 멈춘 해석(`not_converged`)도 버리지 않습니다. 첫 시도와 재시도 중 더 멀리 간 곡선에서 같은 규칙으로 값을 추출합니다.
- **분배:** 기본 768개를 PC 3대, ext1·ext2를 PC 3대씩, 실패분은 재시도 키트로 돌렸습니다. 키트 목록은 `src/cabd/ansys/kits.py`.
- Ansys Student 2026 R1. 형상 1개(2상태)에 약 10–20분(키트 README 추정).
- **개인정보 제거:** 공개본 `ansys_kit/template.wbpz`는 원본에서 Windows 사용자 폴더 경로(`C:\Users\<이름>\` → `C:\Users\user\`, `designPoint.wbdp`와 `.mechdb`)와 STEP 작성자 이름만 지운 것입니다(`cabd.ansys.scrub`, 다른 멤버는 바이트 그대로). Ansys를 여기서 실행할 수 없어 지운 템플릿을 Workbench에서 다시 열어 보지는 않았습니다(가정: Workbench 압축은 위치를 옮겨도 열리고, 저장된 절대 경로는 열 때 새로 잡힙니다).

## 원시 결과 (Release asset)

- Ansys 결과 zip 12개(약 232 MB)를 GitHub Release에 올리고 `artifacts/ansys_raw/`에 풀어 씁니다(`CABD_ANSYS_RAW`로 위치 변경 가능).
- 형상마다 상태별 진행 기록(`progress.csv`), 힘-변위 곡선, 랜드마크 변위가 들어 있습니다.
- 데이터셋 재생성(`cabd.ansys.dataset`)과 검증 채점(`cabd.verify.score`)의 입력입니다. 읽기만 하고 고치지 않습니다.
- **개인정보 제거:** Release의 zip은 키트가 만든 원본 `collect_result_PC<번호>_<컴퓨터 이름>_<날짜>_<시각>.zip`에서 컴퓨터 이름과 사용자 폴더 경로를 지운 사본입니다(`python -m cabd.ansys.scrub zips <원본 폴더> <출력 폴더>`). 파일 이름은 `collect_result_PC<번호>_<날짜>_<시각>.zip`, `machine_<컴퓨터 이름>.txt`는 `machine_HOST.txt`, 상태 파일의 `computer=`와 MAPDL 로그의 Machine Name 표는 `HOST`, 로그의 `C:\Users\<이름>\`는 `C:\Users\user\`가 됩니다. 곡선·랜드마크·`progress.csv`와 상태·메시 값은 그대로라 데이터셋이 바이트 단위로 같게 재생성됩니다(`tests/test_ansys_scrub.py`, `tests/test_ansys_datasets.py`). 같은 규칙으로 `results/verify/ansys_verify_scored.csv`의 `ansys_source` 열(zip 이름)에서도 컴퓨터 이름을 뺐습니다.

## 알려진 데이터 이슈

- **`unknown`:** 학습 512개 중 36개(7%)는 라벨이 없어 분류에서 빠집니다.
- **mono>mono가 적음:** 512개 중 29개. referee의 mono>mono recall이 0.75에 그칩니다.
- **수렴 미완료:** 일부 해석은 15 mm 전에 멈췄습니다. 값은 추출하지만 u₂가 곡선 끝 가까이 있으면 불확실합니다(가정).
- **ext2 Sobol index 1269:** 형상 제약에 걸려 건너뛰었습니다.
- **`geometry_table.csv`:** 최종 키트 실행 전 스냅샷입니다([D23](DECISIONS.md#d23-파일-사건)).
- **자리만 있는 행:** `CABD_dataset.csv`의 `adaptive` 192개와 `inverse_validation` 40개는 계획 단계의 빈 행이며 학습에 쓰지 않습니다.
