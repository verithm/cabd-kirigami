# 재현 방법

## 환경

Python 3.12 환경 두 개를 씁니다. cadquery가 자체 OCC 패키지를 고정하기 때문에 CAD 환경을 따로 둡니다.

| 환경 | 용도 | 설치 |
|---|---|---|
| ML | 학습, 평가, inverse, 검증 채점, 데이터셋, 키트, 앱 | `pip install -e .[ml]` |
| CAD | STEP 생성만 | `pip install -e .[cad]` |

- 버전은 `pyproject.toml`에 고정했습니다(numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, torch 2.15.0 CPU, lightgbm 4.7.0, matplotlib 3.11.2, cadquery 2.8.0). 저장된 가중치는 sklearn 객체를 담고 있어 **sklearn 버전이 다르면 열리지 않을 수 있습니다.**
- 설치하지 않고 쓰려면 레포 루트에서 `$env:PYTHONPATH = "src"`를 지정합니다. 아래 명령은 모두 레포 루트에서 `python -X utf8 -m ...`으로 실행합니다.
- STEP 생성을 ML 환경에서 호출할 때는 CAD 환경의 python 경로를 `CABD_CAD_PYTHON`에 지정합니다.

## Release asset 받기

GitHub Release의 압축 파일 두 개를 레포 루트의 `artifacts/`에 풉니다(git에서 제외됨, `CABD_ARTIFACTS`로 위치 변경 가능). 각 zip 안에 `models/`, `ansys_raw/` 폴더가 들어 있으므로 `artifacts/`를 대상으로 풀면 됩니다. 무결성 확인은 Release의 `SHA256SUMS.txt`로 합니다.

| Release asset | 내용 | 크기 |
|---|---|---|
| `cabd_models.zip` | `artifacts/models/` | 약 241 MB |
| `cabd_ansys_raw.zip` | `artifacts/ansys_raw/` | 약 232 MB |

```powershell
Expand-Archive cabd_models.zip -DestinationPath artifacts
Expand-Archive cabd_ansys_raw.zip -DestinationPath artifacts
```

```
artifacts/
  models/       GP, ROM+GP, referee 전체와 나머지 학습 크기·시드의 가중치 (42개, 약 473 MB)
  ansys_raw/    Ansys 원시 결과 zip 12개 (약 232 MB, 개인정보를 지운 사본: docs/DATA.md)
```

- 데모 앱, MLP·cVAE inverse, 대부분의 테스트는 Release 없이 돌아갑니다.
- GP 계열 inverse, 데이터셋 재생성, 검증 채점에는 Release가 필요합니다.
- 파일 목록과 sha256: `models/MANIFEST.csv`. 모델을 못 찾으면 `cabd.models.load`가 Release를 받으라는 오류를 냅니다.

## 단계별 명령

### 1. 표본과 STEP

```powershell
python -X utf8 -m cabd.geometry.sampling base                          # 학습 512, 시험 256
python -X utf8 -m cabd.geometry.sampling ext --stage 1                 # ext1 512
python -X utf8 -m cabd.geometry.sampling ext --stage 2 --tables build/sampling
python -X utf8 -m cabd.geometry.step build build/step/base params_train.csv params_test.csv   # CAD 환경
```

- 기본 출력은 `build/`(git 제외)입니다. `data/sampling/`의 표와 같은지 테스트가 확인합니다.
- 보조: `cabd.geometry.range_check`(범위 검사), `cabd.geometry.variables_figure`(설계 변수 그림), `cabd.geometry.step verify|compare|fingerprint`.

### 2. Ansys 키트

```powershell
python -X utf8 -m cabd.ansys.kit base        # 또는 ext1, ext2, retry1, final, verify, verify_all, verify2, all
```

- 출력: `build/kits/<kit>/`, `build/kit_zips/<kit>.zip`. STEP이 없으면 `CABD_CAD_PYTHON`으로 만듭니다.
- 검증 키트는 `--inverse-dir`에 전체 정밀도 inverse 결과 폴더를 주면 원본과 바이트 단위로 같습니다. 주지 않으면 9자리 후보 목록을 써서 9번째 자리가 약간 다를 수 있습니다(형상은 같음).
- **Ansys는 키트를 받은 다른 PC에서 실행합니다.** 사용법은 키트 안 `README.md`. 결과 zip은 `artifacts/ansys_raw/`에 둡니다(공개하려면 먼저 `python -m cabd.ansys.scrub zips <원본 폴더> artifacts/ansys_raw`로 컴퓨터 이름·사용자 경로를 지웁니다. 리더는 두 이름 형식을 모두 읽습니다).
- **개인정보 제거:** 공개본 `ansys_kit/template.wbpz`는 원본에서 Windows 사용자 폴더 경로(`C:\Users\<이름>\` → `C:\Users\user\`, `designPoint.wbdp`와 `.mechdb`)와 STEP 작성자 이름만 지운 것입니다(`cabd.ansys.scrub`, 다른 멤버는 바이트 그대로). Ansys를 여기서 실행할 수 없어 지운 템플릿을 Workbench에서 다시 열어 보지는 않았습니다(가정: Workbench 압축은 위치를 옮겨도 열리고, 저장된 절대 경로는 열 때 새로 잡힙니다).

### 3. 데이터셋

```powershell
python -X utf8 -m cabd.ansys.dataset all --out build/dataset    # base, ext, table, share 각각도 가능
```

- `base`: `CABD_dataset.csv`, `CABD_test_results.csv`, 곡선 파일.
- `ext`: `CABD_dataset_ext.csv`, `curves_ext.json.gz`.
- `table`: `geometry_table.csv`(`--final built`가 기본 = 저장된 스냅샷 재현, `--final run` = 최신).
- `share`: 결과 뷰어 묶음(`build/share/`). 뷰어 실행: `python -m cabd.ansys.viewer.app`.

### 4. forward model 학습과 평가

```powershell
python -X utf8 -m cabd.fit --models gp mlp rom_gp cvae referee referee_nogp --train all --seeds 0 1 2 3 4 --jobs 16
python -X utf8 -m cabd.fit --models gp rom_gp cvae referee referee_nogp --train n1536 --seeds 0   # n512, n1024도 같은 방식
python -X utf8 -m cabd.fit --models mlp --train n1536 --seeds 0 1 2 3 4
python -X utf8 -m cabd.evaluate.cv --model gp            # mlp, rom_gp, referee, referee_nogp
python -X utf8 -m cabd.evaluate.learning_curve
python -X utf8 -m cabd.evaluate.joint_output              # --summary-only: 저장된 예측에서 요약만
python -X utf8 -m cabd.evaluate.queries                   # data/queries.csv 확인(--write로 덮어씀)
python -X utf8 -m cabd.plots.correlation                  # distributions, learning_curve
```

- `--train`(하나씩): `all`(512), `fold0`(fold 1–4, dev 질의용), `n512`, `n1024`, `n1536`. 기본 출력은 `artifacts/models/`이고 이미 있는 모델은 건너뜁니다.
- learning curve는 MLP만 시드 5개, 나머지는 시드 0입니다.

### 5. inverse design

```powershell
# v1: 512개 모델, final 질의 106개, 시드 5개 (방법 기본값 gp mlp rom_gp cvae)
python -X utf8 -m cabd.inverse.run --train all --query-set final --seeds 0 1 2 3 4
# v1 dev: fold0 모델, 시작점 수 확인용
python -X utf8 -m cabd.inverse.run --train all --query-set dev --seeds 0 --starts 40
# v2: 1,024·1,536개 모델, Ansys 검증 질의 40개
python -X utf8 -m cabd.inverse.run --train n1024 n1536 --ansys40 --jobs 1
# 대기 모드에 걸린 질의만 다시 실행
python -X utf8 -m cabd.inverse.patch rom_gp:n1024:TE0194 rom_gp:n1024:TE0211 rom_gp:n1536:TE0102 rom_gp:n1536:TE0186 gp:n1536:TE0221 gp:n1536:TE0224
# referee 채점
python -X utf8 -m cabd.inverse.score --train all --query-set final     # 512 referee
python -X utf8 -m cabd.inverse.score --train n1024 n1536               # 같은 크기 referee
```

- 기존 결과 파일은 덮어쓰지 않습니다. 다른 곳에 쓰려면 `--out DIR`.
- v2 GP·ROM+GP 결과는 `--jobs 1`(순차 실행)로 만들었습니다. 300초 상한이 벽시계 기준이라 동시 실행 수가 결과를 바꿉니다([DECISIONS D17](DECISIONS.md#d17-질의당-시간-상한-300초)).

### 6. Ansys 검증 채점

```powershell
python -X utf8 -m cabd.verify.candidates --inverse-dir results/inverse     # 검증 후보 목록
python -X utf8 -m cabd.verify.score --kits 1                               # 키트 1 (results/verify 재현)
python -X utf8 -m cabd.verify.score                                        # 키트 1, 2 전부
python -X utf8 -m cabd.verify.score --self-test                            # 시험 형상 자체로 파이프라인 확인
```

### 7. 데모 앱

```powershell
python -X utf8 -m cabd.app.bundle                 # models/app_models_n1536.joblib (45.3 MB) 다시 만들기
python -X utf8 -m cabd.app.server --port 8765     # http://localhost:8765, --no-warm-up 가능
python -X utf8 -m cabd.app.share_zip              # build/CABD_demo.zip
```

Windows에서는 `tools/run_app.bat`.

## 걸리는 시간 (16코어 PC 기준, 가정: 다른 작업 없음)

| 단계 | 시간 |
|---|---|
| Ansys 해석 | 형상 1개(2상태)에 약 10–20분, PC 3대로 768개에 수일 |
| GP 학습(512, 출력 1개) | 약 4–5분 |
| 학습 1,536개 전체(GP 계열) | 수 시간 |
| GP CV 일부(시드 1, fold 1) | 약 24분 |
| inverse v1(질의당) | MLP·cVAE 수십 초 이하, GP 약 2분, ROM+GP 약 4–5분(상한 300초) |
| 앱 inverse 요청 | 시작점 20개 병렬, 수십 초 |

## 테스트

```powershell
python -X utf8 tests/run_tests.py                 # 전부
python -X utf8 tests/run_tests.py test_ml         # 필터: test_ml, test_ansys, test_geometry, inverse, app
```

- pytest 없이 돌아가는 작은 실행기입니다. 테스트 함수는 pytest 형식이라 pytest로도 돌릴 수 있습니다.
- `CABD_LEGACY`: 원본(레포 이전) 프로젝트 폴더. 지정하면 원본 코드·결과와 직접 비교하고, 없으면 그 테스트는 건너뜁니다. 대부분의 비교는 `tests/fixtures/`의 작은 기준값으로 원본 없이도 돌아갑니다.
- `CABD_SLOW=1`: 다시 학습하는 무거운 테스트(수 분–수 시간)까지 실행.
- `CABD_RAW_ORIGINAL`: 지우기 전 원시 결과 zip 폴더(선택). `test_ansys_scrub`가 공개본 = scrub(원본)을 원본 프로젝트 폴더에 없는 zip(PC1, final)까지 확인합니다.
- 확인하는 것: 변환한 가중치 58개의 예측(차이 ≤ 1e-10, 대부분 0), OOF 지표·learning curve·joint output 요약, 그림(픽셀 단위), Sobol 표, 데이터셋과 곡선(바이트 단위), 과거 키트 18개, 검증 채점, inverse 목적함수와 질의 결과, 앱 출력.

### BLAS 스레드 수

GP 학습 결과는 BLAS 스레드 수에 따라 최대 약 1e-4 달라집니다. 원본과 비트 단위로 맞추려면 원본과 같은 스레드 수를 프로세스 시작 전에 정해야 합니다.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"     # 전체 학습은 1, GP·ROM+GP CV는 3이었음
```

무거운 재현 테스트는 이 조건으로 별도 프로세스에서 돌립니다([DECISIONS D22](DECISIONS.md#d22-재현성-blas-스레드-수)).

## Windows 대기 모드 주의

- inverse의 300초 상한은 벽시계 기준이라 **노트북이 대기 모드에 들어간 시간도 셉니다.** 실제로 질의 6개가 이 때문에 시작점을 덜 돌았습니다([DECISIONS D19](DECISIONS.md#d19-실행-환경-사건-cpu-경합과-대기-모드)).
- 긴 실행 중에는 전원을 연결하고 덮개를 열어 둡니다.
- 끝난 뒤 시간이 420초를 넘은 질의가 있으면 시스템 로그의 대기 기록과 대조하고, 필요하면 `cabd.inverse.patch`로 그 질의만 다시 돌립니다.
