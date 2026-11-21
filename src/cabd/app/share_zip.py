"""Bundle the demo app into a self-contained zip for teammates (build/CABD_demo.zip).

Copies only what the app imports, keeping the package layout (the zip root plays the repo root, so cabd.config finds
models/ next to src/):
  CABD_demo/src/cabd/                      config, data, metrics, geometry/params, models/*, inverse/core, app/*
  CABD_demo/models/app_models_n1536.joblib (bundle written by cabd.app.bundle)
plus setup.bat / run_app.bat / requirements.txt / README.md written here. The sources are only read.

usage: python -m cabd.app.share_zip [--out build/CABD_demo.zip]
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from .. import config
from .bundle import BUNDLE

NAME = "CABD_demo"
PKG = Path(__file__).resolve().parents[1]           # src/cabd
MODULES = ["__init__.py", "config.py", "data.py", "metrics.py",
           "geometry/__init__.py", "geometry/params.py",
           "models/__init__.py", "models/io.py", "models/gp.py", "models/mlp.py", "models/rom.py", "models/rom_gp.py",
           "models/referee.py", "models/cvae.py",
           "inverse/__init__.py", "inverse/core.py",
           "app/__init__.py", "app/engine.py", "app/server.py", "app/bundle.py"]


def files():
    out = {f"src/cabd/{m}": PKG / m for m in MODULES}
    out.update({f"src/cabd/app/static/{p.name}": p for p in (PKG / "app" / "static").iterdir() if p.is_file()})
    out[f"models/{BUNDLE.name}"] = BUNDLE
    return out


REQUIREMENTS = """\
# exact versions the models were saved with (scikit-learn pickles need the same version)
numpy==2.5.4
pandas==3.1.1
scipy==1.18.1
scikit-learn==1.9.1
joblib==1.6.0
torch==2.15.0
"""

SETUP = r"""@echo off
rem One-time setup: creates .venv next to this file and installs the packages (needs Python 3.12 and internet).
cd /d "%~dp0"
where py >nul 2>nul && (py -3.12 -m venv .venv) || (python -m venv .venv)
if not exist ".venv\Scripts\python.exe" (
  echo Python 3.12 was not found. Install it from python.org, then run setup.bat again.
  pause & exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
echo.
echo Setup finished. Double-click run_app.bat to start the app.
pause
"""

RUN = r"""@echo off
rem Starts the app at http://localhost:8765 and opens the browser. Close this window to stop it.
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
set PYTHONPATH=%~dp0src
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8765"
"%PY%" -X utf8 -m cabd.app.server --port 8765
pause
"""

README = """\
# Bistable particle designer (CABD 데모 앱)

치수 → 성능 예측(Forward)과 목표 성능 → 치수 역설계(Inverse)를 보여 주는 로컬 웹 앱입니다.
모델은 Ansys 학습 형상 1,536개로 학습한 것을 그대로 씁니다(다시 학습하지 않음).

## 처음 한 번: 설치
1. Python 3.12를 설치합니다(python.org, 설치할 때 "Add python.exe to PATH" 체크).
2. 이 폴더를 한글·공백 없는 경로에 풀면 안전합니다. 예: `C:\\CABD_demo`
3. `setup.bat`을 더블클릭합니다. 패키지(약 1 GB, 주로 PyTorch)를 내려받아 `.venv` 폴더에 설치합니다.

## 실행
- `run_app.bat`을 더블클릭하면 서버가 뜨고 브라우저가 열립니다(http://localhost:8765).
- 브라우저가 먼저 열려 빈 화면이면 몇 초 뒤 새로고침하세요.
- 끝낼 때는 검은 창을 닫습니다.

## 화면
- Forward: 치수를 입력하면 형상이 바로 바뀌고, Compute를 누르면 쌍안정 판정, 두 번째 안정점(각도, 변위),
  에너지 장벽과 변형 애니메이션이 나옵니다.
- Inverse: θ₂, u₂, ΔE 중 원하는 값만 넣고 Find dimensions를 누르면 치수와 허용 범위가 나옵니다(30초~3분; 목표를 하나만 넣으면 더 오래 걸림).

## 모델
출력마다 시험 세트 오차가 가장 낮았던 모델을 씁니다.
- θ₂, u₂: MLP 앙상블(시드 5개 × 5개 = 25개) / ΔE: ROM+GP / 쌍안정 판정: GP 분류기 / 역설계: 같은 GP + SLSQP(시작점 20개)
- 가운데·오른쪽 형상은 유한요소 결과가 아니라, 예측값에 맞춘 운동학적 도식입니다(강체 팔, 휘는 힌지와 플랜지).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.BUILD / f"{NAME}.zip")
    a = ap.parse_args()
    src = files()
    missing = [str(s) for s in src.values() if not s.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}  (build the bundle first: python -m cabd.app.bundle)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / NAME
        for rel, s in src.items():
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dst)
        (root / "requirements.txt").write_text(REQUIREMENTS, encoding="ascii")
        (root / "setup.bat").write_text(SETUP.replace("\n", "\r\n"), encoding="ascii")
        (root / "run_app.bat").write_text(RUN.replace("\n", "\r\n"), encoding="ascii")
        (root / "README.md").write_text(README, encoding="utf-8")
        if a.out.exists():
            a.out.unlink()
        shutil.make_archive(str(a.out.with_suffix("")), "zip", root_dir=tmp, base_dir=NAME)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB, {len(src) + 4} files)")


if __name__ == "__main__":
    main()
