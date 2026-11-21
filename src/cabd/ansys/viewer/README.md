# CABD 형상 뷰어와 데이터셋

형상 1,000개의 해석 결과를 검색하고 필터링해서 볼 수 있는 폴더입니다.

## 준비물

- Python 3.10 이상
- `numpy`, `pandas`, `matplotlib`

```bash
pip install numpy pandas matplotlib
```

## 실행

1. 폴더를 원하는 곳에 둡니다.
2. `Geometry_Viewer.bat`을 더블클릭합니다. 브라우저가 열립니다.
   - Mac이나 Linux에서는 이 폴더에서 `python viewer.py`를 실행합니다.
3. 검은 창은 쓰는 동안 열어 두고, 끝나면 닫습니다.

## 파일

| 파일 | 내용 |
|---|---|
| `CABD_dataset.csv` | 형상 1개가 1행인 데이터셋 (pandas로 바로 읽을 수 있음) |
| `CABD_dataset_columns.csv` | 각 열의 뜻, 형식, 단위 |
| `CABD_viewer_data.json.gz` | 뷰어가 그리는 힘, 에너지 곡선과 변형 형상 |
| `viewer.py`, `viewer.html`, `param_geom_v3.py` | 뷰어 프로그램 |

```python
import pandas as pd
df = pd.read_csv("CABD_dataset.csv")
```

## 알아 둘 점

- 시험 세트(TE로 시작하는 형상)의 결과는 모델 개발이 끝날 때까지 봉인되어 있습니다. 이 폴더에는 설계값과 해석 진행 상태만 들어 있습니다.
- 보강 학습(AD)과 역설계 검증(IV) 행은 아직 형상이 정해지지 않은 자리입니다.
- 데이터는 계속 갱신됩니다. 새 버전을 받으면 폴더를 통째로 바꾸면 됩니다. 뷰어 왼쪽 위의 `Updated` 시각으로 버전을 확인할 수 있습니다.
