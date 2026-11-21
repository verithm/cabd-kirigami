# 학습 데이터 위치

원본 해석 ZIP과 학습 CSV는 이번 코드 공유 커밋에 포함하지 않았습니다.
로컬 프로젝트 `output/통합 전처리 데이터 v4/`의 `scalar_forward_targets.csv`와 `jobs.csv`를 이 폴더에 복사하세요.
루트 `data/CABD_dataset.csv`와 열 이름·판정·분할이 다르므로 그대로 대체하지 마세요.
형상 변수는 `t_h,L_h,t_a,L_a,L,t_f,c,rho`, 상태는 `partial/full`, 역할은 `train/test`입니다.
스칼라 타깃과 힌지 응력의 정의는 상위 README와 `scripts/preprocess_energy_targets.py`, `scripts/preprocess_hinge_stress.py`를 참고하세요.
