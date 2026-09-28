# Diffusion 실험 모음

노이즈 분포(Gaussian / Student-t / Gamma)를 바꾼 확산모형과, 확산모형을 이용한 불균형 데이터 증강 실험.


## 설치

```bash
pip install -r requirements.txt
```


## 데이터

모든 코드는 저장소 루트의 `data/`를 읽는다.

```
data/
├── MNIST/              자동 다운로드 (torchvision)
├── FashionMNIST/       자동 다운로드 (torchvision)
└── 256/                직접 준비 — ImageFolder 형식
    ├── Normal/
    ├── Underfill_50FR/
    └── Underfill_Fan/
```

- MNIST · FashionMNIST는 처음 실행할 때 `data/`에 자동으로 내려받는다.
- `256/`(공정 이미지)은 공개 데이터가 아니므로 위 구조로 직접 넣어야 한다. `diffusion_3player/`만 사용한다.

## 실행

노트북과 스크립트는 **자기 폴더에서** 실행한다 (상대경로 기준). 예:

```bash
cd diffusion_t_distribution/t_ddpm
jupyter notebook 1_nu_sweep.ipynb
```

```bash
cd diffusion_t_distribution/t_discrete_tail
python run.py embed
python run.py train --nu 3.0 --out results/nu3
```

- 스크립트 인자와 실행 순서는 각 폴더 README의 「실행」 절을 따른다.
- 실험 결과표(`results/*.xlsx`, `results/*.csv`)는 저장소에 포함한다. 학습한 가중치(`*.pt`, `*.pth`), 생성 샘플(`samples/`), 데이터(`data/`)는 포함하지 않는다.
- 원래 설정으로 돌리면 노트북 하나에 수십 분 ~ 수 시간이 걸린다.
