# Diffusion 실험 모음

노이즈 분포(Gaussian / Student-t / Gamma)를 바꾼 확산모형과, 확산모형을 이용한 불균형 데이터 증강 실험.


## 구성

| 폴더 | 내용 |
|---|---|
| `1_score_matching` | 2D 혼합 분포의 스코어 함수와 t-노이즈 Langevin 샘플링 |
| `2_diffusion_3player` | 공정 이미지 불균형 증강: baseline / conditional / joint / decision boundary diffusion 비교 |
| `3_diffusion_gamma_distribution` | Gamma 노이즈 diffusion과 불균형 Fashion-MNIST 증강 |
| `4_diffusion_t_distribution` | Student-t 노이즈 확산모형 (`t_ddpm`, `t_discrete_coverage`, `t_discrete_tail`) |
| `5_diffusion_step_300` | DDPM T=300에서 baseline / 오버샘플링 / joint 비교 (`256` 데이터) |
| `6_diffusion_step_change_experiment` | 오버샘플링에서 diffusion 스텝 수 T(22 · 30 · 100 · 300)를 학습과 샘플링 모두 바꿔 비교 (`256` 데이터) |
| `7_diffusion_image_filter` | AM 데이터에서 respacing 스텝 수 비교, fake 품질 필터, joint 학습 (`AM_image_data_3split`) |
| `8_classifier_guidance` | AM 데이터에서 classifier guidance를 붙인 오버샘플링, 30 seed 반복 (`AM_image_data_3split`) |

각 폴더의 README에 설정과 결과를 정리했다. 5~8번 폴더에는 연구노트 형식으로 적었다.


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
├── 256/                직접 준비 — ImageFolder 형식
│   ├── Normal/
│   ├── Underfill_50FR/
│   └── Underfill_Fan/
└── AM_image_data_3split/   직접 준비 — ImageFolder 형식, 한 장을 3조각(k0/k1/k2) 낸 이미지
    ├── Normal/
    ├── Underfill_50FR/
    └── Underfill_Fan/
```

- MNIST · FashionMNIST는 처음 실행할 때 `data/`에 자동으로 내려받는다.
- `256/`, `AM_image_data_3split/`(공정 이미지)은 공개 데이터가 아니므로 위 구조로 직접 넣어야 한다. `256/`은 `2_`, `5_`, `6_` 폴더가, `AM_image_data_3split/`은 `7_`, `8_` 폴더가 쓴다.
- 5~8번 폴더의 노트북은 기본 경로 대신 환경변수 `DATA_ROOT`로 데이터 위치를 지정할 수 있다.

## 실행

노트북과 스크립트는 **자기 폴더에서** 실행한다 (상대경로 기준). 예:

```bash
cd 4_diffusion_t_distribution/t_ddpm
jupyter notebook 1_nu_sweep.ipynb
```

```bash
cd 4_diffusion_t_distribution/t_discrete_tail
python run.py embed
python run.py train --nu 3.0 --out results/nu3
```

- 스크립트 인자와 실행 순서는 각 폴더 README의 「실행」 절을 따른다.
- 실험 결과표(`results/*.xlsx`, `results/*.csv`)는 저장소에 포함한다. 학습한 가중치(`*.pt`, `*.pth`), 생성 샘플(`samples/`), 데이터(`data/`)는 포함하지 않는다.
- 원래 설정으로 돌리면 노트북 하나에 수십 분 ~ 수 시간이 걸린다.
