# classifier_guidance 연구노트

## 1. 목적

소수 클래스 fake를 만드는 diffusion 샘플링에 classifier guidance를 붙이면 오버샘플링 성능이 더 좋아지는지 확인한다. 가이던스 없는 같은 diffusion 모델, 같은 분할, 같은 스텝 수에서 가이던스만 추가해 seed별로 짝지어 비교한다.

## 2. 실험 구성

| 폴더 | 방법 |
|---|---|
| `exp0_baseline` | 증강 없이 분류기만 학습 |
| `exp1_oversampling` | diffusion 500 epoch 학습 후 respacing 10스텝으로 fake 풀을 만들어 오버샘플링 (가이던스 없음) |
| `exp2_classifier_guidance` | exp1의 EMA diffusion으로 풀을 만들 때 classifier guidance를 적용 (scale 0.5 · 1 · 2 · 4 · 10) |

## 3. 실험 설정

**데이터** (`AM_image_data_3split`)
- 한 장을 3조각(k0/k1/k2) 낸 데이터라 **원본 단위로 분할**한다 (`build_imbalanced_group_split`). 64×64로 resize 후 [-1, 1]로 정규화.
- train 800 / 80 / 80장, test 클래스당 100장. seed 0~29(30회 반복)마다 분할을 새로 뽑는다.
- 데이터 경로는 환경변수 `DATA_ROOT`로 지정한다. 기본값은 노트북 기준 상대경로 `../../data/AM_image_data_3split`이다.

**Diffusion / fake 풀 (exp1)**
- 조건부 U-Net(dim=32), T=1000, Adam lr 2e-4, EMA 0.999, 500 epoch 학습.
- timestep respacing 10스텝으로 소수 클래스(1, 2)를 클래스당 720장씩 생성해 풀을 만든다.

**Classifier guidance (exp2)**
- diffusion은 새로 학습하지 않고 exp1의 T=1000 EMA 체크포인트(seed별)를 쓴다.
- 샘플링 매 스텝에서 평균 μ에 `s · Σ · ∇ log p(y | x_t)`를 더한다. 분산 Σ는 학습하지 않고 posterior 분산을 고정해 쓴다.
- 가이던스용 분류기 `p(y | x_t)`는 노이즈 낀 이미지를 받는 `NoisyClassifier`다. seed별 train 이미지로 100 epoch 학습하고(Adam lr 3e-4), 클래스 빈도에 반비례하는 가중치를 둔 CE를 쓴다.
- 초기 노이즈는 scale마다 같다.

**평가 분류기 (세 실험 공통)**
- 작은 CNN, Adam lr 2e-4 (betas 0.5, 0.9), batch 40, 100 epoch(= 2400 step).
- 매 step 실제 배치의 Normal 수에 맞춰 부족한 소수 클래스를 풀에서 꺼내 채운다.

## 4. 결과 (30 seed, macro, 평균 ± 표준편차)

| 실험 | Accuracy | F1 |
|---|---|---|
| exp0 baseline | 0.751 ± 0.038 | 0.722 ± 0.052 |
| exp1 오버샘플링 (가이던스 없음) | 0.795 ± 0.052 | 0.782 ± 0.066 |
| exp2 guidance s=0.5 | 0.801 ± 0.052 | 0.792 ± 0.063 |
| exp2 guidance s=1 | 0.802 ± 0.050 | 0.792 ± 0.058 |
| exp2 guidance s=2 | 0.812 ± 0.052 | 0.804 ± 0.063 |
| exp2 guidance s=4 | **0.816 ± 0.042** | **0.811 ± 0.050** |
| exp2 guidance s=10 | 0.810 ± 0.040 | 0.804 ± 0.047 |

**대응표본 t-test (macro F1, seed 30쌍)**

| 비교 | 평균 차이 | p | 우세 seed |
|---|---|---|---|
| exp1 − exp0 | +0.060 | 0.0004 | - |
| s=0.5 − exp1 | +0.009 | 0.29 | 20/30 |
| s=1 − exp1 | +0.010 | 0.25 | 20/30 |
| s=2 − exp1 | +0.022 | 0.039 | 24/30 |
| s=4 − exp1 | +0.028 | 0.0044 | 22/30 |
| s=10 − exp1 | +0.022 | 0.089 | 19/30 |

모든 scale에서 baseline보다는 유의하게 높았다(p < 0.001).

**클래스별 Recall**

| 실험 | Normal | Underfill_50FR | Underfill_Fan |
|---|---|---|---|
| exp0 baseline | 0.985 | 0.337 | 0.933 |
| exp1 오버샘플링 | 0.930 | 0.537 | 0.917 |
| s=2 | 0.934 | 0.584 | 0.917 |
| s=4 | 0.931 | 0.609 | 0.909 |
| s=10 | 0.931 | 0.593 | 0.905 |
