# entropy_guidance 연구노트

## 1. 목적

소수 클래스 fake를 만드는 diffusion 샘플링에 classifier guidance 항과 함께 **분류기 엔트로피를 키우는 항**을 더하면 오버샘플링 성능이 더 좋아지는지 확인한다. 엔트로피 항은 fake를 분류기의 결정 경계 쪽으로 밀어, 분류기가 헷갈려하는 영역의 샘플을 더 만들도록 한다. 같은 diffusion 모델, 같은 분할, 같은 스텝 수에서 샘플링 방식만 바꿔 seed별로 짝지어 비교한다.

$$\nabla_{x_t}\log \hat p_{\gamma,\omega}(x_t)=\nabla_{x_t}\log \hat p_\theta(x_t)+\gamma\,\nabla_{x_t}\log p_\phi(y\mid x_t,t)+\omega\,\nabla_{x_t}H\big(p_\phi(\cdot\mid x_t,t)\big)$$

## 2. 실험 구성

| 폴더 | 방법 |
|---|---|
| `diffusion_train` | 조건부 DDPM을 seed별로 500 epoch 학습해 EMA 체크포인트 저장 (exp1~3이 공유) |
| `exp0_baseline` | 증강 없이 분류기만 학습 |
| `exp1_oversampling` | respacing 10스텝으로 fake 풀을 만들어 오버샘플링 (가이던스 없음) |
| `exp2_classifier_guidance` | 풀을 만들 때 classifier guidance 적용 (scale 1) |
| `exp3_entropy_guidance` | classifier guidance(γ=1)에 엔트로피 항을 더해 풀 생성 (ω = 0.1 · 1 · 3 · 10) |

공통 코드는 폴더 루트의 `data.py`, `diffusion.py`, `models.py`, `utils.py`에 있다.

## 3. 실험 설정

**데이터** (`AM_image_data_3split`)
- 한 장을 3조각(k0/k1/k2) 낸 데이터라 **원본 단위로 분할**한다 (`build_imbalanced_group_split`). 64×64로 resize 후 [-1, 1]로 정규화.
- train 800 / 80 / 80장, test 클래스당 100장. seed 0~9(10회 반복)마다 분할을 새로 뽑는다.
- 데이터 경로는 환경변수 `DATA_ROOT`로 지정한다. 기본값은 노트북 기준 상대경로 `../../data/AM_image_data_3split`이다.

**Diffusion / fake 풀** (`diffusion_train`, exp1)
- 조건부 U-Net(dim=32), T=1000, Adam lr 2e-4, EMA 0.999, 500 epoch 학습. 체크포인트는 `diffusion_train/checkpoints/ddpm_ema_seed{seed}.pt`.
- timestep respacing 10스텝으로 소수 클래스(1, 2)를 클래스당 720장(= 800 − 80)씩 생성해 풀을 만든다.

**가이던스용 분류기** (exp2, exp3 공통)
- 노이즈 낀 이미지 `x_t`와 `t`를 받는 `NoisyClassifier`. seed별 train 이미지에 임의의 t로 노이즈를 입혀 100 epoch 학습한다(Adam lr 2e-4, betas 0.5, 0.9, CE).
- diffusion은 새로 학습하지 않고 위 EMA 체크포인트를 그대로 쓴다.

**Classifier guidance (exp2)**
- 샘플링 매 스텝에서 평균 μ에 `s · Σ · ∇ log p(y | x_t)`를 더한다 (s=1). Σ는 posterior 분산을 고정해 쓴다.

**Entropy guidance (exp3)**
- 매 스텝 평균 μ에 `Σ · ∇ [γ · log p(y | x_t) + ω · H(p(· | x_t))]`를 더한다 (`sample_ddpm_respaced_ent`). γ=1로 고정하고 ω만 바꾼다.
- 엔트로피 H를 키우는 방향이라 fake가 분류기의 결정 경계 쪽으로 이동한다. γ=1, ω=0이면 exp2(s=1)와 같다.
- 초기 노이즈는 ω마다 같다.

**평가 분류기 (모든 실험 공통)**
- 작은 CNN, Adam lr 2e-4 (betas 0.5, 0.9), batch 40, 100 epoch(= 2400 step).
- 매 step 실제 배치의 Normal 수에 맞춰 부족한 소수 클래스를 풀에서 꺼내 채운다.

## 4. 결과 (10 seed, macro, 평균 ± 표준편차)

| 실험 | Accuracy | F1 |
|---|---|---|
| exp0 baseline | 0.752 ± 0.036 | 0.726 ± 0.046 |
| exp1 오버샘플링 (가이던스 없음) | 0.787 ± 0.062 | 0.776 ± 0.074 |
| exp2 classifier guidance s=1 | 0.806 ± 0.060 | 0.797 ± 0.070 |
| exp3 entropy ω=0.1 | 0.800 ± 0.057 | 0.790 ± 0.068 |
| exp3 entropy ω=1 | 0.806 ± 0.061 | 0.797 ± 0.071 |
| exp3 entropy ω=3 | **0.815 ± 0.051** | **0.809 ± 0.057** |
| exp3 entropy ω=10 | 0.807 ± 0.052 | 0.798 ± 0.061 |

**대응표본 t-test (macro F1, seed 10쌍)**

| 비교 | 평균 차이 | p | 우세 seed |
|---|---|---|---|
| exp1 − exp0 | +0.049 | 0.11 | 7/10 |
| exp2 − exp0 | +0.070 | 0.018 | 8/10 |
| ω=3 − exp0 | +0.083 | 0.0038 | 8/10 |
| exp2 − exp1 | +0.021 | 0.089 | 7/10 |
| ω=0.1 − exp1 | +0.015 | 0.24 | 7/10 |
| ω=1 − exp1 | +0.021 | 0.13 | 6/10 |
| ω=3 − exp1 | +0.034 | 0.023 | 9/10 |
| ω=10 − exp1 | +0.023 | 0.15 | 5/10 |
| ω=0.1 − exp2 | −0.007 | 0.095 | 3/10 |
| ω=1 − exp2 | +0.000 | 0.95 | 6/10 |
| ω=3 − exp2 | +0.013 | 0.22 | 7/10 |
| ω=10 − exp2 | +0.002 | 0.89 | 4/10 |

exp3의 모든 ω가 baseline보다 유의하게 높았다(p < 0.03). 가이던스 없는 오버샘플링보다 유의하게 높은 것은 ω=3뿐이고, classifier guidance(exp2)와는 어느 ω에서도 유의한 차이가 없었다.

**클래스별 Recall**

| 실험 | Normal | Underfill_50FR | Underfill_Fan |
|---|---|---|---|
| exp0 baseline | 0.984 | 0.350 | 0.922 |
| exp1 오버샘플링 | 0.925 | 0.533 | 0.904 |
| exp2 s=1 | 0.947 | 0.556 | 0.915 |
| ω=0.1 | 0.932 | 0.558 | 0.909 |
| ω=1 | 0.948 | 0.560 | 0.909 |
| ω=3 | 0.947 | **0.591** | 0.908 |
| ω=10 | 0.947 | 0.560 | 0.913 |
