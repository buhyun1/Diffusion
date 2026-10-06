# diffusion_image_filter 연구노트

## 1. 목적

AM 공정 이미지(Normal / Underfill_50FR / Underfill_Fan)에서 소수 클래스를 class-conditional DDPM으로 생성해 채우는 방법을 비교한다. 특히 다음 두 가지를 확인한다.

1. 샘플링 스텝 수(respacing)를 줄여도 오버샘플링 성능이 유지되는가.
2. 깨진 fake를 걸러내는 품질 필터와, 분류기와 diffusion을 함께 학습하는 joint 방식이 도움이 되는가.


## 2. 실험 구성

| 폴더 | 방법 |
|---|---|
| `exp0_baseline` | 증강 없이 분류기만 학습 |
| `exp1_oversampling_step_10` | diffusion 500 epoch 학습 후 fake 풀로 오버샘플링. 샘플링 스텝 10 · 20 · 30 · 100 · 300 · 1000 비교 |
| `exp2_joint_step_10` | 필터 없는 joint. 매 배치 fake를 새로 생성(10스텝)하고 분류 손실을 diffusion에도 공유 |
| `exp3_filtered_oversampling` | exp1의 EMA 모델로 풀을 만들 때 품질 필터를 통과한 이미지만 사용 (10스텝) |
| `exp4_filtered_joint` | 필터를 쓰는 joint. 사전학습 400 epoch 후 joint 100 epoch, 협력 항 λ를 0에서 0.001까지 선형 증가 |
| `exp5_filtered_joint_lam0.005` | exp4에서 λ 최댓값만 0.005로 변경 (실험했지만 생성한 fake가 필터에 모두 걸려서 학습이 12시간을 넘어 중단했다) |
| `exp6_filtered_joint_lam0.01` | exp4에서 λ 최댓값만 0.01로 변경 (실험했지만 생성한 fake가 필터에 모두 걸려서 학습이 12시간을 넘어 중단했다) |
| `diffusion_generation` | diffusion 500 epoch 학습 후 생성 샘플 확인 |

`exp3_filtered_oversampling/generate_filter.ipynb`는 필터가 fake를 얼마나 제외하는지 확인하는 노트북이다.

## 3. 실험 설정

**데이터** (`AM_image_data_3split`)
- 한 장을 3조각(k0/k1/k2) 낸 데이터라, 같은 원본의 조각이 train/test에 나뉘지 않도록 **원본 단위로 분할**한다 (`build_imbalanced_group_split`).
- 64×64로 resize 후 [-1, 1]로 정규화. train 800 / 80 / 80장, test 클래스당 100장.
- seed마다 분할을 새로 뽑는다. 데이터 경로는 환경변수 `DATA_ROOT`로 지정한다. 기본값은 노트북 기준 상대경로 `../../data/AM_image_data_3split`이다.

**Diffusion**
- 조건부 U-Net(dim=32), T=1000, Adam lr 2e-4, EMA decay 0.999.
- 샘플링은 timestep respacing(`sample_ddpm_respaced`)으로 스텝 수를 줄인다.
- 소수 클래스(1, 2)를 클래스당 720장(= 800 − 80)씩 생성해 fake 풀을 만든다.

**분류기**
- 작은 CNN, Adam lr 2e-4 (betas 0.5, 0.9), batch 40, 100 epoch(= 2400 step). 모든 실험 동일.
- 매 step 실제 배치의 Normal 수에 맞춰 부족한 소수 클래스를 fake로 채운다.

**품질 필터** (`FakeQualityFilter`)
- 실제 이미지끼리의 최근접 거리의 95% 분위수를 기준으로 삼는다. 생성 이미지가 가장 가까운 실제 이미지와 그 기준보다 멀면 깨진 이미지로 보고 제외한다.
- 통과분이 모자라면 모자란 수의 2배씩 다시 생성해 채운다.

## 4. 결과

### 4-1. 샘플링 스텝 수 (exp1, macro F1, 10 seed)

| 샘플링 스텝 | F1 (평균 ± 표준편차) | baseline 대비 차이 | p |
|---|---|---|---|
| 10 | 0.757 ± 0.096 | +0.012 | 0.73 |
| 20 | 0.804 ± 0.058 | +0.059 | 0.025 |
| 30 | 0.818 ± 0.075 | +0.072 | 0.025 |
| 100 | **0.850 ± 0.036** | +0.105 | < 0.001 |
| 300 | 0.841 ± 0.069 | +0.096 | 0.008 |
| 1000 | 0.846 ± 0.048 | +0.101 | 0.002 |

baseline(exp0)의 macro F1은 0.745 ± 0.047이다.

### 4-2. 방법 비교 (10스텝, macro, 같은 7 seed)

exp4는 필터를 통과한 fake를 채우느라 학습 시간이 너무 오래 걸려서 7개 seed만 돌았다. 그래서 모든 실험을 exp4가 돈 seed(0, 3, 4, 5, 7, 8, 9)로 맞춰 비교했다.

| 실험 | Accuracy | F1 |
|---|---|---|
| exp0 baseline | 0.764 ± 0.032 | 0.740 ± 0.046 |
| exp1 오버샘플링 (10스텝) | 0.771 ± 0.066 | 0.750 ± 0.086 |
| exp3 필터 오버샘플링 | **0.832 ± 0.051** | **0.825 ± 0.058** |
| exp4 필터 joint | 0.770 ± 0.037 | 0.752 ± 0.048 |

| 비교 (macro F1) | 평균 차이 | p |
|---|---|---|
| 필터 오버샘플링 − baseline | +0.085 | 0.012 |
| 필터 joint − 필터 오버샘플링 | −0.074 | 0.029 |
| 필터 오버샘플링 − 오버샘플링 | +0.075 | 0.091 |
| 오버샘플링 − baseline | +0.010 | 0.79 |
| 필터 joint − 오버샘플링 | +0.002 | 0.94 |
| 필터 joint − baseline | +0.012 | 0.53 |
