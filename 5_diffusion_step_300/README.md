# diffusion_step_300 연구노트

## 1. 목적

소수 클래스 데이터가 부족한 불균형 설정에서, 학습과 샘플링 모두 300스텝(T=300)인 class-conditional DDPM이 생성한 이미지로 소수 클래스를 채우면 분류 성능이 좋아지는지 확인한다. 아래 세 가지 방식을 같은 데이터 분할과 같은 분류기 학습량으로 비교한다.

T=1000이 아니라 300스텝을 쓴 이유는 joint 방식의 학습 시간 때문이다. joint는 매 배치마다 fake를 새로 생성하므로, 1000스텝으로 샘플링하면 시간이 너무 오래 걸린다.

| 실험 | 폴더 | 방법 |
|---|---|---|
| exp1 | `exp1_baseline` | 불균형 train 그대로 분류기 학습 (baseline) |
| exp2 | `exp2_oversampling_step_300` | Diffusion을 먼저 학습하고, 소수 클래스 fake 풀(DDPM 300스텝)을 한 번 만든 뒤 분류기 학습 시 풀에서 꺼내 채움 |
| exp3 | `exp3_joint_step_300` | Diffusion 사전학습 후, 분류기와 Diffusion을 함께 학습(joint). 매 배치 EMA 모델로 fake를 새로 생성(DDPM 300스텝)해 채우고, 분류 손실을 Diffusion에도 일부 공유 |

## 2. 실험 설정

**데이터**
- 3클래스: Normal(0), Underfill_50FR(1), Underfill_Fan(2). 64×64로 resize 후 [-1, 1]로 정규화.
- 불균형 train: 800 / 80 / 80장. 균형 test: 클래스당 100장.
- seed마다 train/test 분할을 새로 뽑는다 (seed 0~9, 10회 반복).
- 데이터 경로는 환경변수 `DATA_ROOT`로 지정한다. 기본값은 노트북 기준 상대경로 `../../data/256`이다.

**분류기**
- 작은 CNN (Conv 3층 + GAP + Linear), Adam lr 1e-3, batch 40.
- 학습량은 gradient update 횟수로 고정: 50 epoch × 24 step = **1200 step**. 세 실험 모두 동일.

**Diffusion**
- 조건부 U-Net(dim=32), Adam lr 2e-4, EMA decay 0.999.
- 학습·샘플링 모두 T=300. β 스케줄은 `GaussianDiffusion`이 1000/T를 곱해 T=1000일 때와 스케일을 맞춘다.
- exp2: 700 epoch 학습 후 seed당 소수 클래스 720장씩(= 800 − 80) fake 풀 생성.
- exp3: 650 epoch 사전학습(700 − 50) 후 50 epoch joint 학습. 공유 손실 계수 λ는 0에서 0.01까지 선형 증가, 공유 샘플 최대 16장, 공유 시점 t < 90.

**채우기 규칙**: 매 step 실제 배치의 Normal 수에 맞춰 소수 클래스의 부족분만큼 fake를 추가해 클래스 균형 배치를 만든다.

## 3. 결과 (10 seed 평균 ± 표준편차)

**Macro 지표**

| 실험 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| exp1 baseline | 0.512 ± 0.129 | 0.626 ± 0.121 | 0.512 ± 0.129 | 0.442 ± 0.178 |
| exp2 oversampling | **0.880 ± 0.052** | **0.887 ± 0.047** | **0.880 ± 0.052** | **0.880 ± 0.054** |
| exp3 joint | 0.779 ± 0.043 | 0.839 ± 0.022 | 0.779 ± 0.043 | 0.773 ± 0.045 |

**클래스별 Recall**

| 실험 | Normal | Underfill_50FR | Underfill_Fan |
|---|---|---|---|
| exp1 baseline | 0.991 | 0.268 | 0.276 |
| exp2 oversampling | 0.827 | 0.886 | 0.928 |
| exp3 joint | 0.971 | 0.536 | 0.830 |

**대응표본 t-test (macro F1, seed 10쌍)**

| 비교 | 평균 차이 | t | p | 우세 seed |
|---|---|---|---|---|
| exp2 − exp1 | +0.437 | 9.53 | < 0.001 | 10/10 |
| exp3 − exp1 | +0.331 | 5.94 | 0.0002 | 10/10 |
| exp3 − exp2 | −0.107 | −5.37 | 0.0005 | 1/10 |

Accuracy 기준도 같은 방향이다 (exp2−exp1 +0.369, exp3−exp1 +0.267, exp3−exp2 −0.101, 모두 p < 0.001).