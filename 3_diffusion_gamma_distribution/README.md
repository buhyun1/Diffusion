# diffusion_gamma — Gamma 노이즈 diffusion과 불균형 데이터 증강

Diffusion의 노이즈를 Gaussian 대신 **Gamma 분포**로 바꾼 DDGM을 구현하고, 이를 불균형 Fashion-MNIST 이상 탐지의 소수 클래스 증강에 적용해 Gaussian과 비교한다.

## 참고 논문

- **Denoising Diffusion Gamma Models** — Nachmani, San Roman & Wolf, 2021
- **Anomaly detection in additive manufacturing processes using supervised classification with imbalanced sensor data based on GAN** — Chung, Shen & Kong, J. Intell. Manuf., 2024

## 파일

| 파일 | 내용 |
|---|---|
| `ddpm_reference.ipynb` | 참고한 DDPM 구현 (MNIST) |
| `ddpm_gaussian_mnist.ipynb` | 표준 DDPM (Gaussian 노이즈, MSE 손실) — 비교 기준 |
| `ddpm_gamma_mnist.ipynb` | DDGM (Gamma 노이즈, L1 손실) — U-Net·스케줄·하이퍼파라미터는 위와 동일 |
| `gamma_vs_gaussian_fit.ipynb` | Fashion-MNIST 픽셀 분포가 Gamma와 Gaussian 중 어디에 더 맞는지 (히스토그램 MSE, AIC, BIC, KS) |
| `augmentation_conditional.ipynb` | 2-player 증강: 조건부 diffusion으로 소수 클래스 생성 → 분류기 학습 |
| `augmentation_cooperative.ipynb` | 2-player 증강: 생성기와 분류기를 번갈아 학습 (공유 분류 손실) |
| `experiments_sequential_alternating.ipynb` | 10회 × {sequential, alternating} × {gaussian, gamma} 자동 실험 + paired t-test → `results/experiment_results.xlsx` |
| `experiments_paper_matched.ipynb` | 논문 실험 조건(64×64, 300 epoch, batch 100, 불균형 테스트셋)에 맞춘 자동 실험 → `results/experiment_results_paper.xlsx` |

## Gaussian vs Gamma (DDPM → DDGM)

| | Gaussian | Gamma |
|---|---|---|
| 순방향 노이즈 | `ε ~ N(0, I)` | `ḡ_t ~ Gamma(k̄_t, θ_t)`, 평균 0으로 이동 |
| 손실 | MSE | L1 |
| 샘플링 | `N(0, I)`에서 시작, Gaussian 노이즈 주입 | 평균 0 Gamma에서 시작, Gamma 노이즈 주입 |

평균을 뺀 Gamma 노이즈의 분산은 `1 − ᾱ_t`로, Gaussian의 총 노이즈 분산과 같다.

## 증강 실험 설정

- 데이터: Fashion-MNIST T-shirt / Pullover / Dress, 학습 800 / 80 / 80 (비율 0.10)
- 원 논문은 Generator + Discriminator + Classifier의 3-player GAN이다. 여기서는 diffusion이 생성기 역할을 하므로 discriminator를 뺀 2-player 구성이다.
- 지표: macro precision / recall / F-score, accuracy, 클래스별 F1
- 반복: 10회. 같은 run 안의 설정들은 같은 데이터와 seed를 공유하므로 paired t-test를 쓴다.
- 데이터 경로: `experiments_*`는 `../data`, 나머지 노트북은 `./data` (없으면 자동 다운로드)
