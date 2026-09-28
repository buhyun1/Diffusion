# diffusion — Diffusion 기반 불균형 데이터 증강

불균형한 공정 이미지(Normal / Underfill_50FR / Underfill_Fan)에서 diffusion 모델로 소수 클래스를 생성해 분류기 성능을 높일 수 있는지 비교한다. 같은 폴더에 score matching 입문 노트북도 있다.

## 파일

| 파일 | 내용 |
|---|---|
| `diffusion_augmentation.ipynb` | 증강 실험 본 노트북 (결과 출력 포함) |
| `diffusion_augmentation_rgb.ipynb` | 위 노트북의 이전 버전 (출력 없음) |
| `score_matching_tutorial.ipynb` | Score matching 튜토리얼 — Philippe Esling (IRCAM) 작성본 |
| `score_matching_mnist.ipynb` | MNIST에서 score network 학습 |
| `results/*.xlsx` | 방법별 10회 반복 분류 성능 (아래 표) |

## 데이터

`../data/256/` (`ImageFolder`, 64×64로 리사이즈, 3채널)

| 클래스 | 학습 | 테스트 |
|---|---|---|
| 0 Normal (다수) | 800 | 100 |
| 1 Underfill_50FR | 80 | 100 |
| 2 Underfill_Fan | 80 | 100 |

## 비교한 방법 — `diffusion_augmentation.ipynb`

| 절 | 방법 | 결과 파일 |
|---|---|---|
| 5 | baseline — 증강 없이 분류기만 | `results/baseline_diffusion.xlsx` |
| 6 | 디퓨전 생성 모델 — 생성 샘플 확인 | `samples/diffusion_only/` |
| 7 | conditional diffusion — 클래스 조건부 생성으로 소수 클래스 oversampling | `results/conditional_diffusion.xlsx` |
| 8 | joint diffusion — 생성기와 분류기를 함께 학습 | `results/joint_diffusion.xlsx` |
| 9 | decision boundary — 결정 경계 근처 샘플을 생성하도록 분류기 손실을 추가 | `results/decisionboundary_diffusion.xlsx` |
| 10–11 | 통계 — 방법 간 paired t-test (`scipy.stats.ttest_rel`), precision / recall / F1 평균 | — |

- 공통 설정: DDPM T=1000, 생성은 DDIM 50 스텝, seed 0–9 반복.
- 결과 xlsx는 `results/`, 생성 샘플은 `samples/`에 저장된다.
