# score 
2D 혼합 분포의 스코어 함수와 t-노이즈 Langevin 샘플링

2D 혼합 분포(mode 3개)에서 밀도 `p(x)`와 스코어 함수 `∇ₓ log p(x)`를 시각화한다. 이어서 스코어를 따라가는 Langevin dynamics에서 노이즈를 Gaussian 대신 **t-분포**로 바꾸면 샘플 분포, 특히 극단값이 어떻게 달라지는지 본다.

## 파일

| 파일 | 내용 |
|---|---|
| `score_gmm.ipynb` | 가우시안 혼합(GMM)의 밀도, 스코어, 혼합 비율 변경, 스코어 따라 이동(Langevin), step size 안정성 |
| `score_t_mixture.ipynb` | 성분을 다변량 t-분포로 바꾼 t-혼합 모델의 밀도·스코어와 Langevin 샘플링 |
| `score_gmm_t_noise.ipynb` | 같은 GMM에서 노이즈만 다변량 t로 바꾼 샘플링 — 자유도별 꼬리, Langevin 궤적, 여러 시드 paired t-test, prior에서 출발하는 diffusion식 샘플링 |

## 공통 설정

- mode 3개, 등방성 스케일 `σ`, 가중치는 균등 또는 비균등(예: 0.7 / 0.2 / 0.1)
- 스코어: `∇ₓ log p(x) = Σ_k r_k(x) · (−(x − µ_k) / σ²)`, 여기서 `r_k`는 성분 책임도(responsibility)
- Langevin 업데이트: `x_{t+1} = x_t + η ∇ₓ log p(x_t) + √(2η) ε_t`, `ε_t`는 Gaussian 또는 t(df)
- 외부 데이터 없이 `numpy`, `matplotlib`, `scipy`만 사용한다.
