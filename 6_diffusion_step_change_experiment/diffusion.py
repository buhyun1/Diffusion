"""DDPM 스케줄과 표준 ancestral sampling.

timesteps/image_size/channels는 실험 설정에 따라 바뀔 수 있어 GaussianDiffusion
생성자에서 받는다. diffusion 모델(U-Net) 자체는 매개변수로 받아서 쓰므로 이 모듈은
어떤 U-Net 구현과도 결합해서 쓸 수 있다.
"""

import torch
import torch.nn.functional as F


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)


class GaussianDiffusion:
    def __init__(self, timesteps, image_size, channels):
        self.timesteps = timesteps
        self.image_size = image_size
        self.channels = channels

        # DDPM 원 논문의 linear 스케줄(1e-4 → 0.02)은 T=1000 기준이다. T가 다르면 전체 노이즈 양이
        # 달라지므로, improved DDPM/guided-diffusion처럼 1000/T를 곱해 스케일을 맞춘다.
        # (T=1000이면 기존과 완전히 같고, T=300이어도 마지막 시점 ᾱ_T가 T=1000일 때와 거의 같아진다.)
        scale = 1000 / timesteps
        betas = torch.linspace(scale * 1e-4, scale * 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.alphas_cumprod = alphas_cumprod  # respacing은 인접하지 않은 timestep 사이도 오가야 해서 필요
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # x0를 클리핑한 뒤 posterior q(x_{t-1}|x_t,x0)의 평균을 구하기 위한 계수
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    def predict_x0(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
                * (x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_mean_variance(self, diffusion, x, t, y, clip_denoised=True):
        eps = diffusion(x, t, y)

        x0 = self.predict_x0(x, t, eps)

        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)

        coef1 = extract(self.posterior_mean_coef1, t, x.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x.shape)
        mean = coef1 * x0 + coef2 * x
        var = extract(self.posterior_variance, t, x.shape)
        return mean, var, eps

    @torch.no_grad()
    def p_sample(self, diffusion, x, t, t_index, y):
        mean, var, _ = self.p_mean_variance(diffusion, x, t, y)
        if t_index == 0:
            return mean
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_ddpm(self, diffusion, y, device):
        diffusion.eval()
        n = y.size(0)
        x = torch.randn((n, self.channels, self.image_size, self.image_size), device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x = self.p_sample(diffusion, x, t, i, y)

        return x

