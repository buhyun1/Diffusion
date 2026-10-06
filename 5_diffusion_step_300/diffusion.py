"""DDPM 스케줄과 sampling."""

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

        scale = 1000 / timesteps
        betas = torch.linspace(scale * 1e-4, scale * 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.alphas_cumprod = alphas_cumprod 
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
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

