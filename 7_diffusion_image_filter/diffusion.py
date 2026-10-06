"""DDPM 스케줄과 sampling"""

import torch
import torch.nn.functional as F
import inspect
from torch.utils.checkpoint import checkpoint

_CKPT_HAS_REENTRANT_ARG = "use_reentrant" in inspect.signature(checkpoint).parameters


def _checkpoint(fn, *args):
    if _CKPT_HAS_REENTRANT_ARG:
        return checkpoint(fn, *args, use_reentrant=False)
    return checkpoint(fn, *args)


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

        self.alphas_cumprod = alphas_cumprod  # respacing은 인접하지 않은 timestep 사이도 오가야 해서 필요
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

    def respaced_schedule(self, steps):
        """학습 timestep 중 steps개만 골라 건너뛴 간격에 맞는 샘플링 계수를 만든다 (steps별 캐싱)."""
        cache = self.__dict__.setdefault("_respaced_cache", {})
        if steps not in cache:
            use_t = torch.linspace(0, self.timesteps - 1, steps).round().long().unique()
            ac = self.alphas_cumprod.double()[use_t]
            ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)
            betas = 1. - ac / ac_prev
            alphas = 1. - betas
            dtype = self.alphas_cumprod.dtype
            cache[steps] = {
                "use_t": use_t,
                "posterior_variance": (betas * (1. - ac_prev) / (1. - ac)).to(dtype),
                "posterior_mean_coef1": (betas * torch.sqrt(ac_prev) / (1. - ac)).to(dtype),
                "posterior_mean_coef2": ((1. - ac_prev) * torch.sqrt(alphas) / (1. - ac)).to(dtype),
            }
        return cache[steps]

    def _ddpm_respaced_loop(self, diffusion, y, device, steps, use_checkpoint=False):
        """respacing 스케줄로 노이즈에서 이미지까지 steps번만 역과정을 돈다. use_checkpoint=True면 gradient를 흘린다."""
        model = diffusion
        sch = self.respaced_schedule(steps)
        use_t = sch["use_t"]
        n = y.size(0)
        x = torch.randn((n, self.channels, self.image_size, self.image_size), device=device)
        if use_checkpoint and not _CKPT_HAS_REENTRANT_ARG:
            x.requires_grad_(True)

        for i in reversed(range(len(use_t))):
            t = torch.full((n,), use_t[i].item(), device=device, dtype=torch.long)  # 모델용: 원래 timestep
            idx = torch.full((n,), i, device=device, dtype=torch.long)  # respaced 계수용: 새 체인의 번호
            eps = _checkpoint(model, x, t, y) if use_checkpoint else model(x, t, y)
            x0 = self.predict_x0(x, t, eps).clamp(-1.0, 1.0)
            mean = (extract(sch["posterior_mean_coef1"], idx, x.shape) * x0
                    + extract(sch["posterior_mean_coef2"], idx, x.shape) * x)
            if i == 0:
                x = mean
            else:
                var = extract(sch["posterior_variance"], idx, x.shape)
                x = mean + torch.sqrt(var) * torch.randn_like(x)

        return x

    @torch.no_grad()
    def sample_ddpm_respaced(self, diffusion, y, device, steps=100):
        """gradient 없이 respacing 샘플링."""
        diffusion.eval()
        return self._ddpm_respaced_loop(diffusion, y, device, steps)

    def sample_ddpm_respaced_grad(self, diffusion, y, device, steps=30):
        """gradient를 흘리는 respacing 샘플링 (joint, 협력 항용)."""
        return self._ddpm_respaced_loop(diffusion, y, device, steps, use_checkpoint=True)

