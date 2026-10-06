import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------- schedule
class Schedule:
    """Linear-beta DDPM schedule.  mu_t = sqrt(abar_t),  sigma_t^2 = 1 - abar_t."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        beta = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float64)
        alpha = 1.0 - beta
        abar = torch.cumprod(alpha, dim=0)
        abar_prev = torch.cat([torch.ones(1, device=device, dtype=torch.float64), abar[:-1]])

        mu = abar.sqrt()                                  # mu_t
        sigma = (1.0 - abar).sqrt()                       # sigma_t
        sigmabar2 = beta * (1.0 - abar_prev) / (1.0 - abar)   # sigmabar_t^2
        sigma21_2 = alpha.sqrt() * (1.0 - abar_prev)          # sigma21^2(t)

        ct = sigma21_2 / (1.0 - abar)                     # sigma21^2 / sigma_t^2
        c0 = abar_prev.sqrt() - ct * mu                   # mu_{t-dt} - ct * mu_t

        self.beta, self.alpha, self.abar = beta.float(), alpha.float(), abar.float()
        self.mu, self.sigma = mu.float(), sigma.float()
        self.sigmabar2, self.sigma21_2 = sigmabar2.float(), sigma21_2.float()
        self.ct, self.c0 = ct.float(), c0.float()


# ---------------------------------------------------------------- noise
def sample_kappa(shape, nu, device):
    """kappa ~ InvGamma(nu/2, nu/2)."""
    a = torch.tensor(nu / 2.0, device=device)
    return 1.0 / torch.distributions.Gamma(a, a).sample(shape)


def sample_mvt_noise(n, d, nu, device):
    """
    draw from the isotropic multivariate t,  t_d(0, I_d, nu).

    A single kappa is shared by the d coordinates -- that is what makes the
    vector multivariate-t rather than a product of d independent t's.  The two
    coincide only at d = 1.
    """
    kappa = sample_kappa((n, 1), nu, device)
    return kappa.sqrt() * torch.randn(n, d, device=device)


# ---------------------------------------------------------------- model
class TimeEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):                                  # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), half, device=t.device))
        ang = t[:, None] * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class Denoiser(nn.Module):
    """x0-prediction network for d-dimensional data.

    Conditioned on the step index through t_idx / T (DDPM-style), not on
    sigma_t.  T is fixed at construction so training and sampling cannot
    disagree about the normalisation.
    """

    def __init__(self, d=1, hidden=128, temb=64, n_layers=3, T=1000):
        super().__init__()
        self.d = d
        self.T = T
        self.temb = TimeEmbed(temb)
        layers, d_in = [], d + temb
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers += [nn.Linear(d_in, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t_idx):
        return self.net(torch.cat([x, self.temb(t_idx.float() / self.T)], dim=-1))


# ---------------------------------------------------------------- training
def train(x0, nu, T=1000, steps=20000, batch=128, lr=1e-3, hidden=128,
          device="cpu", seed=0, log_every=0, beta_end=0.02):
    """
    x0 : (N, d) training data
    nu : degrees of freedom of the diffusion noise

    One kappa is drawn per sample and shared across the d coordinates, matching
    the isotropic multivariate-t forward kernel.

    Loss is the unweighted x0-regression  L = E‖D_theta(x_t) - x_0‖^2, whose
    minimiser is the posterior mean E[x_0 | x_t] that mu_theta expects.
    """
    torch.manual_seed(seed)
    sch = Schedule(T, beta_end=beta_end, device=device)
    d = x0.shape[-1]
    model = Denoiser(d=d, hidden=hidden, T=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    x0 = x0.to(device)
    N = x0.shape[0]

    for it in range(steps):
        xb = x0[torch.randint(0, N, (batch,), device=device)]
        t = torch.randint(0, T, (batch,), device=device)

        kappa = sample_kappa((batch, 1), nu, device)          # shared across the d coords
        eps = torch.randn(batch, d, device=device)
        xt = sch.mu[t][:, None] * xb + sch.sigma[t][:, None] * kappa.sqrt() * eps

        loss = ((model(xt, t) - xb) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lr_sched.step()

        if log_every and (it + 1) % log_every == 0:
            print(f"  step {it + 1:6d}  loss {loss.item():.5f}")

    model.eval()
    return model, sch


# ---------------------------------------------------------------- sampling
@torch.no_grad()
def sample(model, sch, nu, n=10000, d=None, device="cpu", seed=0):
    """
    Ancestral sampler for
        p_theta(x_{t-dt}|x_t) = t_d(mu_theta, s_t * sigmabar_t^2 I_d, nu + d),
    s_t = (nu + delta_t) / (nu + d),  delta_t = ||x_t - mu_t D||^2 / sigma_t^2.

    d defaults to the dimension the model was built with.  Returns (n,) for d = 1
    and (n, d) otherwise.
    """
    torch.manual_seed(seed)
    T = sch.T
    d = getattr(model, "d", 1) if d is None else d

    # prior:  x_T ~ t_d(0, sigma_T^2 I_d, nu)
    x = sample_mvt_noise(n, d, nu, device) * sch.sigma[T - 1]

    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        D = model(x, t)                                     # D_theta(x_t, t)
        mu_theta = sch.ct[i] * x + sch.c0[i] * D
        if i == 0:
            x = mu_theta
            break
        # conditional scale of the multivariate t: (nu + delta_t) / (nu + d).
        # delta_t uses D_theta as a plug-in for the unknown x_0.
        delta = ((x - sch.mu[i] * D) / sch.sigma[i]).pow(2).sum(-1, keepdim=True)
        var = sch.sigmabar2[i] * (nu + delta) / (nu + d)
        x = mu_theta + var.sqrt() * sample_mvt_noise(n, d, nu + d, device)

    x = x.cpu()
    return x.squeeze(-1) if d == 1 else x
