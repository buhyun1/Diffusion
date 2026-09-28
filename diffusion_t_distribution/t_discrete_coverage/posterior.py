import torch


def _col(v, x):
    """A scalar or per-sample value, shaped to broadcast against x = (B, ...)."""
    v = torch.as_tensor(v)
    return v.reshape(-1, *([1] * (x.dim() - 1))) if v.dim() else v


def _row(v):
    """A scalar or per-sample value, shaped to broadcast against a (B,) vector."""
    return torch.as_tensor(v).reshape(-1)


def perturbation_kernel(x0, mu_t, sigma_t, nu):
    """q(x_t|x_0) = t_d(mu_t x_0, sigma_t^2 I_d, nu)."""
    scale_sq = torch.as_tensor(sigma_t, dtype=x0.dtype, device=x0.device) ** 2
    return _col(mu_t, x0) * x0, scale_sq.reshape(-1).expand(x0.shape[0]).contiguous(), nu


def prior(shape, nu, device=None, dtype=None):
    """p(x_T) = q(x_T|x_0) = t_d(0, I_d, nu), i.e. the kernel at mu_T = 0, sigma_T = 1."""
    mu = torch.zeros(shape, device=device, dtype=dtype)
    return mu, torch.ones(shape[0], device=device, dtype=dtype), nu


def reverse_posterior(xt, D, mu_t, mu_prev, sigma_t, sigma_prev,
                      sigma_21_sq, sigma_12_sq, nu):
    """Eqn. 7: p_theta(x_{t-dt}|x_t) = t_d(mu_theta, sigmabar_t^2 I_d, nu + d),
    assembling mu_theta from the network output D_theta."""
    d = xt[0].numel()
    a = _col(sigma_21_sq, xt) / _col(sigma_t, xt) ** 2
    mu_theta = a * xt + (_col(mu_prev, xt) - a * _col(mu_t, xt)) * D
    sigmabar_sq = _row(sigma_prev) ** 2 - _row(sigma_21_sq) * _row(sigma_12_sq) / _row(sigma_t) ** 2
    # exactly 0 at |corr| = 1, where rounding can make it slightly negative
    return mu_theta, sigmabar_sq.clamp(min=0).expand(xt.shape[0]).contiguous(), nu + d


def sample(mu, scale_sq, dof):
    """Draw from t_d(mu, scale_sq * I_d, dof)."""
    eps = torch.randn_like(mu)
    kappa = torch.distributions.Chi2(torch.full_like(scale_sq, dof)).sample() / dof
    scale = (scale_sq / kappa).sqrt().view((-1,) + (1,) * (mu.dim() - 1))
    return mu + scale * eps
