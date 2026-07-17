"""Функции потерь для обучения CVAE."""

import math

import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
    prior_var: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO-лосс: (total, recon, kl). Оба терма — сумма на сэмпл.

    prior_var — дисперсия априорного p(z) = N(0, prior_var · I), единая для всех
    латентных размерностей (диагональная ковариация). При prior_var=1 формула KL
    сводится к стандартной для N(0, I).
    """
    recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
    # KL между q(z|x) = N(mu, exp(log_var)) и p(z) = N(0, prior_var): лог-член
    # нужен, т.к. априори уже не единичный, а отношения дисперсий входят в KL.
    kl_per_dim = 0.5 * (
        math.log(prior_var) - log_var
        + log_var.exp() / prior_var
        + mu.pow(2) / prior_var
        - 1.0
    )
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
