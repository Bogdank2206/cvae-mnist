import torch
import torch.nn as nn

from app.models.decoder import Decoder
from app.models.encoder import Encoder


class CVAE(nn.Module):
    """Условный вариационный автокодировщик."""

    def __init__(self, latent_dim: int, prior_var: float = 1.0) -> None:
        super().__init__()
        # p(z) = N(0, prior_var · I): единая дисперсия по всем латентным осям
        self.prior_var = prior_var
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        # reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z, label)
        return recon, mu, log_var

    def generate(
        self, label: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """Генерация из априорного p(z) = N(0, prior_var · I)."""
        latent_dim = self.encoder.mu_head.out_features
        # sqrt(prior_var) — масштаб стандартной нормали до нужной дисперсии априори
        z = torch.randn(1, latent_dim, device=device) * (self.prior_var ** 0.5)
        return self.decoder(z, label)
