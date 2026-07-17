import torch
import torch.nn as nn

from app.models.decoder import Decoder
from app.models.encoder import Encoder


class CVAE(nn.Module):
    """Условный вариационный автокодировщик."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
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
        """Генерация из стандартного нормального распределения."""
        latent_dim = self.encoder.mu_head.out_features
        z = torch.randn(1, latent_dim, device=device)
        return self.decoder(z, label)
