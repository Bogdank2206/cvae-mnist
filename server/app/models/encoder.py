import torch
import torch.nn as nn


class Encoder(nn.Module):
    """CNN-энкодер: 2×28×28 → (mu, log_sigma_sq) размерности latent_dim."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),  # глобальный avg pooling
        )
        self.flatten = nn.Flatten()
        # adaptive pooling всегда даёт 32-канальный вектор
        self.mu_head = nn.Linear(32, latent_dim)
        self.log_var_head = nn.Linear(32, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.flatten(self.features(x))
        return self.mu_head(features), self.log_var_head(features)
