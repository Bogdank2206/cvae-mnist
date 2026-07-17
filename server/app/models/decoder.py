import torch
import torch.nn as nn


class Decoder(nn.Module):
    """CNN-декодер: [z, one_hot_label] → 1×28×28."""

    NUM_CLASSES: int = 10

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        input_dim = latent_dim + self.NUM_CLASSES  # 42
        self.fc = nn.Linear(input_dim, 49)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1, 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(z.size(0), self.NUM_CLASSES, device=z.device)
        one_hot.scatter_(1, label.unsqueeze(1), 1.0)
        combined = torch.cat([z, one_hot], dim=1)
        x = self.fc(combined).view(-1, 1, 7, 7)
        return self.deconv(x)
