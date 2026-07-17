import torch
import torch.nn as nn


class Decoder(nn.Module):
    """CNN-декодер: [z, one_hot_label] → 1×28×28."""

    NUM_CLASSES: int = 10

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        input_dim = latent_dim + self.NUM_CLASSES
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(z.size(0), self.NUM_CLASSES, device=z.device)
        one_hot.scatter_(1, label.unsqueeze(1), 1.0)
        combined = torch.cat([z, one_hot], dim=1)
        
        h = self.fc(combined)
        x = h.view(-1, 16, 7, 7)
        return self.deconv(x)