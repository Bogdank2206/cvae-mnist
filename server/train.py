"""CLI-скрипт обучения CVAE на MNIST.

Использование:
    python train.py --epochs 50 --batch-size 128 --lr 1e-3
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from app.config import LATENT_DIM, MODEL_PATH, WEIGHTS_DIR
from app.models.cvae import CVAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_condition(
    images: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Формирует 2-канальный вход: канал 0 — яркость, канал 1 — лейбл."""
    b, _, h, w = images.shape
    condition = labels.float().view(b, 1, 1, 1).expand(b, 1, h, w)
    return torch.cat([images, condition], dim=1)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO-лосс: (total, recon, kl)."""
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss


def train(args: argparse.Namespace) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Устройство: %s", device)

    # Данные MNIST из Hugging Face
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = load_dataset("mnist")
    train_data = ds["train"]

    class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, tfm) -> None:
            self.ds = hf_ds
            self.tfm = tfm

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int):
            item = self.ds[idx]
            img = item["image"].convert("L")
            return self.tfm(img), item["label"]

    dataset = MNISTDataset(train_data, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = CVAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            x_input = prepare_condition(images, labels)
            recon, mu, log_var = model(x_input, labels)
            loss, recon_l, kl_l = vae_loss(recon, images, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()
            count += 1

        scheduler.step()
        n = args.epochs
        logger.info(
            "Эпоха %d/%d  loss=%.1f  recon=%.1f  kl=%.1f  lr=%.6f",
            epoch, n,
            total_loss / count, total_recon / count, total_kl / count,
            scheduler.get_last_lr()[0],
        )

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Веса сохранены: %s", MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение CVAE на MNIST")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
