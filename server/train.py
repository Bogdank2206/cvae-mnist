"""CLI-скрипт обучения CVAE на MNIST.

Использование:
    python train.py --epochs 20 --batch-size 128 --lr 1e-4
"""

import argparse
import logging

import torch

from app.config import LATENT_DIM, MODEL_PATH, PRIOR_VAR, WEIGHTS_DIR
from app.models.cvae import CVAE
from app.services.data import create_train_loader, prepare_condition
from app.services.losses import vae_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args: argparse.Namespace) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Устройство: %s", device)

    loader = create_train_loader(args.batch_size)

    model = CVAE(latent_dim=LATENT_DIM, prior_var=PRIOR_VAR).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    for epoch in range(1, args.epochs + 1):
        kl_weight = 0.05 * epoch / args.epochs if epoch > args.epochs / 2 else 0        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            x_input = prepare_condition(images, labels)
            recon, mu, log_var = model(x_input, labels)
            loss, recon_l, kl_l = vae_loss(
                recon, images, mu, log_var, kl_weight, prior_var=PRIOR_VAR
            )

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
            "Эпоха %d/%d  loss=%.6f  recon=%.6f  kl=%.6f  lr=%.6f klw=%.6f",
            epoch, n,
            total_loss / count, total_recon / count, total_kl / count,
            scheduler.get_last_lr()[0], kl_weight,
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
