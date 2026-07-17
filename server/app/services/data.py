"""Загрузка и подготовка датасета MNIST из Hugging Face."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MNISTDataset(torch.utils.data.Dataset):
    """Обёртка над HuggingFace MNIST для PyTorch DataLoader."""

    def __init__(self, hf_ds, tfm: transforms.Compose) -> None:
        self.ds = hf_ds
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        img = item["image"].convert("L")
        return self.tfm(img), item["label"]


def prepare_condition(
    images: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Формирует 11-канальный вход: канал 0 — яркость, каналы 1–10 — one-hot лейбла."""
    b, _, h, w = images.shape
    one_hot = torch.zeros(b, 10, device=images.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    condition = one_hot.view(b, 10, 1, 1).expand(b, 10, h, w)
    return torch.cat([images, condition], dim=1)


def create_train_loader(batch_size: int) -> DataLoader:
    """Создаёт DataLoader для обучающей выборки MNIST."""
    transform = transforms.Compose([transforms.ToTensor()])
    ds = load_dataset("ylecun/mnist")
    dataset = MNISTDataset(ds["train"], transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
