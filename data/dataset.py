"""
Generic image dataset loader with train/val/test split.
Supports folder layouts: root/class_name/images or root/category/class_name/images.
Test set is held out for final inference evaluation (disjoint from train/val).
"""

from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Common image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_samples(
    root_dir: str,
    subdir_depth: int,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Scan root_dir for images and infer labels from folder structure.
    Returns (samples, class_names) where samples = [(path, class_name), ...]
    and class_names is sorted unique class list.
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    path_to_class: List[Tuple[str, str]] = []

    if subdir_depth == 1:
        # root / class_name / images
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for p in class_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    path_to_class.append((str(p), class_name))
    elif subdir_depth == 2:
        # root / category / class_name / images (e.g. LC25000: lung_image_sets/lung_aca/)
        for category_dir in root.iterdir():
            if not category_dir.is_dir():
                continue
            for class_dir in category_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                for p in class_dir.iterdir():
                    if p.suffix.lower() in IMAGE_EXTENSIONS:
                        path_to_class.append((str(p), class_name))
    else:
        raise ValueError("subdir_depth must be 1 or 2")

    if not path_to_class:
        raise ValueError(f"No images found under {root} with subdir_depth={subdir_depth}")

    class_names = sorted({c for _, c in path_to_class})
    return path_to_class, class_names


def _stratified_split_three_way(
    path_to_class: List[Tuple[str, str]],
    class_names: List[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split (path, class_name) into train/val/test by class index, stratified."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be in [0,1) with val_ratio + test_ratio < 1")

    name_to_idx = {c: i for i, c in enumerate(class_names)}
    by_class: List[List[str]] = [[] for _ in class_names]
    for path, c in path_to_class:
        by_class[name_to_idx[c]].append(path)

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []

    generator = torch.Generator().manual_seed(seed)
    for idx, paths in enumerate(by_class):
        n = len(paths)
        perm = torch.randperm(n, generator=generator).tolist()
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_val = max(0, n - 2)
            n_test = 1
            n_train = n - n_val - n_test
        val_indices = set(perm[:n_val])
        test_indices = set(perm[n_val : n_val + n_test])
        for i, p in enumerate(paths):
            if i in val_indices:
                val_samples.append((p, idx))
            elif i in test_indices:
                test_samples.append((p, idx))
            else:
                train_samples.append((p, idx))

    return train_samples, val_samples, test_samples


class ImageDataset(Dataset):
    """
    Generic image dataset returning (image, label).
    Labels are integer indices; use .class_names to map to names.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        subdir_depth: int = 1,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root_dir: Path to dataset root (see subdir_depth for structure).
            split: "train", "val", or "test".
            val_ratio: Fraction of data used for validation (stratified).
            test_ratio: Fraction of data held out for test (stratified).
            seed: Random seed for reproducible split.
            subdir_depth: 1 = root/class/images, 2 = root/category/class/images.
            transform: Optional torchvision transforms (applied to PIL image).
        """
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")

        path_to_class, class_names = _collect_samples(root_dir, subdir_depth)
        train_samples, val_samples, test_samples = _stratified_split_three_way(
            path_to_class, class_names, val_ratio, test_ratio, seed
        )

        self.class_names = class_names
        self.num_classes = len(class_names)
        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        else:
            self.samples = test_samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_default_transforms(
    image_size: int = 224,
    is_training: bool = True,
) -> transforms.Compose:
    """Standard transforms (resize, crop, flip, normalize)."""
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transform


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    subdir_depth: int = 1,
    image_size: int = 224,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders with standard transforms.
    Test set uses the same eval transforms as val and is disjoint for final inference.
    Returns (train_loader, val_loader, test_loader).
    """
    train_transform = get_default_transforms(image_size, is_training=True)
    eval_transform = get_default_transforms(image_size, is_training=False)

    train_ds = ImageDataset(
        root_dir,
        split="train",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        subdir_depth=subdir_depth,
        transform=train_transform,
    )
    val_ds = ImageDataset(
        root_dir,
        split="val",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        subdir_depth=subdir_depth,
        transform=eval_transform,
    )
    test_ds = ImageDataset(
        root_dir,
        split="test",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        subdir_depth=subdir_depth,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_lc25000_root() -> str:
    """
    Download LC25000 (lung/colon cancer) via kagglehub and return the dataset root.
    For use in Colab or any environment with Kaggle credentials.
    Root is the directory that contains lung_image_sets and colon_image_sets
    (use with subdir_depth=2 in ImageDataset / get_dataloaders).
    """
    import kagglehub

    path = Path(kagglehub.dataset_download("andrewmvd/lung-and-colon-cancer-histopathological-images"))
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"kagglehub download path is not a directory: {path}")
    # Dataset zip may have one top-level folder (e.g. lung_colon_image_set)
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / "lung_image_sets").is_dir() and (subdirs[0] / "colon_image_sets").is_dir():
        return str(subdirs[0])
    if (path / "lung_image_sets").is_dir() and (path / "colon_image_sets").is_dir():
        return str(path)
    raise FileNotFoundError(
        f"LC25000 structure not found under {path}. "
        "Expected 'lung_image_sets' and 'colon_image_sets' subdirs."
    )
