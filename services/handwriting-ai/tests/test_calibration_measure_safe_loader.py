from __future__ import annotations

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

import handwriting_ai.training.calibrate as cal
from handwriting_ai._test_hooks import AugmentKnobsDict
from handwriting_ai.training.dataset import AugmentConfig, DataLoaderConfig, PreprocessDataset


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), int(idx)


def test_safe_loader_workers_positive_branch() -> None:
    cfg_aug: AugmentConfig = {
        "batch_size": 1,
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
        "morph_kernel_px": 1,
    }
    ds: PreprocessDataset = PreprocessDataset(_TinyBase(), cfg_aug)
    cfg_loader = DataLoaderConfig(
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    loader = cal._safe_loader(ds, cfg_loader)
    # Exercising the path where num_workers > 0 (prefetch/persistent set)
    assert loader.batch_size == 2
    assert loader.num_workers == 1


class _NotPreprocessDataset:
    """Fake dataset that is NOT a PreprocessDataset for type validation test."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 28, 28), torch.tensor(int(idx))

    @property
    def knobs(self) -> AugmentKnobsDict:
        return {
            "enable": False,
            "rotate_deg": 0.0,
            "translate_frac": 0.0,
            "noise_prob": 0.0,
            "noise_salt_vs_pepper": 0.5,
            "dots_prob": 0.0,
            "dots_count": 0,
            "dots_size_px": 1,
            "blur_sigma": 0.0,
            "morph_mode": "none",
            "morph_kernel_px": 1,
        }


def test_safe_loader_rejects_non_preprocess_dataset() -> None:
    """Test that _safe_loader raises TypeError for non-PreprocessDataset inputs."""
    from handwriting_ai._test_hooks import PreprocessDatasetProtocol

    cfg_loader = DataLoaderConfig(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=0,
    )
    # Create a fake dataset that is NOT a PreprocessDataset
    fake_ds = _NotPreprocessDataset()

    # Get a reference to the function that accepts PreprocessDatasetProtocol
    # and call it with something that matches the Protocol but isn't PreprocessDataset
    def call_safe_loader(ds: PreprocessDatasetProtocol) -> None:
        cal._safe_loader(ds, cfg_loader)

    with pytest.raises(TypeError, match="Expected PreprocessDataset"):
        # fake_ds satisfies PreprocessDatasetProtocol structurally but isn't PreprocessDataset
        call_safe_loader(fake_ds)
