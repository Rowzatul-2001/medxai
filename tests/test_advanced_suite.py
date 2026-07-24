import numpy as np
import torch
import torch.nn as nn

from medxai.losses import DeepSupervisionLoss, DiceBCELoss
from medxai.transforms import add_rician_noise, apply_elastic_deform_2d
from medxai.uncertainty import compute_mc_dropout_uncertainty


class DummyModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.drop = nn.Dropout2d(p=0.5)

    def forward(self, x):
        return self.drop(self.conv(x))


def test_mc_dropout_uncertainty():
    model = DummyModelWithDropout()
    x = torch.randn(1, 1, 16, 16)
    mean, var = compute_mc_dropout_uncertainty(model, x, num_samples=5)
    assert mean.shape == (1, 1, 16, 16)
    assert var.shape == (1, 1, 16, 16)


def test_medical_transforms():
    img = np.random.rand(64, 64).astype(np.float32)
    deformed = apply_elastic_deform_2d(img)
    noisy = add_rician_noise(img)
    assert deformed.shape == (64, 64)
    assert noisy.shape == (64, 64)


def test_deep_supervision_loss():
    base_loss = DiceBCELoss()
    ds_loss = DeepSupervisionLoss(base_loss)

    preds = [
        torch.randn(2, 1, 32, 32),  # Full resolution
        torch.randn(2, 1, 16, 16),  # Half resolution
    ]
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss = ds_loss(preds, target)
    assert loss.item() > 0
