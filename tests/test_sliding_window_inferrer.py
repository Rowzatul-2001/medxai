import torch
import torch.nn as nn

from medxai.inference import SlidingWindowInferrer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_output_shape_matches_input():
    model = DummyModel().eval()
    inferrer = SlidingWindowInferrer(
        roi_size=(32, 32, 32), overlap=0.5, mode="gaussian"
    )
    vol = torch.randn(1, 1, 70, 65, 80)
    out = inferrer(model, vol)
    assert out.shape[2:] == vol.shape[2:]


def test_handles_volume_smaller_than_roi():
    model = DummyModel().eval()
    inferrer = SlidingWindowInferrer(
        roi_size=(96, 96, 96), overlap=0.25, mode="constant"
    )
    vol = torch.randn(1, 1, 50, 50, 50)
    out = inferrer(model, vol)
    assert out.shape[2:] == vol.shape[2:]
