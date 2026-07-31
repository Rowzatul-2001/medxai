import os

import torch
import torch.nn as nn

from medxai.deployment import export_onnx


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 4)

    def forward(self, x):
        return self.linear(x)


def test_export_onnx_fp32(tmp_path):
    model = DummyModel()
    sample_input = torch.randn(2, 10)
    export_path = str(tmp_path / "model.onnx")

    result = export_onnx(model, sample_input, export_path, fp16=False)

    assert result == export_path
    assert os.path.exists(export_path)


def test_export_onnx_dynamic_batch(tmp_path):
    model = DummyModel()
    sample_input = torch.randn(1, 10)
    export_path = str(tmp_path / "model_dynamic.onnx")

    export_onnx(model, sample_input, export_path, fp16=False)

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(export_path)
    batch5_input = np.random.randn(5, 10).astype(np.float32)
    output = session.run(None, {"input": batch5_input})[0]

    assert output.shape[0] == 5
