from copy import deepcopy

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    export_path: str,
    fp16: bool = True,
    opset_version: int = 17,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> str:
    model = deepcopy(model)
    model.eval()

    device = "cuda" if (fp16 and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    sample_input = sample_input.to(device)

    if fp16:
        model = model.half()
        sample_input = sample_input.half()

    with torch.no_grad():
        torch_output = model(sample_input)
        if isinstance(torch_output, (tuple, list)):
            torch_output = torch_output[0]

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        sample_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    ort_session = ort.InferenceSession(export_path, providers=providers)

    ort_inputs = {"input": sample_input.detach().cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    torch_output_np = torch_output.detach().cpu().float().numpy()
    ort_output_np = ort_output.astype(np.float32)

    if not np.allclose(torch_output_np, ort_output_np, atol=atol, rtol=rtol):
        max_diff = np.max(np.abs(torch_output_np - ort_output_np))
        raise RuntimeError(
            f"ONNX output verification failed. Max abs diff: {max_diff:.6f} "
            f"(atol={atol}, rtol={rtol})"
        )

    return export_path
