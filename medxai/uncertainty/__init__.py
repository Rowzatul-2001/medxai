import torch
import torch.nn as nn


def compute_mc_dropout_uncertainty(
    model: nn.Module, input_tensor: torch.Tensor, num_samples: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes mean prediction and epistemic uncertainty (variance) using Monte Carlo Dropout.

    Uncertainty formula per pixel:
    $$\sigma^2 = \frac{1}{T} \sum_{t=1}^{T} (p_t - \bar{p})^2$$
    """
    model.train()  # Keep dropout layers active during evaluation
    predictions = []

    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            predictions.append(probs)

    stacked_preds = torch.stack(predictions, dim=0)
    mean_prediction = torch.mean(stacked_preds, dim=0)
    uncertainty_map = torch.var(stacked_preds, dim=0)

    return mean_prediction, uncertainty_map
