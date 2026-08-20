import itertools

import numpy as np
import torch
import torch.nn.functional as F


class SlidingWindowInferrer:
    def __init__(
        self,
        roi_size,
        overlap=0.5,
        mode="gaussian",
        sigma_scale=0.125,
        sw_batch_size=4,
        device=None,
    ):
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1)")
        if mode not in ("gaussian", "constant"):
            raise ValueError("mode must be 'gaussian' or 'constant'")

        self.roi_size = tuple(roi_size)
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.sw_batch_size = sw_batch_size
        self.device = device

        self._importance_map_cache = {}

    def __call__(self, model, inputs):
        model.eval()
        device = self.device or inputs.device
        inputs = inputs.to(device)

        batch_size, channels, *image_size = inputs.shape
        roi_size = tuple(min(r, s) for r, s in zip(self.roi_size, image_size))

        # Pad the volume if it's smaller than the ROI in any dimension.
        pad_size = []
        for k in range(3):
            diff = max(self.roi_size[k] - image_size[k], 0)
            half = diff // 2
            pad_size.extend([half, diff - half])
        # F.pad expects reverse spatial order (W, H, D)
        pad_size = pad_size[4:6] + pad_size[2:4] + pad_size[0:2]
        inputs = F.pad(inputs, pad_size, mode="constant", value=0)
        padded_size = inputs.shape[2:]

        # Compute sliding window start coordinates for each spatial dim.
        scan_intervals = [
            int(r * (1 - self.overlap)) if r * (1 - self.overlap) > 0 else 1
            for r in roi_size
        ]
        slices = self._get_scan_positions(padded_size, roi_size, scan_intervals)

        importance_map = self._get_importance_map(roi_size, device)

        out_channels = None
        output_sum = None
        weight_sum = torch.zeros(
            (1, 1, *padded_size), device=device, dtype=torch.float32
        )

        # Run patches through the model in mini-batches.
        for i in range(0, len(slices), self.sw_batch_size):
            batch_slices = slices[i : i + self.sw_batch_size]
            patches = torch.cat(
                [inputs[:, :, s[0], s[1], s[2]] for s in batch_slices], dim=0
            )

            with torch.no_grad():
                pred = model(patches)

            if out_channels is None:
                out_channels = pred.shape[1]
                output_sum = torch.zeros(
                    (batch_size, out_channels, *padded_size),
                    device=device,
                    dtype=torch.float32,
                )

            for b_idx, s in enumerate(batch_slices):
                p = pred[b_idx * batch_size : (b_idx + 1) * batch_size]
                output_sum[:, :, s[0], s[1], s[2]] += p * importance_map
                weight_sum[:, :, s[0], s[1], s[2]] += importance_map

        output = output_sum / weight_sum.clamp_min(1e-8)

        # Remove padding to get back to the original image size.
        d0, d1 = pad_size[4], pad_size[5]
        h0, h1 = pad_size[2], pad_size[3]
        w0, w1 = pad_size[0], pad_size[1]
        output = output[
            :,
            :,
            d0 : padded_size[0] - d1,
            h0 : padded_size[1] - h1,
            w0 : padded_size[2] - w1,
        ]
        return output

    def _get_scan_positions(self, image_size, roi_size, scan_intervals):
        """Generate list of (slice_d, slice_h, slice_w) covering the volume."""
        starts_per_dim = []
        for dim in range(3):
            size, roi, interval = image_size[dim], roi_size[dim], scan_intervals[dim]
            starts = list(range(0, size - roi + 1, interval))
            if not starts or starts[-1] + roi < size:
                starts.append(size - roi)
            starts_per_dim.append(sorted(set(starts)))

        slices = []
        for d, h, w in itertools.product(*starts_per_dim):
            slices.append(
                (
                    slice(d, d + roi_size[0]),
                    slice(h, h + roi_size[1]),
                    slice(w, w + roi_size[2]),
                )
            )
        return slices

    def _get_importance_map(self, roi_size, device):
        """Build (and cache) the per-patch weighting map."""
        key = (roi_size, self.mode, device)
        if key in self._importance_map_cache:
            return self._importance_map_cache[key]

        if self.mode == "constant":
            weight = np.ones(roi_size, dtype=np.float32)
        else:
            # Separable 3D Gaussian: outer product of 1D Gaussians per axis.
            weight = np.ones(roi_size, dtype=np.float32)
            for dim, size in enumerate(roi_size):
                sigma = size * self.sigma_scale
                center = size // 2
                coords = np.arange(size, dtype=np.float32) - center
                gauss_1d = np.exp(-(coords**2) / (2 * sigma**2))
                shape = [1, 1, 1]
                shape[dim] = size
                weight = weight * gauss_1d.reshape(shape)
            # Avoid exact-zero weights at the edges (safer normalization).
            weight = np.clip(weight, a_min=1e-4, a_max=None)

        weight_t = torch.from_numpy(weight).to(device).unsqueeze(0).unsqueeze(0)
        self._importance_map_cache[key] = weight_t
        return weight_t
