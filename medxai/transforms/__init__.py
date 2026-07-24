import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def apply_elastic_deform_2d(
    image: np.ndarray, 
    alpha: float = 34.0, 
    sigma: float = 4.0
) -> np.ndarray:
    """
    Applies 2D Elastic Deformation to simulate organic soft-tissue morphology changes.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D single-channel NumPy array.")

    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted = map_coordinates(image, indices, order=1, mode="reflect")
    return distorted.reshape(shape)


def add_rician_noise(image: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Simulates Rician noise, which is characteristic of raw Magnetic Resonance Imaging (MRI) acquisitions.
    """
    noise1 = np.random.normal(0, noise_level, image.shape)
    noise2 = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.sqrt((image + noise1) ** 2 + noise2 ** 2)
    return noisy_image