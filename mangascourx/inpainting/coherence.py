import numpy as np

from .base import Inpainter
from ..core.tensor import structure_tensor


class CoherenceTransport(Inpainter):

    def __init__(
        self,
        iterations=3,
        step_size=0.25
    ):
        self.iterations = iterations
        self.step_size = step_size

    def run(self, image, mask):

        result = image.astype(np.float32).copy()

        gray = (
            0.299 * result[..., 0]
            + 0.587 * result[..., 1]
            + 0.114 * result[..., 2]
        )

        _, _, _, l1, _ = structure_tensor(gray)

        gy, gx = np.gradient(gray)

        magnitude = np.sqrt(
            gx * gx +
            gy * gy
        ) + 1e-6

        tx = -gy / magnitude
        ty = gx / magnitude

        h, w = mask.shape

        for _ in range(self.iterations):

            ys, xs = np.where(mask)

            for y, x in zip(ys, xs):

                px = x - tx[y, x] * self.step_size
                py = y - ty[y, x] * self.step_size

                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)

                ix = int(px)
                iy = int(py)

                result[y, x] = result[iy, ix]

        return result.astype(np.uint8)