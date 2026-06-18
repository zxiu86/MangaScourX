import numpy as np


def perona_malik_diffusion(
    image,
    iterations=5,
    k=30.0,
    dt=0.15
):

    result = image.astype(np.float32)

    for _ in range(iterations):

        gy, gx = np.gradient(result)

        grad = np.sqrt(
            gx * gx +
            gy * gy
        )

        c = np.exp(
            -(grad / k) ** 2
        )

        result += dt * c * grad

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def curvature_diffusion(
    image,
    iterations=3,
    dt=0.05
):

    result = image.astype(np.float32)

    for _ in range(iterations):

        lap = (
            np.roll(result, 1, 0)
            + np.roll(result, -1, 0)
            + np.roll(result, 1, 1)
            + np.roll(result, -1, 1)
            - 4 * result
        )

        result += dt * lap

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)