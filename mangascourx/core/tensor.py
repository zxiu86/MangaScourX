import numpy as np


def structure_tensor(gray):

    gy, gx = np.gradient(gray)

    jxx = gx * gx
    jxy = gx * gy
    jyy = gy * gy

    trace = jxx + jyy

    det = jxx * jyy - jxy * jxy

    disc = np.sqrt(
        np.maximum(
            trace * trace / 4.0 - det,
            0.0
        )
    )

    lambda1 = trace / 2.0 + disc
    lambda2 = trace / 2.0 - disc

    return (
        jxx,
        jxy,
        jyy,
        lambda1,
        lambda2
    )