import numpy as np

try:
    from numba import njit
except:
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


@njit(cache=True)
def euclidean_distance_transform(mask):
    """
    Approximate Euclidean Distance Transform.

    Parameters
    ----------
    mask : bool ndarray
        True = foreground

    Returns
    -------
    float32 ndarray
    """

    h, w = mask.shape

    dist = np.where(mask, 0.0, 1e20).astype(np.float32)

    for y in range(h):
        for x in range(w):

            if y > 0:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y - 1, x] + 1.0
                )

            if x > 0:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y, x - 1] + 1.0
                )

            if y > 0 and x > 0:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y - 1, x - 1] + 1.4142135
                )

            if y > 0 and x < w - 1:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y - 1, x + 1] + 1.4142135
                )

    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):

            if y < h - 1:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y + 1, x] + 1.0
                )

            if x < w - 1:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y, x + 1] + 1.0
                )

            if y < h - 1 and x < w - 1:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y + 1, x + 1] + 1.4142135
                )

            if y < h - 1 and x > 0:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y + 1, x - 1] + 1.4142135
                )

    return dist