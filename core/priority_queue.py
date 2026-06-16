import numpy as np


class PriorityQueue:

    def __init__(self, capacity):

        self.capacity = capacity

        self.size = 0

        self.dist = np.zeros(
            capacity,
            dtype=np.float32
        )

        self.y = np.zeros(
            capacity,
            dtype=np.int32
        )

        self.x = np.zeros(
            capacity,
            dtype=np.int32
        )

    def push(self, d, yy, xx):

        idx = self.size

        self.dist[idx] = d
        self.y[idx] = yy
        self.x[idx] = xx

        self.size += 1

        while idx > 0:

            parent = (idx - 1) // 2

            if self.dist[parent] <= self.dist[idx]:
                break

            self._swap(idx, parent)

            idx = parent

    def pop(self):

        if self.size == 0:
            raise IndexError()

        d = self.dist[0]
        y = self.y[0]
        x = self.x[0]

        self.size -= 1

        self.dist[0] = self.dist[self.size]
        self.y[0] = self.y[self.size]
        self.x[0] = self.x[self.size]

        self._heapify(0)

        return d, y, x

    def empty(self):

        return self.size == 0

    def _swap(self, a, b):

        self.dist[a], self.dist[b] = (
            self.dist[b],
            self.dist[a]
        )

        self.y[a], self.y[b] = (
            self.y[b],
            self.y[a]
        )

        self.x[a], self.x[b] = (
            self.x[b],
            self.x[a]
        )

    def _heapify(self, idx):

        while True:

            left = idx * 2 + 1
            right = idx * 2 + 2

            smallest = idx

            if (
                left < self.size
                and self.dist[left]
                < self.dist[smallest]
            ):
                smallest = left

            if (
                right < self.size
                and self.dist[right]
                < self.dist[smallest]
            ):
                smallest = right

            if smallest == idx:
                break

            self._swap(idx, smallest)

            idx = smallest