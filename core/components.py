import numpy as np


class UnionFind:

    def __init__(self, size):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=np.int32)

    def find(self, x):

        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb

        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra

        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def connected_components(binary):

    h, w = binary.shape

    labels = np.zeros((h, w), dtype=np.int32)

    uf = UnionFind(h * w)

    next_label = 1

    for y in range(h):

        for x in range(w):

            if not binary[y, x]:
                continue

            neighbors = []

            if y > 0 and labels[y - 1, x]:
                neighbors.append(labels[y - 1, x])

            if x > 0 and labels[y, x - 1]:
                neighbors.append(labels[y, x - 1])

            if len(neighbors) == 0:

                labels[y, x] = next_label
                next_label += 1

            else:

                smallest = min(neighbors)

                labels[y, x] = smallest

                for n in neighbors:
                    uf.union(smallest, n)

    remap = {}
    current = 1

    for y in range(h):
        for x in range(w):

            if labels[y, x] == 0:
                continue

            root = uf.find(labels[y, x])

            if root not in remap:
                remap[root] = current
                current += 1

            labels[y, x] = remap[root]

    return labels, current - 1