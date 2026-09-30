import numpy as np


def build_grid_adjacency(rows: int, cols: int) -> np.ndarray:
    """4-connected grid graph, row-major node indexing (index = row * cols + col),
    matching the layout used by the hardcoded 4x4 matrix in test.py."""
    n = rows * cols
    adjacency_matrix = np.zeros((n, n), dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                neighbor = row * cols + (col + 1)
                adjacency_matrix[node, neighbor] = 1
                adjacency_matrix[neighbor, node] = 1
            if row + 1 < rows:
                neighbor = (row + 1) * cols + col
                adjacency_matrix[node, neighbor] = 1
                adjacency_matrix[neighbor, node] = 1
    return adjacency_matrix


def hop_distances(adjacency_matrix: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path length in hops, by BFS from every node.

    Used to stratify the test metrics by how far an observer sits from the node
    it is reporting on: distance 1 means a direct neighbour, which hears about a
    failure through its own keep-alive, while distance >= 2 can only learn of it
    through the channel. Unreachable pairs are +inf; the diagonal is 0.
    """
    a = np.asarray(adjacency_matrix)
    n = a.shape[0]
    distances = np.full((n, n), np.inf, dtype=np.float32)
    neighbours = [np.flatnonzero(a[i]) for i in range(n)]
    for source in range(n):
        distances[source, source] = 0.0
        frontier = [source]
        depth = 0
        while frontier:
            depth += 1
            nxt = []
            for node in frontier:
                for neighbour in neighbours[node]:
                    if distances[source, neighbour] == np.inf:
                        distances[source, neighbour] = depth
                        nxt.append(neighbour)
            frontier = nxt
    return distances
