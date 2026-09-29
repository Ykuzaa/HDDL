"""Exact NumPy fallback for the small Annoy index used by the RAG TP."""

import numpy as np


class AnnoyIndex:
    """Subset of AnnoyIndex used in the notebook, without compiled extensions."""

    def __init__(self, f, metric="angular"):
        if metric != "angular":
            raise ValueError("This fallback supports only metric='angular'.")
        self.f = int(f)
        self._items = {}

    def add_item(self, index, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.f,):
            raise ValueError(f"Expected shape ({self.f},), got {vector.shape}.")
        self._items[int(index)] = vector

    def build(self, n_trees=10):
        return True  # Exact search needs no tree-building step.

    def get_nns_by_vector(self, vector, n, search_k=-1, include_distances=False):
        if not self._items:
            return ([], []) if include_distances else []
        query = np.asarray(vector, dtype=np.float32)
        ids = np.fromiter(self._items.keys(), dtype=np.int64)
        matrix = np.stack([self._items[i] for i in ids])
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        cosine = np.divide(matrix @ query, denom, out=np.zeros_like(denom), where=denom != 0)
        distances = 1.0 - cosine
        order = np.argsort(distances)[: int(n)]
        result_ids = ids[order].tolist()
        if include_distances:
            return result_ids, distances[order].tolist()
        return result_ids
