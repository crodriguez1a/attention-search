import time

import nmslib
import numpy as np


def create_float_index(encoded: np.ndarray, path: str) -> nmslib.dist.FloatIndex:
    float_index: nmslib.dist.FloatIndex = nmslib.init(method="hnsw", space="l2")

    try:
        float_index.loadIndex(path)
    except Exception:
        float_index.addDataPointBatch(encoded)
        float_index.createIndex({"post": 2}, print_progress=True)
        float_index.saveIndex(path, save_data=True)

    return float_index


def knn_search(
    query_vector: np.ndarray, float_index: nmslib.dist.FloatIndex, k: int = 3
) -> tuple:

    tic = time.perf_counter()
    indices, distances = float_index.knnQuery(query_vector, k=k)
    results = (indices, distances)

    toc = time.perf_counter()
    print(f"Completed search in {toc - tic:0.4f} seconds")

    return results
