import time
from typing import Any, List, Sequence, Tuple

import numpy as np

# Inspired by:
# https://towardsdatascience.com/learning-attention-mechanism-from-scratch-f08706aaf6b6


def indices_from_weights(
    weights: np.ndarray, n_results: int, full_attention: bool = False
) -> np.ndarray:
    """
    Returns the indices that would sort the array, preserving order
    """
    # first row, faster to slice than to sort on axis
    X: np.ndarray = weights
    ndim = X.ndim

    if ndim == 4:
        # argsort, max on top, then retrieve a 1D sub-array from a 4D-array
        return np.argsort(X[:, 0, 0, 0], axis=0)[::-1]
    elif ndim == 3:
        return np.argsort(X[:, 0, 0], axis=0)[::-1]
    else:
        # # argsort, max on top
        if not full_attention:
            return np.argsort(X[0, :])[::-1]

        return np.argsort(X)[::-1]


def scale_dot_product(matmul_qsi: np.ndarray, si: np.ndarray) -> np.ndarray:
    """
    Attention is all you Need (Vaswani, et al., 2017):
    "We suspect that for large values of dk, the dot products grow
    large in magnitude, pushing the softmax function into regions where it has
    extremely small gradients. To counteract this effect, we scale the dot
    products by 1/sqrt of dk
    """
    # divide the scores the square root of the dimension of the key vectors
    dk: int = si.shape[-1]
    return matmul_qsi / np.sqrt(dk)


def softmax(x: np.ndarray, axis: int = -1):
    """
    Compute corresponding probabilities for each set of scores in x
    """
    # TODO speed this up? https://github.com/numpy/numpy/issues/8233
    e_x = np.exp(x - np.max(x))
    return np.divide(e_x, e_x.sum(axis=axis, keepdims=True))


def dot_product(query: np.ndarray, si: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean magnitudes of the two vectors and the cosine
    of the angle between them for all the possible search results
    in one step using matrix multiplication (this is the magic)
    """
    return np.matmul(query, np.transpose(si))


def einstein_summation(q: np.ndarray, si: np.ndarray, optim_threshold: int = 30000):
    """
    Einstein summation convention is a notational convention that implies
    summation over a set of indexed terms
    """
    # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    # TODO: Explain this further

    # optimization plateaus at about 30000 indices at which the cost
    # of computing the optimization is too high
    above_threshold: bool = si.shape[0] > optim_threshold
    path = (
        np.einsum_path("ijkl,jkl->ijkl", si, q, optimize="optimal")[0]
        if above_threshold
        else None
    )

    # shapes of input matrices a,b -> shape of output matrix z
    return np.einsum("ijkl,jkl->ijkl", si, q, optimize=path)


def mat_mult(
    query: np.ndarray, si: np.ndarray, n_results: int = 3
) -> Tuple[Any, List[Any]]:
    "Compute Matrix Multiplication"

    matmul_qsi: np.ndarray = None
    if query.ndim == 3:
        matmul_qsi = einstein_summation(query, si)
    else:
        matmul_qsi = dot_product(query, si)

    return matmul_qsi


def scaled_attention_weights(matmul_qsi: np.ndarray, si: np.ndarray) -> np.ndarray:
    """
    Stabilize gradients, normalize with softmax
    """
    # Apply scaling to raw values
    scaled_attn_logits: np.ndarray = scale_dot_product(matmul_qsi, si)

    # Apply a softmax function to obtain weights for all scaled values
    attn_weights: np.ndarray = softmax(scaled_attn_logits, axis=-1)  # axis=-1 last dim

    return attn_weights


def apply_self_attention(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Multiply embeddings matrix X by the weight matrices
    """

    if X.ndim > 2:
        return np.einsum("ijkl,ijkl->ijkl", X, weights, optimize="greedy")

    return np.multiply(weights.T, X)


def find_attn_vector(applied_attention):
    """
    The intuition here is to keep intact the values of the word(s) we want to
    focus on, and drown-out irrelevant words (by multiplying them by tiny
    numbers like 0.001, for example).
    """
    # Sum up the four columns to produce a single attention context vector
    return np.sum(applied_attention, axis=1)


def map_values(values: Sequence[Any], indices: List[int]) -> list:
    if values and max(indices) < len(values):
        return [values[i] for i in indices]
    else:
        return []


def attention_search(
    query: np.ndarray,
    si: np.ndarray,
    values: Sequence[Any] = None,
    n_results: int = 3,
    full_attention: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Apply scaled dot product attention, then map indices to values
    """
    tic = time.perf_counter()

    mat_product = mat_mult(query, si, n_results=n_results)

    full_index: list = []
    indices: list = []
    mapped_values: list = []

    if not full_attention:
        # Dot product is all you need
        indices = indices_from_weights(
            mat_product, n_results, full_attention=full_attention
        )
    else:
        # Compared to full attention steps
        attn_weights: np.ndarray = scaled_attention_weights(mat_product, si)
        attended_weights: np.ndarray = apply_self_attention(attn_weights, si)
        attended_vector: np.ndarray = find_attn_vector(attended_weights)

        # NOTE: full atention creates less complex ouput which sorts faster
        # for a large number of results, full attention is more performant
        # TODO: explain this further
        indices = indices_from_weights(
            attended_vector, n_results, full_attention=full_attention
        )

    mapped_values = map_values(values, indices)

    if verbose:
        toc = time.perf_counter()
        print(f"Searched {si.shape[0]} records in {toc - tic:0.4f} seconds")

    return mapped_values[:n_results], indices[:n_results]


if __name__ == "__main__":
    import random

    vecs: np.ndarray = np.load("notebooks/data/svhn.npy")
    labels: list = np.load("notebooks/data/svhn_labels.npy").tolist()

    qidx = 73256  # random.randrange(0, vecs.shape[0], 100)
    q: np.ndarray = vecs[qidx]
    si: np.ndarray = vecs
    attention_search(
        q, si[: qidx + 1], labels, n_results=5, full_attention=False, verbose=True
    )
