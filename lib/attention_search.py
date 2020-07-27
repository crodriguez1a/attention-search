import time
from typing import Any, List, Sequence, Tuple

# import scipy.linalg.blas TODO: consider using blas directly
import numpy as np

# Inspired by:
# https://towardsdatascience.com/learning-attention-mechanism-from-scratch-f08706aaf6b6


def indices_from_weights(attn_weights: np.ndarray) -> list:
    """
    Index weights to preserve order
    """
    aw_row: np.array = attn_weights[0, :]  # first row
    indices = np.argsort(aw_row)[::-1]  # max on top
    return indices


def softmax(z: np.ndarray, axis: int = -1):
    """Compute corresponding probabilities for each set of scores in z"""
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=axis)


def scaled_dot_product_attention(
    query: np.ndarray, si: np.ndarray, n_results: int = 3
) -> Tuple[Any, List[Any]]:
    "Compute `Scaled Dot Product Attention`"

    # Calculate the scores of all the possible search results
    # in one step using matrix multiplication (this is the magic)
    matmul_qk = np.matmul(query, np.transpose(si))

    # Attention is all you Need (Vaswani, et al., 2017):
    # "We suspect that for large values of dk, the dot products grow
    # large in magnitude, pushing the softmax function into regions where it has
    # extremely small gradients. To counteract this effect, we scale the dot
    # products by 1/sqrt of dk"

    dk: int = si.shape[-1]
    scaled_attn_logits = matmul_qk / np.sqrt(dk)

    # Apply a softmax function to obtain weights for all values
    attn_weights = softmax(scaled_attn_logits, axis=-1)  # axis=-1 last dim

    # Index the weights, truncate to n_results
    indices: list = indices_from_weights(attn_weights)[:n_results]

    return attn_weights, indices


def attention_search(
    query: np.ndarray,
    index: np.ndarray,
    values: List[str] = None,
    n_results: int = 3,
    verbose: bool = False,
) -> tuple:
    """
    Apply scaled dot product attention, then map indices to values
    """
    tic = time.perf_counter()

    _, indices = scaled_dot_product_attention(query, index, n_results=n_results)

    mapped_values: list = []
    if values and max(indices) < len(values):
        mapped_values = [values[i] for i in indices]

    if verbose:
        toc = time.perf_counter()
        print(f"Searched {index.shape[0]} records in {toc - tic:0.4f} seconds")

    return mapped_values, indices
