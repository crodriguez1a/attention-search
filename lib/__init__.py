"""
import numpy as np

# For the purposes of search these remaining steps in the attention function aren't needed
# Keep this for reference


def find_attn_vector(attn_weights: np.ndarray, si: np.ndarray) -> np.ndarray:
    # Multiply the annotations by their weights
    applied_attention = np.multiply(attn_weights, si)
    # Sum up the four columns to produce a single attention context vector
    attn_vector = np.sum(applied_attention, axis=1)

    return attn_vector
"""
