"""
import numpy as np

# For the purposes of search these remaining steps in the attention function
# weren't needed since the output is more useful as a ranked sequence of weights instead
of a single class

# Keep these for reference

def scale_dot_product(matmul_qsi: np.ndarray, si: np.ndarray) -> np.ndarray:
    \"""
    Attention is all you Need (Vaswani, et al., 2017):
    "We suspect that for large values of dk, the dot products grow
    large in magnitude, pushing the softmax function into regions where it has
    extremely small gradients. To counteract this effect, we scale the dot
    products by 1/sqrt of dk
    \"""
    dk: int = si.shape[-1]
    return matmul_qsi / np.sqrt(dk)

def softmax(x: np.ndarray, axis: int = -1):
    \"""
    Compute corresponding probabilities for each set of scores in x
    \"""
    # TODO speed this up? https://github.com/numpy/numpy/issues/8233
    e_x = tnp.exp(x - np.max(x))
    return np.divide(e_x, e_x.sum(axis=axis, keepdims=True))

# Apply scaling to raw values
scaled_attn_logits: np.ndarray = scale_dot_product(matmul_qsi, si)

# Apply a softmax function to obtain weights for all scaled values
# attn_weights = softmax(scaled_attn_logits, axis=-1)  # axis=-1 last dim

# Multiply the annotations by their weights
applied_attention = np.multiply(attn_weights, si)

# Sum up the four columns to produce a single attention context vector
attn_vector = np.sum(applied_attention, axis=1)

"""
