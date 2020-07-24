import numpy as np
import tensorflow as tf

cos = tf.keras.losses.CosineSimilarity(axis=0)


def cos_similarity(x: np.ndarray, y: np.ndarray) -> float:
    return cos(x, y).numpy()
