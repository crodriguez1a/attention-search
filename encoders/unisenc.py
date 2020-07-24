import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

usenc4 = "https://tfhub.dev/google/universal-sentence-encoder/4"
usenc_module = tfhub.load(usenc4)
usenc = usenc_module.signatures["serving_default"]


def usenc_vec(text: str) -> np.ndarray:
    return usenc(tf.convert_to_tensor([text]))["outputs"].numpy()
