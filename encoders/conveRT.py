import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_text

convert_v1 = "http://models.poly-ai.com/convert/v1/model.tar.gz"
convert_module = tfhub.load(convert_v1)
conv = convert_module.signatures["default"]


def conv_vec(text: str) -> np.ndarray:
    return conv(tf.convert_to_tensor([text]))["default"].numpy()
