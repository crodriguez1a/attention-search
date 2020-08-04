from img2vec_pytorch import Img2Vec
from PIL import Image, ImageFile
import numpy as np

img2vec: Img2Vec = Img2Vec(cuda=False, model="alexnet")


def open_img(path: str, size: tuple = (200, 200)) -> ImageFile:
    return Image.open(path).resize(size)


def vec_img(img: ImageFile) -> np.ndarray:
    vec = img2vec.get_vec(img, tensor=False)
    print(vec.shape)
    return vec
