from lib.attention_search import *


def test_can_compute_softmax():
    assert (softmax(np.ones(3)) == np.ones(3) / 3).all()
    assert (softmax(np.ones((1, 3))) == np.ones((1, 3)) / 3).all()


def test_can_compute_sorted_indices_from_weights():
    w = softmax(np.array([np.ones(3)]))
    assert (indices_from_weights(w) == [2, 1, 0]).all()


# TODO: test all the things
