from lib.attention_search import *


def test_can_compute_softmax():
    assert (softmax(np.ones(3)) == np.ones(3) / 3).all()
    assert (softmax(np.ones((1, 3))) == np.ones((1, 3)) / 3).all()


def test_can_compute_sorted_indices_from_weights():
    w = softmax(np.array([np.ones(3)]))
    assert (indices_from_weights(w) == [2, 1, 0]).all()


def test_can_safely_map_indices_to_values():

    res = attention_search(
        np.zeros(4).reshape((1, 4)), np.zeros(24).reshape((6, 4)), n_results=3
    )
    assert res is not None


def test_can_safely_ignore_mapping_with_bad_values():

    mv, i = attention_search(
        np.zeros(4).reshape((1, 4)),
        np.zeros(24).reshape((6, 4)),
        ["a", "b"],
        n_results=3,
    )
    assert mv == []
    assert (i == np.array([5, 4, 3])).all()


def test_can_safely_map_indices_to_values_good_values():
    mv, i = attention_search(
        np.array([np.arange(4)]),
        np.arange(24).reshape((6, 4)),
        ["a"] * 24,
        n_results=3,
    )
    assert mv == ["a", "a", "a"]
    assert (i == np.array([5, 4, 3])).all()


# TODO: test all the things
