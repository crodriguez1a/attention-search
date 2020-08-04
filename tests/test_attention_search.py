from lib.attention_search import *


def test_dot_product():
    a = np.zeros((1, 1024))
    b = np.zeros((100, 1024))
    output = np.zeros((1, 100))
    assert (dot_product(a, b) == output).all()


def test_einstein_summation():
    a = np.ones(64).reshape(2, 4, 8)

    path = np.einsum_path("ijk,ilm->ijk", a, a, optimize="optimal")[0]
    for iteration in range(500):
        res = np.einsum("ijk,ilm->ijk", a, a, optimize=path)

    assert res is not None


def test_scale_dot_product():
    pass


def test_can_compute_softmax():
    assert (softmax(np.ones(3)) == np.ones(3) / 3).all()
    assert (softmax(np.ones((1, 3))) == np.ones((1, 3)) / 3).all()


def test_can_compute_softmax_ndims():
    input = np.ones((3, 100, 100))
    assert (softmax(input) == np.ones((3, 100, 100)) / 100).all()


def test_nd_indices_from_weights():
    x = np.zeros((20, 3, 2, 2))
    x[2][0][0] = 1
    x[2][0][1] = 1
    x[3][0][0] = -1
    x[3][0][1] = -1
    x[9][0][0] = 1
    x[9][0][1] = 1

    _, indices = attention_search(np.ones((3, 2, 2)), x, n_results=2, verbose=True)
    assert (indices == np.array([9, 2])).all()


def test_2d_indices_from_weights():
    x = np.zeros((10, 2))
    x[2][0] = 1
    x[2][1] = 1
    x[3][0] = -1
    x[3][1] = -1
    x[9][0] = 1
    x[9][1] = 1

    _, indices = attention_search(np.ones((1, 2)), x, n_results=2, verbose=True)
    assert (indices == np.array([9, 2])).all()


def test_can_compute_sorted_indices_from_weights():
    w = softmax(np.array([np.ones(3)]))
    assert (indices_from_weights(w, 3) == [2, 1, 0]).all()


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
