import numpy as np
import pytest

import forge.raw.linear_algebra.vectors as vectors


def test_dot_product():
    assert vectors.dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_dot_product_orthogonal():
    assert vectors.dot_product([1, 0], [0, 1]) == 0


def test_dot_product_negative():
    assert vectors.dot_product([1, -2], [3, 4]) == -5


def test_dot_product_matches_numpy():
    a = np.array([1.5, -2.0, 0.5])
    b = np.array([4.0, 3.0, -1.0])
    assert np.allclose(vectors.dot_product(a.tolist(), b.tolist()), np.dot(a, b))


def test_vector_norm_l1():
    assert vectors.vector_norm([3, -4], 1) == 7


def test_vector_norm_l2():
    assert vectors.vector_norm([3, -4], 2) == 5


def test_vector_norm_matches_numpy():
    vector = np.array([1.5, -2.0, 0.5])
    assert np.allclose(vectors.vector_norm(vector.tolist(), 1), np.linalg.norm(vector, 1))
    assert np.allclose(vectors.vector_norm(vector.tolist(), 2), np.linalg.norm(vector, 2))


def test_vector_norm_unsupported_p():
    with pytest.raises(ValueError):
        vectors.vector_norm([1, 2, 3], 3)
