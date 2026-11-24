import numpy as np

import forge.raw.linear_algebra.eigen_decomposition as eigen


def _is_parallel(vector, expected):
    vector = np.asarray(vector, dtype=float)
    expected = np.asarray(expected, dtype=float)
    cosine = np.dot(vector, expected) / (np.linalg.norm(vector) * np.linalg.norm(expected))
    return np.allclose(abs(cosine), 1.0)


def test_eigenvalues_2x2_symmetric():
    matrix = [
        [2, 1],
        [1, 2],
    ]
    assert eigen.eigenvalues_2x2(matrix) == (3.0, 1.0)


def test_eigenvalues_2x2_diagonal():
    matrix = [
        [4, 0],
        [0, -1],
    ]
    assert eigen.eigenvalues_2x2(matrix) == (4.0, -1.0)


def test_eigenvalues_2x2_repeated():
    matrix = [
        [5, 0],
        [0, 5],
    ]
    assert eigen.eigenvalues_2x2(matrix) == (5.0, 5.0)


def test_eigenvalues_2x2_characteristic_equation():
    matrix = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    trace = 5.0
    determinant = -2.0
    values = eigen.eigenvalues_2x2(matrix)
    assert values[0] >= values[1]
    for value in values:
        assert np.allclose(value**2 - trace * value + determinant, 0.0)


def test_eigenvalues_2x2_matches_numpy():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    values = np.array(eigen.eigenvalues_2x2(matrix.tolist()))
    expected = np.sort(np.linalg.eigvals(matrix).real)[::-1]
    assert np.allclose(values, expected)


def _groups(matrix, eigenvalues, vectors):
    # Two vectors when both formulas apply, one when a denominator is zero.
    a = matrix[0][0]
    c = matrix[1][0]
    offset = 0
    groups = []
    for value in eigenvalues:
        count = 2 if (a - value) != 0 and c != 0 else 1
        group = vectors[offset:offset + count]
        assert len(group) == count
        groups.append((value, group))
        offset += count
    assert offset == len(vectors)
    return groups


def test_eigenvector_symmetric():
    matrix = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    eigenvalues = [3.0, 1.0]
    directions = [[1.0, 1.0], [-1.0, 1.0]]
    vectors = eigen.eigenvector(matrix.tolist(), eigenvalues)
    for (value, group), direction in zip(_groups(matrix, eigenvalues, vectors), directions):
        for vector in group:
            vector = np.asarray(vector, dtype=float)
            assert _is_parallel(vector, direction)
            assert np.allclose(matrix @ vector, value * vector)


def test_eigenvector_diagonal():
    matrix = np.array([
        [4.0, 0.0],
        [0.0, -1.0],
    ])
    eigenvalues = [4.0, -1.0]
    directions = [[1.0, 0.0], [0.0, 1.0]]
    vectors = eigen.eigenvector(matrix.tolist(), eigenvalues)
    for (value, group), direction in zip(_groups(matrix, eigenvalues, vectors), directions):
        for vector in group:
            vector = np.asarray(vector, dtype=float)
            assert _is_parallel(vector, direction)
            assert np.allclose(matrix @ vector, value * vector)


def test_eigenvector_from_computed_eigenvalues():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    eigenvalues = eigen.eigenvalues_2x2(matrix.tolist())
    vectors = eigen.eigenvector(matrix.tolist(), eigenvalues)
    for value, group in _groups(matrix, eigenvalues, vectors):
        for vector in group:
            vector = np.asarray(vector, dtype=float)
            assert np.linalg.norm(vector) > 0
            assert np.allclose(matrix @ vector, value * vector)


def test_eigenvector_repeated_eigenvalue():
    matrix = np.array([
        [5.0, 0.0],
        [0.0, 5.0],
    ])
    vectors = eigen.eigenvector(matrix.tolist(), [5.0, 5.0])
    for value, group in _groups(matrix, [5.0, 5.0], vectors):
        for vector in group:
            vector = np.asarray(vector, dtype=float)
            assert np.linalg.norm(vector) > 0
            assert np.allclose(matrix @ vector, value * vector)


def test_eigenvector_jordan_block():
    matrix = np.array([
        [2.0, 1.0],
        [0.0, 2.0],
    ])
    vectors = eigen.eigenvector(matrix.tolist(), [2.0, 2.0])
    for value, group in _groups(matrix, [2.0, 2.0], vectors):
        for vector in group:
            vector = np.asarray(vector, dtype=float)
            assert _is_parallel(vector, [1.0, 0.0])
            assert np.allclose(matrix @ vector, value * vector)


def _assert_qr(matrix):
    orthogonal, upper = eigen.qr_decomposition(matrix.tolist())
    orthogonal = np.asarray(orthogonal, dtype=float)
    upper = np.asarray(upper, dtype=float)
    size = matrix.shape[0]
    assert np.allclose(orthogonal.T @ orthogonal, np.eye(size))
    assert np.allclose(np.tril(upper, k=-1), 0)
    assert np.allclose(orthogonal @ upper, matrix)


def test_qr_decomposition_2x2():
    _assert_qr(np.array([
        [1.0, 0.0],
        [2.0, 3.0],
    ]))


def test_qr_decomposition_symmetric():
    _assert_qr(np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ]))


def test_qr_decomposition_3x3():
    _assert_qr(np.array([
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [5.0, 6.0, 0.0],
    ]))
