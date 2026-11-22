import pytest

import forge.raw.matrix as matrix

def test_sum_operation():
    matrix_first = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_second = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = matrix.add_matrices(matrix_first, matrix_second)
    assert result == [[2, 4, 6], [8, 10, 12], [14, 16, 18]]

def test_subtract_operation():
    matrix_first = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_second = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = matrix.subtract_matrices(matrix_first, matrix_second)
    assert result == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
