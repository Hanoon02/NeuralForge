import numpy as np

import forge.basic.ops.pooling as pooling

def test_max_pool2d_forward():
    image = np.array([
        [1, 2, 0, 4],
        [5, 6, 7, 8],
        [3, 1, 2, 0],
        [1, 2, 9, 1],
    ])
    result = pooling.max_pool2d(image, kernel_shape=(2, 2), stride=2, padding=0)
    assert result.tolist() == [
        [6, 8],
        [3, 9],
    ]

def test_max_pool2d_backward():
    image = np.array([
        [1.0, 2, 0, 4],
        [5, 6, 7, 8],
        [3, 1, 2, 0],
        [1, 2, 9, 1],
    ])
    output_gradient = np.array([
        [0.1, 0.5],
        [1.0, 0.2],
    ])
    input_gradient = pooling.max_pool2d(
        image, kernel_shape=(2, 2), stride=2, padding=0, output_gradient=output_gradient
    )
    assert np.allclose(
        input_gradient,
        [
            [0, 0, 0, 0],
            [0, 0.1, 0, 0.5],
            [1.0, 0, 0, 0],
            [0, 0, 0.2, 0],
        ],
    )

def test_avg_pool2d_forward():
    image = np.array([
        [1, 2, 0, 4],
        [5, 6, 7, 8],
        [3, 1, 2, 0],
        [1, 2, 9, 1],
    ])
    result = pooling.avg_pool2d(image, kernel_shape=(2, 2), stride=2, padding=0)
    assert np.allclose(
        result,
        [
            [3.5, 4.75],
            [1.75, 3],
        ],
    )

def test_avg_pool2d_backward():
    image = np.array([
        [1.0, 2, 0, 4],
        [5, 6, 7, 8],
        [3, 1, 2, 0],
        [1, 2, 9, 1],
    ])
    output_gradient = np.array([
        [0.1, 0.5],
        [1.0, 0.2],
    ])
    input_gradient = pooling.avg_pool2d(
        image, kernel_shape=(2, 2), stride=2, padding=0, output_gradient=output_gradient
    )
    assert np.allclose(
        input_gradient,
        [
            [0.025, 0.025, 0.125, 0.125],
            [0.025, 0.025, 0.125, 0.125],
            [0.25, 0.25, 0.05, 0.05],
            [0.25, 0.25, 0.05, 0.05],
        ],
    )

def test_max_pool2d_padding():
    image = np.array([
        [-2.0, -3],
        [-4, -5],
    ])
    result = pooling.max_pool2d(image, kernel_shape=(2, 2), stride=2, padding=1)
    assert result.tolist() == [
        [0, 0],
        [0, 0],
    ]
    input_gradient = pooling.max_pool2d(
        image, kernel_shape=(2, 2), stride=2, padding=1, output_gradient=np.ones((2, 2))
    )
    assert np.allclose(
        input_gradient,
        [
            [0, 0],
            [0, 0],
        ],
    )

def test_avg_pool2d_padding():
    image = np.array([
        [1.0, 2],
        [3, 4],
    ])
    result = pooling.avg_pool2d(image, kernel_shape=(2, 2), stride=2, padding=1)
    assert np.allclose(
        result,
        [
            [0.25, 0.5],
            [0.75, 1],
        ],
    )
    input_gradient = pooling.avg_pool2d(
        image, kernel_shape=(2, 2), stride=2, padding=1, output_gradient=np.ones((2, 2))
    )
    assert np.allclose(
        input_gradient,
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ],
    )
