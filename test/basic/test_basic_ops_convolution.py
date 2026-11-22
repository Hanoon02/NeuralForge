import numpy as np

import forge.basic.ops.convolution as convolution

def test_conv2d_forward():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[1, 0], [0, 1]])
    result = convolution.conv2d(image, kernel, stride=1, padding=1)
    assert result.tolist() == [
        [1, 2, 3, 0],
        [4, 6, 8, 3],
        [7, 12, 14, 6],
        [0, 7, 8, 9],
    ]

def test_conv2d_backward():
    image = np.array([
        [1.0, 3, 2, 4],
        [5, 1, 6, 2],
        [2, 7, 3, 1],
        [4, 2, 5, 8],
    ])
    kernel = np.array([[2.0, -1], [1, 3]])
    output_gradient = np.array([
        [0.4, -1.2, 0.7],
        [1.5, 0.3, -0.6],
        [-0.8, 2.1, 0.5],
    ])
    input_gradient, kernel_gradient = convolution.conv2d(
        image, kernel, stride=1, padding=0, output_gradient=output_gradient
    )
    assert np.allclose(
        kernel_gradient,
        [[17.0, 4.9], [11.8, 18.3]],
    )
    assert np.allclose(
        input_gradient,
        [
            [0.8, -2.8, 2.6, -0.7],
            [3.4, -0.9, -4.4, 2.7],
            [-0.1, 9.8, -0.8, -2.3],
            [-0.8, -0.3, 6.8, 1.5],
        ],
    )
