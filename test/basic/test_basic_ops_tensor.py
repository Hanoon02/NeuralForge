import numpy as np

import forge.basic.ops.tensor_ops as tensor_ops

def test_flatten_forward():
    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    result = tensor_ops.flatten(image)
    assert result.tolist() == [1, 2, 3, 4, 5, 6]

def test_flatten_backward():
    image = np.array([
        [1.0, 2, 3],
        [4, 5, 6],
    ])
    output_gradient = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    input_gradient = tensor_ops.flatten(image, output_gradient=output_gradient)
    assert np.allclose(
        input_gradient,
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )
