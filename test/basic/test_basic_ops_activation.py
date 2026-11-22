import numpy as np

import forge.basic.ops.activation as activation

def test_relu_forward():
    image = np.array([
        [-1, 2],
        [0, -3],
    ])
    result = activation.relu(image)
    assert result.tolist() == [
        [0, 2],
        [0, 0],
    ]

def test_relu_backward():
    image = np.array([
        [-1.0, 2],
        [0, 3],
    ])
    output_gradient = np.array([
        [0.5, -0.2],
        [1.0, 0.4],
    ])
    input_gradient = activation.relu(image, output_gradient=output_gradient)
    assert np.allclose(
        input_gradient,
        [
            [0, -0.2],
            [0, 0.4],
        ],
    )

def test_softmax_forward():
    scores = np.array([
        [1.0, 2, 3],
        [1, 1, 1],
    ])
    result = activation.softmax(scores)
    assert np.allclose(result.sum(axis=-1), [1, 1])
    assert np.allclose(
        result[0],
        np.exp(scores[0] - 3) / np.sum(np.exp(scores[0] - 3)),
    )
    assert np.allclose(result[1], [1 / 3, 1 / 3, 1 / 3])

def test_softmax_backward():
    scores = np.array([
        [1.0, 2, 3],
        [0, 0, 0],
    ])
    output_gradient = np.array([
        [0.1, 0.2, 0.3],
        [1.0, 0, -1],
    ])
    input_gradient = activation.softmax(scores, output_gradient=output_gradient)

    numeric = np.zeros_like(scores)
    eps = 1e-6
    for index in np.ndindex(scores.shape):
        plus = scores.copy()
        minus = scores.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric[index] = np.sum(
            output_gradient * (activation.softmax(plus) - activation.softmax(minus)) / (2 * eps)
        )
    assert np.allclose(input_gradient, numeric, atol=1e-7)
    assert np.allclose(
        input_gradient,
        activation.softmax_backward_easy(scores, output_gradient),
    )

def test_softmax_integer_input():
    result = activation.softmax(np.array([1, 2, 3]))
    assert result.dtype.kind == "f"
    assert np.allclose(result, activation.softmax(np.array([1.0, 2, 3])))
