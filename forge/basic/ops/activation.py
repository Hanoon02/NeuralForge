# Functional activations used by layers.activation modules. Done: relu, softmax.
# TODO: sigmoid_forward / sigmoid_backward / sigmoid - For VAE Bernoulli heads and gates.
# TODO: tanh_forward / tanh_backward / tanh - Bounded activations for MLPs / older CNN stacks.
import numpy as np

def relu_forward(input):
    image = np.asarray(input)
    return np.maximum(image, 0)

'''
ReLu's backward pass is direct and simple.
A positive input cell passes the complaint through. A zero or negative input cell gets 0 in dL/dInput.
'''
def relu_backward(input, output_gradient):
    values = np.asarray(input)
    complaint = np.asarray(output_gradient)
    gradient = np.zeros(values.shape, dtype=np.result_type(values, complaint))
    gradient[values > 0] = complaint[values > 0]
    return gradient


def relu(input, output_gradient=None):
    if output_gradient is None:
        return relu_forward(input)
    return relu_backward(input, output_gradient)


def softmax_forward(input):
    scores = np.asarray(input)
    dtype = np.result_type(scores)
    if np.issubdtype(dtype, np.integer): # Softmax can change int to float, so we need to handle that.
        dtype = np.result_type(dtype, np.float64)
    scores = scores.astype(dtype, copy=False)

    shifted = scores - np.max(scores, axis=-1, keepdims=True) # Subtracting the max to avoid overflow.
    exponents = np.exp(shifted) # Exponentiation to make the values positive.
    softmax = exponents / np.sum(exponents, axis=-1, keepdims=True) # Normalizing to make the values sum to 1.
    return softmax


'''
Unlike ReLU, Softmax's backward pass is more complex.
We need to calculate dL/dInput.

First step is to create the Jacobian matrix.
1. for i == j, ie for the diagonal elements, it is output_i * (1 - output_i)
2. for i != j, ie for the off-diagonal elements, it is (-output_i * output_j)

This is a square matrix with the same number of rows and columns as the number of classes.
Simply multiplying this matrix with the complaint to get dl/dInput.
'''

def softmax_backward(input, output_gradient):
    complaint = np.asarray(output_gradient)
    output = softmax_forward(input)

    classes = output.shape[-1]
    output_i = np.expand_dims(output, axis=-1) # -1 adds axis at the end, make this a column vector
    output_j = np.expand_dims(output, axis=-2) # -2 adds axis at the beginning, make this a row vector
    jacobian = -output_i * output_j # i != j, -output_i * output_j
    
    diagonal = np.arange(classes) # [..., diag, diag] creates a matrix of diagonal indices
    jacobian[..., diagonal, diagonal] = output * (1 - output) # i == j, output_i * (1 - output_i)

    # First [..., None] makes the complaint a column vector, then @ does matrix multiplication, t
    # Then [..., 0] gets the first element of the resulting matrix, a row vector.
    matrix_gradient = (jacobian @ complaint[..., None])[..., 0]
    
    return matrix_gradient.reshape(np.shape(output_gradient))


'''
Direct form of the same softmax backward pass.
dL/dInput_j = output_j * (complaint_j - sum over i of complaint_i * output_i)
'''
def softmax_backward_easy(input, output_gradient):
    complaint = np.asarray(output_gradient)
    output = softmax_forward(input)
    summed = np.sum(complaint * output, axis=-1, keepdims=True)
    gradient = output * (complaint - summed)
    return gradient.reshape(np.shape(output_gradient))


def softmax(input, output_gradient=None):
    if output_gradient is None:
        return softmax_forward(input)
    return softmax_backward(input, output_gradient)
