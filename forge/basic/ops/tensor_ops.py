# Stateless tensor shaping helpers. Done: flatten. Layers own Parameters; ops stay pure functions.
# TODO: reshape_forward / reshape_backward - General reshape with cached input shape.
# TODO: transpose_forward / transpose_backward - Axis permutation + inverse perm for grads.
import numpy as np

def flatten_forward(input):
    values = np.asarray(input)
    return values.reshape(-1)


'''
Flatten's backward pass is simple.
We only reshape the output gradient back into the original input shape.
'''

def flatten_backward(input, output_gradient):
    values = np.asarray(input)
    complaint = np.asarray(output_gradient)
    return complaint.reshape(values.shape)


def flatten(input, output_gradient=None):
    if output_gradient is None:
        return flatten_forward(input)
    return flatten_backward(input, output_gradient)
