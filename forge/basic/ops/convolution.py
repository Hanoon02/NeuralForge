import numpy as np

'''
For a convolutional operation to be valid, both output dimensions must be positive.
output = (input - kernel + 2 * padding) // stride + 1
Height is axis 0 and width is axis 1.
'''
def check_valid_conv_params_2d(input_shape, kernel_shape, padding, stride):
    output_height = (input_shape[0] - kernel_shape[0] + (2 * padding)) // stride + 1
    output_width = (input_shape[1] - kernel_shape[1] + (2 * padding)) // stride + 1
    valid = output_height > 0 and output_width > 0
    return valid
    
    
'''
Padding applies layers of zeros at the outer layer
Symmetric padding is the one used here, for now, which pads same number of zeros on all sides
Assymetric padding is often used in computer vision tasks, where different padding is applied to different sides
'''
def apply_padding_2d(input, padding):
    array = np.asarray(input) # Making it numpy arr
    symmetric_padding_matrix = [(padding, padding), (padding, padding)] # 4 for each sides, rows, then columns
    
    # In constant mode, we need to define the value through constant_values
    padded_matrix = np.pad(array, symmetric_padding_matrix, mode="constant", constant_values=0)
    
    return padded_matrix


def do_convolution_2d(padded_input, kernel, stride):
    image = np.asarray(padded_input) # Making it numpy arr
    weights = np.asarray(kernel) # Making it numpy arr
    kernel_height, kernel_width = weights.shape
    
    # Define the output shape
    output_height = (image.shape[0] - kernel_height) // stride + 1
    output_width = (image.shape[1] - kernel_width) // stride + 1

    # result_type is used to determine the data type of the output array
    output = np.zeros((output_height, output_width), dtype=np.result_type(image, weights))
    
    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            start_col = col * stride
            window = image[start_row:start_row + kernel_height, start_col:start_col + kernel_width]
            output[row, col] = np.sum(window * weights)
    return output


def conv2d_forward(input, kernel, stride, padding):
    is_valid = check_valid_conv_params_2d(input.shape, kernel.shape, padding, stride)
    if not is_valid:
        raise ValueError("Invalid convolution parameters")
    
    padded_input = apply_padding_2d(input, padding)
    output = do_convolution_2d(padded_input, kernel, stride)
    return output


'''
Convolution backward pass is more like a complain resolution task, the complain itself being the gradient from the next layer.
Its written in the form of dL/dOutput - how much loss can change if we change output of this layer.

We need to find dL/dMatrix and dL/dKernel. 

Couple of rules before we can proceed
1. The dims of dl/dOutput is the same as dims of the original output.
2. Number of windows from a matrix, kernel, stride and padding is equal to output_height * output_width.

The kernel gradient will be the same size as the kernel itself. Computed from complain * window
When we multiply every object in complain with the window, we get a set of matrices which is the same size as window and equal to the number of items in complain matrix or the output.
Summing them all up, ie stacking them all together, gets you the new kernel gradient.

The matrix gradient will be the same size as original matrix. Computed from complain * kernel.
Similiarly we will trace the output creation and stack everything up in the end to get the final input matrix gradient.

With that, we proceed on to the next steps.
1. New kernel = kernel - learning_rate * dL/dKernel
2. The dl/dMatrix leaves and then goes backward as the complain itself to whatever produced it in the first place.

Naturally, the first layer, ie the original image itself wont be updated.

In case of padding, flow is slightly different.
1. Recreate the padded matrix from the original matrix.
2. Continue as normal with the gradient calculation for dl/dMatrix and dl/dKernel.
3. Drop the padding from the final dl/dMatrix one to get the final dl/dMatrix.

'''
def conv2d_backward(input, kernel, stride, padding, output_gradient):
    image = np.asarray(input)
    weights = np.asarray(kernel)
    complain = np.asarray(output_gradient)
    padded_input = apply_padding_2d(image, padding)
    kernel_height, kernel_width = weights.shape

    kernel_gradient = np.zeros(weights.shape, dtype=np.result_type(image, weights, complain))
    padded_input_gradient = np.zeros(padded_input.shape, dtype=np.result_type(image, weights, complain))

    output_height, output_width = complain.shape
    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            start_col = col * stride
            window = padded_input[start_row:start_row + kernel_height, start_col:start_col + kernel_width]
            kernel_gradient += complain[row, col] * window
            padded_input_gradient[start_row:start_row + kernel_height, start_col:start_col + kernel_width] += (
                complain[row, col] * weights
            )

    if padding:
        input_gradient = padded_input_gradient[padding:-padding, padding:-padding]
    else:
        input_gradient = padded_input_gradient
    return input_gradient, kernel_gradient


def conv2d(input, kernel, stride=1, padding=0, output_gradient=None):
    if output_gradient is None:
        return conv2d_forward(input, kernel, stride, padding)
    return conv2d_backward(input, kernel, stride, padding, output_gradient)
