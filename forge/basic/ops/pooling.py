import numpy as np


'''
For a pooling operation to be valid, both output dimensions must be positive.
output = (input - kernel + 2 * padding) // stride + 1
Height is axis 0 and width is axis 1.
'''
def check_valid_pool_params_2d(input_shape, kernel_shape, padding, stride):
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

def do_max_pool2d(padded_input, kernel_shape, stride):
    image = np.asarray(padded_input) # Making it numpy arr
    kernel_height, kernel_width = kernel_shape
    output_height = (image.shape[0] - kernel_height) // stride + 1
    output_width = (image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.result_type(image))
    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            start_col = col * stride
            window = image[start_row:start_row + kernel_height, start_col:start_col + kernel_width]
            output[row, col] = np.max(window)
    return output

def do_avg_pool2d(padded_input, kernel_shape, stride):
    image = np.asarray(padded_input) # Making it numpy arr
    kernel_height, kernel_width = kernel_shape
    output_height = (image.shape[0] - kernel_height) // stride + 1
    output_width = (image.shape[1] - kernel_width) // stride + 1

    dtype = np.result_type(image)
    if np.issubdtype(dtype, np.integer): # Average pool can change int to float, so we need to handle that.
        dtype = np.result_type(dtype, np.float64)
    output = np.zeros((output_height, output_width), dtype=dtype)
    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            start_col = col * stride
            window = image[start_row:start_row + kernel_height, start_col:start_col + kernel_width]
            output[row, col] = np.mean(window)
    return output

def max_pool2d_forward(input, kernel_shape, stride, padding):
    is_valid = check_valid_pool_params_2d(input.shape, kernel_shape, padding, stride)
    if not is_valid:
        raise ValueError("Invalid pooling parameters")
    
    padded_input = apply_padding_2d(input, padding)
    output = do_max_pool2d(padded_input, kernel_shape, stride)
    return output

def avg_pool2d_forward(input, kernel_shape, stride, padding):
    is_valid = check_valid_pool_params_2d(input.shape, kernel_shape, padding, stride)
    if not is_valid:
        raise ValueError("Invalid pooling parameters")
    
    padded_input = apply_padding_2d(input, padding)
    output = do_avg_pool2d(padded_input, kernel_shape, stride)
    return output

'''
In pooling operation, there is no updation of weights step, we simply create the next complaint that needs to be send back.

Based on the type of pooling, we have different ways to calculate the complaint.
As usual, the complain size is the same as the output size.

We need to calculate dl/dInput.

In MaxPooling backward pass, we need to find the index of the maximum value in the window.
Then we set that index as 1, and perform the regular window stack process by multiplying the complaint with the window.
This becomes the new complaint for the next layer.

In AveragePooling backward pass, the value of the gradient is distributed evenly across all the cells in the window.
This is done by dividing the gradient by the number of cells in the window.
This becomes the new complaint for the next layer.

In case of padding, flow is slightly different.
1. Recreate the padded matrix from the original matrix.
2. Continue as normal with the gradient calculation for dl/dInput.
3. Drop the padding from the final dl/dInput one to get the final dl/dInput.
'''
def max_pool2d_backward(input, kernel_shape, stride, padding, output_gradient):
    image = np.asarray(input)
    complaint = np.asarray(output_gradient)
    padded_input = apply_padding_2d(image, padding)
    kernel_height, kernel_width = kernel_shape

    padded_input_gradient = np.zeros(padded_input.shape, dtype=np.result_type(image, complaint))

    output_height, output_width = complaint.shape
    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            start_col = col * stride
            window = padded_input[start_row:start_row + kernel_height, start_col:start_col + kernel_width]
            mask = np.zeros(window.shape, dtype=padded_input_gradient.dtype)
            mask.flat[np.argmax(window)] = 1
            padded_input_gradient[start_row:start_row + kernel_height, start_col:start_col + kernel_width] += (
                complaint[row, col] * mask
            )

    if padding:
        input_gradient = padded_input_gradient[padding:-padding, padding:-padding]
    else:
        input_gradient = padded_input_gradient
    return input_gradient


def avg_pool2d_backward(input, kernel_shape, stride, padding, output_gradient):
    image = np.asarray(input)
    complaint = np.asarray(output_gradient)
    padded_input = apply_padding_2d(image, padding)
    kernel_height, kernel_width = kernel_shape
    window_size = kernel_height * kernel_width

    dtype = np.result_type(image, complaint)
    if np.issubdtype(dtype, np.integer): # Average pool can change int to float, so we need to handle that.
        dtype = np.result_type(dtype, np.float64)
    padded_input_gradient = np.zeros(padded_input.shape, dtype=dtype)

    output_height, output_width = complaint.shape
    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            start_col = col * stride
            padded_input_gradient[start_row:start_row + kernel_height, start_col:start_col + kernel_width] += (
                complaint[row, col] / window_size
            )

    if padding:
        input_gradient = padded_input_gradient[padding:-padding, padding:-padding]
    else:
        input_gradient = padded_input_gradient
    return input_gradient


def max_pool2d(input, kernel_shape=(2, 2), stride=None, padding=0, output_gradient=None):
    if stride is None:
        stride = kernel_shape[0]
    if output_gradient is None:
        return max_pool2d_forward(input, kernel_shape, stride, padding)
    return max_pool2d_backward(input, kernel_shape, stride, padding, output_gradient)


def avg_pool2d(input, kernel_shape=(2, 2), stride=None, padding=0, output_gradient=None):
    if stride is None:
        stride = kernel_shape[0]
    if output_gradient is None:
        return avg_pool2d_forward(input, kernel_shape, stride, padding)
    return avg_pool2d_backward(input, kernel_shape, stride, padding, output_gradient)