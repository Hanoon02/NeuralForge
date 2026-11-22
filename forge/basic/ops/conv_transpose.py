# Functional conv-transpose (upsample) math for ConvTranspose2D / U-Net / conv decoders.
# TODO: conv_transpose2d_forward(input, kernel, stride, padding, output_padding) - Upsample convolution.
# TODO: conv_transpose2d_backward(input, kernel, stride, padding, output_padding, output_gradient) - grad_input, grad_kernel.
# TODO: conv_transpose2d(...) - Unified forward/backward entry like conv2d.
