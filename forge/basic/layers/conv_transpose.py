# Learnable upsampling convolution for ConvAutoencoder / ConvVAE / U-Net upsizing paths.
# TODO: class ConvTranspose2D(Module) - in/out channels, kernel, stride, padding, output_padding; weight Parameter.
# TODO: ConvTranspose2D.forward(x) - Fractionally strided conv (ops.conv_transpose2d); cache for backward.
# TODO: ConvTranspose2D.backward(grad_output) - Accumulate Parameter grads; return grad_input.
