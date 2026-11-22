# Learnable 2D convolution: owns kernel/bias Parameters and delegates math to ops.convolution.
# TODO: class Conv2D(Module) - in_channels, out_channels, kernel_size, stride, padding, optional bias.
# TODO: Conv2D.__init__ - Create weight / bias Parameters; initialize (He for ReLU stacks).
# TODO: Conv2D.forward(x) - Call conv ops per channel/batch as designed; cache input for backward.
# TODO: Conv2D.backward(grad_output) - Use ops.conv2d backward; accumulate Parameter grads; return grad_input.
