These files contain the actual mathematical implementations, independent of neural network layer classes.

conv2d() performs convolution and, on the backward pass, returns both the input gradient and the kernel gradient. It does not own trainable weights. That responsibility belongs to Conv2D.
max_pool2d() and avg_pool2d() only route the incoming gradient back to the input. Pooling has no weights to update.

Defaults:
1. Convolution uses stride 1 and padding 0.
2. Pooling uses a 2×2 window and padding 0. The default stride is the window height, kernel_shape[0]. For the usual square window that stride is 2, so each window starts where the previous one ended.
