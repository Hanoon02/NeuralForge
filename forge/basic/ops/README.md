These files contain the actual mathematical implementations, independent of neural network layer classes.

For example, conv2d() knows how to perform convolution, but it doesn't own trainable weights. That responsibility belongs to Conv2D.

