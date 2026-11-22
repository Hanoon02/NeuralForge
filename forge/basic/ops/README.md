# Ops

Pure NumPy forward/backward **functions**. They do not own weights. `layers/` wraps them in `Module` subclasses that hold `Parameter`s and can be composed into ResNet, VAE, etc.

| Op module | Role | Layer consumer |
|-----------|------|----------------|
| `convolution` | conv2d ± grad | `Conv2D` |
| `pooling` | max/avg pool ± grad | `MaxPool2D`, `AvgPool2D` |
| `activation` | relu, softmax (+ sigmoid/tanh TODOs) | `ReLU`, `Softmax`, … |
| `tensor_ops` | flatten (+ reshape TODOs) | `Flatten` |
| `linear` | affine ± grad | `Linear` |
| `batchnorm` | normalize ± grad | `BatchNorm1D/2D` |
| `dropout` | mask ± grad | `Dropout` |
| `conv_transpose` | upsample conv ± grad | `ConvTranspose2D` |

Defaults:

1. Convolution uses stride 1 and padding 0.
2. Pooling uses a 2×2 window and padding 0; default stride is `kernel_shape[0]`.
