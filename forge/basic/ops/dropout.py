# Functional dropout; layer.Dropout owns the training flag and calls this.
# TODO: dropout_forward(x, p, training, seed=None) - Mask + scale; return output and mask cache.
# TODO: dropout_backward(grad_output, mask, p) - Apply the same mask/scale to grads.
# TODO: dropout(x, p, training, output_gradient=None, mask=None) - Unified forward/backward entry.
