# Functional batch norm math used by BatchNorm1D/2D layers.
# TODO: batchnorm_forward(x, gamma, beta, running_mean, running_var, training, momentum, eps, axis) - Normalize; update running stats in train.
# TODO: batchnorm_backward(grad_output, cache) - Grads for x, gamma, beta from cached batch stats.
# TODO: batchnorm(...) - Unified entry mirroring other ops (forward when grad is None).
