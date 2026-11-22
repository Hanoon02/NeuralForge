# Functional affine transform: math behind layers.Linear (no Parameter ownership here).
# TODO: linear_forward(x, weight, bias=None) - x @ weight.T + bias; support batched x.
# TODO: linear_backward(x, weight, bias, grad_output) - Return grad_x, grad_weight, grad_bias.
# TODO: linear(x, weight, bias=None, output_gradient=None) - Unified forward/backward entry like conv2d.
