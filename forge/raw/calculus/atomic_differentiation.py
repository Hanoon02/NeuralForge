# Tiny autodiff engine: computational graphs, local derivatives, forward and reverse mode.
# TODO: create_node(value, operation, parents) - Create a computational graph node with its dependencies.
# TODO: topological_sort(output_node) - Order graph nodes according to their dependencies.
# TODO: forward_pass(output_node) - Evaluate the computational graph.
# TODO: backward_pass(output_node) - Propagate gradients from output to inputs (reverse mode).
# TODO: accumulate_gradient(node, gradient) - Accumulate gradients from multiple paths (multivariable chain rule).
# TODO: add(a, b) - Graph addition with local derivative tracking.
# TODO: multiply(a, b) - Graph multiplication with local derivative tracking.
# TODO: power(a, exponent) - Exponentiation with derivative tracking.
# TODO: exp(x) - Exponential with derivative tracking.
# TODO: log(x) - Natural logarithm with derivative tracking.
# TODO: relu(x) - ReLU with subgradient tracking.
# TODO: sigmoid(x) - Sigmoid with derivative tracking.
# TODO: apply_function(x, function, derivative) - Apply a custom elementary function while tracking its derivative.
# TODO: forward_mode(f, x) - Derivatives via forward-mode automatic differentiation.
# TODO: reverse_mode(f, x) - Derivatives via reverse-mode automatic differentiation.
# TODO: compute_gradients(output, inputs) - Gradients of one output with respect to many inputs.
