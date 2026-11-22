# Fully connected layer: the workhorse for MLPs and VAE encoder/decoder heads.
# TODO: class Linear(Module) - in_features, out_features, optional bias.
# TODO: Linear.__init__ - Create weight [out, in] and bias [out] as Parameter; apply init scheme.
# TODO: Linear.forward(x) - y = x @ W.T + b; cache x for backward.
# TODO: Linear.backward(grad_output) - Return grad_input; accumulate grad_weight / grad_bias on Parameters.
