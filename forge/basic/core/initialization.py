# Weight init schemes layers call when creating Parameters (fan-in / fan-out aware).
# TODO: set_random_seed(seed) - Make NumPy (and any future RNG) reproducible.
# TODO: zeros(shape) / ones(shape) - Constant Parameter initializers.
# TODO: uniform(shape, a, b) - Uniform random init.
# TODO: normal(shape, mean, std) - Gaussian random init.
# TODO: xavier_uniform(shape) / xavier_normal(shape) - Glorot init for tanh/sigmoid-style layers.
# TODO: he_uniform(shape) / he_normal(shape) - Kaiming init for ReLU networks.
# TODO: initialize_weights(module, scheme) - Walk a Module and init Linear/Conv Parameters by scheme.
