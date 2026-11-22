# Regularization layer shared by MLPs, VAE bottlenecks, and classifier heads.
# TODO: class Dropout(Module) - probability p; active only in training mode.
# TODO: Dropout.forward(x) - Sample Bernoulli mask, scale by 1/(1-p); cache mask.
# TODO: Dropout.backward(grad_output) - Apply the same mask/scale to the incoming gradient.
