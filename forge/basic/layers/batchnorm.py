# Batch normalization with running stats; critical for stable ResNet (and many deep stacks).
# TODO: class BatchNorm2D(Module) - num_features; Parameters gamma/beta; buffers running_mean/running_var.
# TODO: BatchNorm2D.forward(x) - Train: batch stats + update running; Eval: use running stats; cache for backward.
# TODO: BatchNorm2D.backward(grad_output) - Gradients w.r.t. input, gamma, beta.
# TODO: class BatchNorm1D(Module) - Same idea for (batch, features) — MLP / VAE latent paths.
