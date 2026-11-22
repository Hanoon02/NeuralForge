# Classification losses: prefer fused softmax+CE for numerical stability (torch-style).
# TODO: class CrossEntropyLoss(Module) - Logits + integer class indices; fused stable softmax + NLL.
# TODO: CrossEntropyLoss.forward(logits, targets) - Numerically stable CE; cache softmax/one-hot needs for backward.
# TODO: CrossEntropyLoss.backward() - (softmax - one_hot) / N; return grad_logits.
# TODO: class SoftmaxCrossEntropy(Module) - Explicit softmax-then-CE path for learning; same grads as fused when stable.
# TODO: class BCELoss(Module) - Binary targets in [0,1] with sigmoid probabilities (or logits via BCEWithLogits).
# TODO: class BCEWithLogitsLoss(Module) - Stable BCE from logits (VAE Bernoulli decoder / binary heads).
