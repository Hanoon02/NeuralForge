# Trainable tensor wrapper used by every layer and optimizer (torch.nn.Parameter analogue).
# TODO: class Parameter - Hold a NumPy data array and a matching .grad buffer.
# TODO: Parameter.zero_grad() - Reset .grad to zeros (or None) before a new backward.
# TODO: Parameter.shape / Parameter.dtype - Expose data metadata for init and optimizers.
