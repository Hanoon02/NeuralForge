# Shape bridge between conv feature maps and Linear layers.
# TODO: class Flatten(Module) - Flatten all dims after batch (or from start_dim).
# TODO: Flatten.forward(x) - Reshape to (batch, -1); cache original shape.
# TODO: Flatten.backward(grad_output) - Reshape grad back to cached input shape.
