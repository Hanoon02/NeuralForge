# SGD family: default trainer step for CNNs/MLPs before switching to Adam.
# TODO: class SGD - params, lr, optional weight_decay; step() applies -lr * grad (+ decay).
# TODO: class MomentumSGD(SGD) - velocity buffer per Parameter; step() with momentum / Nesterov flag.
# TODO: SGD.zero_grad() - Delegate to zeroing each Parameter.grad.
# TODO: SGD.state_dict() / load_state_dict() - Persist lr and momentum buffers for checkpoints.
