# Activation modules: thin Module wrappers over ops so Sequential/ResNet/VAE can stack them.
# TODO: class ReLU(Module) - forward/backward via ops.relu; cache mask/input.
# TODO: class Sigmoid(Module) - forward/backward via ops.sigmoid (VAE outputs, gates).
# TODO: class Tanh(Module) - forward/backward via ops.tanh.
# TODO: class Softmax(Module) - forward/backward via ops.softmax along dim (prefer fused loss when training classifiers).
