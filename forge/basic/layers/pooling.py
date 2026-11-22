# Pooling modules wrapping ops.pooling; used by ResNet stem / classifier heads.
# TODO: class MaxPool2D(Module) - kernel_size, stride, padding; forward/backward via max_pool2d.
# TODO: class AvgPool2D(Module) - kernel_size, stride, padding; forward/backward via avg_pool2d.
# TODO: class GlobalAvgPool2D(Module) - Spatial mean → (batch, channels); backward spreads grad uniformly.
