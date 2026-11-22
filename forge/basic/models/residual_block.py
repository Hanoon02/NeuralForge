# Reusable residual pieces; ResNet (and variants) import these instead of reimplementing skips.
# TODO: class ProjectionShortcut(Module) - 1×1 Conv2D + BatchNorm2D to match channel/spatial dims.
# TODO: class BasicBlock(Module) - Two 3×3 Conv-BN-ReLU paths + residual add; optional downsampling shortcut.
# TODO: class BottleneckBlock(Module) - 1×1 → 3×3 → 1×1 with expansion; optional ProjectionShortcut.
# TODO: Residual forward contract - main(x) + shortcut(x), then optional post-ReLU; backward through both branches.
