# Full ResNet Module assembled from layers + residual blocks (consumable by trainer).
# TODO: class ResNetStem(Module) - Conv → BatchNorm → ReLU → optional MaxPool2D.
# TODO: class ResNetStage(Module) - Sequence of BasicBlock/BottleneckBlock with given stride/channels (ModuleList).
# TODO: class ResNet(Module) - stem + four stages + GlobalAvgPool2D + Flatten + Linear classifier.
# TODO: ResNet.__init__(block_type, layers, num_classes, in_channels) - Build stages from config (e.g. ResNet-18).
# TODO: factory helpers - resnet18(num_classes), resnet34(...), resnet50(...) returning configured ResNet modules.
