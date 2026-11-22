# Small CNN (LeNet-style): learn conv stacks before jumping to ResNet.
# TODO: class ConvBlock(Module) - Conv2D → optional BatchNorm2D → ReLU → optional pool.
# TODO: class LeNet(Module) - Classic-ish: conv blocks → Flatten → MLP head for 28×28 / 32×32 images.
# TODO: class SimpleConvNet(Module) - Configurable list of ConvBlocks + classifier Linear; for CIFAR experiments.
# TODO: factories - lenet_mnist(num_classes), simple_cifar_cnn(num_classes).
