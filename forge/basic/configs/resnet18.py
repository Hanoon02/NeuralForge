# Default hyperparameters and architecture spec for training ResNet-18 with the basic trainer.
# TODO: ARCH - block=BasicBlock, layers=(2, 2, 2, 2), num_classes, in_channels.
# TODO: TRAIN - batch_size, epochs, optimizer (SGD+momentum), lr, momentum, weight_decay.
# TODO: SCHEDULE - StepLR step_size / gamma (or cosine).
# TODO: DATA - normalize mean/std, train augmentations (flip, crop).
# TODO: build_config() - Return a plain dict (or simple namespace) consumed by scripts/trainer.
