# Dataset-side transforms composed into pipelines (sklearn-ish preprocessing + vision augments).
# TODO: class Compose - Apply a list of transforms in order.
# TODO: class Normalize - Subtract mean / divide std (per-channel for images).
# TODO: class ToFloat - Cast image arrays to float in [0, 1] or similar.
# TODO: class RandomHorizontalFlip - Flip with probability p (training aug).
# TODO: class RandomCrop - Pad then crop to size (CIFAR-style).
# TODO: class StandardScalerTransform - Fit mean/std on train; transform train/test (tabular / feature paths).
