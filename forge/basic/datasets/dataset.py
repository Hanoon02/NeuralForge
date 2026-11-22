# Dataset protocol + wrappers shared by all concrete datasets.
# TODO: class Dataset - Abstract __len__ and __getitem__(index) → sample dict or (x, y).
# TODO: class TensorDataset(Dataset) - Wrap NumPy X [, y] for MLP / linear models / VAE arrays.
# TODO: class Subset(Dataset) - Index view for train/val splits.
# TODO: random_split(dataset, lengths, seed) - Shuffle indices into Subset pieces.
# TODO: concat_datasets(datasets) - Optional ConcatDataset for multi-source training.
