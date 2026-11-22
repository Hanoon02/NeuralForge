# Sequence models built from Embedding + RNN/LSTM layers.
# TODO: class RNNClassifier(Module) - Embedding → RNN → last (or mean) hidden → Linear logits.
# TODO: class LSTMClassifier(Module) - Same pattern with LSTM for longer dependencies.
# TODO: class SimpleRNNLM(Module) - Embedding → RNN/LSTM → Linear vocab projection (next-token teaching model).
# TODO: packing contract - Batches as (batch, seq) indices; document truncate/pad convention with datasets.
