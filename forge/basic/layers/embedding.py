# Token / categorical embedding table used by RNN and Transformer models.
# TODO: class Embedding(Module) - num_embeddings, embedding_dim; Parameter weight [N, D].
# TODO: Embedding.forward(indices) - Gather rows by integer index; cache indices for backward.
# TODO: Embedding.backward(grad_output) - Scatter-add grads into weight.grad.
