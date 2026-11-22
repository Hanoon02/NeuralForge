# Attention primitives for Transformer-style models in basic.models.attention.
# TODO: class LayerNorm(Module) - Normalize over feature dim; gamma/beta Parameters.
# TODO: scaled_dot_product_attention(q, k, v, mask=None) - Functional softmax(QKᵀ/√d) V (+ mask).
# TODO: class MultiHeadAttention(Module) - Project Q/K/V, split heads, attend, merge, out projection.
# TODO: class TransformerEncoderLayer(Module) - MHA + residual + LayerNorm + MLP + residual + LayerNorm.
# TODO: class PositionalEncoding(Module) - Sinusoidal or learned positions added to embeddings.
