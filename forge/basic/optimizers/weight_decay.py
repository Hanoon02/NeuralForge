# Explicit weight decay helper when not baked into the optimizer (teaching / custom loops).
# TODO: apply_weight_decay(params, weight_decay, lr=None) - In-place decay on Parameter.data (L2 or decoupled).
# TODO: l2_penalty(params, weight_decay) - Scalar regularizer value to add to the loss if using loss-based L2.
