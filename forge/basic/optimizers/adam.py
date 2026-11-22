# Adam / AdamW: default for many MLPs and VAEs; optional for ResNet experiments.
# TODO: class Adam - lr, betas, eps; per-Parameter m/v buffers; bias correction; step().
# TODO: class AdamW(Adam) - Decoupled weight decay applied to parameters separately from the adaptive step.
# TODO: Adam.zero_grad() - Clear Parameter grads.
# TODO: Adam.state_dict() / load_state_dict() - Save step count and moment buffers.
