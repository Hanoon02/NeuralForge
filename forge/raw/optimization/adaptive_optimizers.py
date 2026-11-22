# Adaptive per-parameter step sizes: AdaGrad, RMSProp, Adam, and AdamW from first principles.
# TODO: adagrad_update(params, gradients, accumulators, lr, epsilon) - Scale steps by accumulated squared gradients.
# TODO: rmsprop_update(params, gradients, squared_avg, lr, beta, epsilon) - RMSProp with exponential moving average of squares.
# TODO: adam_moments(gradients, m, v, beta1, beta2) - First and second moment estimates.
# TODO: bias_correction(m, v, beta1, beta2, timestep) - Correct Adam moment initialization bias.
# TODO: adam_update(params, gradients, m, v, timestep, lr, epsilon) - Full Adam parameter update.
# TODO: adamw_update(params, gradients, m, v, timestep, lr, weight_decay, epsilon) - Adam with decoupled weight decay.
# TODO: reset_optimizer_state(params) - Initialize empty optimizer state for a parameter set.
