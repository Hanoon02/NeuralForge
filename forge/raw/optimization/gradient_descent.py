# Gradient descent variants: the workhorse training loop before adaptive optimizers.
# TODO: batch_gradient_descent(gradient, params, lr, epochs) - Update using the full dataset gradient.
# TODO: stochastic_gradient_descent(gradient, params, lr, epochs) - Update using one sample at a time.
# TODO: mini_batch_gradient_descent(gradient, params, lr, batch_size, epochs) - Update using mini-batches.
# TODO: compute_velocity(gradient, previous_velocity, beta) - Momentum velocity update.
# TODO: momentum_update(params, velocity, lr) - Apply a momentum step to parameters.
# TODO: nesterov_update(params, gradient, velocity, lr, beta) - Nesterov accelerated gradient step.
# TODO: clip_gradients(gradients, max_norm) - Clip gradient norm to limit exploding gradients.
# TODO: l2_regularization_gradient(params, weight_decay) - Gradient contribution of L2 / weight decay.
# TODO: step_decay(lr, epoch, drop_every, factor) - Periodic learning-rate decay schedule.
# TODO: check_convergence(loss_history, tolerance) - Decide whether training has converged.
# TODO: optimize(gradient, params, update_fn, epochs) - Run an optimization loop and track loss.
