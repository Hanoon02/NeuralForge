# Regression / reconstruction loss (VAE decoder MSE variant, simple regressors).
# TODO: class MSELoss(Module) - reduction: mean | sum | none.
# TODO: MSELoss.forward(prediction, target) - (pred - target)^2 reduced; cache for backward.
# TODO: MSELoss.backward() - dL/dpred = 2*(pred - target) / N (for mean); return grad_prediction.
