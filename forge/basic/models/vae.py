# Variational autoencoder: reuse MLP/Conv building blocks + losses.kl (β-VAE optional).
# TODO: class VAEEncoder(Module) - MLP or conv stem → Linear mean and Linear log_var heads.
# TODO: reparameterize(mean, log_var) - z = mean + std * eps; cache eps so grads flow to mean/log_var.
# TODO: class VAEDecoder(Module) - latent z → reconstruction (Sigmoid Bernoulli or linear Gaussian).
# TODO: class VAE(Module) - encoder + reparameterize + decoder; forward returns recon, mean, log_var.
# TODO: class ConvVAE(Module) - Conv encoder / decoder variant for image datasets.
# TODO: vae_loss(recon, target, mean, log_var, beta=1.0) - recon (MSE|BCE) + beta * KL for the trainer.
