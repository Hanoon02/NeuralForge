# Deterministic autoencoder: reconstruction without KL (stepping stone to VAE).
# TODO: class AEEncoder(Module) - MLP or conv stem → latent code z (no sampling).
# TODO: class AEDecoder(Module) - latent z → reconstruction (Sigmoid / linear head).
# TODO: class Autoencoder(Module) - encoder + decoder; forward returns recon, z.
# TODO: class ConvAutoencoder(Module) - Conv encoder + ConvTranspose/MLP decoder for images.
# TODO: ae_loss(recon, target) - MSE or BCE reconstruction for the trainer.
