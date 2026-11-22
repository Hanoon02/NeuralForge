# Minimal GAN: two Modules + adversarial objectives (composition practice, not SOTA).
# TODO: class Generator(Module) - noise z → fake sample (MLP or small conv stack).
# TODO: class Discriminator(Module) - sample → logit (real vs fake).
# TODO: class GAN(Module) - Holds G and D; helpers for fake = G(z).
# TODO: discriminator_loss(real_logits, fake_logits) - BCE-with-logits style real/fake loss.
# TODO: generator_loss(fake_logits) - Non-saturating (or saturating) generator objective.
# TODO: train_step_gan(...) - Alternate D step then G step using basic optimizers (documented loop, not magic).
