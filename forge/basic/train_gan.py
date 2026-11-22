"""
MLP GAN on MNIST — wiring pseudocode for the Basic checklist.

Fill in imports and calls as each checklist file lands.
See forge/basic/README.md → Checklists → GAN.
"""

# --- config ---
# cfg = configs.gan_mnist.build_config()
#   ARCH: z_dim, g_hidden, d_hidden, image_dim=784
#   TRAIN: batch_size, epochs, lr_g, lr_d, k_discriminator (D steps per G step)
#   DATA: MNIST in [0,1] (Sigmoid G) or [-1,1] (Tanh G)

# --- data ---
# tf = Compose([ToFloat()])                  # match Generator output range
# train_ds = MNIST(root=..., train=True, transform=tf, download=True)
# train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

# --- model ---
# gan = GAN(
#     z_dim=cfg.z_dim,
#     g_hidden=cfg.g_hidden,
#     d_hidden=cfg.d_hidden,
#     image_dim=cfg.image_dim,
# )                                           # models.gan
#   Generator: z → Linear/ReLU stack → Tanh|Sigmoid → flat image
#   Discriminator: flat image → Linear/ReLU stack → logit
#   ← layers: Linear, ReLU, Tanh/Sigmoid, Flatten
#   ← ops: linear, activation, tensor_ops
# initialize_weights(gan, scheme="xavier")

# --- loss / optim ---
# # discriminator_loss(real_logits, fake_logits)  — BCEWithLogits real=1, fake=0
# # generator_loss(fake_logits)                   — non-saturating BCEWithLogits fake=1
# #   losses.cross_entropy.BCEWithLogitsLoss (or helpers in models.gan)
# opt_d = Adam(gan.discriminator.parameters(), lr=cfg.lr_d)
# opt_g = Adam(gan.generator.parameters(), lr=cfg.lr_g)

# --- train (alternate steps; not a single shared train_one_epoch) ---
# for epoch in range(cfg.epochs):
#     for real, _ in train_loader:
#         real = real.reshape(real.shape[0], -1)
#         batch = real.shape[0]
#
#         # --- k steps Discriminator ---
#         for _ in range(cfg.k_discriminator):
#             z = sample_noise(batch, cfg.z_dim)          # N(0,I)
#             fake = gan.generator(z)                     # detach path for D-only if needed
#             real_logits = gan.discriminator(real)
#             fake_logits = gan.discriminator(fake)
#             d_loss = discriminator_loss(real_logits, fake_logits)
#             gan.discriminator.zero_grad()
#             d_loss.backward()
#             opt_d.step()
#
#         # --- one step Generator ---
#         z = sample_noise(batch, cfg.z_dim)
#         fake = gan.generator(z)
#         fake_logits = gan.discriminator(fake)
#         g_loss = generator_loss(fake_logits)
#         gan.generator.zero_grad()
#         g_loss.backward()
#         opt_g.step()
#
#     save_checkpoint(
#         path=f"checkpoints/gan_epoch{epoch}.npz",
#         model=gan,                                    # G + D state_dicts
#         optimizer={"g": opt_g, "d": opt_d},
#         epoch=epoch,
#         metrics={"d_loss": ..., "g_loss": ...},
#     )
#     # optional: dump generated grid from fixed z for visual check

# --- later: real entrypoint ---
# def main():
#     ...
#
# if __name__ == "__main__":
#     main()
