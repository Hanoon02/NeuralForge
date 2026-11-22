"""
MLP VAE on MNIST — wiring pseudocode for the Basic checklist.

Fill in imports and calls as each checklist file lands.
See forge/basic/README.md → Checklists → VAE.
"""

# --- config ---
# cfg = configs.vae.build_config()
#   ARCH: input_dim=784, hidden_dims, latent_dim, observation="bernoulli"|"gaussian"
#   TRAIN: batch_size, epochs, lr, beta (KL weight)
#   DATA: flatten + ToFloat; optional binarize for Bernoulli

# --- data ---
# tf = Compose([ToFloat()])                  # values in [0, 1]
# # optional: binarize with threshold 0.5 for Bernoulli decoder
# train_ds = MNIST(root=..., train=True, transform=tf, download=True)
# test_ds  = MNIST(root=..., train=False, transform=tf, download=True)
# train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
# test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

# --- model ---
# model = VAE(
#     input_dim=cfg.input_dim,
#     hidden_dims=cfg.hidden_dims,
#     latent_dim=cfg.latent_dim,
#     observation=cfg.observation,
# )                                           # models.vae
#   uses: Flatten, Linear, ReLU, Sigmoid (Bernoulli head)
#         ← ops: linear, activation, tensor_ops
#   forward(x) -> recon, mean, log_var
# initialize_weights(model, scheme="xavier")  # or he if ReLU-heavy

# --- loss / optim ---
# # vae_loss = recon (BCE or MSE) + beta * kl_divergence_gaussian(mean, log_var)
# #   losses.kl + losses.cross_entropy (BCE) or losses.mse
# optimizer = Adam(model.parameters(), lr=cfg.lr)

# --- train ---
# for epoch in range(cfg.epochs):
#     model.train()
#     for x, _ in train_loader:               # labels unused
#         x = x.reshape(x.shape[0], -1)       # or Flatten inside VAE
#         recon, mean, log_var = model(x)
#         loss = vae_loss(recon, x, mean, log_var, beta=cfg.beta)
#         model.zero_grad()
#         loss.backward()                     # via Module API / cached graph
#         optimizer.step()
#
#     model.eval()
#     val = evaluate_vae(model, test_loader, beta=cfg.beta)
#     # metrics: avg recon, avg KL, avg ELBO
#     save_checkpoint(
#         path=f"checkpoints/vae_epoch{epoch}.npz",
#         model=model,
#         optimizer=optimizer,
#         epoch=epoch,
#         metrics=val,
#     )
#
# # optional: sample z ~ N(0,I), decode to visualize generations

# --- later: real entrypoint ---
# def main():
#     ...
#
# if __name__ == "__main__":
#     main()
