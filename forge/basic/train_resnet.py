"""
ResNet-18 on CIFAR-10 — wiring pseudocode for the Basic checklist.

Fill in imports and calls as each checklist file lands.
See forge/basic/README.md → Checklists → ResNet.
"""

# --- config ---
# cfg = configs.resnet18.build_config()
#   ARCH: block, layers=(2,2,2,2), num_classes=10, in_channels=3
#   TRAIN: batch_size, epochs, lr, momentum, weight_decay
#   SCHEDULE: step_size, gamma (or cosine)
#   DATA: mean/std, augmentations

# --- data ---
# train_tf = Compose([
#     RandomCrop(32, padding=4),
#     RandomHorizontalFlip(),
#     ToFloat(),
#     Normalize(cifar10_mean_std()),
# ])
# test_tf = Compose([ToFloat(), Normalize(cifar10_mean_std())])
#
# train_ds = CIFAR10(root=..., train=True, transform=train_tf, download=True)
# test_ds  = CIFAR10(root=..., train=False, transform=test_tf, download=True)
# train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
# test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

# --- model ---
# model = resnet18(num_classes=cfg.num_classes)   # models.resnet
#   uses: Conv2D, BatchNorm2D, ReLU, MaxPool2D, BasicBlock,
#         GlobalAvgPool2D, Flatten, Linear
#         ← ops: convolution, batchnorm, activation, pooling, linear, flatten
# initialize_weights(model, scheme="he")          # core.initialization

# --- loss / optim / schedule ---
# loss_fn = CrossEntropyLoss()                    # losses.cross_entropy
# optimizer = MomentumSGD(
#     model.parameters(),
#     lr=cfg.lr,
#     momentum=cfg.momentum,
#     weight_decay=cfg.weight_decay,              # or apply_weight_decay
# )
# scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

# --- train ---
# for epoch in range(cfg.epochs):
#     train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer)
#     val_stats   = evaluate(model, test_loader, loss_fn, metric_fns=[accuracy])
#     scheduler.step()
#     save_checkpoint(
#         path=f"checkpoints/resnet18_epoch{epoch}.npz",
#         model=model,
#         optimizer=optimizer,
#         epoch=epoch,
#         metrics=val_stats,
#         scheduler=scheduler,
#     )
#
# # Or one-shot:
# # fit(model, train_loader, test_loader, loss_fn, optimizer,
# #     epochs=cfg.epochs, scheduler=scheduler, checkpoint_dir=...)

# --- later: real entrypoint ---
# def main():
#     ...
#
# if __name__ == "__main__":
#     main()
