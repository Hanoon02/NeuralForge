# Basic

NumPy-backed building blocks just below PyTorch / TensorFlow / sklearn: **classes and modules you compose**, not pure-math drills.

| Stage | Style | Goal |
|-------|--------|------|
| `raw` | Pure Python functions | Learn the math by hand |
| `basic` | NumPy + reusable classes | Build real models from pieces |

`ops/` holds functional forward/backward math. `layers/`, `losses/`, `optimizers/`, and `core/` wrap those into modules with state. `models/` composes them into MLP, ConvNet, ResNet, AE/VAE, GAN, linear classifiers, RNN/LSTM, tiny Transformers, U-Net, etc. `datasets/` feeds those experiments (MNIST, CIFAR, synthetic, tabular, text).

## When to use NumPy?

- Element-wise ops over arrays
- Linear algebra without hand-written nested loops
- Vectorized implementations that layers and trainers call

## When to use Pandas?

- Tabular datasets and simple preprocessing pipelines

## Checklists

Track which building blocks are ready for an end-to-end run. Each checklist is the **minimum file set** that recipe needs. Entry scripts under `basic/` (`train_*.py`) hold pseudocode that wires them; fill implementations later.

### ResNet

Goal: train ResNet-18 on CIFAR-10 via `train_resnet.py`.

**Core**
- [ ] `core/parameter.py`
- [ ] `core/module.py`
- [ ] `core/initialization.py`

**Ops** (functional math layers call)
- [x] `ops/convolution.py`
- [x] `ops/pooling.py`
- [x] `ops/activation.py` (ReLU)
- [x] `ops/tensor_ops.py` (flatten)
- [ ] `ops/batchnorm.py`
- [ ] `ops/linear.py`

**Layers**
- [ ] `layers/conv.py`
- [ ] `layers/batchnorm.py`
- [ ] `layers/activation.py` (ReLU module)
- [ ] `layers/pooling.py` (MaxPool2D, GlobalAvgPool2D)
- [ ] `layers/flatten.py`
- [ ] `layers/linear.py`

**Model**
- [ ] `models/residual_block.py` (BasicBlock, ProjectionShortcut)
- [ ] `models/resnet.py` (stem, stages, `resnet18`)

**Loss / optim / schedule**
- [ ] `losses/cross_entropy.py`
- [ ] `optimizers/sgd.py` (MomentumSGD)
- [ ] `optimizers/weight_decay.py`
- [ ] `training/scheduler.py` (StepLR or CosineAnnealingLR)

**Data**
- [ ] `datasets/dataset.py`
- [ ] `datasets/cifar10.py`
- [ ] `training/transforms.py` (Normalize, RandomHorizontalFlip, RandomCrop, Compose)
- [ ] `training/dataloader.py`

**Train loop**
- [ ] `training/metrics.py` (accuracy)
- [ ] `training/checkpoint.py`
- [ ] `training/trainer.py`
- [ ] `configs/resnet18.py`
- [ ] `train_resnet.py` (pseudocode → real script)

### VAE

Goal: train an MLP VAE on MNIST via `train_vae.py` (Bernoulli decoder + β-KL).

**Core**
- [ ] `core/parameter.py`
- [ ] `core/module.py`
- [ ] `core/initialization.py`

**Ops**
- [x] `ops/activation.py` (ReLU; need Sigmoid)
- [x] `ops/tensor_ops.py` (flatten)
- [ ] `ops/linear.py`
- [ ] `ops/activation.py` — `sigmoid` forward/backward

**Layers**
- [ ] `layers/linear.py`
- [ ] `layers/activation.py` (ReLU, Sigmoid)
- [ ] `layers/flatten.py`
- [ ] `layers/dropout.py` (optional in encoder/decoder MLPs)

**Model / loss**
- [ ] `models/mlp.py` (optional shared MLP blocks) or inline Linear stacks in VAE
- [ ] `models/vae.py` (VAEEncoder, reparameterize, VAEDecoder, VAE, `vae_loss`)
- [ ] `losses/kl.py` (`kl_divergence_gaussian`)
- [ ] `losses/cross_entropy.py` (BCEWithLogits / BCELoss) **or** `losses/mse.py` (Gaussian decoder)

**Optim**
- [ ] `optimizers/adam.py`

**Data**
- [ ] `datasets/dataset.py`
- [ ] `datasets/mnist.py`
- [ ] `training/transforms.py` (ToFloat, optional Normalize / binarize)
- [ ] `training/dataloader.py`

**Train loop**
- [ ] `training/metrics.py` (`mean_squared_error` or recon BCE as metric)
- [ ] `training/checkpoint.py`
- [ ] `training/trainer.py` (or a VAE-specific epoch loop in the script)
- [ ] `configs/vae.py`
- [ ] `train_vae.py` (pseudocode → real script)

### GAN

Goal: train a tiny MLP GAN on MNIST via `train_gan.py` (alternate D then G steps).

**Core**
- [ ] `core/parameter.py`
- [ ] `core/module.py`
- [ ] `core/initialization.py`

**Ops**
- [x] `ops/activation.py` (ReLU; need Sigmoid / Tanh for heads)
- [x] `ops/tensor_ops.py` (flatten)
- [ ] `ops/linear.py`
- [ ] `ops/activation.py` — `sigmoid` / `tanh` forward/backward

**Layers**
- [ ] `layers/linear.py`
- [ ] `layers/activation.py` (ReLU, Tanh or Sigmoid on G output)
- [ ] `layers/flatten.py`
- [ ] `layers/dropout.py` (optional on Discriminator)

**Model / loss**
- [ ] `models/gan.py` (Generator, Discriminator, GAN, `discriminator_loss`, `generator_loss`)
- [ ] `losses/cross_entropy.py` (BCEWithLogitsLoss for real/fake logits)

**Optim**
- [ ] `optimizers/adam.py` (separate optimizers for G and D)

**Data**
- [ ] `datasets/dataset.py`
- [ ] `datasets/mnist.py`
- [ ] `training/transforms.py` (ToFloat; scale to [0,1] or [-1,1] to match G)
- [ ] `training/dataloader.py`

**Train loop**
- [ ] `training/checkpoint.py` (save G + D + both optimizers)
- [ ] `configs/gan_mnist.py`
- [ ] `train_gan.py` (pseudocode → real script; custom alternate-step loop)
