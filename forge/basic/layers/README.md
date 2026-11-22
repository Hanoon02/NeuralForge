# Layers

Every layer subclasses `Module`, owns `Parameter`s when needed, calls `ops/` for math, and implements `forward` / `backward` so `models/` can stack them into MLPs, CNNs, ResNets, VAEs, GANs, RNNs, U-Nets, and tiny Transformers.
