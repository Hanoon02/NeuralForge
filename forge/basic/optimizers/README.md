# Optimizers

All optimizers take `model.parameters()` (or an explicit Parameter list), read `.grad`, and update `.data`. Keep step logic here; keep L2/AdamW decay explicit so models stay framework-agnostic.
