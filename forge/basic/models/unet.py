# U-Net style encoder-decoder with skip connections (segmentation / dense prediction teaching model).
# TODO: class UNetConvBlock(Module) - Conv → BN → ReLU repeated; used on both paths.
# TODO: class UNetDown(Module) - Conv block + MaxPool (or strided conv); returns skip feature + downsampled map.
# TODO: class UNetUp(Module) - Upsample/conv-transpose → concat skip → conv block.
# TODO: class UNet(Module) - Symmetric down/up stack + 1×1 head to num_classes; forward returns dense logits.
