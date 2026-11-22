# Base module API: compose layers into networks that ResNet, VAE, and trainers can share.
# TODO: class Module - Base with train()/eval(), training flag, and forward(*args) / backward(grad).
# TODO: Module.__call__ - Route through forward; cache activations needed for backward.
# TODO: Module.parameters(recurse=True) - Yield Parameter objects from self and child modules.
# TODO: Module.named_parameters(recurse=True) - Yield (name, Parameter) pairs for checkpoints.
# TODO: Module.children() / Module.modules() - Iterate registered submodules.
# TODO: Module.add_module(name, module) - Register a child module by name.
# TODO: Module.zero_grad() - Zero grads on all parameters.
# TODO: Module.state_dict() / Module.load_state_dict(state) - Serialize / restore learnable state.
# TODO: class Sequential(Module) - Ordered container; forward chains children, backward reverses them.
# TODO: class ModuleList(Module) - List-like container for a dynamic number of children (ResNet stages).
