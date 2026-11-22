# Generic multilayer perceptron — default backbone for tabular, MNIST, and VAE-style heads.
# TODO: class MLP(Module) - Modular stack: Linear → activation → optional Dropout, repeated; final Linear.
# TODO: MLP.__init__(in_features, hidden_dims, out_features, activation, dropout) - Build via Sequential/ModuleList.
# TODO: MLP.forward(x) - Flatten last dims if needed; return logits or features.
# TODO: factory mlp_classifier(...) / mlp_regressor(...) - Classification vs regression output sizes.
