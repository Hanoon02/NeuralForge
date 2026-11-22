# sklearn-analogue linear models as Module subclasses (same trainer/optimizer path as deep nets).
# TODO: class LinearRegression(Module) - Single Linear(in, 1); train with MSELoss.
# TODO: class LogisticRegression(Module) - Linear(in, 1) + BCEWithLogits (binary).
# TODO: class SoftmaxRegression(Module) - Linear(in, num_classes) + CrossEntropyLoss (multiclass).
# TODO: class RidgeRegression(Module) - LinearRegression + explicit L2 via optimizer weight_decay (document the link).
