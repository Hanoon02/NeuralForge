# Recurrent cells/layers for sequence models (NumPy; teaching scale).
# TODO: class RNNCell(Module) - Single-step: h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b).
# TODO: class RNN(Module) - Scan RNNCell over time; return all hidden states and/or last h.
# TODO: class LSTMCell(Module) - Gates i,f,g,o; cell state c_t and hidden h_t.
# TODO: class LSTM(Module) - Scan LSTMCell over time; return outputs, (h_n, c_n).
# TODO: class GRUCell(Module) / class GRU(Module) - Optional lighter recurrent alternative.
