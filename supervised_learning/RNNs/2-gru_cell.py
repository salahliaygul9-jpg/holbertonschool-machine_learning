import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell."""

    def __init__(self, i, h, o):
        self.i = i
        self.h = h
        self.o = o

        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(concat @ self.Wz + self.bz)
        r = self.sigmoid(concat @ self.Wr + self.br)

        r_h = r * h_prev
        concat_r = np.concatenate((r_h, x_t), axis=1)

        h_tilde = np.tanh(concat_r @ self.Wh + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y_linear = h_next @ self.Wy + self.by
        y = self.softmax(y_linear)

        return h_next, y
