import numpy as np

def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    X: (t, m, i)
    h_0: (m, h)

    Returns:
    H: (t, m, h)
    Y: (t, m, o)
    """

    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t, m, h))
    Y = np.zeros((t, m, o))

    h_prev = h_0

    for step in range(t):
        x_t = X[step]

        h_prev, y_t = rnn_cell.forward(h_prev, x_t)

        H[step] = h_prev
        Y[step] = y_t

    return H, Y
