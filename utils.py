from autograd import Value


# First define some vasic functions for matrix-vector products and matmul (naive, not BLAS), which will be useful for the model architecture.
def linear(W, x):
    # include Value(0.0) so sum knows what to do
    return [sum((w_i * x_i for w_i, x_i in zip(w_row, x)), Value(0.0)) for w_row in W]


def transpose(X):
    """
    Transose a matrix for matmul.
    """
    n, m = len(X), len(X[0])
    new_x = [[Value(0.0) for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            new_x[j][i] = X[i][j]

    return new_x


def matmul(A, B):
    B_T = transpose(B)
    prod = [linear(A, b) for b in B_T]
    return transpose(prod)


def softmax(x):
    """
    Computes softmax for a vector x.
    """
    denom = sum(x_i.exp() for x_i in x)
    return [x_i.exp() / denom for x_i in x]


def rmsnorm(x):
    """
    Mean square norm.
    """
    ms = sum((xi * xi for xi in x), Value(0.0)) / Value(float(len(x)))
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
