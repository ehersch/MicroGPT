import math


class Value:
    """
    Autograd class for Value class: native tensor with gradient tracking.
    This is a lightweight version for readability. The version with names, operations, etc. (for easy debugging) is on my GitHub and partially in the notebook.
    """

    def __init__(self, val: float, children=[]):
        self.val = val
        self.children = children  # should be ordered, not sets!
        self.grad = 0  # total gradient (add to this with backprop)
        self._backprop = lambda: None

    def __repr__(self):
        return {self.val}

    # Overrie original add function for Value object
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        new_val = Value(self.val + other.val, children=set([self, other]))

        def _backprop():
            self.grad += new_val.grad
            other.grad += new_val.grad

        new_val._backprop = _backprop

        return new_val

    def __neg__(self):
        return Value(-self.val, children=[self])

    def __sub__(self, other):
        return self.__add__((-other), other_name=other.name)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        new_val = Value(self.val * other.val, children=set([self, other]))

        def _backprop():
            self.grad += new_val.grad * other.val
            other.grad += new_val.grad * self.val

        new_val._backprop = _backprop
        return new_val

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exp: float):
        new_val = Value(self.val**exp, children=[self])
        new_val.local_grads = [exp * self.val ** (exp - 1)]

        def _backprop():
            self.grad += new_val.grad * (exp * self.val ** (exp - 1))

        new_val._backprop = _backprop
        return new_val

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Value(float(other))
        return self * (other ** (-1))

    def exp(self):
        new_val = Value(math.exp(self.val), children=[self])

        def _backprop():
            self.grad += new_val.grad * (math.exp(self.val))

        new_val._backprop = _backprop
        return new_val

    def log(self):
        new_val = Value(math.log(self.val), children=[self])

        def _backprop():
            self.grad += new_val.grad * (self.val ** (-1))

        new_val._backprop = _backprop
        return new_val

    def relu(self):
        new_val = Value(max(0, self.val), children=[self])

        def _backprop():
            self.grad += new_val.grad * (float(self.val > 0))

        new_val._backprop = _backprop
        return new_val

    def backward(self):
        # compute the gradient from the root
        topo = []
        visited = set()

        # first construct a topsort of the computational graph
        def construct_toposort(cur):
            if cur not in visited:
                visited.add(cur)
                for n in cur.children:
                    construct_toposort(n)
                topo.append(cur)

        construct_toposort(self)
        self.grad = 1  # the grad L wrt L is always 1
        for v in reversed(topo):
            v._backprop()
