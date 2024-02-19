import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.grad = 0.0
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        # out.label = self.label + '+' + other.label
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        # out.label = self.label + '*' + other.label
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        # out.label = f'tanh({self.label})'
        return out

    def exp(self):
        x = self.data
        out = Value(x, (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def new_backward(self):
        visited = set()
        topo = []
        stack = [self]

        while stack:
            current_v = stack[-1]
            if current_v in visited:
                stack.pop()  # Remove node from stack if all children have been visited
                continue

            # Process children first (dependencies)
            unvisited_children = [
                child for child in current_v._prev if child not in visited
            ]
            if not unvisited_children:  # If all children have been visited
                visited.add(current_v)  # Mark node as visited
                stack.pop()  # Remove node from stack
                topo.append(current_v)  # Add node to topo list for backpropagation
            else:
                stack.extend(unvisited_children)  # Add unvisited children to stack
        self.grad = 1.0
        # Apply _backward in correct order
        for node in reversed(topo):
            node._backward()

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        self.grad = 1.0
        build_topo(self)
        for node in reversed(topo):
            node._backward()
