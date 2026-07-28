"""
A minimal reverse-mode automatic differentiation engine, built on NumPy only.

Design: every Tensor remembers the operation that created it and a closure
(`_backward`) that knows how to push gradients to its parents (the standard
"tape"/graph approach used by micrograd, PyTorch, etc., just much smaller).

Why this approach instead of hand-deriving one big transformer gradient:
composing a transformer from a dozen primitive ops (add, mul, matmul, exp,
sum, pow, relu, slice, concat, embedding-lookup) means each op's gradient
rule is short and easy to check by hand, and the chain rule wiring is
automatic and uniform. That's a much smaller surface for bugs than one
monolithic hand-derived backward pass for a whole attention block.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting expanded it.

    NumPy broadcasting can (a) prepend extra leading dimensions and
    (b) stretch dimensions of size 1. Both need to be summed away when we
    send a gradient back to the smaller original tensor.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=True, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---------- bookkeeping ----------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def _accumulate(self, grad):
        self.grad = self.grad + _unbroadcast(grad, self.data.shape)

    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

    # ---------- elementwise ops ----------
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if other.requires_grad:
                other._accumulate(out.grad)
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (self._wrap(other) * -1.0)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * other.data)
            if other.requires_grad:
                other._accumulate(out.grad * self.data)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def pow(self, exponent):
        """Elementwise power by a constant python scalar (not a Tensor)."""
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad,
                      _children=(self,), _op=f"pow{exponent}")

        def _backward():
            if self.requires_grad:
                grad = exponent * (self.data ** (exponent - 1)) * out.grad
                self._accumulate(grad)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._wrap(other)
        return self * other.pow(-1)

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward():
            if self.requires_grad:
                self._accumulate(e * out.grad)
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(self.data, 0)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self._accumulate((self.data > 0).astype(self.data.dtype) * out.grad)
        out._backward = _backward
        return out

    # ---------- reductions ----------
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)
                grad = np.broadcast_to(grad, self.data.shape)
                self._accumulate(grad.copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            count = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # ---------- shape ops ----------
    def reshape(self, *shape):
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                      _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(orig_shape))
        out._backward = _backward
        return out

    def swapaxes(self, a, b):
        out = Tensor(np.swapaxes(self.data, a, b), requires_grad=self.requires_grad,
                      _children=(self,), _op="swapaxes")

        def _backward():
            if self.requires_grad:
                self._accumulate(np.swapaxes(out.grad, a, b))
        out._backward = _backward
        return out

    def slice_last(self, start, end):
        """Slice a contiguous chunk out of the last dimension (used to split heads)."""
        idx = tuple([slice(None)] * (self.data.ndim - 1) + [slice(start, end)])
        out = Tensor(self.data[idx], requires_grad=self.requires_grad,
                      _children=(self,), _op="slice_last")

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self._accumulate(grad)
        out._backward = _backward
        return out

    @staticmethod
    def concat(tensors, axis=-1):
        datas = [t.data for t in tensors]
        out = Tensor(np.concatenate(datas, axis=axis),
                      requires_grad=any(t.requires_grad for t in tensors),
                      _children=tuple(tensors), _op="concat")
        sizes = [t.data.shape[axis] for t in tensors]

        def _backward():
            idx = 0
            ndim = out.grad.ndim
            ax = axis if axis >= 0 else ndim + axis
            for t, sz in zip(tensors, sizes):
                sl = [slice(None)] * ndim
                sl[ax] = slice(idx, idx + sz)
                if t.requires_grad:
                    t._accumulate(out.grad[tuple(sl)])
                idx += sz
        out._backward = _backward
        return out

    # ---------- matrix multiply (batched, NumPy-broadcast semantics) ----------
    def __matmul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                dA = out.grad @ np.swapaxes(other.data, -1, -2)
                self._accumulate(dA)
            if other.requires_grad:
                dB = np.swapaxes(self.data, -1, -2) @ out.grad
                other._accumulate(dB)
        out._backward = _backward
        return out

    # ---------- backward driver ----------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)  # seed: d(loss)/d(loss) = 1
        for v in reversed(topo):
            v._backward()


def embedding_lookup(table: Tensor, idx: np.ndarray) -> Tensor:
    """table: Tensor of shape (vocab, dim). idx: int array of any shape.
    Returns Tensor of shape idx.shape + (dim,). Gradient scatters back into
    the rows of `table` that were looked up (rows used more than once
    correctly accumulate via np.add.at)."""
    out_data = table.data[idx]
    out = Tensor(out_data, requires_grad=table.requires_grad,
                 _children=(table,), _op="embed")

    def _backward():
        if table.requires_grad:
            grad = np.zeros_like(table.data)
            np.add.at(grad, idx, out.grad)
            table._accumulate(grad)
    out._backward = _backward
    return out


def softmax(x: Tensor, axis=-1) -> Tensor:
    """Numerically stable softmax, built from primitive ops so it gets
    correct gradients automatically via the chain rule. Subtracting the
    (constant, non-differentiated) row max doesn't change the true gradient
    of softmax, since softmax is shift-invariant."""
    m = np.max(x.data, axis=axis, keepdims=True)
    shifted = x - Tensor(m, requires_grad=False)
    e = shifted.exp()
    s = e.sum(axis=axis, keepdims=True)
    return e / s


def cross_entropy_loss(logits: Tensor, targets: np.ndarray):
    """logits: Tensor (B,T,V). targets: int array (B,T).
    Returns (loss_tensor, scalar_float_loss)."""
    B, T, V = logits.shape
    flat_logits = logits.data.reshape(B * T, V)
    flat_targets = targets.reshape(B * T)

    m = np.max(flat_logits, axis=1, keepdims=True)
    shifted = flat_logits - m
    exp_shifted = np.exp(shifted)
    logsumexp = np.log(np.sum(exp_shifted, axis=1, keepdims=True)) + m
    correct_logit = flat_logits[np.arange(B * T), flat_targets].reshape(-1, 1)
    loss_vec = (logsumexp - correct_logit).reshape(-1)
    loss_val = float(np.mean(loss_vec))

    out = Tensor(loss_val, requires_grad=True, _children=(logits,), _op="cross_entropy")
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def _backward():
        grad = probs.copy()
        grad[np.arange(B * T), flat_targets] -= 1.0
        grad /= (B * T)
        grad = grad.reshape(B, T, V) * out.grad
        logits._accumulate(grad)
    out._backward = _backward
    return out, loss_val
