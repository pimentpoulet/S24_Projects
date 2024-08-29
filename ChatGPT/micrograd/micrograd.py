from graphviz import Digraph

from functions import *


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

    @property
    def prev(self):
        return self._prev

    @property
    def op(self):
        return self._op


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e, e.label = a * b, 'e'
d, d.label = a * b + c, 'd'
f = Value(-2.0, label='f')
L, L.label = d * f, 'L'

draw_dot(d)
