import os
import requests
import tiktoken
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import torch.nn.functional as F

from functions import *


input_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}

BLOCK_SIZE = 3
X, Y = [], []
for w in words:
    context = [0] * BLOCK_SIZE
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

# datasets
X = t.tensor(X)
Y = t.tensor(Y)

g = t.Generator().manual_seed(1789)
C = t.randn((27, 2))
w1 = t.randn((6, 100))
b1 = t.randn(100)
w2 = t.rand((100, 27))
b2 = t.randn(27)

parameters = [C, w1, b1, w2, b2]
num = sum(p.nelement() for p in parameters)

# set gradient requirements
for p in parameters:
    p.requires_grad = True

lre = t.linspace(-3, 0, 1000)
lrs = 10**lre

lri, lossi = [], []
for i in range(100):

    # minibatch construct
    ix = t.randint(0, X.shape[0], (32,))

    if i == 0:
        print(ix.shape)
        print(X.shape)
        print(C.shape)
        print(X[ix][0])

    # forward pass
    emb = C[X[ix]]    # (32, 3, 2)
    h = t.tanh(emb.view(-1, 6) @ w1 + b1)    # (32, 100)
    logits = h @ w2 + b2    # (32, 27)
    loss = F.cross_entropy(logits, Y[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    # lr = lrs[i]
    lr = 10**-0.6
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    # lossi.append(loss.item())

print(loss.item())
# plt.plot(lri, lossi)
# plt.show()
