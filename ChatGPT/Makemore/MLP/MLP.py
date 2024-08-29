import os
import requests
import tiktoken
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

from functions import *
from datasets import *

input_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}

random.seed(1789)
random.shuffle(words)

# create split delimitations
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
block_size = 3

# create splits
Xtr, Ytr = build_dataset(words[:n1], stoi=stoi, itos=itos, block_size=3)        # train
Xdev, Ydev = build_dataset(words[n1:n2], stoi=stoi, itos=itos, block_size=3)    # dev
Xte, Yte = build_dataset(words[n2:], stoi=stoi, itos=itos, block_size=3)        # test

# initialize neural net
g = t.Generator().manual_seed(1789)
C = t.randn((27, 10), generator=g)
w1 = t.randn((30, 200), generator=g)
b1 = t.randn(200, generator=g)
w2 = t.rand((200, 27), generator=g)
b2 = t.randn(27, generator=g)

parameters = [C, w1, b1, w2, b2]
num = sum(p.nelement() for p in parameters)

# set gradient requirements
for p in parameters:
    p.requires_grad = True

lre = t.linspace(-3, 0, 1000)
lrs = 10**lre

lri, lossi, stepi = [], [], []

for i in range(10000):

    # minibatch construct
    ix = t.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[ix]]    # (32, 3, 2)
    h = t.tanh(emb.view(-1, 30) @ w1 + b1)    # (32, 100)
    logits = h @ w2 + b2    # (32, 27)
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    # lr = lrs[i]
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    # lossi.append(loss.item())

# evaluate train split
emb = C[Xtr]                             # (32, 3, 2)
h = t.tanh(emb.view(-1, 30) @ w1 + b1)    # (32, 100)
logits = h @ w2 + b2                     # (32, 27)
loss = F.cross_entropy(logits, Ytr)
print(loss.item())

# evaluate dev split
emb = C[Xdev]                            # (32, 3, 2)
h = t.tanh(emb.view(-1, 30) @ w1 + b1)    # (32, 100)
logits = h @ w2 + b2                     # (32, 27)
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

# sample from model
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        emb = C[t.tensor([context])]
        h = t.tanh(emb.view(1, -1) @ w1 + b1)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        ix = t.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))
