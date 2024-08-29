import os
import requests
import tiktoken
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import torch.nn.functional as F


input_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}

# create the training set of bigrams (x,y)
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = t.tensor(xs)
ys = t.tensor(ys)
num = xs.nelement()
# print(f"number of examples: {num}")

# initialize the network
g = t.Generator().manual_seed(1789)
W = t.randn((27,27), generator=g, requires_grad=True)


# gradient descent
for k in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[t.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    # print(loss.item())

    # backward pass
    W.grad = None
    loss.backward()

    # update
    W.data += -50 * W.grad

# sample from the model
g = t.Generator().manual_seed(1789)
for i in range(5):

    out = []
    ix = 0
    while True:

        xenc = F.one_hot(t.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        ix = t.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out))
