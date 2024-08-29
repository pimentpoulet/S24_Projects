import os
import requests
import tiktoken
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import torch.nn.functional as F


input_file_path = os.path.join(os.path.dirname(__file__), 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}

# create the training set of bigrams (x,y)
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = t.tensor(xs)
ys = t.tensor(ys)

# randomly initialize 27 neurons' weigths. each neuron receives 27 inputs
g = t.Generator().manual_seed(1789)
W = t.randn((27,27), generator=g, requires_grad=True)

"""
# forward pass
xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts/counts.sum(1, keepdim=True)

nlls = t.zeros(5)
for i in range(5):
    # i-th bigram
    x = xs[i].item()    # input character index
    y = ys[i].item()    # label character index
    print("\n----------")
    print(f"bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})")
    print(f"input to the neural net: {x}")
    print(f"output probabilities from the neural net: {probs[i]}")
    print(f"label (actual next character): {y}")
    p = probs[i,y]
    print(f"probability assigned by the net to the correct character: {p.item()}")    logp = t.log(p)
    print(f"log likelihood: {logp.item()}")
    nll = -logp
    print(f"negative log-likelihood: {nll.item()}")
    nlls[i] = nll

print("\n==========\n")
print(f"average negative log-likelihood, i.e. loss={nlls.mean().item()}")
"""

# forward pass
xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts/counts.sum(1, keepdim=True)
loss = -probs[t.arange(5), ys].log().mean()

# backward pass
W.grad = None
loss.backward()






