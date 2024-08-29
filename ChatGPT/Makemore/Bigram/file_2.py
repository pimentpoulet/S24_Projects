import os
import requests
import tiktoken
import numpy as np
import torch as t
import matplotlib.pyplot as plt


input_file_path = os.path.join(os.path.dirname(__file__), 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}

N = t.zeros((27,27), dtype=t.int32)

# matrix with all chars in order, column x layer, and the xy occuring count
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1

P = N.float()
P = P/P.sum(1, keepdim=True)

log_likelihood = 0.0
n = 0
for w in ['clement']:
    chs = ['.'] + list(w) + ['.']
    # print(chs)
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = t.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

print(f"{log_likelihood=}")
nll = -log_likelihood          # negative log-likelihood
print(f"{nll=}")
print(f"{nll/n=}")
