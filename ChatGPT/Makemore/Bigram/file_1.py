import os
import requests
import tiktoken
import numpy as np
import torch as t
import matplotlib.pyplot as plt


# download the names dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'names.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

words = open(input_file_path, 'r').read().splitlines()
# print(words[:10])
# print(min(len(w) for w in words))
# print(max(len(w) for w in words))

"""
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    # print(chs)
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

most_occuring = sorted(b.items(), key=lambda kv: -kv[1])
# print(most_occuring)
"""

N = t.zeros((27,27), dtype=t.int32)

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}

# matrix with all chars in order, column x layer, and the xy occuring count
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1

"""
# visualize the probability map
plt.figure(figsize=(12,12))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray", fontsize=8)
        plt.text(j, i, N[i,j].item(), ha="center", va="top", color="gray", fontsize=8)
plt.axis("off")
plt.show()
"""

g = t.Generator().manual_seed(2147483647)

P = N.float()
P = P/P.sum(1, keepdim=True)

# iterate to sample multiple letters
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = t.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    # print(''.join(out))
