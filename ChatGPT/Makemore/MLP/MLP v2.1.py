import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec

from functions import *
from datasets import *


# get data
input_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}
vocab_size = len(itos)

# build the dataset
block_size = 3

random.seed(1789)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1], stoi=stoi, itos=itos, block_size=block_size)      # 80%
Xdev, Ydev = build_dataset(words[n1:n2], stoi=stoi, itos=itos, block_size=block_size)    # 10%
Xte,  Yte  = build_dataset(words[n2:], stoi=stoi, itos=itos, block_size=block_size)      # 10%

# MLP revisited
n_emb = 10        # dimensionality of the character embedding vectors
n_hidden = 200    # number of neurons in the hidden layer of the MLP

g  = t.Generator().manual_seed(1789)
C  = t.randn((vocab_size, n_emb),            generator=g)
w1 = t.randn((n_emb * block_size, n_hidden), generator=g) * (5/3)/((n_emb * block_size)**0.5)
b1 = t.randn(n_hidden,                       generator=g) * 0.01
w2 = t.randn((n_hidden, vocab_size),         generator=g) * 0.01
b2 = t.randn(vocab_size,                     generator=g) * 0

bngain = t.ones((1, n_hidden))
bnbias = t.zeros((1, n_hidden))
bnmean_running = t.zeros((1, n_hidden))
bnstd_running = t.ones((1, n_hidden))

parameters = [C, w1, w2, b2, bngain, bnbias]
print(f"number of parameters in total: {sum(p.nelement() for p in parameters)}")

for p in parameters:
    p.requires_grad = True

# optimization
max_steps = 100000
batch_size = 32
lossi = []
for i in range(max_steps):

    # minibatch construct
    ix = t.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]                            # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1)    # concatenate the vectors
    hpreact = embcat @ w1  # + b1          # hidden layer pre-activation

    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)

    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    # hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias

    with t.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    h = t.tanh(hpreact)                    # hidden layer
    logits = h @ w2 + b2                   # output layer
    loss = F.cross_entropy(logits, Yb)     # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < max_steps/2 else 0.01    # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 20000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

    if i == 0:
        # plot first iteration data
        fig = plt.figure(figsize=(15,8))

        # create a gridspec
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

        # initial neurons' tanh activation
        ax1 = plt.subplot(gs[0, :])
        ax1.imshow(h.abs() > 0.99, cmap="gray", interpolation="nearest")
        ax1.set_title("initial neurons' tanh activation")

        # tanh layer's inputs
        ax2 = plt.subplot(gs[1, 0])
        ax2.hist(hpreact.view(-1).tolist(), 50)
        ax2.set_title("tanh layer's inputs")

        # tanh layer's activation
        ax3 = plt.subplot(gs[1, 1])
        ax3.hist(h.view(-1).tolist(), 50)
        ax3.set_title("tanh layer's activation")

ax4 = plt.subplot(gs[2, :])
ax4.plot(lossi)
ax4.set_title("loss.log10() evolution")

# tight layout
plt.tight_layout()

split_loss("train",
           Xtr=Xtr, Ytr=Ytr,
           Xdev=Xdev, Ydev=Ydev,
           Xte=Xte, Yte=Yte,
           C=C,
           w1=w1, b1=b1, w2=w2, b2=b2,
           bngain=bngain, bnbias=bnbias,
           bnmean=bnmean_running, bnstd=bnstd_running)
split_loss("val",
           Xtr=Xtr, Ytr=Ytr,
           Xdev=Xdev, Ydev=Ydev,
           Xte=Xte, Yte=Yte,
           C=C,
           w1=w1, b1=b1, w2=w2, b2=b2,
           bngain=bngain, bnbias=bnbias,
           bnmean=bnmean_running, bnstd=bnstd_running)

# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:

        # forward pass the neural net
        emb = C[t.tensor([context])]    # (1, block_size, n_embd)
        h = t.tanh(emb.view(1, -1) @ w1 + b1)
        logits = h @ w2 + b2
        probs = t.softmax(logits, dim=1)

        # sample from the distribution
        ix = t.multinomial(probs, num_samples=1, generator=g).item()

        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)

        # if we sample the special "." token, break
        if ix == 0:
            print(probs.shape)
            break

    # print(''.join(itos[i] for i in out))

# plt.show()
