import torch as t
import math as m
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec

from matplotlib.font_manager import FontProperties

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


class Linear:

    def __init__(self, in_features, out_features, bias=True):
        self.weight = t.randn((in_features, out_features)) / m.sqrt(in_features)
        self.bias = t.zeros(out_features) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):

        self.eps = eps
        self.momentum = momentum
        self.training = True

        # parameters (trained with backprop)
        self.gamma = t.ones(dim)
        self.beta = t.zeros(dim)

        # buffers (*trained with a running "momentum update")
        self.running_mean = t.zeros(dim)
        self.running_var = t.ones(dim)

    def __call__(self, x):

        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)    # batch mean
            xvar = x.var(0, keepdim=True)      # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / t.sqrt(xvar + self.eps)    # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        # update the buffers
        if self.training:
            with t.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:

    def __call__(self, x):
        self.out = t.tanh(x)
        return self.out

    @staticmethod
    def parameters():
        return []


random.seed(1789)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

# create splits
Xtr,  Ytr  = build_dataset(words[:n1], stoi=stoi, itos=itos, block_size=block_size)      # 80%
Xdev, Ydev = build_dataset(words[n1:n2], stoi=stoi, itos=itos, block_size=block_size)    # 10%
Xte,  Yte  = build_dataset(words[n2:], stoi=stoi, itos=itos, block_size=block_size)      # 10%

# model initialization
n_embd = 10       # dimensionality of the character embedding vectors
n_hidden = 200    # number of neurons in the hidden layer
g = t.Generator().manual_seed(1789)

C = t.randn((vocab_size, n_embd))

layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
]

with t.no_grad():
    # last layer : make less confident
    layers[-1].weight *= 0.1

parameters = [C] + [p for layer in layers for p in layer.parameters()]

# total number of parameters
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# optimization
max_steps = 1000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construct
    ix = t.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]    # batch X,Y

    # forward pass
    emb = C[Xb]                      # embeds the characters into vectors
    x = emb.view(emb.shape[0], -1)    # concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < (3*max_steps/4) else 0.01    # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

# reshape lossi into a tensor
lossi = t.tensor(lossi).view(-1, 1000).mean(1)
plt.plot(lossi)

# put all layers into eval mode
for layer in layers:
    layer.training = False

# evaluate the train loss
split_loss("train",
           Xtr=Xtr, Ytr=Ytr,
           Xdev=Xdev, Ydev=Ydev,
           Xte=Xte, Yte=Yte,
           C=C, layers=layers)

# evaluate the validation loss
split_loss("val",
           Xtr=Xtr, Ytr=Ytr,
           Xdev=Xdev, Ydev=Ydev,
           Xte=Xte, Yte=Yte,
           C=C, layers=layers)

# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size

    while True:

        # forward pass the neural net
        emb = C[t.tensor([context])]    # (1, block_size, n_embd)
        x = emb.view(emb.shape[0], -1)    # concatenate the vectors
        for layer in layers:
            x = layer(x)
        logits = x
        probs = F.softmax(logits, dim=1)

        # sample from the distribution
        ix = t.multinomial(probs, num_samples=1, generator=g).item()

        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)

        # if we sample the special "." token, break
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

plt.show()
