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
        self.weight = t.randn((in_features, out_features), generator=g) / m.sqrt(in_features)
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
        self.training = False

        # self.training needs to be True when training and False when sampling from the model
        # ideally, this code would be in a jupyter notebook to be able to run the class defs
        # on their own and change the mode before and after running the training cell
        # I don't know how to do it in a classic PyCharm file

        # parameters (trained with backprop)
        self.gamma = t.ones(dim)
        self.beta = t.zeros(dim)

        # buffers (*trained with a running "momentum update")
        self.running_mean = t.zeros(dim)
        self.running_var = t.ones(dim)

    def __call__(self, x):

        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)                 # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True)    # batch variance
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
n_hidden = 100    # number of neurons in the hidden layer
g = t.Generator().manual_seed(1789)

C = t.randn((vocab_size, n_embd))

layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,            n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,            n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,            n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,            n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,          vocab_size, bias=False), BatchNorm1d(vocab_size),
]

with t.no_grad():
    # last layer : make less confident
    layers[-1].gamma *= 0.1
    # all other layers : apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]

# total number of parameters
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# optimization
max_steps = 100000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):

    # minibatch construct
    ix = t.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]    # batch X,Y

    # forward pass
    emb = C[Xb]                      # embeds the characters into vectors
    x = emb.view(emb.shape[0], -1)    # concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < max_steps/2 else 0.01    # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}\n")
    lossi.append(loss.log10().item())

    with t.no_grad():
        ud.append([(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters])


# set fonts
plt.rcParams.update({
    'font.size': 8,            # Set the font size
    'axes.titlesize': 8,       # Set the font size for axes titles
    'axes.labelsize': 8,       # Set the font size for x and y labels
    'xtick.labelsize': 8,      # Set the font size for x tick labels
    'ytick.labelsize': 8,      # Set the font size for y tick labels
    'legend.fontsize': 8,      # Set the foVVnt size for legend
})

# visualize stats
plt.subplot(4, 1, 1)

legends = []
for i, layer in enumerate(layers[:-1]):    # excludes the output layer
    if isinstance(layer, Tanh):
        tt = layer.out
        print(f"layer %d (%{len(layer.__class__.__name__)}s): mean %+.2f, std %.2f, saturated: %.2f%%" % (i, layer.__class__.__name__, tt.mean(), tt.std(), (tt.abs() > 0.97).float().mean()*100))
        hy, hx = t.histogram(tt, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer.__class__.__name__})")

plt.legend(legends)
plt.title("Activation distribution")  # , fontsize=title_font)

plt.subplot(4, 1, 2)
legends = []
for i, layer in enumerate(layers[:-1]):    # excludes the output layer
    if i == 0:
        print()
    if isinstance(layer, Tanh):
        tt = layer.out.grad
        print(f"layer %d (%{len(layer.__class__.__name__)}s): mean %+f, std %e" % (i, layer.__class__.__name__, tt.mean(), tt.std()))
        hy, hx = t.histogram(tt, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer.__class__.__name__})")

plt.legend(legends)
plt.title("Gradient distribution")  # , fontsize=title_font)

plt.subplot(4, 1, 3)
legends = []
for i, p in enumerate(parameters):
    if i == 0:
        print()
    tt = p.grad
    if p.ndim == 2:
        print(f"weight %10s | mean %+f | std %e | grad:data ratio %e" % (tuple(p.shape), tt.mean(), tt.std(), tt.std() / p.std()))
        hy, hx = t.histogram(tt, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"{i} {tuple(p.shape)}")

plt.legend(legends)
plt.title("Weights gradient distribution")  # , fontsize=title_font)

plt.subplot(4, 1, 4)
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append("param %d" % i)

plt.plot([0, len(ud)], [-3, -3], "k")
plt.legend(legends)
plt.title("Update to data ratio")  # , fontsize=title_font)

# adjust figure layout
plt.get_current_fig_manager().window.showMaximized()

# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:

        # forward pass the neural net
        emb = C[t.tensor([context])]    # (1, block_size, n_embd)
        x = emb.view(1, -1)    # concatenate the vectors
        for layer in layers:
            x = layer(x)
        probs = t.softmax(x, dim=1)

        # sample from the distribution
        ix = t.multinomial(probs, num_samples=1, generator=g).item()

        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)

        # if we sample the special "." token, break
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

# plt.show()
