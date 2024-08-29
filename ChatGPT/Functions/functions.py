import torch as t
import requests
import numpy as np
import torch.nn.functional as F
import os

from graphviz import Digraph

from datasets import *

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()


def encode(s):
    """
    takes a string as input, outputs a list of integers
    :param s: string
    :return: list of integers
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return [stoi[c] for c in s]


def decode(l):
    """
    takes a list as input, outputs a string
    :param l: list of integers
    :return: string
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return ''.join([itos[i] for i in l])


def get_batch(split,
              train_data,
              val_data,
              block_size,
              batch_size):
    """
    generates a small batch of data of input x and targets y
    :param split: train or validation dataset
    :param train_data: training dataset
    :param val_data: validation dataset
    :param block_size: int
    :param batch_size: int
    :return: tuple (input data, target data)
    """
    data = train_data if split == 'train' else val_data
    ix = t.randint(len(data) - block_size, (batch_size,))
    x = t.stack([data[i:i + block_size] for i in ix])
    y = t.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y


def trace(root):
    """
    builds a set of all nodes and edges in a graph
    :param root:
    :return:
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """
    :param root:
    :return:
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR: Left --> Right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{%s | data %.4f }" % (n.label, n.data), shape='record')
        if n.op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n.op, label=n.op)
            # and connect this node to it
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot


def build_dataset(data,
                  stoi: dict,
                  itos: dict,
                  block_size: int):
    """
    creates X tensor which contains input tensors, and Y tensor, which contains the associated output
    :param block_size: number of necessary characters to predict the next tone
    :param itos: lookup table from string to int
    :param stoi: lookup table from int to string
    :param data: list of names, nouns, etc.
    :return: tuple of X and Y
    """
    X, Y = [], []
    for w in data:

        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]    # crop and append

    X = t.tensor(X)
    Y = t.tensor(Y)
    # print(X.shape, Y.shape)

    return X, Y


@t.no_grad()    # decorator that disables gradient tracking
def split_loss(split,
               Xtr, Ytr,
               Xdev, Ydev,
               Xte, Yte,
               C, layers):

    x,y = {"train": (Xtr, Ytr),
           "val": (Xdev, Ydev),
           "test": (Xte, Yte)}[split]
    emb = C[x]
    x = emb.view(emb.shape[0], -1)

    for layer in layers:
        x = layer(x)

    logits = x
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
