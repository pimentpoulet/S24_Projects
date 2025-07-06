## Large Language Models ― GPT from scratch

#### Clément P. ― 11/07/24

##### Jupyter Notebook : GPT from scratch

##### PyCharm Script : bigram.py ― GPT_2.py

##### Video link : https://www.youtube.com/watch?v=kCc8FmEb1nY

##### *Attention is all you need* : https://arxiv.org/pdf/1706.03762



#### First steps :

Let's have a look, a brief visualization of the data we're working with.

```python
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

```python
print(f"length of dataset in characters: {len(text)}")
out : length of dataset in characters: 1115394
```

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
out : 
	   !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
	  65
```

The dataset is a file called ``input.txt`` which contains almost all of Shakespeare's works. It is $1115394$ characters long and contains the characters above. The first one is ``\n`` which accounts for the line break after the ``out`` statement and it is followed by a space. The ``vocab_size`` is 65, which means there are 65 unique characters in the dataset.

This data being raw text, it cannot be fed as it is into a learning model, it needs to be tokenized. To do so, we create a mapping from characters to integers and vice-versa. Those mappings are dictionaries, ``stoi`` (string to integers) and ``itos`` (integers to string). Using those mappings, the ``encode`` and ``decode`` functions are used to actively tokenize data.

```python
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))
out : [46, 47, 47, 1, 58, 46, 43, 56, 43]
	  hii there
```

```python
stoi
out : {'\n': 0,
       ' ': 1,
       '!': 2,
         ...
       'x': 62,
       'y': 63,
       'z': 64}
```

Since we'll use ``Pytorch`` to build the transformer, all the data is put into a ``torch.tensor`` object.

```python
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
out : torch.Size([1115394]) torch.int64
```

Using that tensor, the training and validating datasets are created so that $90$% of the data is used for training. It is worth mentioning that it would be computationally inefficient and way too demanding to simply pass in to the model all of the data at once. Hence, it is preferable to use training chunks of data that are randomly sampled from the training dataset. The size of those chunks depends on a variable called ``block_size`` (``context_length`` is equivalent). 

```python
block_size = 8
train_data[:block_size+1]
out : tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])
```

For example, when feeding this chunk to the network, the goal output would be a certain integer that logically follows the sequence. More in depth, if the context were simply ``[18]``, the expected output would be ``47``., if it were ``[18, 47]``, the expected output would be ``56``, etc.

```python
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, the target is {target}")
out : when input is tensor([18]), the target is 47
      when input is tensor([18, 47]), the target is 56
      when input is tensor([18, 47, 56]), the target is 57
      when input is tensor([18, 47, 56, 57]), the target is 58
      when input is tensor([18, 47, 56, 57, 58]), the target is 1
      when input is tensor([18, 47, 56, 57, 58,  1]), the target is 15
      when input is tensor([18, 47, 56, 57, 58,  1, 15]), the target is 47
      when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]), the target is 58
```

The ``for`` loop is interesting because it casts all the possible training routes from this specific chunk, from a context of $1$ to a context of $8$. Above being efficient, the use of chunks for training accustoms the model to different context lengths so that it becomes more capable for every case of given input.

GPUs, or *Graphics Processing Units*, are designed to accelerate computer graphics workloads and similar processes. In AI, GPUs are highly valuable for their parallel processing capabilities. With significantly more CUDA cores than CPUs ($10240$ in the case of the RTX $3080$ Ti), they are extremely efficient for AI computations. To maximize GPU utilization, researchers developed the concept of "batch processing." A batch is a subset of the training data that is processed through the model in a single pass. By using multiple batches and parallelizing the computations across many cores, the training process is greatly improved.

```python
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return ix, x, y

ix, xb, yb = get_batch("train")
print(f"ix: {ix.shape}\n{ix}\n")
print(f"inputs: {xb.shape}\n{xb}\n")
print(f"targets: {yb.shape}\n{yb}")
out : ix: torch.Size([4])
	  tensor([ 76049, 234249, 934904, 560986])

      inputs: torch.Size([4, 8])
      tensor([[24, 43, 58,  5, 57,  1, 46, 43],
              [44, 53, 56,  1, 58, 46, 39, 58],
              [52, 58,  1, 58, 46, 39, 58,  1],
              [25, 17, 27, 10,  0, 21,  1, 54]])

      targets: torch.Size([4, 8])
      tensor([[43, 58,  5, 57,  1, 46, 43, 39],
              [53, 56,  1, 58, 46, 39, 58,  1],
              [58,  1, 58, 46, 39, 58,  1, 46],
              [17, 27, 10,  0, 21,  1, 54, 39]])
```

The ``batch_size`` defines the number of batches to simultaneously train the model with and the ``get_batch`` function creates tensors of training inputs and target outputs, ``xb`` and ``yb``. The ``torch.randint`` method takes in $2$ arguments : the highest possible number to withdraw, in this case the length of the ``train_data`` tensor, and a tuple indicating the shape of the output tensor, in this case ``(batch_size,)``. Since ``batch_size`` is $4$, the output tensor, ``ix`` looks like

```python
tensor([ 76049, 234249, 934904, 560986])
```

and contains a random integer between $0$ and $1003854$, the length of ``train_data``.



#### Baseline model :

The simplest neural network in language modelling is the Bigram Language Model.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)    # 65x65

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)    # B, T, C
        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)
out : torch.Size([4, 8, 65])
```

The ``self.token_embedding_table`` class takes in $2$ arguments, ``num_embeddings`` and ``embedding_dim``.

- **num_embeddings** ([*int*](https://docs.python.org/3/library/functions.html#int)) – size of the dictionary of embeddings
- **embedding_dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – the size of each embedding vector

In this case, they are both ``vocab_size``, which is $65$. This creates a $65$ x $65$ tensor which contains the probabilities for every output for all single inputs. For example, say the input is the $7^{\text{th}}$ token. Looking at the embedding table, the $7^{\text{th}}$ row gets plucked out. This gives a row vector of length $65$ that contains the probabilities for the $65$ possible tokens in the vocabulary. In other words, for example, the third number in that row vector is the normalized probability of getting output ``{!}``, since the first number would be related to a linebreak character and the second, to a space character (see ``''.join(chars)`` above). Using that embedding table, the ``forward`` method is called when passing data through the model. It uses the embedding table and the indices contained in ``idx``. In this case, ``idx`` is the ``xb`` defined previously and that looks like

```python
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])  --> B, T
```

Passing this tensor as an argument to the ``Embedding`` object gives, as an output, the ``logits``, that look like

```python
tensor([[[-1.5101, -0.0948,  1.0927,  ..., -0.6126, -0.6597,  0.7624],
         [ 0.3323, -0.0872, -0.7470,  ..., -0.6716, -0.9572, -0.9594],
         [ 0.2475, -0.6349, -1.2909,  ...,  1.3064, -0.2256, -1.8305],
         ...,
       **[ 0.5978, -0.0514, -0.0646,  ..., -1.4649, -2.0555,  1.8275],** -> 6th element corresponds to second line of emb. table (see below)
         [ 1.0901,  0.2170, -2.9996,  ..., -0.5472, -0.8017,  0.7761],
         [ 0.3323, -0.0872, -0.7470,  ..., -0.6716, -0.9572, -0.9594]],

        [[ 1.0541,  1.5018, -0.5266,  ...,  1.8574,  1.5249,  1.3035],
         [-0.1324, -0.5489,  0.1024,  ..., -0.8599, -1.6050, -0.6985],
         [-0.6722,  0.2322, -0.1632,  ...,  0.1390,  0.7560,  0.4296],
         ...,
         [ 1.0901,  0.2170, -2.9996,  ..., -0.5472, -0.8017,  0.7761],
         [ 1.1513,  1.0539,  3.4105,  ..., -0.5686,  0.9079, -0.1701],
         [ 0.2475, -0.6349, -1.2909,  ...,  1.3064, -0.2256, -1.8305]],

        [[-0.2103,  0.4481,  1.2381,  ...,  1.3597, -0.0821,  0.3909],
         [ 0.2475, -0.6349, -1.2909,  ...,  1.3064, -0.2256, -1.8305],
         [ 0.5978, -0.0514, -0.0646,  ..., -1.4649, -2.0555,  1.8275],
         ...,
         [ 1.1513,  1.0539,  3.4105,  ..., -0.5686,  0.9079, -0.1701],
         [ 0.2475, -0.6349, -1.2909,  ...,  1.3064, -0.2256, -1.8305],
         [ 0.5978, -0.0514, -0.0646,  ..., -1.4649, -2.0555,  1.8275]],

        [[ 0.0691,  0.2990, -1.4717,  ...,  0.1517,  0.8528,  0.0604],
         [-0.4892, -2.5589,  1.4134,  ..., -1.4296,  0.2347, -1.2034],
         [-0.1600,  1.3981, -0.7047,  ..., -1.9908,  0.8574, -2.1603],
         ...,
         [-2.1910, -0.7574,  1.9656,  ..., -0.3580,  0.8585, -0.6161],
         [ 0.5978, -0.0514, -0.0646,  ..., -1.4649, -2.0555,  1.8275],
         [-0.6787,  0.8662, -1.6433,  ...,  2.3671, -0.7775, -0.2586]]],
       grad_fn=<EmbeddingBackward0>)
```

Looking at this for too long would likely provoke a headache, yet it makes a lot of sense. The first row of ``xb`` contains the index $1$ as the $6^{\text{th}}$ element. This index refers to the second row of the embedding table tensor, which looks like

```python
Parameter containing:
tensor([[ 0.1808, -0.0700, -0.3596,  ...,  1.6097, -0.4032, -0.8345],
      **[ 0.5978, -0.0514, -0.0646,  ..., -1.4649, -2.0555,  1.8275],**
        [ 1.3035, -0.4501,  1.3471,  ...,  0.1910, -0.3425,  1.7955],
        ...,
        [ 0.4222, -1.8111, -1.0118,  ...,  0.5462,  0.2788,  0.7280],
        [-0.8109,  0.2410, -0.1139,  ...,  1.4509,  0.1836,  0.3064],
        [-1.4322, -0.2810, -2.2789,  ..., -0.5551,  1.0666,  0.5364]],
       requires_grad=True)
```

The $2$ highlighted rows above match. Further investigation reveals that the shape of ``logits`` is

```python
torch.Size([4, 8, 65])
```

The $4$ is the number of rows in ``xb``, the $8$ is the number of indexes in a single batch, i.e. ``block_size``, and the $65$ is the length of every single row in ``logits`` and in the embedding table, accordingly.

Another way of saying it is to use the ``B, T, C`` formalism, where ``B`` stands for ``batch``, ``T`` for ``Time`` and ``C`` for ``Channels``. There are $4$ batches, $8$ time stamps and $65$ channels. Time stamps can be seen as time increments where, in ``xb``, $24$ comes before $43$, $43$ comes before $58$, etc. With this formalism, ``idx`` and ``targets`` are both of ``B, T`` shape. They contain the data for $4$ batches of $65$ numbers each.

It follows the previous statements that the ``forward`` method also yields the targets' logits, which are then to be used in the ``loss function``.

```python
def forward(self, idx, targets):
	logits = self.token_embedding_table(idx)    # B, T, C
    loss = F.cross_entropy(logits, targets)
    return logits
```

Running the code with this implementation of the ``forward`` method gives an error. The reason for this is that the ``F.cross_entropy`` function expects its arguments in a different formalism than ``B, T, C``. In fact, it requires that the ``logits`` and the ``targets`` be ``B, C, T``.

« The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to $1$, in general). input has to be a Tensor of size $(C)$ for unbatched input, $(minibatch, C)$ or $(minibatch, C)$ or $(minibatch, 𝐶, 𝑑1, 𝑑2, ..., 𝑑𝐾)$ with $K \geq 1$ for the K-dimensional case. The last being useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images. »

```python
def forward(self, idx, targets):
    
    logits = self.token_embedding_table(idx)
    
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    
    return logits
```

The ``forward`` method is again modified so that the ``logits`` and ``targets`` have the correct dimensions. In this case, the batches of ``logits`` have all been stacked together, so it is now a $32$ x $65$ tensor, and ``targets`` have also been stacked together, so that it is of shape $32$; a row tensor (vector).

```python
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
out : torch.Size([32, 65])
	  tensor(4.8786, grad_fn=<NllLossBackward0>)
```

The theoretical loss at initialization can be found using

```python
-ln(1/vocab_size) = -ln(1/65) = 4.17439
```



#### Generating text :

The ``generate`` method is

```python
def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):          # idx is (B, T)  (4x8)
        logits, loss = self(idx)             # logits is (B, T, C)  (4x8x65)
        logits = logits[:, -1, :]            # logits is (B, C)  (4x65)
        probs = F.softmax(logits, dim=-1)    
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

To implement it, the ``forward`` method had to be adjusted so that it wouldn't give an error when ``targets`` are missing.

```python
def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
```

The ``generate`` method gets the ``logits`` and the ``loss`` from a chunk of data, ``idx``. In this case, the ``loss`` is set to ``None``. Using only the ``logits``, all the batches and channels are conserved, but only the last row is kept. This transforms the ``logits`` from ``B, T, C`` to ``B, C`` and leaves us with $4$ row vectors. These $4$ vectors are the rows from the embedding table that correspond to the last index of every row in ``idx``. In the end, they are the raw probabilities for every output token associated to the last input token of every batch. Using ``F.softmax`` on the ``logits`` gives row vectors that are normalized, so that they sum up to $1$. The next integer for each batch is drawn from the distribution with ``torch.multinomial``. Its argument, ``num_samples``, is set to $1$ so that it only yields $1$ integer.

``torch.multinomial`` takes as an input a tensor, in this case a $4$ x $65$ tensor, that it considers being a probabilistic distribution of possibilities and draws ``num_samples`` indexes out of the distribution. For example, if the input tensor is

```python
torch.tensor([10, 1, 2, 10], dtype=torch.float32
```

``torch.multinomial`` returns

```python
print(torch.multinomial(torch.tensor([10, 1, 2, 10], dtype=torch.float32), num_samples=1))
out: tensor([0])
     tensor([3])
     tensor([0])
     tensor([3])
     tensor([1])
     tensor([0])
```

It is easy to see that the function mostly returns indexes $0$ and $3$. This is because those $2$ indexes are associated with the highest values in the input tensor, $10$. In the ``generate`` method, the input tensor is ``probs``, which has shape ``B, C``, so $4$ x $65$.

The ``idx_next`` is drawn probabilistically from the ``probs`` vector and is concatenated to the existing ``idx`` tensor along the first dimension. In that way, each batch get elongated of a ``max_new_tokens`` number of integers. Setting ``max_new_tokens`` as $50$, with ``idx`` as ``xb``, returns

```python
m.generate(xb, max_new_tokens=50)
out : tensor([[24, 43, 58,  5, 57,  1, 46, 43, 60, 60, 12, 55, 28,  7, 29, 35, 49, 58,
               36, 53, 24,  4, 48, 24, 16, 22, 45, 27, 24, 34, 64,  5, 30, 21, 53, 16,
               55, 20, 42, 46, 57, 34,  4, 60, 24, 24, 62, 39, 58, 48, 57, 41, 25, 54,
               61, 24, 17, 30],
              [44, 53, 56,  1, 58, 46, 39, 58, 31,  0, 60, 60, 47, 37, 16, 17, 57, 62,
               63, 44, 55, 53, 47, 53, 15, 54,  3, 26, 64, 40, 48, 59, 19,  4, 60,  4,
               24, 35, 31, 40, 22, 11, 46, 47, 12, 24, 40,  3, 29, 37,  4, 57, 57, 11,
               13,  5, 36,  9],
              [52, 58,  1, 58, 46, 39, 58,  1,  7, 23, 20, 29, 16, 15, 34,  4, 53, 48,
                3,  7, 33, 63, 40, 31,  0, 61, 44, 55,  7, 61, 39, 43, 27, 60, 17, 30,
               51, 57, 17, 28,  1, 44, 11, 20, 11,  7, 61, 60, 46, 57, 10, 15, 23, 31,
               48, 20, 42, 42],
              [25, 17, 27, 10,  0, 21,  1, 54,  4, 29, 10, 17, 22, 46,  3, 44, 30, 18,
               59, 36, 59, 17, 56, 21, 42, 40, 30, 51, 37, 46, 41, 51, 64, 58, 45, 32,
               17, 36, 35, 58, 50, 61, 39,  4, 38, 63,  5, 28, 12, 39,  2, 28, 19, 24,
               23, 21, 22, 53]])
```

For comparison, ``xb`` is

```python
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
```

The $8$ first integers of every batch in ``out`` above correspond to the batch in ``xb``. The rest has been generated using the ``generate`` method.

Using the ``decode`` function from the beginning of this document, the raw new text for the first batch is

```python
print(decode(m.generate(xb, max_new_tokens=50)[0].tolist()))
out : Let's heSPyao.qfzs$Ys$zF-w,;eEkzxjgCKFChs!iWW.ObzDnxA Ms$3
```

Using a linebreak character as a starting point, we get

```python
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
out : 
	  pYUNM$WGJRjxQm.IkRlNCh
	  x:CKD$HDq
	  YUNjX&3:z rVPZN3:nlxqzmSlnYeWTMgQufiPFbGSxI'OSn VSxLI CXIQE-Rw3d,NT
```

This is completely random because the baseline model hasn't been trained yet.



#### Training & Optimizing the baseline model :

« In PyTorch, an optimizer is a specific implementation of the optimization algorithm that is used to update the parameters of a neural network. The optimizer updates the parameters in such a way that the loss of the neural network is minimized. PyTorch provides various built-in optimizers such as SGD, Adam, Adagrad, etc. that can be used out of the box. »

The optimizer that is used to train the Bigram Language Model is ``torch.optim.AdamW``. This optimizer is one of the most advanced and has really good performances. For more info, https://arxiv.org/pdf/1711.05101.

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
```

The training loop is simple, i.e.

```python
batch_size = 32

for steps in range(10000):

    xb, yb = get_batch("train")    # sample random batch
    logits, loss = m(xb, yb)       # evaluate the loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss)
```

After training for $20100$ iterations, the loss comes down, from $4.8786$, to $2.4263$, approximately.

```python
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist()))
out : 
      Tonbe, ll, thouierrde ll, theeeng f rdisod,

      BUSThacat nd s!
      I:
      Manedoughis y? wste ssthasaleeano denongretinknto endsbel he al cat mareybrd o n the?
      KE: will An hile biththomindit y, d alerss, sech y ituthinone, thene hneerneyoy oredouisthavepan wnidscou't y'de; INCOKires         banteroffousein have jon fet ouar?

      Jus, ay ukisend l was ton y:
      Thesowor a ese buncore je yowis auson, tiokiodencrs bl t,
      BOK
```

The ``generate`` method is still bad, but it's a lot better than before.



#### Pycharm Script

Not much changes when the jupyter is transposed into a Pycharm script. A new function appears as well as agnostic code.

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

This function calculates the average loss over ``eval_iters`` training iterations. In this case, ``eval_iters`` is set to $200$. It returns a dictionary that contains $2$ keys, ``train`` and ``val``. The associated values are roughly the same for now because the model doesn't have a mode selection, like training or inference. A tensors of zeros is initialized then filled with the loss value of several training iterations. The tensor's values are averaged using ``losses.mean()`` and this value is associated to the ``train`` and ``val`` keys of ``out``. The decorator at the top is ``@torch.no_grad()``. This decorator tells Pytorch that the gradients don't have to be tracked within this function. It mostly helps for the memory usage of behind-the-scenes Pytorch.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The ``get_batch`` had to be modified to account for the agnostic code, i.e.

```python
def get_batch(split):
    """
    randomly selects block_size long tensors out of the specified dataset and their associated
    targets tensors
    :param split: "train" or None
    :return: training chunks (x) and targets (y)
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

so that the ``x`` and ``y`` tensors are on the same device as the model.

The training loop has also been modified.

```python
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

This way, every $300$ iterations, the average loss is computed for both training datasets. The results show a decrease in the loss value, which is good.

```python
out : step 0: train loss 4.7305, val loss 4.7241
      step 300: train loss 2.8110, val loss 2.8249
      step 600: train loss 2.5434, val loss 2.5682
      step 900: train loss 2.4932, val loss 2.5088
      step 1200: train loss 2.4863, val loss 2.5035
      step 1500: train loss 2.4665, val loss 2.4921
      step 1800: train loss 2.4683, val loss 2.4936
      step 2100: train loss 2.4696, val loss 2.4846
      step 2400: train loss 2.4638, val loss 2.4879
      step 2700: train loss 2.4738, val loss 2.4911
```

The small variations between ``train loss`` and ``val loss`` probably come from the fact that the training dataset is $9$ times as large as the validation dataset.



#### Aggregation :

« Aggregation in self-attention refers to the process of combining information from different parts of an input sequence to produce a contextualized representation for each element (e.g., word or token) in the sequence. This is achieved through the weighted sum of value vectors, where the weights are determined by the attention scores. »

The weakest form of aggregation is done by averaging past context with ``for`` loops, i.e.

```python
# version 1
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)
print(x[0])

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

print(xbow[0])
out : torch.Size([4, 8, 2])
      tensor([[ 0.1808, -0.0700],
              [-0.3596, -0.9152],
              [ 0.6258,  0.0255],
              [ 0.9545,  0.0643],
              [ 0.3612,  1.1679],
              [-1.3499, -0.5102],
              [ 0.2360, -0.2398],
              [-0.9211,  1.5433]])
	  tensor([[ 0.1808, -0.0700],
              [-0.0894, -0.4926],
              [ 0.1490, -0.3199],
              [ 0.3504, -0.2238],
              [ 0.3525,  0.0545],
              [ 0.0688, -0.0396],
              [ 0.0927, -0.0682],
              [-0.0341,  0.1332]])
```

In this example, ``x`` is a chunk of training data, similar to ``xb``, that is passed through the model. Its shape is the same as before excepted for the number of channels, $2$ instead of $65$. ``xbow`` is an acronym for ``x bag of words`` and it contains the average of the channels before and up to the current ones. Accordingly, the $2^{\text{th}}$ element of ``xbow[0]`` is ``[-0.0894, -0.4926]``. These $2$ numbers are the average of the first $2$ elements in both channels of ``x``. Indeed, for the first channel, the first $2$ elements are $0.1808$ and $-0.3596$. Their average, $-0.0894$, is the second element, for the first channel, of ``xbow[0]``.

Another way of doing this is by using matrices.

```python
# version 2
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
```

This outputs the same tensor as ``version 1``, but much more efficiently. The ``torch.tril()`` method returns the lower triangular matrix of an input matrix, in this case a ``torch.ones(T, T)`` matrix. By then dividing the ``wei`` matrix by itself with every row summed together, we get

```python
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```

Since the first row contained only a $1$ at the first index, its sum became $1$ and it didn't affect the ``wei`` matrix. The second row got divided by $2$ and each element became $0.5000$, etc.

A third way of doing it is by using a mask and the ``torch.functional.softmax()`` method.

```python
# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
```

``tril`` is the same as ``wei`` from ``version 2``, it is a lower triangular matrix of only ones. In ``version 3``, ``wei`` is a zeros matrix that looks like

```python
tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```

The ``torch.Tensor.masked_fill()`` is called upon it so that for every index that is $0$, in ``tril``, the corresponding index in ``wei`` becomes ``"-inf"``. ``wei`` now looks like

```python
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```

Using ``torch.functional.softmax()`` on this matrix returns the equivalent of the final ``wei`` matrix in ``version 2``. That is because when passing ``"-inf"`` to a softmax, the output is $0$. Also, when passing a number that is not ``"+/-inf"`` to a softmax, it outputs a number along

<img src="https://cdn.botpenguin.com/assets/website/Softmax_Function_07fe934386.png" alt="Softmax Function: Advantages and Applications | BotPenguin" style="zoom:25%;" />

and it normalizes the row of the input matrix. The final ``wei`` matrix in ``versioŉ 3`` is

```python
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```

The third version is of course the best one because it starts by setting all affinities between tokens to $0$, which the most less entropic method. Then, all affinities with future tokens are set to ``"-inf"``, so that they are aboslutely ignored. This way, depending on the values of every token, they will build stronger or weaker affinities with each other.

This is called weighted aggregation.



#### Position embedding :

It is common practice to also encode the tokens' position inside ``idx``, and not just their identity. To do so, a ``position_embedding_table`` of shape ``block_size`` x ``n_embd`` is used. ``n_embd`` is an hyperparameter that sets the number of dimensions that are used to encode the positions and the identities of the tokens inside ``idx``.

```python
def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

def forward(self, idx, targets=None):

    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embd)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, n_embd)
    x = tok_emb + pos_emb                                                      # (B, T, n_embd)
    logits = self.lm_head(x)    # (B, T, vocab_size)

    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss
```

 The ``forward`` method has been modified again so that both identities and positions are considered for the encoding, as in ``tok_emb`` and ``pos_emb``, respectively. The ``init`` method now has both embedding tables and a linear layer, ``self.lm_head``. This linear layer is necessary to go from token and position embeddings to logits.

The way this goes is the following. ``idx`` is a $32$ x $8$ tensor, with $32$ batches of data. This is given to the model to be embedded using the embedding tables. The tokens are embedded using the ``token_embedding_table``, and the positions using the ``position_embedding_table``. These operations output $2$ things, ``tok_emb`` and ``pos_emb``. The first one, ``tok_emb``, is a $32$ x $8$ x $32$ tensor. which is the same as before, but with each token embedded into a $32$ digits long row vector. A row vector which was plucked out of the correspond embedding table. The same goes for ``pos_emb``, which is now $8$ x $32$. Those embedding then get added together and passed to the ``lm_head`` to get the logits.

##### Important note : for now, ``pos_emb`` is always the same, since it is created using ``pos_emb = self.position_embedding_table(torch.arange(T, device=device))`` and ``T`` is always $8$. The ``torch.arange()`` function is the same as ``np.linspace()``.



#### Self-Attention :

```python
# version 4: self-attention
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)      # (B, T, 16)
q = query(x)    # (B, T, 16)
v = value(x)

wei = q @ k.transpose(-2, -1)    # (B, T, 16) @ (B, 16, T) --> (B, T, T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

out = wei @ v

out.shape
out : torch.Size([4, 8, 16])
```

This code is similar to the one above, version $3$ of the weighted aggregation. Self-attention is done here by adding the ``key``, ``query`` and ``value`` variables. The first one, ``key``, refers to the information describing a single token. It's hard to see how that is since it's not really explicit here, but ``key`` refers to a tensor of values that are related to the token itself. The second one, ``query``, refers to the tokens a single token looks for. For example, let's say the selected token is a vowel, its ``query`` would most likely be any kind of  values related to consonants. The third one, ``value``, is what the token actually shares when it is "matched" with another token.

All this is unprecise and weird to explain, so let's use an example. Say that $2$ tokens are represented by $2$ salesmen. Both sell NHL player cards and both have a ``key`` and ``query`` entities. The first one is a $56$ year old man and the other is a $34$ year old girl, these are their ``keys``, and let's say that they both look to buy one another's card. Once they match, since their ``key`` and ``query`` match, they aggregate, i.e. they trade cards and exchange. The cards exchanged are ``value``, or ``v``, the private cards vendors own. If the girl was selling soccer cards, which the man didn't want, they wouldn't match and wouldn't trade cards. Self-attention is the same but for tokens, since every token has more affinities with certain other tokens, as certain letters often come with certain other letters.

These $3$ objects are linear projections used to calculate ``out``. First, ``wei`` is created by computing the dot product between the ``query`` and the ``key`` vectors. The ``.transpose(-2,-1)`` is needed so that the matrix multiplication actually works. Then, the same as above is done to ignore all future tokens in the batch, with ``torhc.tril()``. In the end, ``out`` is calculated by multiplying the weighted aggregation vector with the actual values that need to be aggregated, which are ``values``, or ``v`` to be precise. This outputs weighted aggregated values for each time step in the batch.

Note $1$ : Attention is a communication mechanism. It can be seen as nodes looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights, in a directed graph. So node $1$ points to itself, node $2$ points to itself and to node $1$, node $3$ points to itself, node $2$ and node $1$, etc. node $8$ points to itself and all the nodes beforehand.

Note $2$ : There is no notion of space. Attention simply acts over a set of vectors. This is why the position embedding code was added.

Note $3$ : This mechanism is called Self-Attention since all objects are created using the same source, ``x``. Indeed, ``k``, ``q`` and ``v`` are all issued from ``x``. Cross-Attention is when those objects are not created using the same instance of indices, if ``k`` and ``v`` is created using an outsider instance, for example.



The code shown above is often called a ``decoder block``.

```python
# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)      # (B, T, 16)
q = query(x)    # (B, T, 16)
v = value(x)

wei = q @ k.transpose(-2, -1)    # (B, T, 16) @ (B, 16, T) --> (B, T, T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

The reason for that is better understood by looking at a different use of transformers, like sentiment detectors. For those models to function properly, they need to analyze the full length of a sentence are analyze every token and their interactions. Such a block of code is called an ``encoder block``. Since, in this case, the sentence are cut to have a single token at first, then $2$ tokens, then $3$, etc. it is called a ``decode block``, as it "decodes" the sentence step by step. For ``encoder block``, this line is removed

```python
wei = wei.masked_fill(tril == 0, float('-inf'))
```

so that all tokens talk to all tokens.



After checking on the *Attention is all you need* paper, it is mentioned that they use scaled attention. This means that they multiply ``wei`` with a $\sqrt{d_k}$ factor. This makes it so that when ``q`` and ``k`` are unit variance, ``wei`` is unit variance as well. This way, the SoftMax at initialization will be diffuse enough and not converge towards a saturated ``one_hot`` vector form.

```python
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1)
print(k.var())
print(q.var())
print(wei.var())
out : tensor(1.0449)
      tensor(1.0700)
      tensor(17.4690)
```

With the $\sqrt{d_k}$ factor :

```python
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
print(wei.var())
out : tensor(0.9957)
```

This way, I mean by allowing the SoftMax to not be too peaky, we allow tokens to effectively aggregate. If it were too peaky, every node would aggregate information with $1$ other single node and they wouldn't all talk to each other accordingly to the context.



#### PyCharm Script :

All the code above is transcribed in the PyCharm script as

```python
class Head(nn.Module):
    """
    one head of self-attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)      # (B, T, C)
        q = self.query(x)    # (B, T, C)

        # compute attention scores ("affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5                         # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                    # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)    # (B, T, C)
        out = wei @ v         # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out
```

which implements a single head of scaled self-attention.

The ``BigramLanguageModel`` class needs to be modified as well

```python
# super simple Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, n_embd)
        x = tok_emb + pos_emb                                                      # (B, T, n_embd)
        x = self.sa_head(x)                                                        # (B, T, n_embd)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

	def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):  # idx is (B, T)
            idx_cond = idx[:, -block_size:]                       # crop idx  to the last block_size tokens
            logits, loss = self(idx_cond)                         # logits is (B, T, C)
            logits = logits[:, -1, :]                             # logits become (B, C)
            probs = F.softmax(logits, dim=-1)                     # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)    # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)        # (B, T+1)

        return idx
```

so that a self-attention head is initialized. The information, once embedded, is passed to the ``sa_head`` which outputs aggregated information that can be decoded into ``logits``. The ``generate`` method was modified to only accept inputs of length ``block_size``. This way, self-attention always works even if ``idx`` has a length greater than ``block_size``.

Iterating $20000$ times, the training yields

```python
out : step 19800: train loss 2.3321, val loss 2.3698

      And thik bry cowinen O la, bth

      Hiset bobe doe.
      Sagr-ans mealilanss:
      Want he us hathe.
      War dilas ate awice my.

      OD:
      O om onou thowns, tof is heing mil; dill, bes iree sen cie lat Het drovets, and Wil ngan goerans!

      Anlind me lllliser onchiry:
      Supr aiss hew ye n's nes normopeengaves homy yuked mothakeeo Windo whre eiiby owouth dourive wen, ime st so mower; th
      To kad nterthinf so;
      Angis! mef thre male ont ffaf Prred my om.

      He-
      LIERLA,
      Sby ake adsad the Ere st hoin cour ay aney Iry ts chan yo vely
```

Better, but still bad.



#### Multi-headed self-attention :

Similar to multi-threading with a processor, multi-head self-attention is the parallelization of self-attention computing in LLMs.

```python
class MultiHeadAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
```

The results of this multi-threading are concatenated over the last dimension, which is the ``channels`` dimension.

The ``BigramLanguageModel`` class's ``__init__`` method needs to be modified accordingly

```python
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = MultiHeadAttention(4, n_embd//4)           # i.e. 4 heads of 8-dimensional self-attention
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

It is useful to compare the before and the after. Beforehand, there was a single head of self-attention which had, as its output, a $4$ x $8$ x $32$ tensor. Now, there are $4$ heads of self-attention which output $4$ $4$ x $8$ x $8$ tensors. After concatenation, these tensor form the same $4$ x $8$ x $32$ tensor as before. This is forced by the ``Linear`` layer (``self.lm_head``) which takes in ``n_embd`` inputs.



#### FeedForward Layers :

Following the *Attention is all you need paper*, it follows that the model is currently missing a FeedForward block. This is added with a simple ``nn.Sequential()`` block, as in

```python
class FeedForward(nn.Module):
    """
    a simple linear layer followed by a non-linearity
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
```

The FeedForward block is in fact a ``linear`` layer followed by a non-linear activation layer. More conceptually, this block lets the tokens learn from the aggregated information they gathered from the self-attention block.

```python
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4)           # i.e. 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, n_embd)
        x = tok_emb + pos_emb                                                      # (B, T, n_embd)
        x = self.sa_heads(x)        # (B, T, n_embd)
        x = self.ffwd(x)            # (B, T, n_embd)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

After going through the ``self.sa_head`` block, they are passed to ``self.ffwd``. In other word, they gather intel about other tokens, then get activated and trained over this new data.

Even with such a structure, the model isn't very capable. The thing that's missing is a loop, or a kind of loop for all those steps to repeat themselves. Indeed, by repeating this cycle over and over, the tokens learn about other tokens, train over the data, learn again, train again, etc.



#### sa and ffwd blocks :

In order to make this loop, the ``Block`` class is created. A ``Block`` object contains $2$ things : a ``MultiHeadAttention`` object and a ``FeedForward`` object. This is done so that the ``Block`` object can be used multiple times to loops those $2$ steps again and again like so

```python
class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
```

```python
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

This way, the initialization is the same, but there a multiple blocks or self-attention and feedforward operations.



#### Residual Connections :

To be continued ...



#### LayerNormalization Layers :

To be continued ...



#### Dropouts :

To be continued ...

















