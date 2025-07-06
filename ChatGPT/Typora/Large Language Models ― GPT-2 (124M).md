## Large Language Models ― GPT-2 (124M)

#### Clément P. ― 15/08/24

##### Jupyter Notebook : GPT-2 (124M)

##### PyCharm Script : 

##### Video link : https://www.youtube.com/watch?v=l8pRSuU81PU&t=5s

##### *Attention is all you need* : https://arxiv.org/pdf/1706.03762

##### Typora reference : Large Language Models - Tokenizer



#### Hugging Face's GPT2 :

Hugging Face is a French-American startup that creates AI tools since 2015. They also help democratizing AI models' structure with a library called Transformers.

```python
from transformers import GPT2LMHeadModel
```

```python
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()
```

Something to mention is that this code imports the $10$M parameters model of GPT-2; to import the biggest and actual GPT-2 model, we need to use ``"gpt2-xl"`` instead of ``"gpt2"``. Also, ``sd_hf`` stands for ``state_dict_hugging_face``.

```python
for k, v in sd_hf.items():
    print(k, v.shape)
out : transformer.wte.weight torch.Size([50257, 768])
      transformer.wpe.weight torch.Size([1024, 768])
      transformer.h.0.ln_1.weight torch.Size([768])
      transformer.h.0.ln_1.bias torch.Size([768])
      transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
      transformer.h.0.attn.c_attn.bias torch.Size([2304])
      transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
      transformer.h.0.attn.c_proj.bias torch.Size([768])
      transformer.h.0.ln_2.weight torch.Size([768])
      transformer.h.0.ln_2.bias torch.Size([768])
      transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
      transformer.h.0.mlp.c_fc.bias torch.Size([3072])
      transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
      transformer.h.0.mlp.c_proj.bias torch.Size([768])
      transformer.h.1.ln_1.weight torch.Size([768])
      transformer.h.1.ln_1.bias torch.Size([768])
      transformer.h.1.attn.c_attn.weight torch.Size([768, 2304])
      transformer.h.1.attn.c_attn.bias torch.Size([2304])
      transformer.h.1.attn.c_proj.weight torch.Size([768, 768])
      transformer.h.1.attn.c_proj.bias torch.Size([768])
      transformer.h.1.ln_2.weight torch.Size([768])
      transformer.h.1.ln_2.bias torch.Size([768])
      transformer.h.1.mlp.c_fc.weight torch.Size([768, 3072])
      transformer.h.1.mlp.c_fc.bias torch.Size([3072])
      transformer.h.1.mlp.c_proj.weight torch.Size([3072, 768])
      ...
      transformer.h.11.mlp.c_proj.bias torch.Size([768])
      transformer.ln_f.weight torch.Size([768])
      transformer.ln_f.bias torch.Size([768])
      lm_head.weight torch.Size([50257, 768])
```

The first output, ``transformer.wte.weight torch.Size([50527, 758])``, is the length of the GPT-2 vocabulary. The GPT-2 tokenizer uses a $768$ dimensional embedding for each one of those tokens, i.e. each token is represented by a $768$ digits-long vector in the "model's point of view". The second output, ``transformer.wpe.weight torch.Size([1024, 768])``, is the positional embedding of those tokens. GPT-2 has a maximum sequence length of $1024$, which is why that LUT is $1024$ x $768$.

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
out :
```

![image-20240816125019548](C:\Users\cleme\AppData\Roaming\Typora\typora-user-images\image-20240816125019548.png)

In this image, each row is the visual representation of a position's embedding. The image has a structure since every position has a unique representation that has been learned by the transformer. Each positional embedding also includes the relative positions of the token with others and its residual connections, may I say. All the relations between tokens are shown in this image.

```python
plt.plot(sd_hf["transformer.wpe.weight"][:, 150])
plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
plt.plot(sd_hf["transformer.wpe.weight"][:, 250])
out :
```

![image-20240816131543649](C:\Users\cleme\AppData\Roaming\Typora\typora-user-images\image-20240816131543649.png)

This graph shows $3$ random columns of the above image, i.e. it is what a channel does as a function of position from $0$ to $1023$. It is useful to see which parts of the spectrum channels respond the most to. For example, the green channel responds more to positions between $200$ and $800$ than it does to positions near the $0^{th}$ position. It's also possible to see that the model is a bit undertrained since the curves are jaggy and not super clean. The reason for that is that the more random a model is, the more random this graph will be. With randomness comes great fluctuations in the interactions of channels with positions, which explains the jaggyness. At initialization, the curve are completely random and tend to get smoother and smoother as the training goes.

##### Note 1 : The curve end up looking like sines and cosines, but why ? This is because of the sinusoidal nature of the positional encoding. These patterns align with the expected sine and cosine functions used in positional embeddings. The differences in the amplitude and phase of the sinusoidal pattern suggest that each curve captures different aspects of the positional information.

The loaded weights can be used to do inference with the model.

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model="gpt2")
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
out : [{'generated_text': "Hello, I'm a language model, but what I'm really doing is making a human-readable document. There are other languages, but those are"},
       {'generated_text': "Hello, I'm a language model, not a syntax model. That's why I like it. I've done a lot of programming projects.\n"},
       {'generated_text': "Hello, I'm a language model, and I'll do it in no time!\n\nOne of the things we learned from talking to my friend"},
       {'generated_text': "Hello, I'm a language model, not a command line tool.\n\nIf my code is simple enough:\n\nif (use (string"},
       {'generated_text': "Hello, I'm a language model, I've been using Language in all my work. Just a small example, let's see a simplified example."}]
```



#### Starting from scratch :

The goal with the base skeleton is to match, in some sort, the scheme of the Hugging Face transformers so that the weights and parameters can be loaded easily. Ideally, the skeleton should reflect the output below.

```python
out : transformer.wte.weight torch.Size([50257, 768])
      transformer.wpe.weight torch.Size([1024, 768])
      transformer.h.0.ln_1.weight torch.Size([768])
      transformer.h.0.ln_1.bias torch.Size([768])
      transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
      transformer.h.0.attn.c_attn.bias torch.Size([2304])
      transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
      transformer.h.0.attn.c_proj.bias torch.Size([768])
      transformer.h.0.ln_2.weight torch.Size([768])
      transformer.h.0.ln_2.bias torch.Size([768])
      transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
      transformer.h.0.mlp.c_fc.bias torch.Size([3072])
      transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
      transformer.h.0.mlp.c_proj.bias torch.Size([768])
      transformer.h.1.ln_1.weight torch.Size([768])
      transformer.h.1.ln_1.bias torch.Size([768])
      transformer.h.1.attn.c_attn.weight torch.Size([768, 2304])
      transformer.h.1.attn.c_attn.bias torch.Size([2304])
      transformer.h.1.attn.c_proj.weight torch.Size([768, 768])
      transformer.h.1.attn.c_proj.bias torch.Size([768])
      transformer.h.1.ln_2.weight torch.Size([768])
      transformer.h.1.ln_2.bias torch.Size([768])
      transformer.h.1.mlp.c_fc.weight torch.Size([768, 3072])
      transformer.h.1.mlp.c_fc.bias torch.Size([3072])
      transformer.h.1.mlp.c_proj.weight torch.Size([3072, 768])
      ...
      transformer.h.11.mlp.c_proj.bias torch.Size([768])
      transformer.ln_f.weight torch.Size([768])
      transformer.ln_f.bias torch.Size([768])
      lm_head.weight torch.Size([50257, 768])
```

The first skeleton ``nn.Modules`` that the model will be built upon are

```python
@dataclass
class GPTConfig():
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
     	self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            ))
    
     	self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

```

with

```python
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
```

In the first place. this creates a ``@dataclass`` class that will be used later on and that mainly stores values for different parameters. A way to use this class would be

```python
config = GPTConfig()
print(config.block_size)  # Output: 256
print(config.n_layer)     # Output: 6
out : 256
	  6
```

In other words, this decorator automatically generates special methods for the class, such as ``__init__``, ``__repr__``, ``__eq__`` and others, based on the class attributes.

The ``GPT`` class is initialized based on ``nn.Module`` and defines crucial attributes, such as ``self.config``, ``self.transformer`` and ``self.lm_head``. The first one is, I expect, an instance of the dataclass above and will contain all the parameters' values of the transformer. The second attribute, ``self.transformer``, is the actual body of the transformer. It's easy to recognize the shape of the GPT-2 transformer, with the embedding tables for the tokens, their position, the blocks of self-attention heads and feedforward layers and the final normalization layer.

```python
h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
```

I suspect this line, which yields an error now because ``Block`` is undefined, is equivalent to

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

from the previous GPT-2 from scratch code. The ``h`` attribute is actually a ``nn.ModuleList`` which contains all the hidden parameters of the model. When looking at the output above, there are lines called ``transformers.h.0-11.x``. This assumes that ``h`` is a submodule that contains $12$ layers that are each identical of structure and that can be indexed to with numbers, as it is the case in a list.

```python
h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
```

In this case, ``config.n_layer`` would be $12$, instead of $6$ as defined above.

``nn.transformers`` is created using ``nn.ModuleDict``. This dictionary will resemble something like

```python
{"wte": value1, "wpe": value2, "h": value3, "ln_f": value4}
```

where each key is a submodule and the values are their associated tensors, or data-like structures.



#### Block class :

As suspected above, the ``Block`` class is similar to the previous one, but not identical.

```python
class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

The inputs are much simpler as everything depends only of ``config`` and the structure was changed a bit. The ``MultiHeadAttention`` class was replaced by a ``CausalSelfAttention`` class and the ``feedforward`` class was replaced by an ``MLP`` class. Those are undefined for now.

##### Note 2 : Notice that the output of each layer, ``attn(ln_1)`` and ``mlp(ln_2)``, is added to the current ``x`` value. This is a technique called *residual connection* that kind of solves the problem of exploding or vanishing gradients in deep neural networks. In fact, it doesn't solve it, it avoids it by having multiple shallow networks in the ensemble, all working as one.

##### More details on residual connections : https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55



#### MLP class :

As used in the ``Block`` class, the ``MLP`` class is a classic multi-layer perceptron.

```python
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
```

``c_fc`` and ``c_proj`` are linear layers and they are based on the desired output, as in ``transformer.h.1.mlp.c_fc.weight`` and ``transformer.h.1.mlp.c_proj.weight``. It is worth noting that the hidden state of these layers is $4$ times bigger than the inputs are outputs. This is because in the *Self Attention Is All You Need* paper, is is mentioned that

« The dimensionality of input and output is $d_{\text{model}} = 512$, and the inner layer has dimensionality $d_{ff} = 2048$. »

while talking about the *Position-wise Feed-Forward Networks*.

 These layers are wrapped around a ``nn.GELU`` activation layer with its ``approximate`` parameter set to "tanh". The ``GELU`` function has $2$ versions, the standard GELU that looks like
$$
\text{GELU}(x) = x\:*\:\Phi(x)
$$
and the approximate version, that looks like
$$
\text{GELU}(x) = 0.5 \: * \: x \: \Big(1 + \text{Tanh}\big(\sqrt{2/\pi} \: * \: (x + 0.044715 \: x \: x^3)\big)\Big)
$$
which is used when ``approximate`` is called as "tanh".

##### Note 3 : GELU stands for *Gaussian Error Linear Unit* and $\Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution. The approximate version was used for GPT-2 as it was faster than the typical GELU activation function. In fact, GPT-2 was made using TensorFlow and the error function (erf), which is used in the exact GELU, was very slow at the time with TensorFlow. To fix this, they came up with the approximate version and sticked with it for GPT-2. A reason to use GELU instead of RELU is that RELU is completely flat up to $0$ where GELU has a slight curve. This slight curve makes it so that all the activations that fall short of $0$ are not left to die there and that their gradient gets optimized at least a little. In more modern models like Llama3, this GELU function is replaced with SwiGLU, an even more curvy version of RELU.



#### Attention class :

The ``CausalSelfAttention`` class has the same shape as before, but instead of computing self-attention on multiple heads at the same time and concatenating their results, the heads are seen as batches in the data batches.

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)     # if the input is (2, 4, 12), the output will be (2, 4, 36)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buff("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B, T, C = x.size()    # batch size, sequence length, embedding dimensionality (n_embd)
        					  # let's say that B = 2, T = 4, C = 12, n_head = 3

        # calculate the query, key and values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size" and C (number of channels) = nh * hs
        # i.e. in GPT-2 124M, n_heads=12, hs=64, so nh*hs=C=768 channels in the transformer

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)    # if the input is (2, 4, 36), the output will be 3 x (2, 4, 12)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)	   # (2, 4, 3, 4) -transpose(1, 2)-> (2, 3, 4, 4)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs) --> (2, 3, 4, 4)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))    # (2, 3, 4, 4) @ (2, 3, 4, 4) --> (2, 3, 4, 4)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v   								   		# (2, 3, 4, 4)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # (2, 4, 3, 4) -view(B, T, C)-> (2, 4, 12)
        y = self.c_proj(y)                                  # (2, 4, 12)
        
        return y
```

Understood, and almost self-explanatory with the help of ChatGPT.

The rest of the code will probably not be explained here. I feel like I'm getting accustomed to it and everything needs less and less explanation. This document will still be useful if a chunk of code is very hard to understand and needs deeper analyzing. 

Further notes : the rest of the code is mostly about new ways to optimize the training of the model. For example, the use of ``torch.autocast(..., dtype=torch.bloat16)`` or the use of the PyTorch compiler, i.e. ``model = torch.compile(model)``. To use this compiler, I needed to use the windows ``wsl`` module in order to run a Linux Conda environment with which ``triton`` is compatible. This package is used by the compiler and isn't supported on windows. The self-attention block was replaced by a flash-attention line, with ``F.scaled_dot_product_attention(q, k, v, is_causal=True)``.

I'll need to come back to this to fully understand the rest of the code. The last 2 hours of the video are somewhat understandable, but everything related to the shards and the FinewebEDU dataset is still blurry. ``DDP`` is another topic that I haven't truly understood and that I just agreed to, in some way, without truly acknowledging it. Just as when I read terms and conditions from big techs' softwares ;).

I understand the training process in its globality, but since I didn't do it on my own, I can't describe it in details. The reason why I didn't do it on my own is for the lack of the required technology and processing power. I happen to have an RTX 3080 ti, which is a good GPU, but not good enough for this kind of model and training.













































