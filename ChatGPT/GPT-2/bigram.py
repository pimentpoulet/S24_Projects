import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}


def encode(s):
    """
    takes a string, outputs a list of integers
    :param s: string to encode
    :return: encoded string
    """
    return [stoi[c] for c in s]


def decode(l):
    """
    takes a list of integers, outputs a string
    :param l: list of integers
    :return: decoded string
    """
    return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

# training and validation datasets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


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


# super simple Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, C)
        x = tok_emb + pos_emb                                                      # (B, T, C)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        # logits = self.token_embedding_table(idx)    # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):  # idx is (B, T)  (4x8)
            logits, loss = self(idx)  # logits is (B, T, C)  (4x8x65)
            logits = logits[:, -1, :]  # logits is (B, C)  (4x65)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
