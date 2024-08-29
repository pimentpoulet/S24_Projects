import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
import sys
import time
import inspect
import numpy as np
import os
import code
import torch.distributed as dist

from dataclasses import dataclass
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

"""
To run this code in terminal:
/home/clement/anaconda3/envs/Pytorch_env_2/bin/python "/mnt/c/Clément PC_t/Code/Code/Python/PyTorch/ChatGPT/GPT-2/train_gpt2.py"

To run this code using wsl terminal:
goto /mnt/c/Clément PC_t/Code/Code/Python/PyTorch/ChatGPT/GPT-2
python train_gpt2.py

To run this code using ddp in wsl terinal
torchrun --standalone --nproc_per_node=1 train_gpt2.py
"""


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):

        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)    # Flash-Attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


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


@dataclass
class GPTConfig():
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):

        B, T = idx.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)    # (T)
        pos_emb = self.transformer.wpe(pos)                              # (T, n_embd)
        tok_emb = self.transformer.wte(idx)                              # (B, T, n_embd)
        
        x = tok_emb + pos_emb
        for block in self.transformer.h:    # forward the blocks of the transformer
            x = block(x)
        
        x = self.transformer.ln_f(x)        # forward the final layernorm and classifier
        logits = self.lm_head(x)            # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        loads pretrained GPT-2 model weights from HuggingFace
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),     # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),    # 350M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),    # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),    # 1558M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]     # >= 2D parameters => weight decay
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]    # biases and layernorms => no decay

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters    # True is "fused" is a parameter
        use_fused = fused_available and "cuda" in device                                # True is fused and cuda.is_available()
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer


def load_tokens(filename):

    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)

    return ptt


class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):

        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]    # +1 to account for the targets' tensor
        x = (buf[:-1]).view(B, T)    # inputs
        y = (buf[1:]).view(B, T)     # targets

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y


ddp = int(os.environ.get("RANK", -1)) != -1    # is it a ddp run ?
if ddp:
    assert torch.cuda.is_available()                  # set up DDP (Distributed Data Parallel)
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])    # the RANK, LOCAL_RANK and WORLD_RANK are variables set by the torchrun command
    device = f"cuda:{ddp_local_rank}"                 # WORLD_SIZE is the number of running processes, each one has its own ddp_rank
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0                    # this process will do logging, checkpointing, etc.
else:
    ddp_rank = 0                                      # vanilla, non-DDP run
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}\n")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288    # 2^19 --> ~0.5M in number of tokens
B = 8                                                     # micro batch size
T = 1024                                                  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0                    # 524288 % 8192 = 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)            # 524288 // 8192 = 64 steps to complete 1 full batch
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision("high")

# initialize the model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model             # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715                                     # in the GPT-3 paper says that they warmup the learning_rate over 375M tokens, which is 375M / total_batch_size (524288) = 715.26
max_steps = 19073                                      # 19073 comes from 10B / total_batch_size = 19073.49
def get_lr(it):
    if it < warmup_steps:                                             # linear learning rate for warmup
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:                                                # final learning rate
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)    # cosine decay of the learning rate
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):

    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0    # time in ms
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} |  lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

sys.exit(0)


num_return_sequences = 5
max_length = 30

# generate
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)                                           # (B, T, vocab_size)
        logits = logits[:, -1, :]                                   # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # (B, 50) & (B, 50)
        ix = torch.multinomial(topk_probs, 1)                       # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix)                   # (B, 1)
        x = torch.cat((x, xcol), dim=1)                             # (B, T+1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)                    # print generated text
