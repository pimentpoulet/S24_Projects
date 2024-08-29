import torch
import math
import os

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    if it < warmup_steps:                                             # linear learning rate for warmup
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:                                                # final learning rate
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    print(f"decay_ratio: {decay_ratio:.4f}")
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    print(f"coeff: {coeff:.4f}")

    return min_lr + coeff * (max_lr - min_lr)

print(8 * 1024)
print(524288 // (8 * 1024))

print(os.cpu_count()//2)



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
        print(f"found {len(shards)} shards for split {split}")

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):

        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B*T + 1]    # +1 to account for the targets' tensor
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y


""""""""""""""""""


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
B = 8                                                                      # micro batch size
T = 1024                                                                   # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0                    # 524288 % 8192 = 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)            # 524288 // 8192 = 64 steps to complete 1 full batch
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
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
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):    # gradient accumulation loop --> 32 iterations to complete 1 batch

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps                                    # the loss needs to be normalized since F.cross_entropy
        loss_accum += loss.detach()                                       # uses the "mean" loss over every batch size
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
    dt = (t1 - t0) * 1000    # time in ms
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)

    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} |  lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

sys.exit(0)


