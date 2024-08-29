# S24_Projects

2024 summer projects of a Physics Engineering Student at Laval University, Quebec

~ Learn the basics of PyTorch and Deep Learning

~ CNNs, WaveNets (briefly) and MLPs

~ Transformers i.e. GPT-2 -> 124M parameters
 - Flash-Attention
 - Distributed Data Parallel (DDP)
 - ``torch.compile()`` with ``torch.triton`` in windows wsl
 - FinewebEDU & HellaSwag Datasets from Huggingface

Note: the ``train_gpt2_ddp.py`` and ``train_gpt2_no_ddp.py`` contain the same code, but the latter runs faster on my end for reasons I haven't investigated yet. All in all, the script with *ddp* enabled can also run on a single gpu machine
