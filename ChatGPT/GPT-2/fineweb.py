import os
import multiprocessing as mp
import numpy as np
import tiktoken

from datasets import load_dataset
from tqdm import tqdm


# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)                                         # create the directory

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")    # download the dataset


enc = tiktoken.get_encoding("gpt2")           # initialize the tokenizer
eot = enc._special_tokens["<|endoftext|>"]    # end of text [eot] token

def tokenize(doc):
    """
    tokenizes a single document and returns a numpy array of uint16 tokens
    """
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)

    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """
    writes a numpy array of uint16 tokens to a binary file
    """
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:                                              # tokenize all documents and write output shards each of shard_size
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)               # preallocate buffer to hold the current shard
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < shard_size:                         # is there enough space in the current shard for the new tokens
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"                                      # write the current shard and start a new one
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

            remainder = shard_size - token_count                                                # split the document into whatever fits in this shard
            progress_bar.update(remainder)                                                      # the remainder goes into the next one

            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]    # populate the next shard with the leftovers of the current doc
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index ==0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
