## Large Language Models ― Tokenizer

##### Clément P.  ―  02/07/24

##### Jupyter Notebook : Tokenizer Notes

##### Video link : https://www.youtube.com/watch?v=zduSFxRajkE&t=1915s



#### Quick breakthrough :

Tokenization, in the realm of Artificial Intelligence (AI), refers to the process of converting input text into smaller units or ‘tokens’ such as words or subwords.

OpenAI's large language models (sometimes referred to as GPT's) process text using **tokens**, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens.

```
Many words map to one token, but some don't: indivisible.

Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾

Sequences of characters commonly found next to each other may be grouped together: 1234567890
```

Tokens : 57  |  Characters : 252

```
[8607, 4339, 2472, 311, 832, 4037, 11, 719, 1063, 1541, 956, 25, 3687, 23936, 382, 35020, 5885, 1093, 100166, 1253, 387, 6859, 1139, 1690, 11460, 8649, 279, 16940, 5943, 25, 11410, 97, 248, 9468, 237, 122, 271, 1542, 45045, 315, 5885, 17037, 1766, 1828, 311, 1855, 1023, 1253, 387, 41141, 3871, 25, 220, 4513, 10961, 16474, 15]
```

Good example : https://tiktokenizer.vercel.app

The current Tokenizer for Chat GPT-4 is cl100k_base, which contains 100k tokens. The GPT-2 Tokenizer had about 50k tokens, as single spaces were a single token. This was a massive flaw of GPT-2 as Python needs a lot of indentation to work. The more tokens the better, but to a certain extent. With more tokens, the context's length gets longer, therefore better, but the Embedding Table and the Softmax layer become larger, too large to handle efficiently at some point.

One way of tokenizing strings would be to use the Unicode Standard, which is a text encoding standard that defines 149 813 characters, such as letters for every language and emojis. The disadvantage of Unicode is that it isn't stable in time and changes regularly. It's a bit too long as well.

```python
Built-in Python function to get a code point's Unicode :

ord("h")
out: 104
```

Another way of tokenizing strings would be to use Unicode Transformation Format encoding (UTF-8), which is part of the Standard Encoding System. UTF-8 encodes code points into byte streams, from 1 to 4 bytes depending on the character to encode.

```python
list("€".encode("utf-8"))
out: [226, 130, 172]

"€" requires 3 bytes for it to be encoded, i.e. 11100010 10000010 10101100 or b'\xe2\x82\xac'
```

Since a byte contains 8 bits, the maximum vocabulary length would be 256, which is very small. Also, it's not very computationnally efficient since every character need between 1 and 4 bytes to be tokenized. Hence, the data needs to be compressed before it is fed to the neural network, making raw UTF-8 encoding impossible.

A solution to this is to use the Byte Pair Encoding Algorithm.

Starting with

```python
aaabdaaabac
```

which contains 11 tokens and a vocabulary length of 4, the most common pairs are grouped up such as

```python
ZYdZYac

Y=ab
Z=aa
```

The vocabulary length is now 6 with 7 tokens. Continuing, we get

```python
XdXac

X = ZY
```

for a total of 5 tokens and a vocabulary length of 7. The final sequence is a compressed form of the initial sequence.

The tokenizer is a completely separate, independant module from the LLM. It has its own training dataset of text (which could be different from that of the LLM), on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. It then translates back and forth between raw text and sequences of tokens. The LLM later only ever sees the tokens and never directly deals with any text.



#### Diving into the actual tokenizer :

```python
def get_stats(ids):
    """
    sorts the pairs of tokens from the most occuring one to the least
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

```python
def merge(ids, pair, idx):
    """
    merges the most occuring pair (top_pair) and replaces it/them with a new idx
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

The first thing that's needed is a string of sequence of characters to encode :

```
text = """A Programmer’s Introduction to Unicode March 3, 2017 · Coding · 22 Comments  Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺\u200c🇳\u200c🇮\u200c🇨\u200c🇴\u200c🇩\u200c🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.  A few months ago, I got interested in Unicode and decided to spend some time learning more about it in detail. In this article, I’ll give an introduction to it from a programmer’s point of view.  I’m going to focus on the character set and what’s involved in working with strings and files of Unicode text. However, in this article I’m not going to talk about fonts, text layout/shaping/rendering, or localization in detail—those are separate issues, beyond my scope (and knowledge) here.  Diversity and Inherent Complexity The Unicode Codespace Codespace Allocation Scripts Usage Frequency Encodings UTF-8 UTF-16 Combining Marks Canonical Equivalence Normalization Forms Grapheme Clusters And More… Diversity and Inherent Complexity As soon as you start to study Unicode, it becomes clear that it represents a large jump in complexity over character sets like ASCII that you may be more familiar with. It’s not just that Unicode contains a much larger number of characters, although that’s part of it. Unicode also has a great deal of internal structure, features, and special cases, making it much more than what one might expect a mere “character set” to be. We’ll see some of that later in this article.  When confronting all this complexity, especially as an engineer, it’s hard not to find oneself asking, “Why do we need all this? Is this really necessary? Couldn’t it be simplified?”  However, Unicode aims to faithfully represent the entire world’s writing systems. The Unicode Consortium’s stated goal is “enabling people around the world to use computers in any language”. And as you might imagine, the diversity of written languages is immense! To date, Unicode supports 135 different scripts, covering some 1100 languages, and there’s still a long tail of over 100 unsupported scripts, both modern and historical, which people are still working to add.  Given this enormous diversity, it’s inevitable that representing it is a complicated project. Unicode embraces that diversity, and accepts the complexity inherent in its mission to include all human writing systems. It doesn’t make a lot of trade-offs in the name of simplification, and it makes exceptions to its own rules where necessary to further its mission.  Moreover, Unicode is committed not just to supporting texts in any single language, but also to letting multiple languages coexist within one text—which introduces even more complexity.  Most programming languages have libraries available to handle the gory low-level details of text manipulation, but as a programmer, you’ll still need to know about certain Unicode features in order to know when and how to apply them. It may take some time to wrap your head around it all, but don’t be discouraged—think about the billions of people for whom your software will be more accessible through supporting text in their language. Embrace the complexity!  The Unicode Codespace Let’s start with some general orientation. The basic elements of Unicode—its “characters”, although that term isn’t quite right—are called code points. Code points are identified by number, customarily written in hexadecimal with the prefix “U+”, such as U+0041 “A” latin capital letter a or U+03B8 “θ” greek small letter theta. Each code point also has a short name, and quite a few other properties, specified in the Unicode Character Database.  The set of all possible code points is called the codespace. The Unicode codespace consists of 1,114,112 code points. However, only 128,237 of them—about 12% of the codespace—are actually assigned, to date. There’s plenty of room for growth! Unicode also reserves an additional 137,468 code points as “private use” areas, which have no standardized meaning and are available for individual applications to define for their own purposes."""
```

such as this one.

The text is encoded into raw bytes using UTF-8 encoding.

```python
tokens = list(text.encode("utf-8"))
```

```
vocab_size : desired final vocabulary size for the tokenizer
num_merges : number of needed merges to get to the desired vocab_size
```

```python
vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)
```

This code merges the most occuring pairs successively into new tokens.

```python
merges = {}    # (int, int) --> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
```

The ``max`` function is given ``stats.get`` key which sets the way it evaluates the max value. In this way, the value assigned to each key is evaluated and the key associated to the highest value is sorted as the maximum of the dictionary. At the end, the ``merges`` dictionary looks like

```python
{(101, 32): 256, (105, 110): 257, (115, 32): 258, (116, 104): 259, (101, 114): 260, (99, 111): 261, (116, 32): 262, (226, 128): 263, (44, 32): 264, (97, 110): 265, (111, 114): 266, (100, 32): 267, (97, 114): 268, (101, 110): 269, (257, 103): 270, (261, 100): 271, (121, 32): 272, (46, 32): 273, (97, 108): 274, (259, 256): 275}
```

```python
print(f"tokens length: {len(tokens)}\nids length: {len(ids)}\ncompression ratio: {len(tokens)/len(ids):.2f}X")
out : tokens length: 24597
	  ids length: 19438
	  compression ratio: 1.27X
```

This gets us from 24597 tokens to 19438.

To actually encode and decode stuff, we need

```python
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
```

and

```python
def decode(ids):
    """
    given list of integers (ids), returns Python string
    """
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
```

```python
def encode(text):
    """
    given a string, returns list of integers (tokens) 
    """
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))    
        if pair not in merges:
            break    # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
```

The vocab variable referes to a dictionary containing tokens as keys and their byte value as values, like

```python
{0: b'\x00', 1: b'\x01', 2: b'\x02', 3: b'\x03', 4: b'\x04', 5: b'\x05', 6: b'\x06', 7: b'\x07', 8: b'\x08'}
```

For example :

```python
print(encode("hello world"))
out : [104, 101, 108, 108, 111, 32, 119, 266, 108, 100]
```

The ``min`` function is given a ``lambda`` function which takes every single tuple from the ``stats`` dictionary and checks if it exists in the ``merges`` dictionary. If a tuple exists in ``merges``, it is associated to the ``pair`` variable which contains the lowest value tuple in merges that is also in ``stats``. In the end, the ``pair`` variable is a tuple that is both in ``stats`` and ``merges`` and that has the lowest associated value from 256 to 275.

The OpenAi researches observed that some words were tokenized differently even though they were the same word, like ``dog``, ``dog!``, ``dog?``, etc. The way they came up to solve this is to force some rules for the tokenization to happen accordingly. Using ``regex``, the ``gp2pat`` is a raw string of characters to force the separation of string the way it is wanted.

```python
import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

print(re.findall(gpt2pat, "hello world've I have your nose123 and YOUR'S shoes   HAahHaHa!!?!   "))
out : ['hello', ' world', "'ve", ' I', ' have', ' your', ' nose', '123', ' and', ' YOUR', "'", 'S', ' shoes', '  ', ' HAahHaHa', '!!?!', '   ']
```

For example, `` ?\p{L}+`` catches a continuous sequence of letters, while `` ?\p{N}+`` catches a continuous sequence of numbers.

Useful resources for ``regex`` documentation : 

https://www.regular-expressions.info/unicode.html

https://coderpad.io/blog/development/the-complete-guide-to-regular-expressions-regex/



« tiktoken is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with OpenAI's models. »

« SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements **subword units** (e.g., *byte-pair-encoding* (BPE) and **unigram language model** with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing. »

The difference between tiktoken and sentencepiece are

- tiktoken encodes to UTF-8 first then BPEs bytes.
- sentencepiece BPEs the code points and optionally falls back to UTF-8 bytes for rare code points (rarity is determined by the ``character_coverage`` hyperparameter), which then get translated to byte tokens.

```python
import sentencepiece as spm

with open("toy.txt", "w", encoding="utf-8") as f:
    f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")
```

The SentencePiece library uses a dictionary of options to accomodate all kinds of training datasets. In this way, the example above was matched with the options down below for better optimization of the text tokenizer and detokenizer algorithm.

```python
import os

options = dict(
    
    # input spec
    input="toy.txt",
    input_format="text",
    
    # output spec
    model_prefix="tok400",    # output filename prefix
    
    # algorithm spec
    model_type="bpe",    # uses Byte Pair Encoding algorithm
    vocab_size=400,
    
    # normalization
    normalization_rule_name="identity",  # keeps the training dataset as is, no normalization applied
    remove_extra_whitespaces=False,      # keeps the space at the end if need is
    input_sentence_size=20000000,        # max number of training sentences
    max_sentence_length=4192,          	 # max number of bytes per sentence
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    
    # rare word treatment (code points)
    character_coverage=0.99995,
    byte_fallback=True,          # refers to byte encoding for unknown pieces
    
    # merge rules
    split_digits=True,               # combines digits in 1 token during training
    split_by_unicode_script=True,    
    split_by_whitespace=True,
    split_by_number=True,            # splits numeric and non-numeric sequences into different tokens
    max_sentencepiece_length=16,
    add_dummy_prefix=True,           # adds a whitespace in front when encoding text
    allow_whitespace_only_pieces=True,

    # special tokens
    unk_id=0,    # the unk token MUST exist, the others are optiona, set to -1 to turn off
    bos_id=1,    # beginning of sequence
    eos_id=2,    # end of sequence
    pad_id=-1,

    # systems
    num_threads=os.cpu_count(),    # use ~all system resources
)

spm.SentencePieceTrainer.train(**options)
```

```python
sp = spm.SentencePieceProcessor()
sp.load("tok400.model")
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
```

The ``options`` contains all the parameters necessary to train a SentencePiece model. This is done using the ``spm.SentencePieceTrainer.train()`` method with ``**options`` as an argument. This is an efficient way of programming dynamically. A ``SentencePieceProcessor`` object is then created as ``sp``. This object has different functionalities, such as loading a model, encoding text, decoding text, setting specific id's to certain words, etc. To load a model, the ``load`` attribute is used with the ``tok400.model``. This model name was defined in the ``options`` dictionary with the ``model_prefix`` parameter.

The ``vocab`` variable is a list of lists that are each 2 elements long. The first element is the ``piece``, the subword unit (whole words, word fragments or single characters), and the second is its associated ``id``. For example, the ``options`` dictionary above forcely assigned 3 pieces to certain ids, which are ``unk_id``, ``bos_id`` and ``eos_id``. 

```python
vocab
out : [['<unk>', 0],
       ['<s>', 1],
       ['</s>', 2],
       ['<0x00>', 3],
       ['<0x01>', 4],
       ...,
       [',', 395],
       ['/', 396],
       ['B', 397],
       ['E', 398],
       ['K', 399]]
```

```python
ids = sp.encode("hello 안녕하세요")
print(ids)
out : [362, 378, 361, 372, 358, 362, 239, 152, 139, 238, 136, 152, 240, 152, 155, 239, 135, 187, 239, 157, 151]
```

```python
print([sp.id_to_piece(idx) for idx in ids])
out : ['▁', 'h', 'e', 'l', 'lo', '▁', '<0xEC>', '<0x95>', '<0x88>', '<0xEB>', '<0x85>', '<0x95>', '<0xED>', '<0x95>', '<0x98>', '<0xEC>', '<0x84>', '<0xB8>', '<0xEC>', '<0x9A>', '<0x94>']
```

Since ``byte_fallback`` is ``True``, unknown characters such as ``안녕하세요`` are encoded as bytes using BPE. Indeed, the byte encoding for ``안`` is ``<0xEC> <0x95> <0x88>``.
