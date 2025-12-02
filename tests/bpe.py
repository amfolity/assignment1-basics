from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from collections import defaultdict
from collections import Counter
import time
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):            
        self.vocab = vocab
        self.inverse_map = {}
        for key, val in self.vocab.items():
            self.inverse_map[val] = key
        self.merges = merges
        self.special_tokens = special_tokens
        
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        #with open(vocab_path) as vocab_f:
        #    gpt2_vocab = json.load(vocab_f)
        #vocab = {
        #    gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        #    for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        #}
        with open(merges_filepath, encoding="utf-8") as mf:
            for line in mf:
                self.merges.append(tuple(line.split()))
                
        with open(vocab_filepath, encoding="utf-8") as vf:
            self.vocab = {}
            for line in vf:
                word, enum = line.split()
                self.vocab[word] = enum
            
        self.special_tokens = special_tokens
        
        
    def encode(self, text):
        #pattern = "|".join(map(re.escape, special_tokens))
        #result = re.split(pattern, text)
        import re

        def split_bytes_except(words, text):
            # Convert keyword list to byte patterns, escape each one
            #print("text", text)
            text = text.encode("utf-8")
            bwords = [re.escape(w.encode("utf-8")) for w in words] if words else []
        
            # Build regex pattern as bytes
            special = b"|".join(bwords)
        
            # Match: special word OR any single byte
            pattern = special + b"|." if special else b"."
            pattern = re.compile(pattern, re.DOTALL)
            
            tokens = []
            for m in re.finditer(pattern, text):
                #print("m", m)
                tokens.append(m.group(0))
            return tokens

        #num_processes = 4
        #boundaries = find_chunk_boundaries(num_processes, b"<|endoftext|>")

        #for start, end in zip(boundaries[:-1], boundaries[1:]):
        #    f.seek(start)
        #    chunk = f.read(end - start).decode("utf-8", errors="ignore")

        
        splittext = split_bytes_except(self.special_tokens, text)
        #print("splittest before", splittext) # "inverse dict", self.inverse_map
        new_splittext = []
        for merge in self.merges:
            part1, part2 = merge
            i = 0
            while i < len(splittext):
                if i+1 < len(splittext) and (splittext[i], splittext[i+1]) == (part1, part2):
                    new_splittext.append(splittext[i] + splittext[i+1])
                    i+=2
                else:
                    new_splittext.append(splittext[i])
                    i+=1
            splittext, new_splittext  = new_splittext, []
        #print("\n" in splittext)
        
        return [self.inverse_map[word] for word in splittext]     

    def encode_iterable(self, iterable):
        for piece in iterable:
            for el in self.encode(piece):
                yield el
            #yield self.encode(piece)
        
    def decode(self, ids: list[int]) -> str:
        idstotext = []
        for idd in ids:
            idstotext.append(self.vocab[idd])
        return b"".join(idstotext).decode("utf-8", errors='replace')
        
def construct_words(text, special_tokens=list()):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    pattern = "|".join(map(re.escape, special_tokens)) 

    result = re.split(pattern, text)

    wordscounter = Counter()
    for words in result:
        wordscounter.update(map(lambda x : x.encode('utf-8'), re.findall(PAT, words)))

    wordscounter = {tuple( map(lambda x : x.to_bytes(), list(key))): value for key, value in wordscounter.items()}
    return wordscounter

def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r") as f:
        text = f.read()
                
    words = construct_words(text, special_tokens)

    merges = []
    tokenizer_vocab = {}
    res = [token.encode('utf-8') for token in special_tokens] + [bytes((_,)) for _ in range(256)]
    
    while len(res) < vocab_size:

        pairs_dict = defaultdict(int)
        for word, repetitions in words.items():
            for i in range(len(word)):
                if i + 1 < len(word):
                    pair = (word[i], word[i+1])
                    pairs_dict[pair] += repetitions
                
        if not pairs_dict:
            break
        max_pair = max(pairs_dict, key=lambda key: (pairs_dict[key], key))
        
        res += [max_pair[0] + max_pair[1]]
        merges.append((max_pair[0], max_pair[1]))
        
        new_words = []
        del_words = []

        for word in words:
            tmp = words[word]
            del_words.append(word)
            new_word = []
            i = 0
            while i < len(word):
                if i+1 < len(word) and (word[i], word[i+1]) == max_pair:
                    new_word.append(word[i] + word[i+1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_words.append((tuple(new_word), tmp))
        for word in del_words:
            del words[word]
        for word, count in new_words:
            words[word] = count

        for i in range(len(res)):
            tokenizer_vocab.update({i : res[i]})
    return tokenizer_vocab, merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
