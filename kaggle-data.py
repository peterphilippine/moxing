import random
import numpy as np
import os
import multiprocessing
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset, Sampler
import transformers

def _tokenize_and_write_chunk(args):
    txt_file_path, tokenizer_file_path, dtype, start_line, end_line, temp_output_file_path, batch_size = args
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_file_path)

    with open(txt_file_path, 'r', encoding='utf-8') as f_in:
        # Skip to start_line
        for _ in range(start_line):
            f_in.readline()

        with open(temp_output_file_path, 'wb') as f_out:
            lines_buffer = []
            for i in range(start_line, end_line):
                line = f_in.readline()
                if not line:
                    break
                lines_buffer.append(line)

                if len(lines_buffer) >= batch_size:
                    tokenized_batches = tokenizer.encode_batch(lines_buffer)
                    for encoding in tokenized_batches:
                        arr = np.array(encoding.ids, dtype=dtype)
                        arr.tofile(f_out)
                    lines_buffer = []
            
            # Process any remaining lines in the buffer
            if lines_buffer:
                tokenized_batches = tokenizer.encode_batch(lines_buffer)
                for encoding in tokenized_batches:
                    arr = np.array(encoding.ids, dtype=dtype)
                    arr.tofile(f_out)

class CustomMambaDataset(Dataset):
    def __init__(
        self,
        txt_file,
        tokenizer_file,
        context_len,
        fim_pad,
        rng_seed=42,
        offset=0,
        length=None,
        batch_size=32,
    ):
        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        # Tokenize the file and memory-map the token IDs
        token_file = txt_file
        
        vocab_size = self.tokenizer.get_vocab_size()
        dtype = np.uint16 if vocab_size < 2**16 else np.uint32
        self.tokenized_data = np.memmap(token_file, dtype=dtype, mode='r')
        self.context_len = context_len
        self.offset = offset
        self.data_length = len(self.tokenized_data) if length is None else length

        self.np_rng = np.random.RandomState(seed=rng_seed)
        self.fim_pad = torch.tensor([fim_pad])
        self.batch_size = batch_size

    def __len__(self):
        return self.data_length -self.context_len

    def __getitem__(self, idx):
        pad_id = int(self.fim_pad)
        sample = self.tokenized_data[idx : idx + self.context_len + 1]
        sample = torch.tensor(sample, dtype=torch.long)
        if len(sample) < self.context_len + 1:
            pad_len = (self.context_len + 1) - len(sample)
            sample = torch.cat([sample, torch.full((pad_len,), pad_id, dtype=torch.long)])

        tokens = sample[:-1]
        labels = sample[1:]

        return {"tokens": tokens, "labels": labels}


class MambaSampler(Sampler):
    def __init__(self, data_source, k=16, start_index=0, indices_file="indices.txt"):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.indices_file = indices_file
        self.state_file = indices_file + ".state"

        # 加载或生成索引
        if os.path.exists(self.indices_file):
            with open(self.indices_file, "r") as f:
                self.available_indices = [int(x) for x in f.read().split()]
        else:
            self.available_indices = list(range(start_index, self.num_samples, k))
            random.shuffle(self.available_indices)
            with open(self.indices_file, "w") as f:
                f.write(" ".join(map(str, self.available_indices)))

        # 加载进度
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                self.pos = int(f.read().strip())
        else:
            self.pos = 0

    def __iter__(self):
        while self.pos < len(self.available_indices):
            idx = self.available_indices[self.pos]
            self.pos += 1
            with open(self.state_file, "w") as f:
                f.write(str(self.pos))
            yield idx

    def __len__(self):
        return len(self.available_indices) - self.pos
