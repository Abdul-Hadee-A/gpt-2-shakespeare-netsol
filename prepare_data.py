import torch
from tokenizers import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer(
    "shakespeare_tokenizer/vocab.json",
    "shakespeare_tokenizer/merges.txt",
)

text = open("input.txt", encoding="utf-8").read()
ids = tok.encode(text).ids
data = torch.tensor(ids, dtype=torch.long)

n = int(0.9 * len(data))
torch.save(
    {"train": data[:n], "val": data[n:], "vocab_size": tok.get_vocab_size()},
    "data.pt",
)

print(f"chars: {len(text)}, tokens: {len(ids)}, chars/token: {len(text)/len(ids):.2f}")