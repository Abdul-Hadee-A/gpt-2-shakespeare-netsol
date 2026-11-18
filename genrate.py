import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from model import GPT, Config

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the tokenizer and the trained model
tok = ByteLevelBPETokenizer(
    "shakespeare_tokenizer/vocab.json",
    "shakespeare_tokenizer/merges.txt",
)
ckpt = torch.load("ckpt.pt", map_location=device)
cfg = Config(**ckpt["cfg"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

@torch.no_grad()
def generate(prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]            # only the last 256 tokens fit
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature        # scores for the next token
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)  # pick one at random by probability
        idx = torch.cat([idx, nxt], dim=1)
    return tok.decode(idx[0].tolist())

for t in [0.5, 0.8, 1.2]:
    print(f"\n===== temperature {t} =====")
    print(generate("ROMEO:", temperature=t))