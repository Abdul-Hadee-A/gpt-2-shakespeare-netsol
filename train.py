import math
import torch
from model import GPT, Config

# ---- settings (you can change these) ----
batch_size = 32
max_iters = 3000
eval_interval = 250
eval_iters = 50
max_lr = 6e-4
min_lr = 6e-5
warmup = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)

# ---- data ----
d = torch.load("data.pt")
train_data, val_data = d["train"], d["val"]
cfg = Config(vocab_size=d["vocab_size"])

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])  # shifted by one
    return x.to(device), y.to(device)

# ---- model + optimizer ----
model = GPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

def get_lr(it):
    if it < warmup:
        return max_lr * (it + 1) / warmup
    progress = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---- training loop ----
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        l = estimate_loss()
        print(f"step {it}: train {l['train']:.3f}, val {l['val']:.3f}")
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "ckpt.pt")

    for g in opt.param_groups:
        g["lr"] = get_lr(it)

    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()