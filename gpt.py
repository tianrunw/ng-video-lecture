import math

import torch
import torch.nn as nn
from torch.nn import functional

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    TEXT = f.read()

# Vocabulary
VOCAB = sorted(list(set(TEXT)))
VOCAB_SIZE = len(VOCAB)

# encode decode
C_TO_I = {c: i for i, c in enumerate(VOCAB)}
I_TO_C = {i: c for i, c in enumerate(VOCAB)}


def encode(string):
    return [C_TO_I[c] for c in string]


def decode(encoding_list):
    return ''.join([I_TO_C[i] for i in encoding_list])


DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'

print(f'Using device: {DEVICE}')

DATA = torch.tensor(encode(TEXT), dtype=torch.long)
DATA_SPLIT_CUTOFF = int(0.9 * len(DATA))
TRAIN_DATA = DATA[:DATA_SPLIT_CUTOFF]
VAL_DATA = DATA[DATA_SPLIT_CUTOFF:]

BATCH_SIZE = 32
BLOCK_SIZE = 8
EMBEDDING_DIM = 32

MAX_ITERS = 5000
EVAL_INTERVAL = MAX_ITERS // 10
EVAL_ITERS = 100

N_LAYER = 3
DROPOUT = 0.2


def get_batch(split):
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)  # B, T, H
        q = self.query(x)  # B, T, H
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # B, T, T
        wei = functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # self attention, B, T, H
        out = wei @ v  # B, T, H
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # B, T, EMBEDDING
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.ReLU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embedding, n_heads):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_heads, n_embedding // n_heads)  # Communication
        self.ffwd = FeedForward(n_embedding)  # Computation
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x: torch.Tensor):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.blocks = nn.Sequential(
            *[Block(EMBEDDING_DIM, n_heads) for _ in range(N_LAYER)],
            nn.LayerNorm(EMBEDDING_DIM)
        )
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # B, T, C
        position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE))  # T, C
        x = token_embeddings + position_embeddings
        x = self.blocks(x)  # B, T, H
        logits = self.lm_head(x)  # B, T, V

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            # At every time step, for every context up to current time step, we have a prediction
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx[:, -BLOCK_SIZE:])
            # For every context/sample in the batch, we have a prediction. The last time step's prediction has the
            # longest context.
            logits = logits[:, -1, :]  # (B, C)
            # softmax
            probs = functional.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append to idx
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


def main():
    M = BigramLanguageModel(4)
    M = M.to(DEVICE)
    optimizer = torch.optim.AdamW(M.parameters(), lr=1e-3)

    loss = -math.log(1/VOCAB_SIZE)
    print(f"Vocab loss: {loss:.4f}")

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(M)
            print(f"Iter: {it}, Train loss: {losses['train']:4f}, Val loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = M(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(M.generate(idx, 500)[0].tolist()))


if __name__ == '__main__':
    main()
