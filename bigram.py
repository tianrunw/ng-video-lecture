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
# if torch.cuda.is_available():
#     DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps'

print(f'Using device: {DEVICE}')

DATA = torch.tensor(encode(TEXT), dtype=torch.long)
DATA_SPLIT_CUTOFF = int(0.9 * len(DATA))
TRAIN_DATA = DATA[:DATA_SPLIT_CUTOFF]
VAL_DATA = DATA[DATA_SPLIT_CUTOFF:]

BATCH_SIZE = 4
BLOCK_SIZE = 8
EMBEDDING_DIM = 32

MAX_ITERS = 10000
EVAL_INTERVAL = MAX_ITERS // 10
EVAL_ITERS = 10


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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # B, T, C
        position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE))  # T, C
        x = token_embeddings + position_embeddings
        logits = self.lm_head(x)  # B, T, V

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # last time step
            logits = logits[:, -1, :]  # (B, C)
            # softmax
            probs = functional.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append to idx
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


def main():
    M = BigramLanguageModel(VOCAB_SIZE)
    M = M.to(DEVICE)
    optimizer = torch.optim.AdamW(M.parameters(), lr=1e-3)

    loss = -math.log(1/VOCAB_SIZE)
    print(f"Initial loss: {loss:.4f}")

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(M)
            print(f"Iter: {it}, Train loss: {losses['train']:4f}, Val loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = M(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Train loss: {loss:.4f}")

    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(M.generate(idx, 500)[0].tolist()))


if __name__ == '__main__':
    main()
