import math

import torch
import torch.nn as nn


class Feedforward(nn.Module):
    def __init__(self, embd_size) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size, embd_size)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Head(nn.Module):
    def __init__(self, embd_size, block_size, head_size) -> None:
        super().__init__()
        self.embd_size = embd_size

        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    # B, T, C -> B, T, H
    def forward(self, x: torch.Tensor):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        W = Q @ K.transpose(1, 2) / self.embd_size ** 0.5
        W = W.masked_fill(self.tril == 0, float('-inf'))
        W = W.softmax(dim=1)
        return W @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_size, block_size, n_head) -> None:
        super().__init__()

        self.heads = nn.ModuleList([Head(embd_size, block_size, embd_size // n_head) for _ in range(n_head)])
        self.project = nn.Linear(embd_size, embd_size)

    def forward(self, x: torch.Tensor):
        out = torch.cat([head(x) for head in self.heads])
        out = self.project(out)
        return out


class Block(nn.Module):
    def __init__(self, embd_size, block_size, n_head) -> None:
        super().__init__()

        self.multi_heads = MultiHeadAttention(embd_size, block_size, n_head)
        self.feed_forward = Feedforward(embd_size)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x: torch.Tensor):
        x = x + self.multi_heads(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embd_size, block_size, n_head, n_blocks) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embd = nn.Embedding(vocab_size, embd_size)
        self.position_embd = nn.Embedding(block_size, embd_size)

        self.blocks = nn.Sequential(*[Block(embd_size, block_size, n_head) for _ in range(n_blocks)])
        self.lm_head = nn.Linear(embd_size, vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        batch_size = x.shape[0]
        x = self.token_embd(x) + self.position_embd(torch.arange(self.block_size))
        x = self.blocks(x)
        logits = self.lm_head(x)

        if y:
            logits = logits.view(batch_size * self.block_size, self.vocab_size)
            y = y.view(batch_size * self.block_size)
            loss = nn.functional.cross_entropy(logits, y)
        else:
            loss = None

        return logits, loss

    def generate(self, x: torch.Tensor, n_tokens: int):
        """
        x.shape = B, T
        """
        for _ in range(n_tokens):
            logits, _ = self(x[:, -self.block_size:])
            logits = logits[:, -1, :]  # B, V
            probs = logits.softmax(dim=1)  # B, V
            tokens = torch.multinomial(probs, 1)  # B, 1
            x = torch.cat([x, tokens])

        return x


def get_batch(batch_size: int, block_size: int):
    xb, yb = torch.ones(batch_size, block_size), torch.ones(batch_size, block_size)
    return xb, yb


def main(vocab_size, batch_size, embd_size, block_size, n_head, n_blocks, n_iter):
    M = Transformer(vocab_size, embd_size, block_size, n_head, n_blocks)
    optim = torch.optim.AdamW(M.parameters(), lr=1e-3)

    loss = -math.log(1/vocab_size)

    for _ in range(n_iter):
        xb, yb = get_batch(batch_size, block_size)
        _, loss = M(xb, yb)  # forward pass
        optim.zero_grad(set_to_none=True)
        loss.backward()  # backward pass
        optim.step()

    tokens = M.generate(torch.ones(1, 1), n_tokens=500)
    return list(tokens[0, :])
