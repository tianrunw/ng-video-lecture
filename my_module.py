import time

import torch
import torch.nn as nn

MSIZE = 2000


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(MSIZE, MSIZE)

    def forward(self, x, targets):
        x = self.layer1(x)
        loss = nn.functional.cross_entropy(x, targets)
        return x, loss


def train_model(module, x, y, times=500):
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)

    start_time = time.time()
    for _ in range(times):
        logits, loss = module(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Ran {times} times, {time.time() - start_time} seconds")


if __name__ == '__main__':
    train_model(MyModule(), torch.rand(MSIZE, MSIZE), torch.arange(MSIZE))
    # train_model(MyModule().to('mps'), torch.rand(MSIZE, MSIZE).to('mps'), torch.arange(MSIZE).to('mps'))
