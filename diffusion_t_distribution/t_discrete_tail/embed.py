import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Embedder(nn.Module):
    """Small CNN; features() returns the 128-d penultimate layer."""

    def __init__(self, dim=128):
        super(Embedder, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(dim, 10)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


def train_embedder(args):
    """Trained on all ten classes so the space separates garment types, not just
    the one class the diffusion model was fitted to."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    tr = datasets.FashionMNIST(args.data, train=True, download=True, transform=tf)
    te = datasets.FashionMNIST(args.data, train=False, download=True, transform=tf)
    ltr = DataLoader(tr, batch_size=256, shuffle=True, num_workers=args.workers, drop_last=True)
    lte = DataLoader(te, batch_size=512, num_workers=args.workers)

    net = Embedder().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(1, args.embed_epochs + 1):
        net.train()
        for x, y in ltr:
            loss = nn.functional.cross_entropy(net(x.to(args.device)), y.to(args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, y in lte:
                ok += int((net(x.to(args.device)).argmax(1).cpu() == y).sum())
                tot += len(y)
        print('epoch %d  test accuracy %.3f' % (epoch, ok / tot), flush=True)

    os.makedirs(os.path.dirname(args.embed) or '.', exist_ok=True)
    torch.save(net.state_dict(), args.embed)
    print('wrote', args.embed)


def load(path, device):
    net = Embedder().to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()


@torch.no_grad()
def features(net, flat, batch=2000):
    """flat: (N, 784) in [-1, 1]  ->  (N, 128)."""
    out = []
    for i in range(0, len(flat), batch):
        out.append(net.features(flat[i:i + batch].reshape(-1, 1, 28, 28)))
    return torch.cat(out)
