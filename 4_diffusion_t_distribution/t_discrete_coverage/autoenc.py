import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """features() returns the dim-d bottleneck."""

    def __init__(self, dim=128):
        super(AutoEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),     # 28 -> 14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),    # 14 -> 7
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # 7 -> 4
            nn.Flatten(), nn.Linear(128 * 4 * 4, dim))
        self.dec_in = nn.Linear(dim, 128 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),                         # 4 -> 7
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),        # 7 -> 14
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),         # 14 -> 28
            nn.Tanh())                                                                   # data in [-1, 1]

    def features(self, x):
        return self.enc(x)

    def forward(self, x):
        return self.dec(self.dec_in(self.enc(x)).reshape(-1, 128, 4, 4))


def train_autoencoder(x, path, dim=128, epochs=100, batch=128, lr=1e-3, val=500,
                      device='cuda', log_every=20, seed=0):
    """x: (N, 1, 28, 28) in [-1, 1].  A held-out split reports overfitting, which
    would make the space dense around the training images and nowhere else."""
    torch.manual_seed(seed)
    perm = torch.randperm(len(x), device=x.device)
    xv, xt = x[perm[:val]], x[perm[val:]]

    net = AutoEncoder(dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        net.train()
        idx = torch.randperm(len(xt), device=xt.device)
        running, nb = 0.0, 0
        for i in range(0, len(xt) - batch + 1, batch):
            xb = xt[idx[i:i + batch]]
            loss = ((net(xb) - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        if epoch % log_every == 0 or epoch == epochs:
            net.eval()
            with torch.no_grad():
                v = float(((net(xv) - xv) ** 2).mean())
            print('    epoch %3d  train %.5f  val %.5f' % (epoch, running / nb, v), flush=True)

    torch.save({'state': net.state_dict(), 'dim': dim, 'epochs': epochs, 'n': len(xt)}, path)
    print('    wrote %s' % path)
    return net.eval()


def load(path, device):
    ck = torch.load(path, map_location=device)
    net = AutoEncoder(ck['dim']).to(device)
    net.load_state_dict(ck['state'])
    return net.eval()


@torch.no_grad()
def features(net, flat, batch=2000):
    """flat: (N, 784) in [-1, 1]  ->  (N, dim)."""
    out = []
    for i in range(0, len(flat), batch):
        out.append(net.features(flat[i:i + batch].reshape(-1, 1, 28, 28)))
    return torch.cat(out)
