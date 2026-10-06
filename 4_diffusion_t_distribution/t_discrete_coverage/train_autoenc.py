import argparse
import os

import torch
from torchvision import datasets, utils

import autoenc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='../../data',
                   help='root holding FashionMNIST/raw')
    p.add_argument('--out', default='results/autoenc.pt', help='where the weights go')
    p.add_argument('--label', type=int, default=0,
                   help='FashionMNIST class to fit; None-less, the metrics use class 0')
    p.add_argument('--dim', type=int, default=128, help='bottleneck width')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val', type=int, default=500,
                   help='held-out images; train and val diverging means the space is '
                        'dense around the training images and nowhere else')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    ds = datasets.FashionMNIST(args.data, train=True, download=True)
    keep = (ds.targets == args.label).nonzero(as_tuple=True)[0]
    x = (ds.data[keep].float() / 127.5 - 1.0).reshape(-1, 1, 28, 28).to(args.device)
    print('training on %d class-%d images (%d held out)' % (len(x), args.label, args.val))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    net = autoenc.train_autoencoder(x, args.out, dim=args.dim, epochs=args.epochs,
                                    batch=args.batch_size, lr=args.lr, val=args.val,
                                    device=args.device)

    # eyeball check: the bottleneck must keep sleeve length, print and silhouette,
    # which is exactly what the reconstructions show
    with torch.no_grad():
        src = x[:8]
        grid = torch.cat([src, net(src)])
    path = os.path.splitext(args.out)[0] + '_recon.png'
    utils.save_image(grid.clamp(-1, 1) * 0.5 + 0.5, path, nrow=8)
    print('wrote %s  (top: real, bottom: reconstruction)' % path)


if __name__ == '__main__':
    main()
