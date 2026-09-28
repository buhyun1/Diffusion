import argparse
import copy
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import posterior
from unet import UNet


def mu(t):
    return 1.0 - t


def sigma(t):
    return t


def sigma_cross(t, t_prev, corr):
    """sigma_12^2(t) = sigma_21^2(t).  |.| <= sigma_t sigma_{t-dt} keeps Sigma PSD,
    so corr in [-1, 1].  corr = 1 makes sigmabar_t vanish (deterministic steps);
    corr = 0 leaves x_t and x_{t-dt} uncorrelated given x_0."""
    return corr * sigma(t) * sigma(t_prev)


def selected(args, transform=None):
    """The training subset chosen by --label / --limit."""
    ds = datasets.FashionMNIST(args.data, train=True, download=True, transform=transform)
    keep = torch.arange(len(ds))
    if args.label is not None:
        keep = (ds.targets == args.label).nonzero(as_tuple=True)[0]
    if args.limit is not None:
        keep = keep[:args.limit]
    return ds, keep


def train(args):
    # no horizontal flip: analysis compares against the un-flipped training set,
    # and a mirrored sample would score as spuriously novel
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    ds, keep = selected(args, tf)
    if len(keep) < len(ds):
        ds = torch.utils.data.Subset(ds, keep)
    print('training on %d images' % len(ds), flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)

    os.makedirs(args.out, exist_ok=True)
    net = UNet().to(args.device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    step = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, running = time.time(), 0.0
        for x0, _ in loader:
            x0 = x0.to(args.device, non_blocking=True)
            # t_min keeps sigma_t away from the degenerate x_t = x_0 point
            t = torch.rand(x0.shape[0], device=x0.device) * (1.0 - args.t_min) + args.t_min
            xt = posterior.sample(*posterior.perturbation_kernel(x0, mu(t), sigma(t), args.nu))
            D = net(xt, t * 1000)
            loss = ((D - x0) ** 2).mean()                  # Eqn. 10

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                decay = min(args.ema, (step + 1) / (step + 10))
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn.detach(), 1 - decay)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
            running += loss.item()
            step += 1

        print('epoch %3d  loss %.4f  %.1fs' % (epoch, running / len(loader),
                                               time.time() - t0), flush=True)
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            x = generate(ema, 64, args.steps, args.nu, args.corr, args.device)
            utils.save_image(x.clamp(-1, 1) * 0.5 + 0.5,
                             os.path.join(args.out, 'samples_ep%03d.png' % epoch), nrow=8)
            torch.save({'ema': ema.state_dict(), 'nu': args.nu, 'epoch': epoch},
                       os.path.join(args.out, 'ckpt.pt'))


@torch.no_grad()
def generate(net, n, steps, nu, corr, device, batch=2000, tag=None):
    """Ancestral sampling from p_theta(x_{t-dt}|x_t) = t_d(mu_theta, sigmabar_t^2 I, nu + d),
    whose mean mu_theta is assembled from the network output D_theta by Eqn. 7."""
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    out, t0 = [], time.time()
    for i in range(0, n, batch):
        m = min(batch, n - i)
        x = posterior.sample(*posterior.prior((m, 1, 28, 28), nu, device))   # sigma_T = 1
        for j in range(steps):
            t, t_prev = ts[j], ts[j + 1]
            D = net(x, (t * 1000).repeat(m))
            s_c = sigma_cross(t, t_prev, corr)
            x = posterior.sample(*posterior.reverse_posterior(
                x, D, mu(t), mu(t_prev), sigma(t), sigma(t_prev), s_c, s_c, nu))
        out.append(x)
        if tag:
            print('\r      %s %5d/%d images  %5.1fs' % (tag, i + m, n, time.time() - t0),
                  end='', flush=True)
    if tag:
        print()
    return torch.cat(out)


def config_line(ck):
    """One-line summary of how a checkpoint was trained."""
    return 'nu=%g  epoch=%s' % (ck['nu'], ck.get('epoch'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('train', 'sample', 'analyze', 'embed'),
                   help="train: fit D_theta.  sample: one grid from --out/ckpt.pt.\n"
                        "analyze: compare --ckpt models.  embed: fit the feature space")
    p.add_argument('--embed', default='results/embed.pt',
                   help='feature extractor; analyze adds feature-space metrics\n'
                        'when this file exists')
    p.add_argument('--embed-epochs', type=int, default=100, help='embed: training epochs')
    p.add_argument('--ckpt', nargs='+', help='analyze: checkpoints to compare')
    p.add_argument('--topk', type=int, default=10,
                   help='analyze: extreme metrics average the k largest values,\n'
                        'which is far steadier than a single order statistic')
    p.add_argument('--repeat', type=int, default=1,
                   help='analyze: repeats, for error bars and a t-test')
    p.add_argument('--data', default='../../data',
                   help='root holding FashionMNIST/raw')
    p.add_argument('--out', default='results',
                   help='train/sample: run directory.  analyze: where the report goes')
    p.add_argument('--label', type=int, default=None,
                   help='train/sample/analyze: keep only this FashionMNIST class')
    p.add_argument('--limit', type=int, default=None,
                   help='use only the first N training images')
    p.add_argument('--nu', type=float, default=None,
                   help='degrees of freedom; 3.0 when training, the checkpoint\n'
                        'value when sampling, unless given explicitly')
    p.add_argument('--corr', type=float, default=0.99,
                   help='sample/analyze: sigma_c^2 = corr * sigma_t * sigma_prev;\n'
                        '1 = deterministic steps, 0 = fully stochastic')
    p.add_argument('--steps', type=int, default=200,
                   help='sample/analyze: denoising steps from t=1 to t=0')
    p.add_argument('--t-min', type=float, default=1e-3,
                   help='train: lower end of t ~ U[t_min, 1]')
    p.add_argument('--epochs', type=int, default=100, help='train: epochs')
    p.add_argument('--batch-size', type=int, default=128, help='train: batch size')
    p.add_argument('--lr', type=float, default=2e-4, help='train: Adam learning rate')
    p.add_argument('--ema', type=float, default=0.9999, help='train: EMA decay')
    p.add_argument('--workers', type=int, default=4,
                   help='train/embed: dataloader workers')
    p.add_argument('--sample-every', type=int, default=10,
                   help='train: epochs between preview grids and checkpoints')
    p.add_argument('-n', type=int, default=64,
                   help='sample: images in the grid.  analyze: images per set')
    p.add_argument('--device', default='cuda', help='cuda or cpu')
    args = p.parse_args()

    if args.mode == 'embed':
        import embed
        embed.train_embedder(args)
    elif args.mode == 'train':
        if args.nu is None:
            args.nu = 3.0
        train(args)
    elif args.mode == 'analyze':
        if not args.ckpt:
            raise SystemExit('analyze needs --ckpt')
        if args.nu is not None:
            print('note: --nu is ignored by analyze; each checkpoint uses its own')
        import analyze
        analyze.analyze(args)
    else:
        ck = torch.load(os.path.join(args.out, 'ckpt.pt'), map_location=args.device)
        print('checkpoint: %s' % config_line(ck))
        net = UNet().to(args.device)
        net.load_state_dict(ck['ema'])
        net.eval()
        if args.nu is None:
            args.nu = ck['nu']
        elif ck['nu'] != args.nu:
            print('note: trained with nu=%g, sampling with nu=%g' % (ck['nu'], args.nu))
        x = generate(net, args.n, args.steps, args.nu, args.corr, args.device)
        nrow = int(args.n ** 0.5)
        path = os.path.join(args.out, 'samples.png')
        utils.save_image(x.clamp(-1, 1) * 0.5 + 0.5, path, nrow=nrow)
        print('wrote', path)


if __name__ == '__main__':
    main()
