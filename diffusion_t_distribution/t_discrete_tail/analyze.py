import csv
import os

import numpy as np
import torch
from scipy import linalg, stats
from torchvision import utils

import run
from unet import UNet


def nn_distance(a, b, chunk=512, exclude_self=False):
    """min_j ||a_i - b_j|| for every i.

    exclude_self skips j == i, which the real data needs when compared against
    itself: otherwise every image matches itself at distance 0."""
    bn = (b * b).sum(1)
    out = []
    for i in range(0, len(a), chunk):
        c = a[i:i + chunk]
        d = (c * c).sum(1, keepdim=True) + bn[None, :] - 2 * c @ b.t()
        if exclude_self:
            r = torch.arange(len(c), device=d.device)
            d[r, i + r] = float('inf')
        out.append(d.clamp(min=0).min(1)[0].sqrt())
    return torch.cat(out).cpu().numpy()


def topk_mean(v, k):
    """Mean of the k largest values."""
    return float(np.sort(v)[-k:].mean())


def fid(a, b):
    """Frechet distance between Gaussian fits of two feature sets."""
    a, b = a.double().cpu().numpy(), b.double().cpu().numpy()
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cov, _ = linalg.sqrtm(ca.dot(cb), disp=False)
    return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * cov.real))


def analyze(args):
    os.makedirs(args.out, exist_ok=True)
    ds, keep = run.selected(args)
    real = (ds.data[keep].float() / 127.5 - 1.0).reshape(len(keep), -1).to(args.device)
    print('real: %d images' % len(real))
    if args.n != len(real):
        print('note: -n %d against %d real images; top-k metrics grow with sample '
              'size, so -n %d compares rows directly' % (args.n, len(real), len(real)))

    keys = ['pixel std', 'NN dist top%d' % args.topk]
    emb = em = None
    if args.embed and os.path.exists(args.embed):
        import embed as em
        emb = em.load(args.embed, args.device)
        f_real = em.features(emb, real)
        keys += ['feat NN top%d' % args.topk, 'FID']
        print('feature space: %s (%d-d)' % (args.embed, f_real.shape[1]))
    else:
        print('no feature space (%s missing); run "python run.py embed" for FID' % args.embed)

    nets, cfg = [], {'data': dict(nu='', tsample='', epoch='')}
    for path in args.ckpt:
        ck = torch.load(path, map_location=args.device)
        net = UNet().to(args.device)
        net.load_state_dict(ck['ema'])
        net.eval()
        label = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if label in [n for n, _, _ in nets] or label == 'data':
            label = '%s#%d' % (label, len(nets) + 1)          # keep dict keys distinct
        print('  %-14s %s' % (label, run.config_line(ck)))
        nets.append((label, net, ck['nu']))
        cfg[label] = dict(nu=ck['nu'], tsample=ck.get('tsample', 'uniform'),
                          epoch=ck.get('epoch'))

    names = ['data'] + [n for n, _, _ in nets]
    acc = dict((n, dict((k, []) for k in keys)) for n in names)

    # the real data is its own reference, each image left out of its own search,
    # so this row uses all N images instead of half of them
    real_np = real.cpu().numpy()                          # the one copy every plot shares
    d_real = nn_distance(real, real, exclude_self=True)
    fd_real = nn_distance(f_real, f_real, exclude_self=True) if emb is not None else None
    shown = None

    for r in range(args.repeat):
        # bootstrap gives the data row an error bar without halving the sample
        bi = np.random.randint(0, len(real), len(real))
        bt = torch.from_numpy(bi).to(real.device)
        acc['data'][keys[0]].append(float(real[bt].std()))
        acc['data'][keys[1]].append(topk_mean(d_real[bi], args.topk))
        if emb is not None:
            acc['data'][keys[2]].append(topk_mean(fd_real[bi], args.topk))
            # the floor: the same N drawn from the data itself.  FID falls steeply
            # with sample size, so halving the data here would not be comparable
            # to a model measured with N samples against N.
            acc['data'][keys[3]].append(fid(f_real[bt], f_real))

        cur = {'data': (real_np, d_real, fd_real if emb is not None else None)}
        for name, net, nu in nets:
            print('  repeat %d/%d  generating %s' % (r + 1, args.repeat, name), flush=True)
            g = run.generate(net, args.n, args.steps, nu, args.corr, args.device, tag=name)
            x = g.reshape(len(g), -1)
            d = nn_distance(x, real)
            acc[name][keys[0]].append(float(x.std()))
            acc[name][keys[1]].append(topk_mean(d, args.topk))
            fd = None
            if emb is not None:
                fx = em.features(emb, x)
                fd = nn_distance(fx, f_real)
                acc[name][keys[2]].append(topk_mean(fd, args.topk))
                acc[name][keys[3]].append(fid(fx, f_real))
            cur[name] = (x.cpu().numpy(), d, fd)
        if shown is None:
            shown = cur
        print('  repeat %d/%d done   ' % (r + 1, args.repeat)
              + '  '.join('%s NN=%.2f%s' % (n, acc[n][keys[1]][-1],
                                            '' if emb is None else ' FID=%.3f' % acc[n][keys[3]][-1])
                          for n in names), flush=True)

    print('\n%-16s %s' % ('', '  '.join('%18s' % k for k in keys)))
    for n in names:
        print('%-16s %s' % (n, '  '.join('%9.3f +- %-6.3f' % (np.mean(acc[n][k]), np.std(acc[n][k]))
                                         for k in keys)))

    if len(nets) == 2 and args.repeat > 1:
        a, b = names[1], names[2]
        print('\nWelch t-test  %s vs %s  (n=%d each)' % (a, b, args.repeat))
        for k in keys:
            tv, pv = stats.ttest_ind(acc[a][k], acc[b][k], equal_var=False)
            print('  %-16s diff %+8.3f   t=%+6.2f   p=%.4f  %s'
                  % (k, np.mean(acc[a][k]) - np.mean(acc[b][k]), tv, pv,
                     'significant' if pv < 0.05 else ''))
    elif args.repeat == 1:
        print('\n(--repeat >= 2 gives error bars and a t-test)')

    write_csv(args, names, keys, acc, cfg)
    plot(args, names, shown, real_np, emb is not None)


def write_csv(args, names, keys, acc, cfg):
    """Raw per-repeat values, self-describing so runs can be compared later."""
    path = os.path.join(args.out, 'metrics.csv')
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['set', 'nu', 'tsample', 'epoch', 'corr', 'steps', 'n', 'label',
                    'limit', 'repeat'] + keys)
        for n in names:
            c = cfg[n]
            for r in range(args.repeat):
                w.writerow([n, c['nu'], c['tsample'], c['epoch'], args.corr, args.steps,
                            args.n, args.label, args.limit, r + 1]
                           + ['%.6f' % acc[n][k][r] for k in keys])

    tpath = os.path.join(args.out, 'ttest.csv')
    with open(tpath, 'w') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'a', 'b', 'mean_a', 'mean_b', 'diff', 't', 'p', 'significant'])
        if len(names) == 3 and args.repeat > 1:
            a, b = names[1], names[2]
            for k in keys:
                tv, pv = stats.ttest_ind(acc[a][k], acc[b][k], equal_var=False)
                ma, mb = np.mean(acc[a][k]), np.mean(acc[b][k])
                w.writerow([k, a, b, '%.6f' % ma, '%.6f' % mb, '%.6f' % (ma - mb),
                            '%.4f' % tv, '%.6f' % pv, int(pv < 0.05)])
    print('wrote %s and %s' % (path, tpath))


def _pair_rows(top, bot, px, bk, d, ncol, label='', exclude_self=False):
    """Fill two axis rows: the ncol most distant samples above their nearest real
    image.  exclude_self is needed when px is bk, or every image pairs with itself."""
    for c, i in enumerate(np.argsort(d)[-ncol:][::-1]):
        sq = ((bk - px[i]) ** 2).sum(1)
        if exclude_self:
            sq[i] = np.inf
        j = int(sq.argmin())
        top[c].imshow(px[i].reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
        top[c].set_title('%s  d=%.1f' % (label if c == 0 else '', d[i]), fontsize=8)
        bot[c].imshow(bk[j].reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
        bot[c].set_title('nearest real', fontsize=7)


def plot(args, names, shown, bk, has_emb):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = ['pixel value', 'distance to nearest real image', 'mean brightness per image']
    if has_emb:
        panels.insert(2, 'feature-space distance to nearest real')
    fig, ax = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.2))
    for name, c in zip(names, ['k', 'C3', 'C0', 'C2']):
        px, d, fd = shown[name]
        ax[0].hist(px.ravel(), bins=100, histtype='step', density=True, log=True, color=c, label=name)
        ax[1].hist(d, bins=80, histtype='step', density=True, color=c, label=name)
        if has_emb:
            ax[2].hist(fd, bins=80, histtype='step', density=True, color=c, label=name)
        ax[-1].hist(px.mean(1), bins=80, histtype='step', density=True, color=c, label=name)
    for a, t in zip(ax, panels):
        a.set_xlabel(t); a.legend(fontsize=7); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'stats.png'), dpi=130)

    fig2, ax2 = plt.subplots(1, len(names), figsize=(4.6 * len(names), 5.0))
    for a, name in zip(np.atleast_1d(ax2), names):
        g = torch.from_numpy(shown[name][0][:64]).reshape(-1, 1, 28, 28)
        a.imshow(utils.make_grid(g.clamp(-1, 1) * 0.5 + 0.5, nrow=8, padding=1)
                 .permute(1, 2, 0).numpy())
        a.set_title(name); a.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'samples.png'), dpi=130)

    ncol = 8
    fig3, ax3 = plt.subplots(2 * len(names), ncol, figsize=(1.5 * ncol, 3.1 * len(names)))
    for r, name in enumerate(names):
        px, d, fd = shown[name]
        _pair_rows(ax3[2 * r], ax3[2 * r + 1], px, bk, d, ncol, name,
                   exclude_self=(name == 'data'))
    for a in ax3.ravel():
        a.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'extremes.png'), dpi=130)
    print('\nwrote %s/{stats,samples,extremes}.png' % args.out)
