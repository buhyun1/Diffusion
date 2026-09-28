import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(ch):
    return nn.GroupNorm(min(32, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, dropout):
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, out_ch)
        self.norm2 = norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(c // self.heads), dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=64, mults=(1, 2, 2), num_res=2, attn_res=(14,),
                 dropout=0.1, img_res=28):
        super().__init__()
        emb_ch = base * 4
        self.emb = nn.Sequential(nn.Linear(base, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))
        self.base = base

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        chans = [base]
        ch, res = base, img_res

        self.down = nn.ModuleList()
        for i, m in enumerate(mults):
            for _ in range(num_res):
                blocks = nn.ModuleList([ResBlock(ch, base * m, emb_ch, dropout)])
                ch = base * m
                if res in attn_res:
                    blocks.append(AttnBlock(ch))
                self.down.append(blocks)
                chans.append(ch)
            if i != len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                chans.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_ch, dropout), AttnBlock(ch),
                                  ResBlock(ch, ch, emb_ch, dropout)])

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for j in range(num_res + 1):
                blocks = nn.ModuleList([ResBlock(ch + chans.pop(), base * m, emb_ch, dropout)])
                ch = base * m
                if res in attn_res:
                    blocks.append(AttnBlock(ch))
                if i != 0 and j == num_res:
                    blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    res *= 2
                self.up.append(blocks)

        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, c_noise):
        emb = self.emb(timestep_embedding(c_noise, self.base))
        h = self.in_conv(x)
        hs = [h]
        for blocks in self.down:
            for b in blocks:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)
            hs.append(h)
        for b in self.mid:
            h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        for blocks in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for b in blocks:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        return self.out(h)
