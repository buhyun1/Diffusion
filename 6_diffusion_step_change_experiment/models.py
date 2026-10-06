"""신경망 구조 정의: diffusion U-Net과 그 빌딩 블록, 분류기(Classifier).

순수 아키텍처 정의만 담고 있고, 학습 루프/하이퍼파라미터는 노트북에 둔다.
"""

import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from utils import exists, default


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(dim, dim_out if dim_out is not None else dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(dim * 4, dim_out or dim, kernel_size=1)

    def forward(self, x):
        return self.conv(self.unshuffle(x))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        w = self.weight
        mean = w.mean(dim=(1, 2, 3), keepdim=True)
        var = w.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        w = (w - mean) / torch.sqrt(var + eps)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim) else None
        )
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            h = self.mlp(time_emb)[:, :, None, None]
            scale_shift = h.chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.view(b, self.heads, -1, h * w)
        k = k.view(b, self.heads, -1, h * w)
        v = v.view(b, self.heads, -1, h * w)
        q = q * self.scale
        q_t = q.permute(0, 1, 3, 2)
        attn = torch.softmax(torch.matmul(q_t, k), dim=-1)
        v_t = v.permute(0, 1, 3, 2)
        out = torch.matmul(attn, v_t)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, -1, h, w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.view(b, self.heads, -1, h * w).softmax(dim=-2)
        k = k.view(b, self.heads, -1, h * w).softmax(dim=-1)
        v = v.view(b, self.heads, -1, h * w)
        q = q * self.scale
        v_t = v.permute(0, 1, 3, 2)
        context = torch.matmul(k, v_t)
        context_t = context.permute(0, 1, 3, 2)
        out = torch.matmul(context_t, q)
        out = out.contiguous().view(b, -1, h, w)
        return self.to_out(out)


class Unet(nn.Module):
    """U-Net. num_classes를 주면 class-conditional(label_emb 사용), None이면
    완전한 unconditional 모델이 되어 라벨을 다룰 방법 자체가 없다."""

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4), channels=1, resnet_block_groups=4, num_classes=None):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        # num_classes가 없으면 label_emb 자체를 만들지 않는다 (unconditional 모델)
        self.label_emb = nn.Embedding(num_classes, time_dim) if exists(num_classes) else None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time, y=None):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        if exists(self.label_emb):
            t = t + self.label_emb(y)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class Classifier(nn.Module):
    """일반 분류기 (실험 1, 2, 그리고 최종 평가에 공용으로 사용)"""

    def __init__(self, num_classes=3, img_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # 14 -> 7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 7 -> 4
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.features(x)
        f = self.pool(f)
        f = torch.flatten(f, 1)
        return self.classifier(f)
