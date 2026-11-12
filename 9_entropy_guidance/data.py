import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLASS_DIRS = ["Normal", "Underfill_50FR", "Underfill_Fan"]
CLASS_NAMES = CLASS_DIRS
NUM_CLASSES = len(CLASS_DIRS)
TRAIN_COUNTS = [800, 80, 80]
TEST_COUNTS = [100, 100, 100]
IMAGE_SIZE = 64


def crop_group(path):
    return re.sub(r"_k\d+$", "", Path(path).stem)


def load_am(root):
    root = Path(root)
    cache = root.parent / f"cache_am_3split_{IMAGE_SIZE}.pt"
    if cache.exists():
        return torch.load(cache)
    xs, ys, groups = [], [], []
    for c, name in enumerate(CLASS_DIRS):
        for f in sorted((root / name).glob("*.png")):
            im = Image.open(f).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            xs.append(np.asarray(im))
            ys.append(c)
            groups.append(crop_group(f))
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    y = torch.tensor(ys)
    torch.save((x, y, groups), cache)
    return x, y, groups


def build_imbalanced_group_split(y, groups, train_counts, test_counts, seed=0):
    rng = random.Random(seed)
    class_groups = defaultdict(lambda: defaultdict(list))
    for idx, (c, g) in enumerate(zip(y.tolist(), groups)):
        class_groups[c][g].append(idx)

    tr, te = [], []
    for c, (n_train, n_test) in enumerate(zip(train_counts, test_counts)):
        glist = [sorted(class_groups[c][g]) for g in sorted(class_groups[c])]
        rng.shuffle(glist)
        for g in glist:
            rng.shuffle(g)
        it = iter(glist)
        for out, n in ((te, n_test), (tr, n_train)):
            taken = 0
            while taken < n:
                g = next(it, None)
                if g is None:
                    raise ValueError(f"class {c}: 원본이 부족해서 train {n_train} / test {n_test}장을 채울 수 없습니다")
                out.extend(g[:n - taken])
                taken += min(len(g), n - taken)
    return torch.tensor(tr), torch.tensor(te)


def draw_from_pool(pool, real_y, device):
    n0 = (real_y == 0).sum().item()
    if n0 == 0:
        return None, None

    fake_x_list, fake_y_list = [], []
    for cls, imgs in pool.items():
        n_real = (real_y == cls).sum().item()
        n_missing = max(0, n0 - n_real)

        if n_missing > 0:
            idx = torch.randint(0, imgs.size(0), (n_missing,), device=device)
            fake_x_list.append(imgs[idx])
            fake_y_list.append(torch.full((n_missing,), cls, dtype=torch.long, device=device))

    if len(fake_x_list) == 0:
        return None, None

    return torch.cat(fake_x_list, dim=0), torch.cat(fake_y_list, dim=0)
