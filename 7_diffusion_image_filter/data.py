"""데이터셋 로딩/불균형 오버샘플링 관련 클래스와 헬퍼."""

import os
import random
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def crop_group(path):
    return re.sub(r"_k\d+$", "", os.path.splitext(os.path.basename(path))[0])


def build_imbalanced_group_split(dataset, train_counts, test_counts, seed=0, group_fn=crop_group):
    """ 한 장을 3조각 낸 원본 단위로 묶어서 분할"""
    rng = random.Random(seed)
    class_groups = defaultdict(lambda: defaultdict(list))
    for idx, (path, label) in enumerate(dataset.samples):
        class_groups[label][group_fn(path)].append(idx)

    train_indices, test_indices = [], []
    for cls, n_train in train_counts.items():
        groups = [sorted(class_groups[cls][g]) for g in sorted(class_groups[cls])]
        rng.shuffle(groups)
        for g in groups:
            rng.shuffle(g)
        it = iter(groups)
        for out, n in ((test_indices, test_counts[cls]), (train_indices, n_train)):
            taken = 0
            while taken < n:
                g = next(it, None)
                if g is None:
                    raise ValueError(f"class {cls}: 원본이 부족해서 train {n_train} / test {test_counts[cls]}장을 채울 수 없습니다")
                out.extend(g[:n - taken])
                taken += min(len(g), n - taken)

    return train_indices, test_indices


class CachedDataset(torch.utils.data.Dataset):
    """디스크 JPEG 디코딩+resize를 한 번만 수행하고 텐서로 메모리에 캐싱해서,
    매 epoch마다 같은 이미지를 반복 디코딩하는 비용을 없앤다."""

    def __init__(self, dataset, batch_size=64):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        self.x = torch.cat(xs, dim=0)
        self.y = torch.cat(ys, dim=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CachedSubset(torch.utils.data.Dataset):
    """CachedDataset에서 일부 인덱스만 잘라낸 view. seed마다 train/test 분할을 바꿀 때
    이미지를 다시 디코딩하지 않도록, 한 번 캐싱한 전체 텐서에서 인덱스로만 잘라낸다."""

    def __init__(self, cached, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        self.x = cached.x[idx]
        self.y = cached.y[idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BalancedDataset(torch.utils.data.Dataset):
    """실제 데이터 + diffusion이 생성한 소수 클래스 fake 데이터를 합쳐 균형 데이터셋을 만든다."""

    def __init__(self, real_dataset, fake_data):
        self.real_dataset = real_dataset
        self.fake_x = []
        self.fake_y = []
        for cls, imgs in fake_data.items():
            self.fake_x.append(imgs)
            self.fake_y.append(torch.full((imgs.size(0),), cls, dtype=torch.long))
        self.fake_x = torch.cat(self.fake_x, dim=0) if self.fake_x else torch.empty(0)
        self.fake_y = torch.cat(self.fake_y, dim=0) if self.fake_y else torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.real_dataset) + len(self.fake_x)

    def __getitem__(self, idx):
        if idx < len(self.real_dataset):
            return self.real_dataset[idx]
        j = idx - len(self.real_dataset)
        return self.fake_x[j].clamp(-1, 1), self.fake_y[j]


@torch.no_grad()
def generate_balanced_fake(ddpm, diffusion, device, steps, target_per_class=800, minority_classes=(1,),
                           batch_size=100):
    """오버샘플링에 쓰인다: 소수 클래스 fake 풀을 미리 생성한다."""
    fake_data = {}
    for cls in minority_classes:
        n_needed = target_per_class
        n_done = 0
        imgs = []
        while n_needed > 0:
            n = min(batch_size, n_needed)
            y = torch.full((n,), cls, dtype=torch.long, device=device)
            x_fake = ddpm.sample_ddpm_respaced(diffusion, y, device, steps=steps)
            imgs.append(x_fake.cpu())
            n_needed -= n
            n_done += n
            print(f"  class {cls}: {n_done}/{target_per_class}장 생성")
        fake_data[cls] = torch.cat(imgs, dim=0)
        print(f"  class {cls}: {fake_data[cls].size(0)}장 생성 완료")
    return fake_data


def draw_from_pool(pool, real_y, device):
    """오버샘플링에 쓰인다: fake 풀에서 부족한 소수 클래스 수만큼 꺼낸다."""
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


@torch.no_grad()
def generate_fill_batch(ddpm, model, real_y, device, minority_classes=(1,), respaced_steps=None):
    """joint에 쓰인다: 매 배치 부족한 소수 클래스 fake를 새로 생성한다."""
    n0 = (real_y == 0).sum().item()
    ys = []
    for cls in minority_classes:
        n_missing = max(0, n0 - (real_y == cls).sum().item())
        if n_missing > 0:
            ys.append(torch.full((n_missing,), cls, dtype=torch.long, device=device))
    if not ys:
        return None, None
    y_fake = torch.cat(ys)
    if respaced_steps is not None:
        x_fake = ddpm.sample_ddpm_respaced(model, y_fake, device, steps=respaced_steps)
    else:
        x_fake = ddpm.sample_ddpm(model, y_fake, device)
    return x_fake, y_fake


class FakeQualityFilter:
    """소수 클래스 fake 중 실제 이미지에서 깨진 이미지를 걸러낸다."""

    def __init__(self, real_x, real_y, classes, far_q=1.0):
        self.ref, self.t_far = {}, {}
        for c in classes:
            f = self.feat(real_x[real_y == c])
            d = torch.cdist(f, f)
            d.fill_diagonal_(float("inf"))
            self.ref[c] = f
            self.t_far[c] = torch.quantile(d.min(1).values, far_q)

    @staticmethod
    def feat(x):
        return F.avg_pool2d(x, 4).flatten(1)

    def dist(self, x, c):
        return torch.cdist(self.feat(x), self.ref[c]).min(1).values

    def accept(self, x, c):
        return self.dist(x, c) <= self.t_far[c]
