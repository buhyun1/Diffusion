"""6_diffusion_step_change_experiment: Diffusion 오버샘플링에서 DDPM 스텝 수(T=22 · 30 · 100 · 300)에 따른 성능 비교 — 터미널 실행 스크립트.

exp_diffusion_step_N/exp_diffusion_step_N.ipynb와 같은 실험을 T별로 실행하고, 결과를 exp_diffusion_step_N/exp_diffusion_step_N.csv에 저장한다.
 - T마다 학습/샘플링 모두 그 T로 한다 (diffusion 700 epoch 학습 → 소수 클래스 720장씩 fake 풀 → 분류기 1200 step).
 - 결과 CSV는 지우지 않고 뒤에 이어서 쓰며, 이미 결과가 있는 seed는 건너뛴다. 체크포인트가 있으면 학습/풀 생성도 건너뛴다.
실행:  python -u exp_diffusion_step.py --bg [--T 22 30 100 300] [--seeds 0 1 2] [--gpu 0]
       --bg를 주면 프로세스 하나를 백그라운드로 띄우고 바로 끝난다. T는 그 안에서 하나씩 순서대로 돈다 (로그: logs/exp_diffusion_step.log).
       --bg 없이 실행하면 지정한 T를 순서대로 현재 프로세스에서 돈다.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--T", type=int, nargs="+", default=[22, 30, 100, 300])
p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
p.add_argument("--gpu", nargs="+", default=None)
p.add_argument("--bg", action="store_true")
p.add_argument("--diffusion-epochs", type=int, default=700)
p.add_argument("--eval-epochs", type=int, default=50)
args = p.parse_args()

HERE = Path(__file__).resolve().parent


def gpu_order():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]).decode()
    rows = [tuple(int(v) for v in l.split(",")) for l in out.strip().splitlines()]
    return [str(r[0]) for r in sorted(rows, key=lambda r: (r[1], r[2]))]


if args.bg:
    gpu = (args.gpu or gpu_order())[0]
    (HERE / "logs").mkdir(exist_ok=True)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--T", *map(str, args.T), "--gpu", gpu,
           "--seeds", *map(str, args.seeds), "--diffusion-epochs", str(args.diffusion_epochs),
           "--eval-epochs", str(args.eval_epochs)]
    log = open(HERE / "logs" / "exp_diffusion_step.log", "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=(os.name != "nt"))
    print(f"T={args.T} 순서대로 실행: GPU {gpu}, pid {proc.pid}, 로그 logs/exp_diffusion_step.log")
    sys.exit(0)

if args.gpu is None:
    args.gpu = gpu_order()[:1]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
print("사용 GPU:", args.gpu[0])

sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T_
from torchvision.datasets import ImageFolder

from models import Unet, Classifier
from utils import set_seed, DEVICE, EMA, evaluate_classifier, metrics_to_rows, append_result_rows
from diffusion import GaussianDiffusion
from data import build_imbalanced_split, CachedDataset, CachedSubset, generate_balanced_fake, draw_from_pool

image_size, channels, batch_size, num_classes = 64, 3, 40, 3
data_root = os.environ.get("DATA_ROOT", str(HERE.parent / "data" / "256"))
CLASS_NAMES = ["Normal", "Underfill_50FR", "Underfill_Fan"]
train_counts = {0: 800, 1: 80, 2: 80}
test_counts = {0: 100, 1: 100, 2: 100}
EMA_DECAY = 0.999
POOL_PER_CLASS = 800 - 80
POOL_GEN_BATCH = 360

transform = T_.Compose([T_.Resize((image_size, image_size)), T_.ToTensor(), T_.Lambda(lambda t: t * 2 - 1)])
full_dataset = ImageFolder(root=data_root, transform=transform)
assert full_dataset.classes == CLASS_NAMES, full_dataset.classes
full_cached = CachedDataset(full_dataset)

train_dataset = test_dataset = train_loader = test_loader = None


def set_split(seed):
    global train_dataset, test_dataset, train_loader, test_loader
    tr, te = build_imbalanced_split(full_dataset, train_counts, test_counts, seed=seed)
    train_dataset, test_dataset = CachedSubset(full_cached, tr), CachedSubset(full_cached, te)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(DEVICE == "cuda"))


def train_diffusion(seed, ddpm, timesteps):
    set_seed(seed)
    diffusion = Unet(dim=32, channels=channels, num_classes=num_classes).to(DEVICE)
    opt_g = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()
    ema = EMA(diffusion, decay=EMA_DECAY)
    t0 = time.time()
    print(f"[Diffusion seed{seed}] 학습 시작: device={DEVICE}, {len(train_loader)} batches/epoch, {args.diffusion_epochs} epochs, T={timesteps}", flush=True)
    for epoch in range(args.diffusion_epochs):
        diffusion.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            t = torch.randint(0, timesteps, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)
            loss = loss_fn(diffusion(ddpm.q_sample(x, t, noise), t, y), noise)
            opt_g.zero_grad()
            loss.backward()
            opt_g.step()
            ema.update(diffusion)
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.diffusion_epochs - 1:
            print(f"[Diffusion seed{seed}] Epoch {epoch+1}/{args.diffusion_epochs} Diff={np.mean(losses):.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    return diffusion, ema


def load_or_train_diffusion(seed, ddpm, timesteps, path, ema_path):
    if os.path.exists(path) and os.path.exists(ema_path):
        model = Unet(dim=32, channels=channels, num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(ema_path, map_location=DEVICE))
        print(f"[Diffusion seed{seed}] 학습 건너뜀: {ema_path} 로드", flush=True)
        return model
    diffusion, ema = train_diffusion(seed, ddpm, timesteps)
    torch.save(diffusion.state_dict(), path)
    torch.save(ema.model.state_dict(), ema_path)
    return ema.model


def load_or_build_pool(seed, ddpm, model, path):
    if os.path.exists(path):
        pool = torch.load(path, map_location="cpu")
        print(f"[Pool seed{seed}] 생성 건너뜀: {path} 로드", flush=True)
    else:
        set_seed(seed)
        t0 = time.time()
        pool = generate_balanced_fake(ddpm, model, DEVICE, target_per_class=POOL_PER_CLASS, minority_classes=(1, 2),
                                      batch_size=POOL_GEN_BATCH)
        torch.save(pool, path)
        print(f"[Pool seed{seed}] 클래스당 {POOL_PER_CLASS}장 생성 완료: {time.time()-t0:.1f}초 → {path}", flush=True)
    return {cls: x.to(DEVICE) for cls, x in pool.items()}


def train_classifier_pool(seed, pool, total_steps):
    set_seed(seed)
    clf = Classifier(num_classes=num_classes, img_channels=channels).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    clf.train()
    it, losses, t0 = iter(train_loader), [], time.time()
    for step in range(total_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        xf, yf = draw_from_pool(pool, y, DEVICE)
        if xf is not None:
            x, y = torch.cat([x, xf]), torch.cat([y, yf])
        loss = loss_fn(clf(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % max(1, total_steps // 10) == 0 or step == total_steps - 1:
            print(f"[Classifier-pool seed{seed}] step {step+1}/{total_steps} Cls={np.mean(losses[-50:]):.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    return clf


for T in args.T:
    name = f"exp_diffusion_step_{T}"
    out_dir = HERE / name
    out_dir.mkdir(exist_ok=True)
    csv = str(out_dir / f"{name}.csv")
    ddpm = GaussianDiffusion(timesteps=T, image_size=image_size, channels=channels)
    done = set(pd.read_csv(csv)["seed"]) if os.path.exists(csv) else set()

    for seed in args.seeds:
        if seed in done:
            print(f"[{name} seed{seed}] 결과 있음 → 건너뜀", flush=True)
            continue
        set_split(seed)
        print(f"\n===== {name} (DDPM T={T}) | Seed {seed} =====", flush=True)
        print(f"[{name} seed{seed}] 분할 완료: train {len(train_dataset)}장, test {len(test_dataset)}장", flush=True)
        t0 = time.time()
        model = load_or_train_diffusion(seed, ddpm, T, str(out_dir / f"{name}_diffusion_seed{seed}.pt"),
                                        str(out_dir / f"{name}_diffusion_ema_seed{seed}.pt"))
        pool = load_or_build_pool(seed, ddpm, model, str(out_dir / f"{name}_fake_pool_seed{seed}.pt"))
        clf = train_classifier_pool(seed, pool, len(train_loader) * args.eval_epochs)
        metrics = evaluate_classifier(clf, test_loader, DEVICE, num_classes, CLASS_NAMES)
        append_result_rows(metrics_to_rows(metrics, seed, name), csv)
        print(f"[{name} seed{seed}] 총 소요 시간: {time.time()-t0:.1f}초", flush=True)
        del pool, model, clf
        torch.cuda.empty_cache()
