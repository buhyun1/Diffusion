"""일반 유틸: 시드 고정, 디바이스 설정, 평가지표 계산 등
모델 구조나 데이터셋에 종속되지 않는 헬퍼들을 모아둔다."""

import copy
import csv
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class EMA:
    """모델 가중치의 지수이동평균(EMA) 사본. 학습(gradient)은 원본 모델로 하고,
    샘플 생성은 매 step 조금씩만 따라오는 EMA 사본으로 해서 가중치 요동에 의한
    샘플 품질 저하를 줄인다 (DDPM 계열의 표준 관행).

    초반에는 decay를 (1+n)/(10+n)으로 낮춰서, 랜덤 초기값이 EMA에 오래 남지 않게 한다."""

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.lerp_(p.detach(), 1 - d)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = list(range(10))  # 실험별 반복 seed. seed마다 train/test 분할과 학습 무작위성이 모두 바뀐다 (논문: 10회 반복 평균)


def evaluate_classifier(model, dataloader, device, num_classes, class_names=None):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y)

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    print("===== Evaluation =====")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision(macro): {precision_macro:.4f}")
    print(f"Recall(macro)   : {recall_macro:.4f}")
    print(f"F1(macro)       : {f1_macro:.4f}")
    print("\nPer-class:")

    if class_names is None:
        class_names = [str(i) for i in range(len(precision))]
    for i, cls in enumerate(class_names):
        print(f"Class {cls} | P:{precision[i]:.3f} R:{recall[i]:.3f} F1:{f1[i]:.3f} N:{support[i]}")

    print("\nConfusion Matrix:\n", cm)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "support": support, "confusion_matrix": cm}


def metrics_to_rows(metrics, seed, method):
    rows = []
    for cls in range(len(metrics["f1"])):
        rows.append({"method": method, "seed": seed, "class": cls,
                     "precision": metrics["precision"][cls], "recall": metrics["recall"][cls],
                     "f1": metrics["f1"][cls], "support": metrics["support"][cls],
                     "accuracy": metrics["accuracy"]})
    rows.append({"method": method, "seed": seed, "class": "macro",
                 "precision": metrics["precision"].mean(), "recall": metrics["recall"].mean(),
                 "f1": metrics["f1"].mean(), "support": metrics["support"].sum(),
                 "accuracy": metrics["accuracy"]})
    return rows


def append_result_rows(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
