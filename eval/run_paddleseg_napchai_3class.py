"""
PaddleSeg training on 3-class napchai dataset.
Classes: 0=background, 1=scratch, 2=mc_deform, 3=ring_fracture
"""
import json
import time
import numpy as np
import cv2
from pathlib import Path

import paddle
from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
from paddleseg.transforms import Resize, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomScaleAspect
from paddleseg.datasets import Dataset
from paddleseg.core import train
import paddle.nn as nn
import paddle.nn.functional as F

DATA_DIR = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\paddleseg_napchai_3class")
RESULTS_DIR = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\results\paddleseg_napchai_3class")

NUM_CLASSES = 4  # bg + scratch + mc_deform + ring_fracture
CLASS_NAMES = {0: "background", 1: "scratch", 2: "mc_deform", 3: "ring_fracture"}
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
EPOCHS = 15
LR = 0.02
SEED = 42


class WeightedCELoss(nn.Layer):
    def __init__(self, weight):
        super().__init__()
        self.weight = paddle.to_tensor(weight, dtype='float32')

    def forward(self, logits, labels):
        loss = F.cross_entropy(logits, labels.squeeze(1).astype('int64'),
                               weight=self.weight, axis=1)
        return paddle.unsqueeze(loss, axis=0)


class DiceLoss(nn.Layer):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        labels = labels.squeeze(1).astype('int64')
        probs = F.softmax(logits, axis=1)
        num_classes = probs.shape[1]
        one_hot = F.one_hot(labels, num_classes).transpose([0, 3, 1, 2]).astype('float32')
        dice = 0.0
        count = 0
        for c in range(1, num_classes):
            p = probs[:, c]
            g = one_hot[:, c]
            intersection = paddle.sum(p * g)
            union = paddle.sum(p) + paddle.sum(g)
            dice += (2.0 * intersection + self.smooth) / (union + self.smooth)
            count += 1
        if count == 0:
            return paddle.zeros([1])
        loss = 1.0 - dice / count
        return paddle.unsqueeze(loss, axis=0)


class CombinedLoss(nn.Layer):
    def __init__(self, weight, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = WeightedCELoss(weight)
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, labels):
        return self.ce_weight * self.ce(logits, labels) + self.dice_weight * self.dice(logits, labels)


def compute_class_weights():
    lines = (DATA_DIR / "train.txt").read_text("utf-8").strip().split("\n")
    pixel_counts = np.zeros(NUM_CLASSES)

    for line in lines[:60]:
        _, mask_rel = line.split(" ")
        mask = cv2.imread(str(DATA_DIR / mask_rel), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        for c in range(NUM_CLASSES):
            pixel_counts[c] += np.sum(mask == c)

    total = pixel_counts.sum()
    freq = pixel_counts / (total + 1e-10)
    print(f"  Pixel freq: {dict(zip(CLASS_NAMES.values(), [f'{f:.4f}' for f in freq]))}")

    weights = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        if freq[c] > 0:
            weights[c] = 1.0 / (freq[c] * NUM_CLASSES)
        else:
            weights[c] = 1.0

    weights[0] = min(weights[0], 0.3)
    for c in range(1, NUM_CLASSES):
        weights[c] = max(weights[c], 50.0)
        weights[c] = min(weights[c], 500.0)

    return weights.tolist()


def manual_evaluate(model, dataset):
    model.eval()
    intersect = np.zeros(NUM_CLASSES)
    pred_total = np.zeros(NUM_CLASSES)
    label_total = np.zeros(NUM_CLASSES)

    for i in range(len(dataset)):
        data = dataset[i]
        img = paddle.to_tensor(data['img']).unsqueeze(0)
        label = data['label'].astype('int64')
        while label.ndim > 2:
            label = label[0]

        with paddle.no_grad():
            logits = model(img)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = paddle.argmax(logits, axis=1).squeeze().numpy().astype('int64')

        if pred.shape != label.shape:
            pred = cv2.resize(pred.astype('float32'), (label.shape[1], label.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype('int64')

        for c in range(NUM_CLASSES):
            p = (pred == c)
            l = (label == c)
            intersect[c] += np.sum(p & l)
            pred_total[c] += np.sum(p)
            label_total[c] += np.sum(l)

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(dataset)}")

    iou_per_class = intersect / (pred_total + label_total - intersect + 1e-10)
    acc = np.sum(intersect) / (np.sum(label_total) + 1e-10)
    miou = np.mean(iou_per_class[label_total > 0])
    return miou, acc, iou_per_class


def generate_predictions(model, results_dir):
    COLORS = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    lines = (DATA_DIR / "val.txt").read_text("utf-8").strip().split("\n")

    for line in lines[:15]:
        img_rel, mask_rel = line.split(" ")
        img = cv2.imread(str(DATA_DIR / img_rel))
        mask_gt = cv2.imread(str(DATA_DIR / mask_rel), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_resized = cv2.resize(img, IMG_SIZE)
        img_norm = (img_resized.astype('float32') / 255.0 - 0.5) / 0.5
        tensor = paddle.to_tensor(img_norm.transpose(2, 0, 1)[None].astype('float32'))

        with paddle.no_grad():
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = paddle.argmax(logits, axis=1).squeeze().numpy().astype('uint8')

        pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        ov_gt = img.copy()
        ov_pred = img.copy()
        for c in range(1, NUM_CLASSES):
            color = COLORS[c]
            if mask_gt is not None:
                m = (mask_gt == c)
                if np.any(m):
                    for ch in range(3):
                        ov_gt[:, :, ch] = np.where(m, ov_gt[:, :, ch] * 0.5 + color[ch] * 0.5, ov_gt[:, :, ch])
            m = (pred_full == c)
            if np.any(m):
                for ch in range(3):
                    ov_pred[:, :, ch] = np.where(m, ov_pred[:, :, ch] * 0.5 + color[ch] * 0.5, ov_pred[:, :, ch])

        disp_h = 400
        scale = disp_h / h
        disp_w = int(w * scale)
        combined = np.hstack([
            cv2.resize(img, (disp_w, disp_h)),
            cv2.resize(ov_gt, (disp_w, disp_h)),
            cv2.resize(ov_pred, (disp_w, disp_h)),
        ])
        cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, 'Ground Truth', (disp_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, 'Prediction', (2 * disp_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fname = Path(img_rel).stem + '_compare.png'
        cv2.imwrite(str(pred_dir / fname), combined)

    print(f"  Saved {min(15, len(lines))} prediction images to {pred_dir}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== PaddleSeg: Napchai 3-class (WeightedCE + Dice, 50ep) ===\n")
    paddle.set_device('cpu')

    print("Computing class weights...")
    class_weights = compute_class_weights()
    print(f"  Weights: {[f'{w:.1f}' for w in class_weights]}")

    train_transforms = [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomScaleAspect(min_scale=0.75, aspect_ratio=0.33),
        Resize(target_size=IMG_SIZE),
        Normalize(),
    ]
    val_transforms = [Resize(target_size=IMG_SIZE), Normalize()]

    train_dataset = Dataset(
        dataset_root=str(DATA_DIR),
        train_path=str(DATA_DIR / "train.txt"),
        num_classes=NUM_CLASSES,
        transforms=train_transforms,
        mode="train",
        separator=" ",
    )
    val_dataset = Dataset(
        dataset_root=str(DATA_DIR),
        val_path=str(DATA_DIR / "val.txt"),
        num_classes=NUM_CLASSES,
        transforms=val_transforms,
        mode="val",
        separator=" ",
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    backbone = STDC1(pretrained=None)
    model = PPLiteSeg(
        num_classes=NUM_CLASSES,
        backbone=backbone,
        arm_out_chs=[32, 64, 128],
        seg_head_inter_chs=[32, 64, 64],
    )

    n_train = len(train_dataset)
    iters = EPOCHS * n_train // BATCH_SIZE

    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=LR, decay_steps=iters, end_lr=1e-5, power=0.9,
    )
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr_scheduler, parameters=model.parameters(),
        momentum=0.9, weight_decay=4e-5,
    )

    combined_loss = CombinedLoss(weight=class_weights, ce_weight=1.0, dice_weight=2.0)
    losses = {'types': [combined_loss] * 3, 'coef': [1.0, 0.4, 0.4]}

    print(f"\nTraining {EPOCHS} epochs, {iters} iters (WeightedCE + Dice, CPU)...")
    start = time.time()

    train(
        model=model, train_dataset=train_dataset, val_dataset=None,
        optimizer=optimizer, save_dir=str(RESULTS_DIR / "checkpoints"),
        iters=iters, batch_size=BATCH_SIZE, save_interval=iters + 1,
        log_iters=100, num_workers=0, losses=losses, use_vdl=False,
    )

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed / 60:.1f} min")

    print("\nEvaluating...")
    miou, acc, iou_per_class = manual_evaluate(model, val_dataset)

    print("\n" + "=" * 50)
    print("RESULTS — PaddleSeg Napchai 3-class")
    print("=" * 50)
    print(f"  mIoU:     {miou:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    for c in range(NUM_CLASSES):
        print(f"  IoU [{c}] {CLASS_NAMES[c]:15s}: {iou_per_class[c]:.4f}")

    metrics = {
        "mIoU": float(miou), "Acc": float(acc),
        "class_iou": {CLASS_NAMES[c]: float(iou_per_class[c]) for c in range(NUM_CLASSES)},
        "epochs": EPOCHS, "lr": LR, "loss": "WeightedCE + Dice (1:2)",
        "class_weights": class_weights,
        "num_classes": NUM_CLASSES,
        "training_time_min": elapsed / 60,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nGenerating prediction images...")
    generate_predictions(model, RESULTS_DIR)
    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
