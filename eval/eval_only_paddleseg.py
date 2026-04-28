"""
Evaluate-only script: loads trained PP-LiteSeg checkpoint and runs evaluation.
No retraining needed.
"""
import json
import numpy as np
import cv2
from pathlib import Path

import paddle
from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
from paddleseg.transforms import Resize, Normalize
from paddleseg.datasets import Dataset

from eval_config import SPLIT_DIR, RESULTS_DIR as _RESULTS_BASE, NUM_CLASSES, SRC_DIR, CLASS_NAMES

RESULTS_DIR = _RESULTS_BASE / "paddleseg"
IMG_SIZE = (512, 512)
CHECKPOINT = RESULTS_DIR / "checkpoints" / "iter_2505" / "model.pdparams"


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
            print(f"  Evaluated {i+1}/{len(dataset)} samples")

    iou_per_class = intersect / (pred_total + label_total - intersect + 1e-10)
    acc = np.sum(intersect) / (np.sum(label_total) + 1e-10)
    miou = np.mean(iou_per_class[label_total > 0])
    return miou, acc, iou_per_class


def main():
    paddle.set_device('cpu')
    print("=== PaddleSeg PP-LiteSeg — Eval Only ===\n")

    ps_dir = RESULTS_DIR.parent / "paddleseg_data"
    transforms = [Resize(target_size=IMG_SIZE), Normalize()]

    val_dataset = Dataset(
        dataset_root=str(SRC_DIR),
        val_path=str(ps_dir / "val.txt"),
        num_classes=NUM_CLASSES,
        transforms=transforms,
        mode="val",
        separator=" ",
    )
    print(f"Val samples: {len(val_dataset)}")

    backbone = STDC1(pretrained=None)
    model = PPLiteSeg(
        num_classes=NUM_CLASSES,
        backbone=backbone,
        arm_out_chs=[32, 64, 128],
        seg_head_inter_chs=[32, 64, 64],
    )

    print(f"Loading checkpoint: {CHECKPOINT}")
    state = paddle.load(str(CHECKPOINT))
    model.set_state_dict(state)

    print("\nEvaluating...")
    miou, acc, iou_per_class = manual_evaluate(model, val_dataset)

    print("\n" + "=" * 50)
    print("RESULTS — PaddleSeg PP-LiteSeg (30 epochs, CPU)")
    print("=" * 50)
    print(f"  mIoU:     {miou:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print()
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"class_{c}")
        print(f"  IoU [{c}] {name:15s}: {iou_per_class[c]:.4f}")

    metrics = {
        "mIoU": float(miou),
        "Acc": float(acc),
        "class_iou": {CLASS_NAMES.get(c, f"class_{c}"): float(iou_per_class[c]) for c in range(NUM_CLASSES)},
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nMetrics saved to {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
