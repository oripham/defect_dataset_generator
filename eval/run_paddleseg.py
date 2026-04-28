"""
PaddleSeg evaluation — PP-LiteSeg on B6/MKA defect dataset.
Supervised semantic segmentation (multi-class masks).
Runs on GPU.
"""
import os
import json
import time
import numpy as np
import cv2
from pathlib import Path

import paddle
from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
from paddleseg.transforms import Compose, Resize, Normalize
from paddleseg.datasets import Dataset
from paddleseg.core import train, evaluate, predict
import paddle.nn as nn
from paddleseg.cvlibs import Config

from eval_config import SPLIT_DIR, RESULTS_DIR as _RESULTS_BASE, NUM_CLASSES, SRC_DIR

RESULTS_DIR = _RESULTS_BASE / "paddleseg"


def manual_evaluate(model, dataset):
    """Fallback evaluation when PaddleSeg's evaluate() crashes."""
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

    iou_per_class = intersect / (pred_total + label_total - intersect + 1e-10)
    acc = np.sum(intersect) / (np.sum(label_total) + 1e-10)
    miou = np.mean(iou_per_class[label_total > 0])
    return (miou, acc, 0.0, iou_per_class.tolist(), [])
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
EPOCHS = 30
LR = 0.005


def prepare_paddleseg_files():
    """Convert split lists to PaddleSeg format (space-separated, relative to SRC_DIR)."""
    ps_dir = RESULTS_DIR.parent / "paddleseg_data"
    ps_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        src = SPLIT_DIR / f"{split}.txt"
        lines = src.read_text(encoding="utf-8").strip().split("\n")
        out_lines = []
        for line in lines:
            img_path, mask_path = line.split("\t")
            out_lines.append(f"{img_path} {mask_path}")
        (ps_dir / f"{split}.txt").write_text(
            "\n".join(out_lines) + "\n", encoding="utf-8"
        )

    return ps_dir


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== PaddleSeg PP-LiteSeg Evaluation ===\n")
    paddle.set_device('cpu')
    print(f"Device: CPU (Tạm thời do thiếu cuDNN 8)")

    ps_dir = prepare_paddleseg_files()

    transforms = [
        Resize(target_size=IMG_SIZE),
        Normalize(),
    ]

    train_dataset = Dataset(
        dataset_root=str(SRC_DIR),
        train_path=str(ps_dir / "train.txt"),
        num_classes=NUM_CLASSES,
        transforms=transforms,
        mode="train",
        separator=" ",
    )

    val_dataset = Dataset(
        dataset_root=str(SRC_DIR),
        val_path=str(ps_dir / "val.txt"),
        num_classes=NUM_CLASSES,
        transforms=transforms,
        mode="val",
        separator=" ",
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")

    backbone = STDC1(pretrained=None)
    model = PPLiteSeg(
        num_classes=NUM_CLASSES,
        backbone=backbone,
        arm_out_chs=[32, 64, 128],
        seg_head_inter_chs=[32, 64, 64],
    )

    base_lr = LR
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=base_lr,
        decay_steps=EPOCHS * len(train_dataset) // BATCH_SIZE,
        end_lr=0.0,
        power=0.9,
    )
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        momentum=0.9,
        weight_decay=4e-5,
    )

    print(f"\nTraining {EPOCHS} epochs (CPU)...")
    start = time.time()

    class MyCrossEntropyLoss(nn.Layer):
        def __init__(self):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(axis=1)
        def forward(self, logits, labels):
            loss = self.ce(logits, labels)
            return paddle.unsqueeze(loss, axis=0)

    # PPLiteSeg trả về 3 outputs (1 main, 2 aux), cần 3 hàm loss
    losses = {}
    losses['types'] = [MyCrossEntropyLoss()] * 3
    losses['coef'] = [1.0, 1.0, 1.0]

    iters = EPOCHS * len(train_dataset) // BATCH_SIZE

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        optimizer=optimizer,
        save_dir=str(RESULTS_DIR / "checkpoints"),
        iters=iters,
        batch_size=BATCH_SIZE,
        save_interval=iters + 1,
        log_iters=50,
        num_workers=0,
        losses=losses,
        use_vdl=False,
    )

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed/60:.1f} min")

    print("\nEvaluating...")
    try:
        results = evaluate(
            model=model,
            eval_dataset=val_dataset,
            num_workers=0,
        )
    except Exception as e:
        print(f"PaddleSeg evaluate() failed: {e}")
        print("Running manual evaluation instead...")
        results = manual_evaluate(model, val_dataset)

    print("\n" + "=" * 50)
    print("RESULTS — PaddleSeg PP-LiteSeg")
    print("=" * 50)
    if isinstance(results, dict):
        for k, v in results.items():
            print(f"  {k}: {v}")
    elif isinstance(results, (list, tuple)):
        metric_names = ["mIoU", "Acc", "Kappa", "Class_IoU", "Class_Acc"]
        for i, v in enumerate(results):
            name = metric_names[i] if i < len(metric_names) else f"metric_{i}"
            print(f"  {name}: {v}")

    metrics_out = {}
    if isinstance(results, (list, tuple)) and len(results) >= 2:
        metrics_out["mIoU"] = float(results[0]) if not isinstance(results[0], (dict, list, np.ndarray)) else results[0]
        metrics_out["Acc"] = float(results[1]) if not isinstance(results[1], (dict, list, np.ndarray)) else results[1]
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nMetrics saved to {RESULTS_DIR / 'metrics.json'}")

    print("\nGenerating prediction images for Validation set...")
    val_images = []
    with open(str(ps_dir / "val.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                img_path = line.strip().split(" ")[0]
                val_images.append(str(SRC_DIR / img_path))
    
    predict_out_dir = RESULTS_DIR / "predictions"
    predict_out_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = RESULTS_DIR / "checkpoints" / "best_model" / "model.pdparams"
    if best_model_path.exists():
        predict(
            model=model,
            model_path=str(best_model_path),
            transforms=transforms,
            image_list=val_images[:20],  # Lưu 20 ảnh demo thôi cho nhanh
            image_dir=str(SRC_DIR),
            save_dir=str(predict_out_dir),
        )
        print(f"\nPrediction overlays saved to {predict_out_dir}")
    else:
        print("\nSkipped prediction: best_model not found.")


if __name__ == "__main__":
    main()
