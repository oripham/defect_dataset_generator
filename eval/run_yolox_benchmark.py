"""
YOLOX fine-tuning for defect detection (Apache 2.0, commercial-safe).
Converts YOLO-seg dataset → COCO format, fine-tunes YOLOX-S, evaluates.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.utils import postprocess

ROOT = Path(r"V:\HondaPlus\defect_dataset_generator")
BATCH_ROOT = ROOT / "batch_output"
RESULTS_ROOT = ROOT / "eval" / "output" / "results"
YOLOX_WEIGHTS = Path(r"V:\HondaPlus\YOLOX\yolox_s.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
EPOCHS = 50
BATCH_SIZE = 4
IMGSZ = 640
CONF_THRESH = 0.3
NMS_THRESH = 0.45


def yolo_seg_to_coco(dataset_yaml_path: Path, exclude_classes: list[str] | None = None) -> tuple[dict, dict, list[str]]:
    import yaml
    with open(dataset_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_root = dataset_yaml_path.parent
    all_class_names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]

    exclude_ids = set()
    if exclude_classes:
        for i, name in enumerate(all_class_names):
            if name in exclude_classes:
                exclude_ids.add(i)
        print(f"  Excluding classes: {exclude_classes} (ids={exclude_ids})")

    kept_names = [n for i, n in enumerate(all_class_names) if i not in exclude_ids]
    old_to_new = {}
    new_id = 0
    for i, name in enumerate(all_class_names):
        if i not in exclude_ids:
            old_to_new[i] = new_id
            new_id += 1

    coco_categories = [{"id": i, "name": name} for i, name in enumerate(kept_names)]

    results = {}
    for split in ["train", "val"]:
        img_dir = ds_root / cfg[split]
        label_dir = ds_root / cfg[split].replace("images", "labels")
        images_list, annotations_list = [], []
        ann_id = 0

        for img_id, img_path in enumerate(sorted(img_dir.glob("*.png"))):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                images_list.append({"id": img_id, "file_name": str(img_path), "width": w, "height": h})
                continue

            img_anns = []
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id in exclude_ids:
                    continue
                coords = list(map(float, parts[1:]))
                xs, ys = coords[0::2], coords[1::2]
                x_min, y_min = min(xs) * w, min(ys) * h
                x_max, y_max = max(xs) * w, max(ys) * h
                bw, bh = x_max - x_min, y_max - y_min
                if bw < 1 or bh < 1:
                    continue
                img_anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": old_to_new[class_id],
                    "bbox": [x_min, y_min, bw, bh], "area": bw * bh, "iscrowd": 0,
                })
                ann_id += 1

            images_list.append({"id": img_id, "file_name": str(img_path), "width": w, "height": h})
            annotations_list.extend(img_anns)

        results[split] = {"images": images_list, "annotations": annotations_list, "categories": coco_categories}
        print(f"  {split}: {len(images_list)} images, {len(annotations_list)} annotations")

    return results["train"], results["val"], kept_names


class DetectionDataset(Dataset):
    def __init__(self, coco_dict: dict, imgsz: int = 640):
        self.images = coco_dict["images"]
        self.imgsz = imgsz
        self.img_to_anns: dict[int, list] = {}
        for ann in coco_dict["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img = cv2.imread(img_info["file_name"])
        h0, w0 = img.shape[:2]

        r = self.imgsz / max(h0, w0)
        img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)))
        rh, rw = img_resized.shape[:2]

        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[:rh, :rw] = img_resized

        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float()

        anns = self.img_to_anns.get(img_info["id"], [])
        targets = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) * r
            cy = (y + bh / 2) * r
            w_s = bw * r
            h_s = bh * r
            targets.append([ann["category_id"], cx, cy, w_s, h_s])

        if targets:
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            targets_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, targets_tensor, (h0, w0, r)


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    max_targets = max(b[1].shape[0] for b in batch)
    if max_targets == 0:
        max_targets = 1
    targets = torch.zeros((len(batch), max_targets, 5))
    for i, b in enumerate(batch):
        n = b[1].shape[0]
        if n > 0:
            targets[i, :n] = b[1]
    infos = [b[2] for b in batch]
    return imgs, targets, infos


def build_yolox_model(num_classes: int) -> YOLOX:
    depth, width = 0.33, 0.50  # YOLOX-S
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
    head = YOLOXHead(num_classes=num_classes, width=width, in_channels=in_channels)
    model = YOLOX(backbone, head)
    return model


def load_pretrained(model: YOLOX, weights_path: Path, num_classes: int):
    ckpt = torch.load(str(weights_path), map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    model_dict = model.state_dict()
    filtered = {}
    for k, v in ckpt.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v

    print(f"  Loaded {len(filtered)}/{len(model_dict)} params from pretrained (skipped mismatched head layers)")
    model_dict.update(filtered)
    model.load_state_dict(model_dict)


def evaluate(model, val_coco, class_names, device, imgsz=640):
    model.eval()
    all_detections = defaultdict(list)
    all_gt = defaultdict(list)
    num_classes = len(class_names)
    inference_times = []

    for img_info in val_coco["images"]:
        img = cv2.imread(img_info["file_name"])
        h0, w0 = img.shape[:2]
        r = imgsz / max(h0, w0)
        img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)))
        rh, rw = img_resized.shape[:2]
        padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        padded[:rh, :rw] = img_resized
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(tensor)
        preds = postprocess(outputs, num_classes=num_classes, conf_thre=CONF_THRESH, nms_thre=NMS_THRESH)
        inference_times.append((time.time() - t0) * 1000)

        if preds[0] is not None:
            det = preds[0].cpu().numpy()
            for d in det:
                x1, y1, x2, y2, obj_conf, cls_conf, cls_id = d
                all_detections[img_info["id"]].append({
                    "box": np.array([x1 / r, y1 / r, x2 / r, y2 / r]),
                    "score": float(obj_conf * cls_conf),
                    "label": int(cls_id),
                })

        anns = [a for a in val_coco["annotations"] if a["image_id"] == img_info["id"]]
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            all_gt[img_info["id"]].append({"box": np.array([x, y, x + bw, y + bh]), "label": ann["category_id"]})

    def iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-10)

    tp, fp, fn = 0, 0, 0
    per_class_tp, per_class_fp, per_class_fn = defaultdict(int), defaultdict(int), defaultdict(int)

    for img_id in set(list(all_gt.keys()) + list(all_detections.keys())):
        dets = sorted(all_detections.get(img_id, []), key=lambda x: -x["score"])
        gts = list(all_gt.get(img_id, []))
        matched = [False] * len(gts)

        for det in dets:
            best_iou, best_j = 0, -1
            for j, gt in enumerate(gts):
                if matched[j] or gt["label"] != det["label"]:
                    continue
                v = iou(det["box"], gt["box"])
                if v > best_iou:
                    best_iou = v; best_j = j
            if best_iou >= 0.5 and best_j >= 0:
                tp += 1; per_class_tp[det["label"]] += 1; matched[best_j] = True
            else:
                fp += 1; per_class_fp[det["label"]] += 1

        for j, m in enumerate(matched):
            if not m:
                fn += 1; per_class_fn[gts[j]["label"]] += 1

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    per_class = {}
    for i, name in enumerate(class_names):
        t, f_p, f_n = per_class_tp[i], per_class_fp[i], per_class_fn[i]
        p = t / (t + f_p + 1e-10); r = t / (t + f_n + 1e-10)
        per_class[name] = {"precision": p, "recall": r, "f1": 2*p*r/(p+r+1e-10), "tp": t, "fp": f_p, "fn": f_n}

    return {
        "precision@0.5": precision, "recall@0.5": recall, "f1@0.5": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "inference_ms_avg": float(np.mean(inference_times)),
        "per_class": per_class,
    }


def train_yolox(dataset_name: str, dataset_yaml: Path, exclude_classes: list[str] | None = None):
    print(f"\n{'='*60}")
    print(f"YOLOX-S Training: {dataset_name}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    results_dir = RESULTS_ROOT / "yolox_batch" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    marker = results_dir / "metrics.json"
    if marker.exists():
        print(f"SKIP — already done ({marker})")
        return

    print("\nConverting dataset...")
    train_coco, val_coco, class_names = yolo_seg_to_coco(dataset_yaml, exclude_classes=exclude_classes)
    num_classes = len(class_names)

    print(f"\nBuilding YOLOX-S (num_classes={num_classes})...")
    model = build_yolox_model(num_classes)
    load_pretrained(model, YOLOX_WEIGHTS, num_classes)
    model = model.to(DEVICE)

    train_dataset = DetectionDataset(train_coco, IMGSZ)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR * 0.1},
        {"params": model.head.parameters(), "lr": LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining {EPOCHS} epochs...")
    t0 = time.time()
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, targets, _ in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(imgs, targets=targets)
            loss = outputs["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_coco, class_names, DEVICE, IMGSZ)
            f1 = metrics["f1@0.5"]
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  P={metrics['precision@0.5']:.4f}  R={metrics['recall@0.5']:.4f}  F1={f1:.4f}", flush=True)

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), str(results_dir / "best.pth"))
            else:
                patience_counter += 5

            if patience_counter >= 20:
                print(f"  Early stopping at epoch {epoch}")
                break
        else:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min")

    print("\nFinal evaluation on best model...")
    model.load_state_dict(torch.load(str(results_dir / "best.pth"), map_location=DEVICE))
    final_metrics = evaluate(model, val_coco, class_names, DEVICE, IMGSZ)
    final_metrics["model"] = "YOLOX-S"
    final_metrics["epochs"] = EPOCHS
    final_metrics["training_time_min"] = elapsed / 60
    final_metrics["license"] = "Apache-2.0"

    marker.write_text(json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved → {marker}")
    print(json.dumps(final_metrics, indent=2, default=str))


def main():
    train_yolox("napchai", BATCH_ROOT / "yolo" / "napchai" / "dataset.yaml")
    train_yolox("mka", BATCH_ROOT / "yolo" / "mka" / "dataset.yaml")
    print("\nALL_DONE")


if __name__ == "__main__":
    main()
