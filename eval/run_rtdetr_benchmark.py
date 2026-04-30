"""
RT-DETR fine-tuning for defect detection (Apache 2.0, commercial-safe).
Converts YOLO-seg dataset → COCO format, fine-tunes RT-DETR-R18, evaluates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

ROOT = Path(r"V:\HondaPlus\defect_dataset_generator")
BATCH_ROOT = ROOT / "batch_output"
RESULTS_ROOT = ROOT / "eval" / "output" / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "PekingU/rtdetr_r18vd_coco_o365"
LR = 1e-3
EPOCHS = 80
BATCH_SIZE = 4
IMGSZ = 640


def yolo_seg_to_coco(dataset_yaml_path: Path) -> tuple[dict, dict, list[str]]:
    """Convert YOLO-seg dataset to COCO format dicts for train/val."""
    import yaml

    with open(dataset_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_root = dataset_yaml_path.parent
    class_names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]

    coco_categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    results = {}
    for split in ["train", "val"]:
        img_dir = ds_root / cfg[split]
        label_dir = ds_root / cfg[split].replace("images", "labels")

        images_list = []
        annotations_list = []
        ann_id = 0

        for img_id, img_path in enumerate(sorted(img_dir.glob("*.png"))):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            images_list.append({
                "id": img_id,
                "file_name": str(img_path),
                "width": w,
                "height": h,
            })

            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]

                x_min = min(xs) * w
                y_min = min(ys) * h
                x_max = max(xs) * w
                y_max = max(ys) * h
                bw = x_max - x_min
                bh = y_max - y_min

                if bw < 1 or bh < 1:
                    continue

                annotations_list.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

        results[split] = {
            "images": images_list,
            "annotations": annotations_list,
            "categories": coco_categories,
        }
        print(f"  {split}: {len(images_list)} images, {len(annotations_list)} annotations")

    return results["train"], results["val"], class_names


class CocoDetectionDataset(Dataset):
    def __init__(self, coco_dict: dict, processor: RTDetrImageProcessor, class_names: list[str], imgsz: int = 640):
        self.images = coco_dict["images"]
        self.processor = processor
        self.imgsz = imgsz

        self.img_to_anns: dict[int, list] = {}
        for ann in coco_dict["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.id2label = {i: name for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img = cv2.imread(img_info["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = self.img_to_anns.get(img_info["id"], [])
        w, h = img_info["width"], img_info["height"]

        coco_anns = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            coco_anns.append({
                "bbox": [x, y, bw, bh],
                "category_id": ann["category_id"],
                "area": bw * bh,
                "iscrowd": 0,
            })

        target = {
            "image_id": img_info["id"],
            "annotations": coco_anns,
        }

        encoding = self.processor(
            images=img,
            annotations=[target],
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels_out = encoding["labels"][0]

        return pixel_values, labels_out


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return pixel_values, labels


def evaluate_coco(model, processor, val_coco, class_names, device, imgsz=640):
    """Simple mAP evaluation at IoU=0.5."""
    from collections import defaultdict

    model.eval()
    all_detections = defaultdict(list)
    all_gt = defaultdict(list)

    for img_info in val_coco["images"]:
        img = cv2.imread(img_info["file_name"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        inputs = processor(images=img_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)

        if results:
            r = results[0]
            for score, label, box in zip(r["scores"], r["labels"], r["boxes"]):
                all_detections[img_info["id"]].append({
                    "box": box.cpu().numpy(),
                    "score": float(score),
                    "label": int(label),
                })

        anns = [a for a in val_coco["annotations"] if a["image_id"] == img_info["id"]]
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            all_gt[img_info["id"]].append({
                "box": np.array([x, y, x + bw, y + bh]),
                "label": ann["category_id"],
            })

    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-10)

    tp_total = 0
    fp_total = 0
    fn_total = 0
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for img_id in set(list(all_gt.keys()) + list(all_detections.keys())):
        dets = sorted(all_detections.get(img_id, []), key=lambda x: -x["score"])
        gts = list(all_gt.get(img_id, []))
        matched = [False] * len(gts)

        for det in dets:
            best_iou = 0
            best_j = -1
            for j, gt in enumerate(gts):
                if matched[j] or gt["label"] != det["label"]:
                    continue
                iou_val = iou(det["box"], gt["box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            if best_iou >= 0.5 and best_j >= 0:
                tp_total += 1
                per_class_tp[det["label"]] += 1
                matched[best_j] = True
            else:
                fp_total += 1
                per_class_fp[det["label"]] += 1

        for j, m in enumerate(matched):
            if not m:
                fn_total += 1
                per_class_fn[gts[j]["label"]] += 1

    precision = tp_total / (tp_total + fp_total + 1e-10)
    recall = tp_total / (tp_total + fn_total + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        tp = per_class_tp[i]
        fp = per_class_fp[i]
        fn = per_class_fn[i]
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f = 2 * p * r / (p + r + 1e-10)
        per_class_metrics[name] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}

    return {
        "precision@0.5": precision,
        "recall@0.5": recall,
        "f1@0.5": f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "per_class": per_class_metrics,
    }


def train_rtdetr(dataset_name: str, dataset_yaml: Path):
    print(f"\n{'='*60}")
    print(f"RT-DETR Training: {dataset_name}")
    print(f"Model: {MODEL_NAME}  Device: {DEVICE}")
    print(f"{'='*60}")

    results_dir = RESULTS_ROOT / "rtdetr_batch" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    marker = results_dir / "metrics.json"
    if marker.exists():
        print(f"SKIP — already done ({marker})")
        return

    print("\nConverting YOLO-seg → COCO format...")
    train_coco, val_coco, class_names = yolo_seg_to_coco(dataset_yaml)

    print(f"\nLoading model: {MODEL_NAME}")
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    model = RTDetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: n for i, n in enumerate(class_names)},
        label2id={n: i for i, n in enumerate(class_names)},
        ignore_mismatched_sizes=True,
    )
    model = model.to(DEVICE)

    train_dataset = CocoDetectionDataset(train_coco, processor, class_names, IMGSZ)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": LR * 0.02},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": LR},
    ], weight_decay=1e-4)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining {EPOCHS} epochs, {len(train_loader)} batches/epoch...")
    t0 = time.time()
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch_idx, (pixel_values, targets) in enumerate(train_loader):
            pixel_values = pixel_values.to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_coco(model, processor, val_coco, class_names, DEVICE, IMGSZ)
            f1 = metrics["f1@0.5"]
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  P={metrics['precision@0.5']:.4f}  R={metrics['recall@0.5']:.4f}  F1={f1:.4f}", flush=True)

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                model.save_pretrained(str(results_dir / "best_model"))
                processor.save_pretrained(str(results_dir / "best_model"))
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
    best_model = RTDetrForObjectDetection.from_pretrained(str(results_dir / "best_model")).to(DEVICE)
    best_processor = RTDetrImageProcessor.from_pretrained(str(results_dir / "best_model"))
    final_metrics = evaluate_coco(best_model, best_processor, val_coco, class_names, DEVICE, IMGSZ)
    final_metrics["model"] = MODEL_NAME
    final_metrics["epochs"] = EPOCHS
    final_metrics["training_time_min"] = elapsed / 60
    final_metrics["license"] = "Apache-2.0"

    marker.write_text(json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved → {marker}")
    print(json.dumps(final_metrics, indent=2, default=str))


def main():
    train_rtdetr("napchai", BATCH_ROOT / "yolo" / "napchai" / "dataset.yaml")
    train_rtdetr("mka", BATCH_ROOT / "yolo" / "mka" / "dataset.yaml")
    print("\nALL_DONE")


if __name__ == "__main__":
    main()
