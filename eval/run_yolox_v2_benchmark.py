"""
YOLOX v2 — Crop-focused training for small defect detection.
Strategy: crop around defects so they fill more of the 640x640 input.
"""
from __future__ import annotations

import logging
import sys

import json
import os
import random
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

# --- Monkey-patch simota_matching to fix topk overflow on tiny defects ---
_orig_simota = YOLOXHead.simota_matching

def _patched_simota(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
    import torch as _torch
    matching_matrix = _torch.zeros_like(cost, dtype=_torch.uint8)
    n_candidate_k = min(10, pair_wise_ious.size(1))
    topk_ious, _ = _torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = _torch.clamp(topk_ious.sum(1).int(), min=1)
    num_anchors = cost.size(1)
    for gt_idx in range(num_gt):
        k = min(int(dynamic_ks[gt_idx].item()), num_anchors)
        if k == 0:
            continue
        _, pos_idx = _torch.topk(cost[gt_idx], k=k, largest=False)
        matching_matrix[gt_idx][pos_idx] = 1
    del topk_ious, dynamic_ks
    anchor_matching_gt = matching_matrix.sum(0)
    if anchor_matching_gt.max() > 1:
        multiple_match_mask = anchor_matching_gt > 1
        _, cost_argmin = _torch.min(cost[:, multiple_match_mask], dim=0)
        matching_matrix[:, multiple_match_mask] *= 0
        matching_matrix[cost_argmin, multiple_match_mask] = 1
    fg_mask_inboxes = anchor_matching_gt > 0
    num_fg = fg_mask_inboxes.sum().item()
    fg_mask[fg_mask.clone()] = fg_mask_inboxes
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]
    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

YOLOXHead.simota_matching = _patched_simota
# --- End monkey-patch ---

ROOT = Path(r"V:\HondaPlus\defect_dataset_generator")
BATCH_ROOT = ROOT / "batch_output"
RESULTS_ROOT = ROOT / "eval" / "output" / "results"
YOLOX_WEIGHTS = Path(r"V:\HondaPlus\YOLOX\yolox_s.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
EPOCHS = 80
BATCH_SIZE = 4
IMGSZ = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45
CROP_PAD_FACTOR = 4.0
NEG_CROP_RATIO = 0.15


def parse_yolo_seg_labels(label_path: Path, img_w: int, img_h: int):
    """Parse YOLO-seg label file → list of (class_id, x1, y1, x2, y2) in pixel coords."""
    anns = []
    if not label_path.exists():
        return anns
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = [c * img_w for c in coords[0::2]]
        ys = [c * img_h for c in coords[1::2]]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            continue
        anns.append((cls_id, x1, y1, x2, y2))
    return anns


def make_crops(img: np.ndarray, anns: list, pad_factor: float, imgsz: int,
               neg_ratio: float = 0.15):
    """Generate defect-centered crops + negative crops from one image."""
    h, w = img.shape[:2]
    crops = []

    clusters = _cluster_nearby(anns, merge_dist_ratio=0.15, img_w=w, img_h=h)

    for cluster in clusters:
        all_x1 = min(a[1] for a in cluster)
        all_y1 = min(a[2] for a in cluster)
        all_x2 = max(a[3] for a in cluster)
        all_y2 = max(a[4] for a in cluster)

        cw = all_x2 - all_x1
        ch = all_y2 - all_y1
        cx = (all_x1 + all_x2) / 2
        cy = (all_y1 + all_y2) / 2

        crop_size = max(cw, ch) * pad_factor
        crop_size = max(crop_size, 80)
        crop_size = min(crop_size, min(w, h))

        jx = random.uniform(-cw * 0.3, cw * 0.3)
        jy = random.uniform(-ch * 0.3, ch * 0.3)
        cx += jx
        cy += jy

        half = crop_size / 2
        crop_x1 = int(max(0, cx - half))
        crop_y1 = int(max(0, cy - half))
        crop_x2 = int(min(w, cx + half))
        crop_y2 = int(min(h, cy + half))

        if crop_x2 - crop_x1 < 30 or crop_y2 - crop_y1 < 30:
            continue

        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        crop_anns = []
        for cls_id, ax1, ay1, ax2, ay2 in cluster:
            nx1 = max(0, ax1 - crop_x1)
            ny1 = max(0, ay1 - crop_y1)
            nx2 = min(crop_x2 - crop_x1, ax2 - crop_x1)
            ny2 = min(crop_y2 - crop_y1, ay2 - crop_y1)
            if nx2 - nx1 < 1 or ny2 - ny1 < 1:
                continue
            crop_anns.append((cls_id, nx1, ny1, nx2, ny2))

        if crop_anns:
            crops.append((crop_img, crop_anns))

    n_neg = max(1, int(len(clusters) * neg_ratio))
    for _ in range(n_neg):
        cs = random.randint(100, min(w, h) // 2)
        rx = random.randint(0, w - cs)
        ry = random.randint(0, h - cs)

        overlap = False
        for a in anns:
            if rx < a[3] and rx + cs > a[1] and ry < a[4] and ry + cs > a[2]:
                overlap = True
                break
        if not overlap:
            crops.append((img[ry:ry+cs, rx:rx+cs], []))

    return crops


def _cluster_nearby(anns, merge_dist_ratio, img_w, img_h):
    """Cluster annotations that are close together."""
    if not anns:
        return []
    merge_dist = max(img_w, img_h) * merge_dist_ratio
    clusters = [[a] for a in anns]
    merged = True
    while merged:
        merged = False
        new_clusters = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            current = list(clusters[i])
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if _clusters_close(current, clusters[j], merge_dist):
                    current.extend(clusters[j])
                    used[j] = True
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters
    return clusters


def _clusters_close(c1, c2, dist):
    cx1 = (min(a[1] for a in c1) + max(a[3] for a in c1)) / 2
    cy1 = (min(a[2] for a in c1) + max(a[4] for a in c1)) / 2
    cx2 = (min(a[1] for a in c2) + max(a[3] for a in c2)) / 2
    cy2 = (min(a[2] for a in c2) + max(a[4] for a in c2)) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 < dist


def build_crop_coco(dataset_yaml_path: Path, pad_factor: float, imgsz: int):
    """Build COCO dict from crop-focused extraction."""
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

        images_list, annotations_list = [], []
        img_id = 0
        ann_id = 0

        for img_path in sorted(img_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            label_path = label_dir / (img_path.stem + ".txt")
            anns = parse_yolo_seg_labels(label_path, w0, h0)

            if split == "train":
                crops = make_crops(img, anns, pad_factor, imgsz, NEG_CROP_RATIO)
                for crop_img, crop_anns in crops:
                    ch, cw = crop_img.shape[:2]
                    crop_path = img_dir.parent / "crops" / f"crop_{img_id:05d}.png"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crop_path), crop_img)

                    images_list.append({"id": img_id, "file_name": str(crop_path),
                                        "width": cw, "height": ch})
                    for cls_id, x1, y1, x2, y2 in crop_anns:
                        bw, bh = x2 - x1, y2 - y1
                        annotations_list.append({
                            "id": ann_id, "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [x1, y1, bw, bh], "area": bw * bh,
                            "iscrowd": 0,
                        })
                        ann_id += 1
                    img_id += 1
            else:
                random.seed(42)
                crops = make_crops(img, anns, pad_factor, imgsz, 0.0)
                random.seed()
                for crop_img, crop_anns in crops:
                    ch, cw = crop_img.shape[:2]
                    crop_path = img_dir.parent / "crops_val" / f"crop_{img_id:05d}.png"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crop_path), crop_img)

                    images_list.append({"id": img_id, "file_name": str(crop_path),
                                        "width": cw, "height": ch})
                    for cls_id, x1, y1, x2, y2 in crop_anns:
                        bw, bh = x2 - x1, y2 - y1
                        annotations_list.append({
                            "id": ann_id, "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [x1, y1, bw, bh], "area": bw * bh,
                            "iscrowd": 0,
                        })
                        ann_id += 1
                    img_id += 1

        results[split] = {"images": images_list, "annotations": annotations_list,
                          "categories": coco_categories}
        print(f"  {split}: {len(images_list)} crops, {len(annotations_list)} annotations")

    return results["train"], results["val"], class_names


class CropDetectionDataset(Dataset):
    def __init__(self, coco_dict: dict, imgsz: int = 640, augment: bool = False):
        self.images = coco_dict["images"]
        self.imgsz = imgsz
        self.augment = augment
        self.img_to_anns: dict[int, list] = {}
        for ann in coco_dict["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img = cv2.imread(img_info["file_name"])
        h0, w0 = img.shape[:2]

        if self.augment:
            img = self._augment(img)

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
            targets.append([ann["category_id"], cx, cy, bw * r, bh * r])

        if targets:
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            targets_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, targets_tensor, (h0, w0, r)

    def _augment(self, img):
        if random.random() < 0.5:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-30, 30)
            img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            sigma = random.uniform(0.5, 1.5)
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.7, 1.3)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img


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
    depth, width = 0.33, 0.50
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
    print(f"  Loaded {len(filtered)}/{len(model_dict)} params from pretrained")
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
        preds = postprocess(outputs, num_classes=num_classes,
                            conf_thre=CONF_THRESH, nms_thre=NMS_THRESH)
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
            all_gt[img_info["id"]].append({
                "box": np.array([x, y, x + bw, y + bh]),
                "label": ann["category_id"],
            })

    def iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-10)

    tp, fp, fn = 0, 0, 0
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

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
        p = t / (t + f_p + 1e-10)
        r_val = t / (t + f_n + 1e-10)
        per_class[name] = {
            "precision": p, "recall": r_val,
            "f1": 2 * p * r_val / (p + r_val + 1e-10),
            "tp": t, "fp": f_p, "fn": f_n,
        }

    return {
        "precision@0.5": precision, "recall@0.5": recall, "f1@0.5": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "inference_ms_avg": float(np.mean(inference_times)),
        "per_class": per_class,
    }


def train_yolox_v2(dataset_name: str, dataset_yaml: Path):
    _log(f"\n{'='*60}")
    _log(f"YOLOX-S v2 (Crop-focused): {dataset_name}")
    _log(f"Device: {DEVICE}, Pad factor: {CROP_PAD_FACTOR}")
    _log(f"{'='*60}")

    results_dir = RESULTS_ROOT / "yolox_v2" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    marker = results_dir / "metrics.json"
    if marker.exists():
        _log(f"SKIP — already done ({marker})")
        return

    _log("\nBuilding crop-focused dataset...")
    train_coco, val_coco, class_names = build_crop_coco(
        dataset_yaml, CROP_PAD_FACTOR, IMGSZ)
    num_classes = len(class_names)

    _log(f"\nBuilding YOLOX-S (num_classes={num_classes})...")
    model = build_yolox_model(num_classes)
    load_pretrained(model, YOLOX_WEIGHTS, num_classes)
    model = model.to(DEVICE)

    train_dataset = CropDetectionDataset(train_coco, IMGSZ, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    WARMUP_EPOCHS = 5
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR * 0.01},
        {"params": model.head.parameters(), "lr": LR * 0.3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, scheduler], milestones=[WARMUP_EPOCHS])

    _log(f"\nTraining {EPOCHS} epochs on {len(train_dataset)} crops...")
    t0 = time.time()
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, targets, _ in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            try:
                outputs = model(imgs, targets=targets)
                loss = outputs["total_loss"]
            except RuntimeError as e:
                if "out of range" in str(e):
                    continue
                raise
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()

        combined_scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_coco, class_names, DEVICE, IMGSZ)
            f1 = metrics["f1@0.5"]
            pc = " | ".join(f"{n}={metrics['per_class'][n]['f1']:.3f}"
                            for n in class_names)
            _log(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
                 f"F1={f1:.4f}  [{pc}]")

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), str(results_dir / "best.pth"))
            else:
                patience_counter += 5

            if patience_counter >= 40:
                _log(f"  Early stopping at epoch {epoch}")
                break
        else:
            _log(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    _log(f"\nTraining done in {elapsed/60:.1f} min")

    _log("\nFinal evaluation on best model...")
    model.load_state_dict(torch.load(str(results_dir / "best.pth"), map_location=DEVICE))
    final_metrics = evaluate(model, val_coco, class_names, DEVICE, IMGSZ)
    final_metrics["model"] = "YOLOX-S-v2-crop"
    final_metrics["epochs"] = EPOCHS
    final_metrics["crop_pad_factor"] = CROP_PAD_FACTOR
    final_metrics["training_time_min"] = elapsed / 60
    final_metrics["license"] = "Apache-2.0"

    marker.write_text(json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")
    _log(f"\nResults saved → {marker}")
    _log(json.dumps(final_metrics, indent=2, default=str))


LOG_FILE = RESULTS_ROOT / "yolox_v2" / "training.log"

def _log(msg):
    print(msg, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("")
    train_yolox_v2("napchai", BATCH_ROOT / "yolo" / "napchai" / "dataset.yaml")
    train_yolox_v2("mka", BATCH_ROOT / "yolo" / "mka" / "dataset.yaml")
    _log("\nALL_DONE")


if __name__ == "__main__":
    main()
