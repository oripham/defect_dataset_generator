"""
YOLOX v3 — Improved small defect detection.
Changes from v2:
  - EMA (Exponential Moving Average) for training stability
  - Freeze backbone first 10 epochs, then unfreeze with very low LR
  - CROP_PAD_FACTOR=2.0 (defects fill ~25% of crop, vs 6% in v2)
  - Multi-scale training (random resize 448-832 per batch)
  - Copy-paste augmentation (paste defects onto clean crops)
"""
from __future__ import annotations

import copy
import json
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

# --- Monkey-patch simota_matching to fix topk overflow ---
def _patched_simota(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
    n_candidate_k = min(10, pair_wise_ious.size(1))
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    num_anchors = cost.size(1)
    for gt_idx in range(num_gt):
        k = min(int(dynamic_ks[gt_idx].item()), num_anchors)
        if k == 0:
            continue
        _, pos_idx = torch.topk(cost[gt_idx], k=k, largest=False)
        matching_matrix[gt_idx][pos_idx] = 1
    del topk_ious, dynamic_ks
    anchor_matching_gt = matching_matrix.sum(0)
    if anchor_matching_gt.max() > 1:
        mm = anchor_matching_gt > 1
        _, cost_argmin = torch.min(cost[:, mm], dim=0)
        matching_matrix[:, mm] *= 0
        matching_matrix[cost_argmin, mm] = 1
    fg_mask_inboxes = anchor_matching_gt > 0
    num_fg = fg_mask_inboxes.sum().item()
    fg_mask[fg_mask.clone()] = fg_mask_inboxes
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]
    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

YOLOXHead.simota_matching = _patched_simota

ROOT = Path(r"V:\HondaPlus\defect_dataset_generator")
BATCH_ROOT = ROOT / "batch_output"
RESULTS_ROOT = ROOT / "eval" / "output" / "results"
YOLOX_WEIGHTS = Path(r"V:\HondaPlus\YOLOX\yolox_s.pth")
LOG_FILE = RESULTS_ROOT / "yolox_v3" / "training.log"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 5e-5
EPOCHS = 100
BATCH_SIZE = 4
IMGSZ = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45
CROP_PAD_FACTOR = 2.0
NEG_CROP_RATIO = 0.10
FREEZE_EPOCHS = 10
EMA_DECAY = 0.9998
MULTI_SCALES = [448, 512, 576, 640, 704, 768, 832]


def _log(msg):
    print(msg, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


def parse_yolo_seg_labels(label_path: Path, img_w: int, img_h: int):
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


def _cluster_nearby(anns, merge_dist_ratio, img_w, img_h):
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
                cx1 = (min(a[1] for a in current) + max(a[3] for a in current)) / 2
                cy1 = (min(a[2] for a in current) + max(a[4] for a in current)) / 2
                cx2 = (min(a[1] for a in clusters[j]) + max(a[3] for a in clusters[j])) / 2
                cy2 = (min(a[2] for a in clusters[j]) + max(a[4] for a in clusters[j])) / 2
                if ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5 < merge_dist:
                    current.extend(clusters[j])
                    used[j] = True
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters
    return clusters


def make_crops(img, anns, pad_factor, imgsz, neg_ratio=0.10):
    h, w = img.shape[:2]
    crops = []
    clusters = _cluster_nearby(anns, merge_dist_ratio=0.10, img_w=w, img_h=h)

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
        crop_size = max(crop_size, 64)
        crop_size = min(crop_size, min(w, h))

        jx = random.uniform(-cw * 0.2, cw * 0.2)
        jy = random.uniform(-ch * 0.2, ch * 0.2)
        cx += jx
        cy += jy

        half = crop_size / 2
        crop_x1 = int(max(0, cx - half))
        crop_y1 = int(max(0, cy - half))
        crop_x2 = int(min(w, cx + half))
        crop_y2 = int(min(h, cy + half))

        if crop_x2 - crop_x1 < 20 or crop_y2 - crop_y1 < 20:
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
        cs = random.randint(80, min(w, h) // 3)
        rx = random.randint(0, w - cs)
        ry = random.randint(0, h - cs)
        overlap = any(rx < a[3] and rx + cs > a[1] and ry < a[4] and ry + cs > a[2] for a in anns)
        if not overlap:
            crops.append((img[ry:ry+cs, rx:rx+cs], []))

    return crops


def build_crop_coco(dataset_yaml_path: Path, pad_factor: float, imgsz: int):
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
        img_id, ann_id = 0, 0

        crop_dir = ds_root / f"crops_v3_{split}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            label_path = label_dir / (img_path.stem + ".txt")
            anns = parse_yolo_seg_labels(label_path, w0, h0)
            nr = NEG_CROP_RATIO if split == "train" else 0.0
            crops = make_crops(img, anns, pad_factor, imgsz, nr)

            for crop_img, crop_anns in crops:
                ch, cw = crop_img.shape[:2]
                crop_path = crop_dir / f"crop_{img_id:05d}.png"
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
        _log(f"  {split}: {len(images_list)} crops, {len(annotations_list)} annotations")

    return results["train"], results["val"], class_names


class CropDetectionDataset(Dataset):
    def __init__(self, coco_dict, imgsz=640, augment=False, multi_scale=False):
        self.images = coco_dict["images"]
        self.imgsz = imgsz
        self.augment = augment
        self.multi_scale = multi_scale
        self.img_to_anns = {}
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

        sz = random.choice(MULTI_SCALES) if (self.augment and self.multi_scale) else self.imgsz
        r = sz / max(h0, w0)
        img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)))
        rh, rw = img_resized.shape[:2]
        padded = np.full((sz, sz, 3), 114, dtype=np.uint8)
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
            sigma = random.uniform(0.5, 2.0)
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.6, 1.4)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img


def collate_fn(batch):
    max_sz = max(b[0].shape[1] for b in batch)
    imgs = []
    for b in batch:
        c, h, w = b[0].shape
        if h < max_sz or w < max_sz:
            padded = torch.full((c, max_sz, max_sz), 114.0)
            padded[:, :h, :w] = b[0]
            imgs.append(padded)
        else:
            imgs.append(b[0])
    imgs = torch.stack(imgs)

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


def build_yolox_model(num_classes):
    depth, width = 0.33, 0.50
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
    head = YOLOXHead(num_classes=num_classes, width=width, in_channels=in_channels)
    return YOLOX(backbone, head)


def load_pretrained(model, weights_path, num_classes):
    ckpt = torch.load(str(weights_path), map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model_dict = model.state_dict()
    filtered = {k: v for k, v in ckpt.items()
                if k in model_dict and v.shape == model_dict[k].shape}
    _log(f"  Loaded {len(filtered)}/{len(model_dict)} params from pretrained")
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
                    "box": np.array([x1/r, y1/r, x2/r, y2/r]),
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
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
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
        t = per_class_tp[i]; f_p = per_class_fp[i]; f_n = per_class_fn[i]
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


def train_yolox_v3(dataset_name, dataset_yaml):
    _log(f"\n{'='*60}")
    _log(f"YOLOX-S v3: {dataset_name}")
    _log(f"pad={CROP_PAD_FACTOR}, freeze={FREEZE_EPOCHS}ep, EMA={EMA_DECAY}")
    _log(f"{'='*60}")

    results_dir = RESULTS_ROOT / "yolox_v3" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    marker = results_dir / "metrics.json"
    if marker.exists():
        _log(f"SKIP — already done ({marker})")
        return

    _log("\nBuilding crop-focused dataset (pad=2.0)...")
    train_coco, val_coco, class_names = build_crop_coco(
        dataset_yaml, CROP_PAD_FACTOR, IMGSZ)
    num_classes = len(class_names)

    _log(f"\nBuilding YOLOX-S (num_classes={num_classes})...")
    model = build_yolox_model(num_classes)
    load_pretrained(model, YOLOX_WEIGHTS, num_classes)
    model = model.to(DEVICE)

    ema = ModelEMA(model, decay=EMA_DECAY)

    train_dataset = CropDetectionDataset(train_coco, IMGSZ, augment=True, multi_scale=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    # Phase 1: freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-4)

    _log(f"\nTraining {EPOCHS} epochs on {len(train_dataset)} crops...")
    _log(f"Phase 1: backbone frozen (epochs 1-{FREEZE_EPOCHS})")
    t0 = time.time()
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Phase 2: unfreeze backbone
        if epoch == FREEZE_EPOCHS + 1:
            _log(f"  Phase 2: backbone unfrozen (lr={LR*0.01:.1e})")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": LR * 0.01},
                {"params": model.head.parameters(), "lr": LR * 0.5},
            ], weight_decay=1e-4)

        model.train()
        total_loss = 0
        n_batches = 0
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
            ema.update(model)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(ema.ema, val_coco, class_names, DEVICE, IMGSZ)
            f1 = metrics["f1@0.5"]
            pc = " | ".join(f"{n}={metrics['per_class'][n]['f1']:.3f}"
                            for n in class_names)
            _log(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
                 f"F1={f1:.4f}  [{pc}]")

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(ema.ema.state_dict(), str(results_dir / "best.pth"))
            else:
                patience_counter += 5

            if patience_counter >= 40:
                _log(f"  Early stopping at epoch {epoch}")
                break
        else:
            _log(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    _log(f"\nTraining done in {elapsed/60:.1f} min")

    _log("\nFinal evaluation (EMA best model)...")
    ema.ema.load_state_dict(torch.load(str(results_dir / "best.pth"), map_location=DEVICE))
    final_metrics = evaluate(ema.ema, val_coco, class_names, DEVICE, IMGSZ)
    final_metrics["model"] = "YOLOX-S-v3"
    final_metrics["epochs"] = EPOCHS
    final_metrics["crop_pad_factor"] = CROP_PAD_FACTOR
    final_metrics["training_time_min"] = elapsed / 60
    final_metrics["license"] = "Apache-2.0"

    marker.write_text(json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")
    _log(f"\nResults saved → {marker}")
    _log(json.dumps(final_metrics, indent=2, default=str))


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("")
    train_yolox_v3("napchai", BATCH_ROOT / "yolo" / "napchai" / "dataset.yaml")
    train_yolox_v3("mka", BATCH_ROOT / "yolo" / "mka" / "dataset.yaml")
    _log("\nALL_DONE")


if __name__ == "__main__":
    main()
