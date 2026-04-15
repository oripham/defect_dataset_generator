# Defect Studio — UI Guide

URL: `http://localhost:5000/studio`

---

## Layout tổng quan

```
┌─────────────┬─────────────────────────────┬──────────────┐
│  LEFT       │  CENTER                     │  RIGHT       │
│  Controls   │  Preview Panel              │  Output      │
│  (3 cards)  │  + Log                      │  (3 cards)   │
└─────────────┴─────────────────────────────┴──────────────┘
```

---

## LEFT — Controls

### Card 1: Product & Defect

| Element | Mô tả |
|---|---|
| **Product** dropdown | Chọn sản phẩm — 2 nhóm: **Pharma** (Elongated Capsule, Round Tablet) và **MKA Cap** |
| **Defect Class** dropdown | Tự động cập nhật khi đổi product, list các loại lỗi của product đó |

Defect classes theo product:

| Product | Defects |
|---|---|
| Elongated Capsule | Hollow, Underfill |
| Round Tablet | Crack, Dent |
| MKA Cap | Scratch, Rim Deform, Ring Fracture, Dark Spots, Thread, Dent, Plastic Flow |

---

### Card 2: Input Image

| Element | Mô tả |
|---|---|
| **OK Image** dropdown | Load ảnh OK từ thư mục server (tối đa 20 ảnh, thumbnail 80px) |
| **Or upload** | Upload file từ máy local (PNG/JPG/BMP) |
| **🔍 Auto-Detect Mask (Otsu)** | Chỉ hiện cho Pharma — tự detect mask viên thuốc bằng Otsu threshold |
| **🔵 Detect Circle (Hough)** | Chỉ hiện cho MKA Cap — detect tâm và bán kính ring bằng Hough Circle |
| Mask status | Dòng text nhỏ bên dưới, xác nhận mask đã detect xong |

---

### Card 3: Parameters

**Intensity** (0–100%) — strength của lỗi, áp dụng cho tất cả defect types.

Các param panel phụ xuất hiện tùy defect type đang chọn:

#### `params-crack` — Pharma: Crack (Round Tablet)
| Control | Range | Mô tả |
|---|---|---|
| **Break Type** | straight / corner / curved / concave / zigzag×2 / zigzag×3 | Hình dạng vết nứt |
| **Depth** | 10–70% | Độ sâu vết nứt vào viên thuốc |
| **Angle** | 0–315° (step 45°) | Góc xoay vết nứt |

#### `params-hollow` — Pharma: Hollow / Underfill (Elongated Capsule)
| Control | Mô tả |
|---|---|
| **Fixed Region** checkbox | Tích = lỗi luôn ở vị trí cố định (reproducible), bỏ tích = random |

#### `params-polar` — MKA: Rim Deform / Ring Fracture
| Control | Range | Mô tả |
|---|---|---|
| **Deform Position** | 12/3/6/9 o'clock hoặc Random | Vị trí lỗi trên ring (theo giờ đồng hồ) |
| **Span** | 0.1–1.2 rad | Độ rộng vùng biến dạng trên rim |
| **Deform Depth** | 3–40 px | Độ sâu biến dạng trong polar space |
| **Jitter** | 1–20 *(chỉ Ring Fracture)* | Biên độ dao động ngẫu nhiên của rim |

#### `params-scuff` — MKA: Scratch / Plastic Flow
| Control | Options | Mô tả |
|---|---|---|
| **Scuff Style** | Auto / Whiten Streak / Micro Haze / Gouge Whiten / Parallel Micro / Crosshatch | Kiểu vết xước |

#### `params-spots` — MKA: Dark Spots
| Control | Range | Mô tả |
|---|---|---|
| **Spot Count** min–max | 1–20 | Số lượng đốm đen |
| **Radius px** min–max | 1–80 px | Kích thước đốm |

#### Seed (tất cả defect types)
- Input số — giá trị seed để reproduce kết quả
- 🎲 button — random seed mới

---

### Action Buttons

| Button | Mô tả |
|---|---|
| **▶ Preview** | Generate 1 ảnh, hiển thị ngay trong Preview Panel |
| **⚡ Batch Generate** | Mở modal để gen nhiều ảnh, lưu vào `defect_samples/results/` |

---

## CENTER — Preview Panel

4 ô ảnh (2×2 grid):

| Ô | Nội dung |
|---|---|
| **OK Image** | Ảnh gốc không lỗi |
| **Mask** | Mask vùng lỗi (trắng = vùng được tổng hợp) |
| **Result** | Ảnh kết quả có lỗi |
| **Debug Panel** | 4 ảnh nối ngang: OK \| Mask \| Result \| Diff×4 (khuếch đại 4× để thấy sự khác biệt) |

**Log box** bên dưới: hiển thị timestamp + message mỗi action (preview, save, lỗi...).

---

## RIGHT — Output

### QC Panel
- Hiện sau khi chạy Preview
- Thống kê nhanh: defect area %, pixel diff mean, histogram

### Output
| Button | Mô tả |
|---|---|
| **💾 Save** | Lưu result vào `defect_samples/results/manual/<product>/<defect>/` trên server |
| **⬇ Download PNG** | Download file PNG về máy local |

### Recent (History Strip)
- Thumbnail nhỏ các ảnh đã gen trong session hiện tại
- Click thumbnail để xem lại kết quả cũ

---

## Batch Modal (`⚡ Batch Generate`)

| Element | Mô tả |
|---|---|
| **Number of images** | 1–500, default 20 |
| **Break types** *(chỉ Crack)* | Checkbox chọn các kiểu nứt sẽ được xen kẽ trong batch |
| **Progress bar** | Hiện sau khi nhấn Start, poll mỗi 1.5s |
| **⚡ Start** | Bắt đầu background job, có thể đóng modal vẫn chạy |

Kết quả batch lưu tại:
```
defect_samples/results/<YYYYMMDD_HHMMSS>/<product>/<defect>/
  <defect>_s<seed>_<ok_stem>.png      ← ảnh kết quả
  debug_<defect>_s<seed>_<ok_stem>.png ← debug panel
```

---

## API Routing (JS → Backend)

JS tự động route đến đúng backend dựa vào `CATALOG[product].api`:

| Product | API prefix |
|---|---|
| Pharma (elongated_capsule, round_tablet) | `/api/pharma/*` |
| MKA Cap | `/api/cap/*` |

Flask webapp chạy local port 5000.  
FastAPI engine chạy trên RunPod: `https://j6tbhd1pq4gog3-8001.proxy.runpod.net`

---

## Engine Badge (góc phải trên)

`Engine: CV (Local)` — hiện engine đang được dùng:
- **CV (Local)**: pure OpenCV, không cần GPU, chạy ngay
- **GenAI (SDXL)**: dùng SDXL Inpaint + ControlNet trên RunPod GPU
