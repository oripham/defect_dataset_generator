# Engine Interface Specification

---

## Unified `generate()` function

Cả `fast_physics.py` (Vuong) và `deep_generative.py` (Oanh) đều phải export hàm:

```python
def generate(
    base_image:  np.ndarray,   # shape (H, W, 3), RGB, dtype uint8
    mask:        np.ndarray,   # shape (H, W),    binary 0/255, dtype uint8
    defect_type: str,          # xem bảng Defect Types bên dưới
    material:    str,          # "metal" | "plastic" | "pharma"
    params:      dict,         # xem bảng Params bên dưới
) -> dict:
    ...
    return {
        "result_image": str,    # base64-encoded PNG string
        "engine":       str,    # "cv" hoặc "genai"  (để log/debug)
        "metadata":     dict,   # bất kỳ info thêm (optional, có thể là {})
    }
```

---

## Defect Types


| `defect_type`   | Mô tả           | Engine mặc định                     |
| --------------- | --------------- | ----------------------------------- |
| `"scratch"`     | Xước bề mặt     | cv                                  |
| `"crack"`       | Nứt             | cv                                  |
| `"dent"`        | Móp, lõm        | genai (metal) / cv (plastic)        |
| `"bulge"`       | Phồng           | genai (metal) / cv (plastic/pharma) |
| `"chip"`        | Sứt mẻ, vỡ cạnh | cv (plastic) / genai (metal)        |
| `"rust"`        | Rỉ sét          | cv                                  |
| `"burn"`        | Cháy bề mặt     | genai                               |
| `"micro_crack"` | Nứt mạng nhện   | genai                               |
| `"foreign"`     | Dị vật          | cv                                  |


---

## Params Dictionary


| Key               | Type  | Range      | Mô tả                         | Map từ UI Slider                  |
| ----------------- | ----- | ---------- | ----------------------------- | --------------------------------- |
| `intensity`       | float | 0.0 – 1.0  | Độ nặng/rõ của lỗi            | Slider "Intensity" / 100          |
| `naturalness`     | float | 0.0 – 1.0  | Độ tự nhiên, blend mềm        | Slider "Naturalness" / 100        |
| `position_jitter` | float | 0.0 – 1.0  | Offset ngẫu nhiên vị trí mask | Slider "Position Variation" / 100 |
| `seed`            | int   | any        | Random seed (optional)        | —                                 |
| `ref_image_b64`   | str   | base64 PNG | Ảnh NG reference (optional)   | — từ NG Refs upload               |


> **Lưu ý:** `ref_image_b64` bắt buộc nếu engine=GenAI và defect_type thuộc nhóm appearance
> (scratch/crack/chip/rust/burn/micro_crack/foreign). Thiếu → server trả HTTP 422.
> Với dent/bulge thì không cần (bị bỏ qua).

---

## Ví dụ call

```python
import numpy as np
from engines.fast_physics import generate   # hoặc deep_generative

base = cv2.imread("ok_metal.jpg")           # uint8 RGB
mask = cv2.imread("mask_scratch.png", 0)    # uint8 grayscale

result = generate(
    base_image  = base,
    mask        = mask,
    defect_type = "scratch",
    material    = "metal",
    params      = {
        "intensity":        0.7,
        "naturalness":      0.6,
        "position_jitter":  0.0,
        "seed":             42,
        "ref_image_b64":    None,
    }
)

print(result["engine"])         # "cv"
print(result["result_image"])   # base64 PNG string
```

---

## API Endpoints (Vuong cung cấp — port 8001)

```
GET  /health
GET  /api/default-engine?defect_type=dent&material=metal  → {"engine":"genai"}
POST /api/generate/preview   → sinh 1 ảnh, trả về ngay
POST /api/generate/batch     → sinh N ảnh nền, trả job_id
GET  /api/generate/status/{job_id}  → poll tiến độ batch
```

**Preview request:**

```json
{
  "base_image":      "<base64 PNG>",
  "mask":            "<base64 PNG grayscale>",
  "defect_type":     "scratch",
  "material":        "metal",
  "intensity":       0.6,
  "naturalness":     0.7,
  "position_jitter": 0.0,
  "engine_override": null,
  "ref_image_b64":   "<base64 PNG hoặc null>",
  "seed":            null
}
```

**Preview response:**

```json
{
  "result_image": "<base64 PNG>",
  "engine_used":  "cv",
  "metadata":     {}
}
```

---

