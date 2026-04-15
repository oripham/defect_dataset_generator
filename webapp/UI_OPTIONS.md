# UI Direction Options — Defect Studio

Trạng thái hiện tại: chọn Product → Defect → OK image (dropdown hoặc upload) → Params → Preview/Batch

---

## ⭐ Option F — Nhóm theo loại vật thể (khuyến nghị mới)

**Ý tưởng:** Tách UI thành 2 nhóm rõ ràng theo vật thể thực tế, không phải tên kỹ thuật:

```
┌─────────────────────────┐   ┌─────────────────────────┐
│  🔩 Nắp / Chai nhựa     │   │  💊 Thuốc               │
│  (MKA Cap)              │   │  (Capsule · Tablet)      │
│                         │   │                          │
│  • Scratch (Xước)       │   │  • Crack (Vỡ)            │
│  • Rim Deform           │   │  • Hollow (Rỗng ruột)    │
│  • Ring Fracture        │   │  • Underfill             │
│  • Dark Spots           │   │  • Dent                  │
│  • Thread               │   │                          │
│  • Dent / Plastic Flow  │   │  → Capsule / Tablet      │
└─────────────────────────┘   └─────────────────────────┘
```

**Flow:** Click nhóm vật thể → Chọn defect → Upload OK → Params → Preview

**Tại sao hợp lý hơn:**
- User trong nhà máy nghĩ theo **vật thể đang kiểm tra**, không theo tên code
- "Nắp chai" và "Thuốc" là 2 pipeline hoàn toàn khác nhau (polar transform vs Otsu mask) → tách rõ tránh nhầm
- Khi thêm sản phẩm mới (vd: nắp kim loại loại 2, viên nang khác size) → chỉ cần thêm vào đúng nhóm
- Detect mask khác nhau: nắp → Hough Circle, thuốc → Auto Otsu → hiện đúng button theo nhóm

**Thay đổi code:**
- 2 tab/card lớn: `cap` và `pharma`
- Click tab → load defect list + hiện đúng detect button (Circle vs Otsu)
- API route gắn với tab, không cần suy từ defect key

**Ưu:** Trực quan nhất với người dùng nhà máy, scale tốt, tách rõ 2 pipeline  
**Nhược:** Thêm 1 bước chọn nhóm (nhưng chỉ 2 lựa chọn → rất nhanh)

---

---

## Option A — Giữ nguyên (minimal change)

**Flow:** Product → Defect → OK image → Params → Preview

**Thay đổi nhỏ:**
- Thêm description ngắn dưới mỗi defect ("vết xước trên mặt rim", "biến dạng mép")
- Thêm thumbnail preview khi hover ảnh OK trong dropdown
- Không động đến logic

**Ưu:** Không tốn công, ổn định  
**Nhược:** Vẫn phải biết trước sản phẩm, hơi bureaucratic

---

## Option B — Flat defect list (bỏ Product dropdown)

**Flow:** Defect (flat list có optgroup) → Upload OK → Params → Preview

```
── MKA Cap ──
  Scratch (Xước)
  Rim Deform (Cấn miệng)
  Ring Fracture
  Dark Spots
  Thread (Dị vật chỉ)
  Dent (Lõm)
  Plastic Flow (Nhựa chảy)
── Pharma ──
  Crack
  Hollow
  Underfill
  Dent
```

**Thay đổi code:**
- Bỏ `sel-product`, gộp CATALOG thành 1 dropdown flat
- API route tự suy từ defect key (mka defects → `/api/cap/`, pharma → `/api/pharma/`)
- OK image dropdown filter theo defect key thay vì product

**Ưu:** Trực quan hơn, ít click hơn, user chỉ nghĩ về "muốn lỗi gì"  
**Nhược:** Mất khái niệm product grouping nếu sau này nhiều sản phẩm

---

## Option C — Upload-first

**Flow:** Upload OK image → (auto-detect product type) → Defect list filter → Params → Preview

Auto-detect bằng aspect ratio + Hough circle:
- Tìm được circle → MKA Cap → show cap defects
- Không tìm được circle, hình oval → Capsule
- Hình tròn nhỏ → Round Tablet

**Thay đổi code:**
- Bước đầu tiên là upload/chọn ảnh
- Gọi `/api/cap/detect-circle` sau khi upload, nếu found → switch sang cap mode
- Defect dropdown populate sau khi detect xong

**Ưu:** Tự động nhất, không cần user biết sản phẩm tên gì  
**Nhược:** Detect sai edge case (ảnh blur, góc chụp lạ), logic phức tạp hơn

---

## Option D — Tab theo sản phẩm (dạng pill/tab ngang)

**Flow:** Click tab sản phẩm → Defect dropdown → Upload OK → Params → Preview

Thay dropdown product bằng tab ngang:
```
[ MKA Cap ]  [ Capsule ]  [ Round Tablet ]
```

**Thay đổi code:**
- Thay `<select id="sel-product">` bằng Bootstrap nav-pills
- Còn lại giữ nguyên

**Ưu:** Visual hơn, dễ hiểu product context, 1 click thay vì dropdown  
**Nhược:** Nếu nhiều sản phẩm thì tab bị dài

---

## Option E — Card grid chọn sản phẩm (landing page nhỏ)

**Flow:** Chọn product card → Defect dropdown → Upload OK → Params → Preview

Thay vì dropdown, hiện grid card mỗi sản phẩm có icon + tên + số defect types:

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  🔩 MKA Cap  │  │  💊 Capsule  │  │  ⬤ Tablet   │
│  7 defects   │  │  2 defects   │  │  2 defects   │
└──────────────┘  └──────────────┘  └──────────────┘
```

Sau khi click card → collapse cards, show defect + params

**Ưu:** Đẹp, onboarding rõ ràng, dễ thêm sản phẩm mới  
**Nhược:** Thêm 1 click so với flat list, cần thêm CSS/JS

---

## So sánh nhanh

| Option | Số click đến Preview | Code change | Phù hợp khi |
|---|---|---|---|
| A — Giữ nguyên | 5 | Minimal | Prototype nội bộ |
| B — Flat defect | 4 | Nhỏ | Upload-first workflow |
| C — Upload-first | 4 | Trung bình | User không biết product |
| D — Tab ngang | 4 | Nhỏ | Ít sản phẩm (≤5) |
| E — Card grid | 5 | Trung bình | Demo / presentable UI |
| **F — Nhóm vật thể** | **4** | **Nhỏ** | **User nhà máy, 2 pipeline rõ** |

---

## Khuyến nghị

- **Ngắn hạn (demo sớm):** Option F — 2 nhóm vật thể (Nắp/Chai · Thuốc), ít code, sát thực tế nhà máy nhất
- **Dài hạn (nhiều sản phẩm):** Option F + E kết hợp — mỗi nhóm là card grid, click vào expand defect list
