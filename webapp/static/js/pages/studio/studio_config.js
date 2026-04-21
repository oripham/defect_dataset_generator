// static/js/pages/studio/studio_config.js

const GROUPS = {
  cap: {
    api: "cap",
    product: "mka",
    defects: [
      { key: "scratch", label: "Scratch / 傷", params: "scuff" },
      { key: "dent", label: "Dent / 凹み", params: null },
      { key: "plastic_flow", label: "Plastic Flow / 樹脂流れ", params: "plastic-flow-tune" },
      { key: "dark_spots", label: "Dark Spots / 黒異物", params: "spots" },
      { key: "thread", label: "Thread / 糸異物", params: null },
      { key: "can_mieng", label: "Rim Crush / 縁つぶれ", params: "can-mieng-tune" },
    ],
  },
  pharma: {
    api: "pharma",
    product: null,
    defects: [
      { key: "crack", label: "Crack / 割れ (Tablet)", params: "crack", product: "round_tablet" },
      { key: "dent", label: "Dent / 欠け (Tablet)", params: "dent", product: "round_tablet" },
    ],
  },
  metal_cap: {
    api: "metal_cap",
    product: null,
    defects: [
      { key: "mc_deform", label: "MC Deform / MC変形", params: "polar" },
      { key: "ring_fracture", label: "Ring Fracture / 割れ輪", params: "napchai-ring" },
      { key: "scratch", label: "Scratch / 傷", params: "napchai-scratch" },
    ],
  },
};

const DEFAULT_PROMPTS = {
  // Cap (via deep_generative / sdxl_refiner)
  "cap|scratch":      { pos: "industrial metal surface defect, realistic surface scratch mark, professional quality inspection photograph, sharp focus, metallic surface", neg: "cartoon, painting, blurry, low quality, plastic, text" },
  "cap|dent":         { pos: "industrial metal surface defect, realistic physical dent depression with shadow shading, surface deformation, concentric ring distortion, directional light shadow, professional quality inspection photograph, sharp focus, metallic surface", neg: "cartoon, painting, blurry, low quality, plastic, text" },
  "cap|plastic_flow": { pos: "plastic component surface defect, realistic molten plastic flow smear, glossy streak, flow mark haze, quality control photography, sharp focus, matte plastic surface", neg: "cartoon, painting, blurry, low quality, metal, shiny, text" },
  "cap|dark_spots":   { pos: "industrial metal surface defect, realistic foreign particle contamination, professional quality inspection photograph, sharp focus, metallic surface", neg: "cartoon, painting, blurry, low quality, plastic, text" },
  "cap|thread":       { pos: "industrial metal surface defect, realistic foreign particle contamination, professional quality inspection photograph, sharp focus, metallic surface", neg: "cartoon, painting, blurry, low quality, plastic, text" },
  "cap|can_mieng":    { pos: "plastic bottle rim crush defect, local deformation on dark oval rim, bright streak with adjacent shadow, micro surface damage, industrial inspection, sharp focus", neg: "cartoon, painting, blurry, low quality, text, large crack" },
  // Pharma (capsule_experiments SDXL)
  "pharma|crack":     { pos: "pharmaceutical tablet surface defect, realistic surface crack line, quality inspection close-up, sharp focus, uniform background", neg: "cartoon, painting, blurry, low quality, text" },
  "pharma|dent":      { pos: "pharmaceutical tablet surface defect, realistic physical dent depression, quality inspection close-up, sharp focus, uniform background", neg: "cartoon, painting, blurry, low quality, text" },
  // Metal Cap (mc_deform / ring_fracture / scratch)
  "metal_cap|mc_deform":     { pos: "irregular industrial metal defect, crushed rim, jagged metallic edges, deep dent, heavy specular reflections, polished chrome, photorealistic, high contrast, non-geometric damage", neg: "smooth, perfect circle, plastic, matte, flat, low quality, sphere" },
  "metal_cap|ring_fracture": { pos: "extremely sharp industrial metal surface, hyper-detailed steel grain, microscopic metallic scratches, high contrast, 8k, ultra sharp focus", neg: "blur, soft, out of focus, bokeh, smooth, plastic, paint, fog, glowing edge, noise, compression artifacts" },
  "metal_cap|scratch":       { pos: "realistic metal scratch, deep industrial gouge, torn raw steel, irregular jagged edges, harsh specular glints, metallic burrs, highly detailed metallic texture, industrial damage, 8k, harsh lighting", neg: "paint, drawing, plastic, blur, soft edges, uniform texture, artificial, flat, cartoon" },
};
