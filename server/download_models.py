"""
download_models.py — Pre-download all model weights for HondaPlus defect generation.

Run ONCE on a new pod before starting the server:
    python download_models.py

Models & verified snapshot IDs (as of 2026-03-28):
  - diffusers/stable-diffusion-xl-1.0-inpainting-0.1   115134f363124c53c7d878647567d04daf26e41e  (19.4 GB)
  - diffusers/controlnet-depth-sdxl-1.0                17bb97973f29801224cd66f192c5ffacf82648b4  (4.7 GB)
  - h94/IP-Adapter  (sdxl_models + image_encoder)      018e402774aeeddd60609b4ecdb7e298259dc729  (3.3 GB)
  - Intel/dpt-large                                     bc15f29aa3a80d532f2ed650b5e16ac48d8958f9  (1.3 GB)
  - black-forest-labs/FLUX.1-dev                        main (gated)                             (~35.0 GB)

Total: ~65 GB — estimated download time 60-90 min on RunPod.

Cache location: $HF_HOME/hub (set HF_HOME=/workspace/models on RunPod)
"""

import os
import sys

from huggingface_hub import snapshot_download, hf_hub_download

# ── Verified snapshot IDs (pinned for reproducibility) ───────────────────────
SNAPSHOTS = {
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": "115134f363124c53c7d878647567d04daf26e41e",
    "diffusers/controlnet-depth-sdxl-1.0":              "17bb97973f29801224cd66f192c5ffacf82648b4",
    "h94/IP-Adapter":                                   "018e402774aeeddd60609b4ecdb7e298259dc729",
    "Intel/dpt-large":                                  "bc15f29aa3a80d532f2ed650b5e16ac48d8958f9",
}

CACHE_DIR = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def dl(repo, revision=None, allow_patterns=None, ignore_patterns=None):
    rev = revision or SNAPSHOTS.get(repo)
    print(f"\n[DL] {repo}  (revision={rev}) ...")
    try:
        path = snapshot_download(
            repo,
            revision=rev,
            cache_dir=CACHE_DIR,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        print(f"     ✓ {path}")
        return path
    except Exception as e:
        print(f"     ✗ ERROR: {e}")
        return None


def dl_file(repo, filename, subfolder=None, revision=None):
    rev = revision or SNAPSHOTS.get(repo)
    label = f"{subfolder}/{filename}" if subfolder else filename
    print(f"\n[DL] {repo} / {label}  (revision={rev}) ...")
    try:
        kw = dict(cache_dir=CACHE_DIR, revision=rev)
        if subfolder:
            kw["subfolder"] = subfolder
        path = hf_hub_download(repo, filename, **kw)
        print(f"     ✓ {path}")
        return path
    except Exception as e:
        print(f"     ✗ ERROR: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("HondaPlus — Model Download Script")
print(f"Cache dir: {CACHE_DIR}")
print("=" * 60)

# 1. SDXL Inpainting (base pipeline)
print("\n[1/4] SDXL Inpainting base model (~19.4 GB)")
dl(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ignore_patterns=["*.msgpack", "flax_model*", "*.ot", "rust_model*"],
)

# 2. ControlNet Depth SDXL
print("\n[2/4] ControlNet Depth SDXL (~4.7 GB)")
dl(
    "diffusers/controlnet-depth-sdxl-1.0",
    ignore_patterns=["*.msgpack", "flax_model*"],
)

# 3. IP-Adapter: image encoder (ViT-H) + ip-adapter-plus weights
print("\n[3/4] IP-Adapter ViT-H image encoder + Plus weights (~3.3 GB)")
dl(
    "h94/IP-Adapter",
    allow_patterns=[
        "models/image_encoder/*",        # ViT-H encoder (config.json + model.safetensors)
        "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",  # IP-Adapter Plus weights
    ],
)

# 4. Intel DPT-Large (depth estimator)
print("\n[4/5] Intel DPT-Large depth estimator (~1.3 GB)")
dl("Intel/dpt-large")

# 5. FLUX.1-dev
print("\n[5/5] FLUX.1-dev (~35.0 GB)")
print("Note: Requires HF_TOKEN environment variable for gated access.")
dl(
    "black-forest-labs/FLUX.1-dev", 
    revision="main",
    ignore_patterns=["*.msgpack", "flax_model*", "*.ot", "rust_model*", "*.tfevents*"]
)

print("\n" + "=" * 60)
print("✅ All downloads complete!")
print("=" * 60)
print("\nNext step: start the server")
print("  cd /app && nohup uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/server.log 2>&1 &")
