# scripts/generate_dataset.py
import argparse
import yaml
import os
import random
import shutil
import datetime
from PIL import Image

from generator import DefectGenerator
from generator_classical import ClassicalDefectGenerator
from generator_src import DefectGenerator as SrcDefectGenerator
from mask_sampler import sample_mask
from ref_loader import load_reference_image
from yolo_export import mask_to_yolo_polygons
from sdxl_refiner import SDXLRefiner


def main(cfg_path):
    # ---------- LOAD CONFIG ----------
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------- CHECK PATHS ----------
    paths = cfg["paths"]
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] Path not found: {k} -> {p}")

    # ---------- OUTPUT FOLDER ----------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(paths["output_root"], f"run_{ts}")
    img_out = os.path.join(out_root, "images")
    lbl_out = os.path.join(out_root, "labels")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    shutil.copy(cfg_path, os.path.join(out_root, "config.yaml"))

    # ---------- LOAD GOOD IMAGES ----------
    good_imgs = [
        os.path.join(paths["good_images"], f)
        for f in os.listdir(paths["good_images"])
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not good_imgs:
        raise RuntimeError("[ERROR] No good images found")

    print(f"[INFO] Found {len(good_imgs)} good images")

    # ---------- INIT GENERATOR ----------
    gen_type = cfg.get("model", {}).get("type", "sdxl")
    if gen_type == "classical":
        gen = ClassicalDefectGenerator(cfg)
    elif gen_type == "src":
        gen = SrcDefectGenerator(cfg)
    else:
        gen = DefectGenerator(cfg)

    # ---------- INIT SDXL REFINER (optional) ----------
    refine_cfg = cfg.get("sdxl_refine", {})
    refiner = None
    if refine_cfg.get("enabled", False):
        device = cfg.get("model", {}).get("device", "cuda")
        refiner = SDXLRefiner(refine_cfg, device=device)

    img_id = 0
    image_size = tuple(cfg["product"]["image_size"])

    # ---------- GENERATE ----------
    for cls in cfg["classes"]:
        cls_name = cls["name"]
        num_images = cls["generation"]["num_images"]

        print(f"\n[INFO] Generating class '{cls_name}' ({num_images} images)")

        for _ in range(num_images):
            base_path = random.choice(good_imgs)
            base = Image.open(base_path).convert("RGB").resize(image_size, Image.LANCZOS)

            gen_cfg = cls["generation"]
            rotation_range = (
                float(gen_cfg.get("mask_rotation_min", 0.0)),
                float(gen_cfg.get("mask_rotation_max", 0.0)),
            )
            offset_range = (
                int(gen_cfg.get("mask_offset_min", -15)),
                int(gen_cfg.get("mask_offset_max",  15)),
            )
            mask = sample_mask(
                paths["mask_root"],
                cls["mask_dir"],
                image_size,
                rotation_range=rotation_range,
                offset_range=offset_range,
            )

            ref = load_reference_image(
                paths["defect_refs"],
                cls["ref_dir"]
            )

            # Support both legacy single "prompt" and new "prompts" list
            prompts = gen_cfg.get("prompts") or [gen_cfg.get("prompt", "a defect on a product surface")]
            chosen_prompt = random.choice(prompts)
            effective_cfg = dict(gen_cfg)
            effective_cfg["prompt"] = chosen_prompt

            # src generator expects PIL mask, classical/sdxl expect np.ndarray
            mask_for_gen = Image.fromarray(mask) if gen_type == "src" else mask

            out_img = gen.generate(
                base_image=base,
                mask=mask_for_gen,
                ref_image=ref,
                gen_cfg=effective_cfg,
            )

            # ---------- PER-CLASS SDXL INPAINTING (high strength, defect prompt) --
            # Skip for src: pipeline already runs SDXL internally
            inpaint_cfg = gen_cfg.get("sdxl_inpaint", {})
            if gen_type != "src" and refiner is not None and inpaint_cfg.get("enabled", False):
                print(f"[INPAINT] Running high-strength SDXL inpaint on defect mask …")
                out_img = refiner.inpaint_defect(out_img, mask, inpaint_cfg,
                                                 ref_image=ref)

            # ---------- SDXL REFINEMENT (last step, appearance only) ----------
            # Skip for src: pipeline already runs SDXL internally
            if gen_type != "src" and refiner is not None:
                print(f"[REFINE] Running SDXL refinement …")
                out_img = refiner.refine_with_sdxl(out_img)

            img_name = f"{img_id:06d}.jpg"
            lbl_name = f"{img_id:06d}.txt"

            out_img.save(os.path.join(img_out, img_name))

            polygons = mask_to_yolo_polygons(mask)
            with open(os.path.join(lbl_out, lbl_name), "w") as f:
                for poly in polygons:
                    f.write(
                        f"{cls['class_id']} " +
                        " ".join(map(str, poly)) + "\n"
                    )

            print(f"[OK] {img_name}")
            img_id += 1

    # ---------- DATASET.YAML ----------
    with open(os.path.join(out_root, "dataset.yaml"), "w") as f:
        f.write(f"""path: {out_root}
train: images
val: images

names:
""")
        for cls in cfg["classes"]:
            f.write(f"  {cls['class_id']}: {cls['name']}\n")

    print("\n✅ Dataset generation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
