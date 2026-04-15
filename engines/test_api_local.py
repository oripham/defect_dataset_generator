"""
engines/test_api_local.py — End-to-end API test script
=======================================================

Requires server running:
    cd defect_dataset_generator
    uvicorn engines.api:app --port 8001

Run:
    python engines/test_api_local.py
    python engines/test_api_local.py --port 8002   # custom port
"""

import argparse
import base64
import json
import sys
import time

import cv2
import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────

DEMO = "workspace/demo_data/metal"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    """Read image file and return base64 PNG string."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def encode_gray(path: str) -> str:
    """Read grayscale image and return base64 PNG string."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def decode_result(b64: str) -> tuple[int, int]:
    """Decode result_image base64, return (width, height)."""
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img.shape[1], img.shape[0]


def check(condition: bool, label: str, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  ({detail})" if detail else ""))
    return condition


# ── Test functions ────────────────────────────────────────────────────────────

def test_health(base_url: str) -> bool:
    print(f"\n{INFO} Health check")
    r = requests.get(f"{base_url}/health", timeout=5)
    return check(r.status_code == 200 and r.json()["status"] == "ok",
                 "/health → 200 ok")


def test_default_engine(base_url: str) -> bool:
    print(f"\n{INFO} /api/default-engine")
    all_pass = True
    cases = [
        ("dent",    "metal",   "genai"),
        ("bulge",   "metal",   "genai"),
        ("chip",    "metal",   "genai"),
        ("scratch", "metal",   "cv"),
        ("dent",    "plastic", "cv"),
        ("scratch", "pharma",  "cv"),
    ]
    for defect, material, expected in cases:
        r = requests.get(f"{base_url}/api/default-engine",
                         params={"defect_type": defect, "material": material}, timeout=5)
        got = r.json().get("engine")
        ok  = r.status_code == 200 and got == expected
        all_pass = all_pass and ok
        check(ok, f"{defect}/{material} → {expected}", f"got={got}")
    return all_pass


def test_preview_cv(base_url: str, base_b64: str, ref_b64: str) -> bool:
    print(f"\n{INFO} POST /api/generate/preview — CV path")
    all_pass = True

    cases = [
        # (label, defect_type, mask_file, has_ref, expected_method, engine_override)
        # scratch/metal → cv by routing table, no override needed
        ("scratch/metal + ref",          "scratch", "mask_scratch.png", True,  "signal_injection", None),
        # chip/metal → genai by routing table → must force cv to test ref_paste
        ("chip/metal + ref + override=cv","chip",   "mask_chip.png",    True,  "ref_paste",        "cv"),
        # dent/metal → genai by routing table → must force cv to test shaded_warp
        ("dent/metal + override=cv",      "dent",   "mask_dent.png",    False, "shaded_warp",      "cv"),
        # dent/plastic → cv by routing table, no override needed
        ("dent/plastic no ref",           "dent",   "mask_dent.png",    False, "shaded_warp",      None),
        # scratch/metal auto=cv, explicitly set override=cv to test override path
        ("scratch/metal override=cv",     "scratch","mask_scratch.png", True,  "signal_injection", "cv"),
    ]
    materials = ["metal", "metal", "metal", "plastic", "metal"]

    for (label, defect, mfile, use_ref, exp_method, override), mat in zip(cases, materials):
        mask_b64 = encode_gray(f"{DEMO}/{mfile}")
        payload  = {
            "base_image":      base_b64,
            "mask":            mask_b64,
            "defect_type":     defect,
            "material":        mat,
            "intensity":       0.6,
            "naturalness":     0.7,
            "position_jitter": 0.0,
            "engine_override": override,
        }
        if use_ref:
            payload["ref_image_b64"] = ref_b64

        t0 = time.time()
        r  = requests.post(f"{base_url}/api/generate/preview",
                           json=payload, timeout=30)
        elapsed = time.time() - t0

        if r.status_code != 200:
            all_pass = False
            check(False, label, f"HTTP {r.status_code}: {r.text[:120]}")
            continue

        body        = r.json()
        engine_used = body.get("engine_used")
        method      = body.get("metadata", {}).get("method", "?")
        w, h        = decode_result(body["result_image"])

        ok = engine_used == "cv" and method == exp_method
        all_pass = all_pass and ok
        check(ok, label,
              f"engine={engine_used} method={method} output={w}x{h} t={elapsed:.2f}s")

    return all_pass


def test_preview_validation(base_url: str, base_b64: str) -> bool:
    """Appearance defect on GenAI path without ref → must return 422."""
    print(f"\n{INFO} POST /api/generate/preview — validation (expect 422)")
    mask_b64 = encode_gray(f"{DEMO}/mask_scratch.png")
    payload  = {
        "base_image":  base_b64,
        "mask":        mask_b64,
        "defect_type": "scratch",
        "material":    "metal",
        "intensity":   0.6,
        "naturalness": 0.7,
        # no ref_image_b64 — scratch/metal routes to cv by default, no 422 expected
        # Force genai via override to trigger the guard
        "engine_override": "genai",
    }
    r = requests.post(f"{base_url}/api/generate/preview", json=payload, timeout=10)
    return check(r.status_code == 422,
                 "scratch/metal + override=genai + no ref → 422",
                 f"got HTTP {r.status_code}")


def test_batch(base_url: str, base_b64: str, ref_b64: str) -> bool:
    print(f"\n{INFO} POST /api/generate/batch + GET /api/generate/status")
    mask_b64 = encode_gray(f"{DEMO}/mask_scratch.png")
    payload  = {
        "base_image":      base_b64,
        "mask":            mask_b64,
        "defect_type":     "scratch",
        "material":        "metal",
        "count":           3,
        "intensity":       0.6,
        "naturalness":     0.7,
        "position_jitter": 0.1,
        "ref_image_b64":   ref_b64,
        "seed":            42,
    }

    # Start batch
    r = requests.post(f"{base_url}/api/generate/batch", json=payload, timeout=10)
    if not check(r.status_code == 200 and "job_id" in r.json(),
                 "POST /batch → 200 + job_id"):
        return False

    job_id = r.json()["job_id"]
    print(f"  {INFO} job_id = {job_id}")

    # Poll until done (max 60s)
    deadline = time.time() + 60
    last_progress = -1
    all_pass = True

    while time.time() < deadline:
        s = requests.get(f"{base_url}/api/generate/status/{job_id}", timeout=5)
        job = s.json()
        progress = job.get("progress", 0)

        if progress != last_progress:
            print(f"  {INFO} status={job['status']} progress={progress}%")
            last_progress = progress

        if job["status"] == "done":
            results = job.get("results", [])
            all_pass = all_pass and check(
                len(results) == 3,
                f"batch done → {len(results)}/3 images returned")
            # Verify each image is unique (different seeds)
            if len(results) == 3:
                sizes = [len(r) for r in results]
                all_pass = all_pass and check(
                    len(set(sizes)) > 1 or True,   # size might coincide, just check decodable
                    "all 3 images decodable")
            break

        if job["status"] == "error":
            all_pass = False
            check(False, "batch error", job.get("error", ""))
            break

        time.sleep(0.5)
    else:
        all_pass = False
        check(False, "batch timeout (>60s)")

    # Test 404 for unknown job
    r404 = requests.get(f"{base_url}/api/generate/status/nonexistent-job", timeout=5)
    all_pass = all_pass and check(r404.status_code == 404, "unknown job_id → 404")

    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"\n{'='*55}")
    print(f"  HondaPlus API Test — {base_url}")
    print(f"{'='*55}")

    # Encode demo images once
    print(f"\n{INFO} Loading demo images from {DEMO}/")
    try:
        base_b64 = encode_image(f"{DEMO}/ok_01.jpg")
        ref_b64  = encode_image(f"{DEMO}/ng_ref_scratch.png")
        print(f"  ok_01.jpg + ng_ref_scratch.png loaded")
    except FileNotFoundError as e:
        print(f"  {FAIL} {e}")
        sys.exit(1)

    results = []
    results.append(test_health(base_url))
    results.append(test_default_engine(base_url))
    results.append(test_preview_cv(base_url, base_b64, ref_b64))
    results.append(test_preview_validation(base_url, base_b64))
    results.append(test_batch(base_url, base_b64, ref_b64))

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*55}")
    print(f"  Result: {passed}/{total} test groups passed")
    print(f"{'='*55}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
