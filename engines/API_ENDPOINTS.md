# HondaPlus Defect Generation API - Endpoint Summary

This document describes endpoints from `engines/api.py` (standalone FastAPI server).

## Run

```bash
cd defect_dataset_generator
uvicorn engines.api:app --port 8001 --reload
```

Base URL (default): `http://127.0.0.1:8001`

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check + GPU info |
| GET | `/api/default-engine` | Get default engine by defect/material |
| POST | `/api/generate/preview` | Generate 1 image synchronously |
| POST | `/api/generate/batch` | Start background batch job |
| GET | `/api/generate/status/{job_id}` | Poll batch job status |

## GET `/health`

Response `200`:

```json
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "NVIDIA ...",
  "gpu_memory": "12288 MB"
}
```

## GET `/api/default-engine`

Query params:

- `defect_type` (string, required)
- `material` (string, required)

Response `200`:

```json
{
  "engine": "genai",
  "defect_type": "dent",
  "material": "metal"
}
```

## POST `/api/generate/preview`

Request body (`application/json`):

```json
{
  "base_image": "<base64 PNG>",
  "mask": "<base64 PNG grayscale>",
  "defect_type": "scratch",
  "material": "metal",
  "intensity": 0.6,
  "naturalness": 0.7,
  "position_jitter": 0.0,
  "engine_override": null,
  "ref_image_b64": null,
  "seed": null
}
```

Notes:

- `engine_override`: `"cv"` or `"genai"` or `null`.
- `ref_image_b64` is required when request is routed to `genai` for non-shape defects.
- Shape defects (`dent`, `bulge`) do not require `ref_image_b64`.
- Missing required ref on GenAI path returns `422`.

Response `200`:

```json
{
  "result_image": "<base64 PNG>",
  "engine_used": "cv",
  "metadata": {}
}
```

## POST `/api/generate/batch`

Request body is same as preview, plus:

- `count` (integer, default `10`)

Response `200`:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## GET `/api/generate/status/{job_id}`

Response `200`:

```json
{
  "status": "running",
  "progress": 30,
  "total": 10,
  "results": ["<base64 PNG>"],
  "engine": "genai",
  "error": null
}
```

Status values:

- `queued`
- `running`
- `done`
- `error`

Error cases:

- `404` if `job_id` is not found.

## Useful auto docs

- `GET /docs`
- `GET /openapi.json`

