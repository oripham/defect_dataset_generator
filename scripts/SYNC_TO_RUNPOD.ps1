# scripts/SYNC_TO_RUNPOD.ps1
# ==============================================================================
# HondaPlus — RunPod Code Sync Script
# Target: /workspace/defect_generator
# Usage: .\scripts\SYNC_TO_RUNPOD.ps1
# ==============================================================================

$SSH_HOST = "runpod-flux"
$REMOTE_DIR = "/workspace/defect_generator"

Write-Host "🚀 [SYNC] Starting HondaPlus Backend sync to $SSH_HOST..." -ForegroundColor Cyan

# 1. Ensure remote directory exists
ssh $SSH_HOST "mkdir -p $REMOTE_DIR"

# 2. Sync core folders
Write-Host "📂 [SYNC] engines/ ..."
scp -r "engines" "${SSH_HOST}:${REMOTE_DIR}/"

Write-Host "📂 [SYNC] server/ ..."
scp -r "server" "${SSH_HOST}:${REMOTE_DIR}/"

Write-Host "📂 [SYNC] scripts/ ..."
scp -r "scripts" "${SSH_HOST}:${REMOTE_DIR}/"

# 3. Sync critical root files
Write-Host "📄 [SYNC] requirements.txt and tests ..."
scp "requirements.txt" "${SSH_HOST}:${REMOTE_DIR}/"
scp "test_backend.py" "${SSH_HOST}:${REMOTE_DIR}/"

Write-Host "==============================================================================" -ForegroundColor Green
Write-Host "✅ [DONE] Backend successfully synced to ${SSH_HOST}:${REMOTE_DIR}" -ForegroundColor Green
Write-Host "Next steps on RunPod:" -ForegroundColor Yellow
Write-Host "  1. cd $REMOTE_DIR"
Write-Host "  2. pip install -r server/requirements_server.txt"
Write-Host "  3. export HF_TOKEN=your_key"
Write-Host "  4. python3 server/download_models.py"
Write-Host "  5. uvicorn engines.api:app --host 0.0.0.0 --port 8000"
Write-Host "==============================================================================" -ForegroundColor Green
