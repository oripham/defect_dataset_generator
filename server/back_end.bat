@echo off
title Run Backend + Cloudflare Tunnel

echo ============================
echo CHECK cloudflared...
echo ============================

cloudflared --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Chua cai cloudflared!
    echo Dang cai bang winget...

    winget install Cloudflare.cloudflared

    echo.
    echo Cai xong, vui long CHAY LAI FILE BAT nay!
    pause
    exit
)

echo OK - cloudflared da san sang
echo.

echo ============================
echo START BACKEND (port 8000)
echo ============================

REM 👉 SUA DONG NAY neu file backend khac
start cmd /k "uvicorn main:app --host 0.0.0.0 --port 8000"

echo Cho backend khoi dong...
timeout /t 5 >nul

echo.
echo ============================
echo START CLOUDFARE TUNNEL
echo ============================

cloudflared tunnel --url http://localhost:8000

pause