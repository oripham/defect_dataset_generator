@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set TOOL=V:\HondaPlus\defect_dataset_generator\tools\mask_annotator.py
set NG=V:\dataHondatPlus\test_samples\ng_mka
set OUT=V:\dataHondatPlus\test_samples\masks

if not exist "%OUT%" mkdir "%OUT%"

echo ============================================================
echo   MKA Defect Mask Annotator  (6 classes)
echo   Lan luot ve mask cho tung class loi MKA
echo.
echo   S / Enter  =  Luu va chuyen sang anh ke tiep
echo   Q / Esc    =  Bo qua anh nay
echo   Scroll     =  Doi kich thuoc brush
echo   Z = Undo   R = Reset
echo ============================================================
echo.

call :annotate "ng_foreign.bmp"  "[1/6] Di Vat (Foreign Object)"
call :annotate "ng_crater.bmp"   "[2/6] Crater Di Vat (Crater Foreign)"
call :annotate "ng_mouth.jpg"    "[3/6] Can Mieng (Mouth Dent)"
call :annotate "ng_plastic.bmp"  "[4/6] Loi Nhua (Plastic Defect)"
call :annotate "ng_thread.bmp"   "[5/6] Soi Chi (Thread Mark)"
call :annotate "ng_scratch.bmp"  "[6/6] Loi Tray (Scratch)"

echo.
echo ============================================================
echo   Xong! Masks da luu tai: %OUT%
echo.
for %%M in ("%OUT%\*_mask.png") do echo   %%~nxM
echo ============================================================
echo.
pause
goto :eof


:annotate
set FILE=%~1
set LABEL=%~2
set SRC=%NG%\%FILE%
set FNAME=%~n1
set MASK=%OUT%\%FNAME%_mask.png

echo.
echo %LABEL%
echo   Anh : %SRC%
echo   Mask: %MASK%

if not exist "%SRC%" (
    echo   [SKIP] Khong tim thay file: %SRC%
    goto :eof
)

if exist "%MASK%" (
    echo.
    echo   [!] Mask da ton tai. Ve lai? Y=ve lai, Enter=bo qua
    set /p REDO=  ^>
    if /i "!REDO!" neq "Y" (
        echo   Bo qua.
        goto :eof
    )
)

python "%TOOL%" "%SRC%" --out "%MASK%"
goto :eof
