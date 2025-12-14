@echo off
setlocal
cd /d %~dp0

set "PIXI=%~dp0pixi.exe"
if not exist "%PIXI%" (
  echo Downloading pixi...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$url='https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.zip';" ^
    "$zip=Join-Path $env:TEMP 'pixi.zip';" ^
    "Invoke-WebRequest -Uri $url -OutFile $zip;" ^
    "Expand-Archive -Path $zip -DestinationPath '%~dp0' -Force;" ^
    "Remove-Item $zip -Force"
)

REM Installs env (downloads deps) if needed, then runs inside it
"%PIXI%" run --manifest-path "%~dp0pixi.toml" python src\client.py
