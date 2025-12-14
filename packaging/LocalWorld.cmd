@echo off
setlocal
cd /d %~dp0

set "APP=%LOCALAPPDATA%\LocalWorld"
if not exist "%APP%" mkdir "%APP%"

set "PIXI_HOME=%APP%\.pixi-home"
set "PIXI_CACHE_DIR=%APP%\.pixi-cache"
set "PIXI=%APP%\pixi.exe"

if not exist "%PIXI%" (
  echo Downloading pixi...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$url='https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.zip';" ^
    "$zip=Join-Path $env:TEMP 'pixi.zip';" ^
    "Invoke-WebRequest -Uri $url -OutFile $zip;" ^
    "Expand-Archive -Path $zip -DestinationPath '%APP%' -Force;" ^
    "Remove-Item $zip -Force"
)

copy /Y "%~dp0pixi.toml" "%APP%\pixi.toml" >nul
copy /Y "%~dp0client.py" "%APP%\client.py" >nul

pushd "%APP%"
"%PIXI%" install --manifest-path pixi.toml || exit /b 1
"%PIXI%" run --manifest-path pixi.toml python client.py
popd
