@echo off
REM Dev server launcher (Windows). Default port 11500.
REM Usage: scripts\dev.bat   (run from repo root)
REM Uses myenv\Scripts\python.exe explicitly so it does not pick up
REM whichever `python` happens to be first on PATH.

setlocal
set REPO_ROOT=%~dp0..
set PY=%REPO_ROOT%\myenv\Scripts\python.exe

if not exist "%PY%" (
    echo [aistack] myenv not found at %PY%
    echo Run: uv venv --python 3.12 myenv ^&^& uv pip install --python myenv\Scripts\python.exe -e ".[dev]"
    exit /b 1
)

REM Point each runtime's model cache at the shared D:\AI_Models tree so
REM ASR backends find pre-downloaded weights instead of re-pulling GBs
REM into %USERPROFILE%\.cache\<vendor>. See docs/selection/runtimes.md
REM for the full mapping (HF -> faster-whisper / Parakeet / Qwen3-TTS,
REM ModelScope -> SenseVoice).
if "%HF_HOME%"=="" (
    set HF_HOME=D:\AI_Models\hf
)
if "%MODELSCOPE_CACHE%"=="" (
    set MODELSCOPE_CACHE=D:\AI_Models\modelscope
)
if "%NEMO_CACHE_DIR%"=="" (
    set NEMO_CACHE_DIR=D:\AI_Models\nemo
)

cd /d "%REPO_ROOT%"
"%PY%" -m uvicorn aistack.main:app --host 127.0.0.1 --port 11500 --reload
endlocal
