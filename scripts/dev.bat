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

REM Point HuggingFace cache at the shared model dir so faster-whisper /
REM other ASR backends find pre-downloaded weights instead of re-pulling
REM into %USERPROFILE%\.cache\huggingface.
if "%HF_HOME%"=="" (
    set HF_HOME=D:\AI_Models\hf
)

cd /d "%REPO_ROOT%"
"%PY%" -m uvicorn aistack.main:app --host 127.0.0.1 --port 11500 --reload
endlocal
