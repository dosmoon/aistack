@echo off
REM Dev server launcher (Windows). Default port 11500.
REM Usage: scripts\dev.bat
python -m uvicorn aistack.main:app --host 127.0.0.1 --port 11500 --reload
