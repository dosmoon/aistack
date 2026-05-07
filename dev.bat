@echo off
REM Quick launcher (root-level). Calls scripts\dev.bat for the actual
REM uvicorn invocation. Place at repo root for one-click double-click.
REM
REM This launcher uses BASELINE settings — no torch.cuda.empty_cache()
REM between requests. For the cache-clear A/B variant, run dev-clearcache.bat.

call "%~dp0scripts\dev.bat"
