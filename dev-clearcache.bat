@echo off
REM Variant launcher: enables torch.cuda.empty_cache() between every
REM Parakeet ASR request via AISTACK_PARAKEET_CLEAR_CACHE_BETWEEN=1.
REM
REM Use this for the memory-dynamics A/B experiments. See
REM docs/research-note/_wip-parakeet-memory-dynamics.md for the
REM hypothesis being tested.
REM
REM Side effect: loses warm-cache pool reuse between same-shape
REM requests, may make best-case wall time worse but worst-case
REM (size-mismatch fragmentation) wall time better. Compare against
REM baseline runs from dev.bat with the same audio + warm-up history.

set AISTACK_PARAKEET_CLEAR_CACHE_BETWEEN=1
echo [aistack] CLEAR_CACHE_BETWEEN mode = ON
call "%~dp0scripts\dev.bat"
