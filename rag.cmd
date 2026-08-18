@echo off
REM Launcher that works around the broken .venv (its pyvenv.cfg points at a
REM base interpreter that does not exist on this machine). The site-packages
REM directory itself is fine, so we borrow it via PYTHONPATH instead of
REM reinstalling ~2GB of torch.
REM
REM The win32 entries are for pywin32, which Qdrant's embedded mode needs for
REM file locking. A proper `pip install -e .` wires those up via a .pth file.
REM
REM Usage:  .\rag.cmd ask "How long do refunds take?"
REM         .\rag.cmd eval --all
setlocal
set "PYTHONPATH=%~dp0.venv\Lib\site-packages;%~dp0.venv\Lib\site-packages\win32\lib;%~dp0.venv\Lib\site-packages\win32;%~dp0src"
python -m rag_app %*
