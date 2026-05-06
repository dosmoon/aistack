"""Admin Web UI — server-rendered Jinja templates + HTMX fragments.

Mounted at /admin by aistack/main.py. Read-only in this MVP: shows
models, GPU memory, lock state, model cache state, and a tail of
recent log lines. Model install / uninstall is intentionally out of
scope for the first iteration.
"""
