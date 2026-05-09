"""Generate the configuration reference markdown from aistack/config.py.

Source of truth is the dataclasses in `aistack/config.py`. Each
domain dataclass (ModelCacheConfig / ParakeetConfig / BackendsConfig /
ObservabilityConfig) has a `from_env` classmethod that reads
`AISTACK_*` environment variables and applies defaults.

This generator parses the file's AST to extract:
  - dataclass name + docstring (becomes section header + intro)
  - per-field name (becomes the dataclass attribute)
  - per-field type annotation (str / bool / int / float / Path)
  - inline comment on the field declaration (becomes the "effect" cell)
  - env variable name + default + parser function from the
    matching `_env*("NAME", default)` call inside `from_env`

Output goes to docs/public/reference/configuration.md, which is
committed to the repo (CI cannot regenerate — no aistack venv there).

Run automatically by site/scripts/sync-docs.mjs prebuild, or directly:

    python scripts/gen_config_reference.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_FILE = ROOT / "aistack" / "config.py"
OUT_FILE = ROOT / "docs" / "public" / "reference" / "configuration.md"

# Friendly section labels keyed by dataclass name. Order = sidebar order
# in the rendered markdown.
SECTIONS: list[tuple[str, str]] = [
    ("ModelCacheConfig", "Model lifecycle"),
    ("ParakeetConfig", "Parakeet ASR"),
    ("BackendsConfig", "Backend upstream URLs"),
    ("ObservabilityConfig", "Observability"),
]

# Map _env* parser function names to a friendly type tag.
PARSER_TO_TYPE = {
    "_env": "string",
    "_env_bool": "bool",
    "_env_int": "int",
    "_env_float": "float",
}


def _read_source_with_lines() -> tuple[str, list[str], ast.Module]:
    src = CONFIG_FILE.read_text(encoding="utf-8")
    return src, src.splitlines(), ast.parse(src)


def _classes_by_name(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """Top-level dataclasses keyed by class name."""
    out: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = node
    return out


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        # Match either `@dataclass` or `@dataclass(frozen=True)`.
        target = d.func if isinstance(d, ast.Call) else d
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
    return False


def _field_inline_comments(cls: ast.ClassDef, src_lines: list[str]) -> dict[str, str]:
    """Map field-name → inline comment text (without the `#`).

    Inline comment is whatever follows a `#` on the same source line as
    the field's annotation. Returns "" for fields without a comment.
    """
    out: dict[str, str] = {}
    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            line = src_lines[item.lineno - 1]
            if "#" in line:
                comment = line.split("#", 1)[1].strip()
            else:
                comment = ""
            out[item.target.id] = comment
    return out


def _format_default(node: ast.AST) -> str:
    """Render a default-value AST node as a human string.

    Constants render as `repr(value)`. Anything else falls back to
    `ast.unparse` so calls like `_default_payload_dir()` show up
    verbatim instead of as opaque object references.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return repr(node.value)
    return ast.unparse(node)


def _walk_env_calls(from_env: ast.FunctionDef) -> list[tuple[str | None, str, str, str]]:
    """Find every `_env*("NAME", default)` call inside `from_env`.

    Returns tuples of (kwarg_name, env_name, default_str, parser_name).
    `kwarg_name` is the field this env value populates when the env
    call sits directly inside a keyword argument; None when the call
    is buried in a deeper expression (e.g. `int(_env_float(...) * GB)`)
    — the field is then identified by string-search through the
    enclosing keyword.
    """
    results: list[tuple[str | None, str, str, str]] = []

    # Find the `cls(...)` call (the dataclass constructor) inside the return.
    target_keywords: list[ast.keyword] = []
    for node in ast.walk(from_env):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            target_keywords.extend(node.value.keywords)

    for kw in target_keywords:
        env_calls = _find_env_calls_in(kw.value)
        if not env_calls:
            # Field that does not read env directly (e.g. log_dir uses a
            # helper that wraps the env read). We'll handle these via
            # post-processing below.
            continue
        for parser, env_name, default in env_calls:
            results.append((kw.arg, env_name, default, parser))
    return results


def _find_env_calls_in(expr: ast.AST) -> list[tuple[str, str, str]]:
    """Return (parser_name, env_name, default_str) for each _env*()
    call reachable from `expr`."""
    found: list[tuple[str, str, str]] = []
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name):
            continue
        if not func.id.startswith("_env"):
            continue
        if len(node.args) < 2:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        env_name = first.value
        default_str = _format_default(node.args[1])
        found.append((func.id, env_name, default_str))
    return found


def _helper_env_calls(tree: ast.Module) -> list[tuple[str, str, str, str]]:
    """Pick up env vars read inside helper functions (path computers).

    Returns (parser_name, env_name, default_str, helper_name). The
    helper_name lets the renderer point readers at the function that
    decides how the default falls out (e.g. `_default_payload_dir`
    cascades AISTACK_OBS_PAYLOAD_DIR → HF_HOME → cwd/captures).
    """
    out: list[tuple[str, str, str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("_default_"):
            continue
        # Catch both _env*("NAME", default) and os.environ.get("NAME").
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            # Pattern A: _env*("NAME", default)
            if isinstance(call.func, ast.Name) and call.func.id.startswith("_env"):
                if (
                    len(call.args) >= 2
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    out.append((
                        call.func.id,
                        call.args[0].value,
                        _format_default(call.args[1]),
                        node.name,
                    ))
            # Pattern B: os.environ.get("NAME")
            if isinstance(call.func, ast.Attribute) and call.func.attr == "get":
                if (
                    isinstance(call.func.value, ast.Attribute)
                    and call.func.value.attr == "environ"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    out.append((
                        "os.environ",
                        call.args[0].value,
                        f"<see {node.name}>",
                        node.name,
                    ))
    return out


def _render_section(cls: ast.ClassDef, src_lines: list[str], label: str) -> str:
    """Render one dataclass as a markdown section."""
    lines: list[str] = []
    lines.append(f"## {label}")
    lines.append("")
    docstring = (ast.get_docstring(cls) or "").strip()
    if docstring:
        lines.append(docstring)
        lines.append("")

    inline = _field_inline_comments(cls, src_lines)

    from_env = next(
        (b for b in cls.body if isinstance(b, ast.FunctionDef) and b.name == "from_env"),
        None,
    )
    rows: list[tuple[str, str, str, str, str]] = []  # field, env, type, default, effect
    if from_env is not None:
        for field, env, default, parser in _walk_env_calls(from_env):
            type_str = PARSER_TO_TYPE.get(parser, parser)
            effect = inline.get(field or "", "") if field else ""
            rows.append((field or "", env, type_str, default, effect))

    if not rows:
        lines.append("_(No env-driven fields in this section.)_")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Env variable | Type | Default | Effect |")
    lines.append("|---|---|---|---|")
    for field, env, type_str, default, effect in rows:
        cell_effect = effect.replace("|", "\\|") if effect else ""
        lines.append(f"| `{env}` | {type_str} | `{default}` | {cell_effect} |")
    lines.append("")
    lines.append(
        f"Source: `aistack/config.py` → `{cls.name}.from_env()` "
        f"(line {cls.lineno})"
    )
    lines.append("")
    return "\n".join(lines)


def _render_helper_section(helpers: list[tuple[str, str, str, str]]) -> str:
    """Render env vars consumed by helper functions (path computers)."""
    if not helpers:
        return ""
    lines: list[str] = []
    lines.append("## Path overrides (resolved by helpers)")
    lines.append("")
    lines.append(
        "These env vars are read inside helper functions in "
        "`aistack/config.py`, not directly inside a `from_env` "
        "classmethod, because their default behaviour cascades over "
        "multiple env vars. The **Default** column shows the literal "
        "default; the **Resolver** column points at the helper "
        "function that decides the cascade order."
    )
    lines.append("")
    lines.append("| Env variable | Type | Default | Resolver |")
    lines.append("|---|---|---|---|")
    for parser, env, default, helper in helpers:
        type_str = PARSER_TO_TYPE.get(parser, "string")
        lines.append(f"| `{env}` | {type_str} | `{default}` | `{helper}` |")
    lines.append("")
    return "\n".join(lines)


FRONTMATTER = """\
---
title: Configuration Reference
description: "Auto-generated reference for every AISTACK_* environment variable, with types, defaults, and effects. Source — aistack/config.py."
sidebar:
  order: 0
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/config.py dataclasses,
rendered by scripts/gen_config_reference.py. -->

# Configuration reference

Every aistack environment variable, organised by feature domain. The
**Default** column shows the value applied when the variable is unset
(or set to a value the parser cannot decode). The **Effect** column
is the inline comment from the source dataclass field — concise by
design; for design rationale (why the default is what it is, when to
deviate), see [the configuration narrative](../../configuration/).

aistack reads configuration once at process start. Changing an env
variable after the worker is up requires a restart. The three
observability toggles (metrics / access_log / payload) are the only
exception — they can also be flipped live from the `/admin` dashboard,
but that is session-only and not persisted.

"""


def main() -> int:
    src, src_lines, tree = _read_source_with_lines()
    classes = _classes_by_name(tree)
    helpers = _helper_env_calls(tree)

    # Helpers point at env vars that are *also* read by from_env
    # (AISTACK_OBS_PAYLOAD_DIR + HF_HOME). Dedup against from_env env
    # names so we only render helper-only ones.
    direct_env_names: set[str] = set()
    for cls_name, _label in SECTIONS:
        cls = classes.get(cls_name)
        if cls is None:
            continue
        from_env = next(
            (b for b in cls.body if isinstance(b, ast.FunctionDef) and b.name == "from_env"),
            None,
        )
        if from_env is not None:
            for _f, env, _d, _p in _walk_env_calls(from_env):
                direct_env_names.add(env)
    helpers_filtered = [(p, e, d, h) for (p, e, d, h) in helpers if e not in direct_env_names]

    parts: list[str] = [FRONTMATTER]
    for cls_name, label in SECTIONS:
        cls = classes.get(cls_name)
        if cls is None:
            parts.append(f"## {label}\n\n_(class `{cls_name}` not found in aistack/config.py)_\n")
            continue
        if not _is_dataclass(cls):
            parts.append(f"## {label}\n\n_(class `{cls_name}` is not a dataclass)_\n")
            continue
        parts.append(_render_section(cls, src_lines, label))

    parts.append(_render_helper_section(helpers_filtered))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"[gen_config_reference] wrote {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
