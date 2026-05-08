"""Generate Starlight markdown reference docs from FastAPI's OpenAPI spec.

Source of truth is the aistack source code itself: Pydantic models in
aistack/api/_schemas.py, Form()/Body() declarations on the route
handlers, and the docstrings on those handlers. FastAPI compiles all
of this into an OpenAPI 3.x spec at app.openapi(). This script reads
that spec and renders one Starlight markdown page per route group
into docs/public/api/reference/.

Run automatically by site/scripts/sync-docs.mjs prebuild, or directly:

    python scripts/gen_api_reference.py

The output directory is gitignored — the markdown is build output.
Editing the rendered files has no lasting effect because the next
build overwrites them. Edit the source code (docstrings + Pydantic
models + Form descriptions) instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Make the aistack package importable when this script is run directly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Importing aistack.main starts logging + registers middleware. That's
# fine for a one-shot script — we just want app.openapi().
from aistack.main import app  # noqa: E402

OUT_DIR = ROOT / "docs" / "public" / "api" / "reference"

# Logical groups for the reference pages. Each group becomes one
# markdown file at docs/public/api/reference/<group>.md. Endpoints
# whose path matches the prefix are rendered into that group.
#
# Order matters for sidebar layout (sidebar.order in frontmatter).
GROUPS: list[dict[str, Any]] = [
    {
        "slug": "models",
        "title": "Inventory & health",
        "description": "Auto-generated reference for GET /health and GET /v1/models — what the gateway can serve right now.",
        "order": 10,
        "paths": ["/health", "/v1/models"],
    },
    {
        "slug": "asr",
        "title": "ASR — speech to text",
        "description": "Auto-generated reference for POST /v1/audio/transcriptions. OpenAI Whisper API compatible.",
        "order": 11,
        "paths": ["/v1/audio/transcriptions"],
    },
    {
        "slug": "tts",
        "title": "TTS — text to speech",
        "description": "Auto-generated reference for the /v1/audio/{path} reverse proxy to Qwen3-TTS.",
        "order": 12,
        "paths_prefix": "/v1/audio/{path}",
    },
    {
        "slug": "llm",
        "title": "LLM — chat completion",
        "description": "Auto-generated reference for POST /v1/chat/completions, reverse-proxied to local Ollama.",
        "order": 13,
        "paths": ["/v1/chat/completions"],
    },
    {
        "slug": "admin",
        "title": "Admin runtime controls",
        "description": "Auto-generated reference for the runtime control endpoints under /admin/api/.",
        "order": 14,
        "paths_prefix": "/admin/api",
    },
]


def _resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve `#/components/schemas/Foo` → schema dict."""
    parts = ref.lstrip("#/").split("/")
    obj: Any = spec
    for p in parts:
        obj = obj[p]
    return obj


def _render_type(prop: dict) -> str:
    """Compact type string for a JSON Schema fragment."""
    if "$ref" in prop:
        name = prop["$ref"].split("/")[-1]
        return f"[`{name}`](#schema-{name.lower()})"
    if "anyOf" in prop:
        non_null = [sub for sub in prop["anyOf"] if sub.get("type") != "null"]
        if not non_null:
            return "null"
        rendered = " \\| ".join(_render_type(sub) for sub in non_null)
        if len(non_null) < len(prop["anyOf"]):
            rendered = f"{rendered} \\| null"
        return rendered
    if "oneOf" in prop:
        return " \\| ".join(_render_type(sub) for sub in prop["oneOf"])
    if "enum" in prop:
        vals = ", ".join(f"`{v!r}`" for v in prop["enum"])
        return f"enum ({vals})"
    if prop.get("type") == "array":
        items = prop.get("items", {})
        return f"array of {_render_type(items)}"
    if prop.get("type") == "object":
        return "object"
    t = prop.get("type", "any")
    fmt = prop.get("format")
    return f"{t} ({fmt})" if fmt else t


def _escape_table_cell(s: str) -> str:
    return (s or "").replace("\n", " ").replace("|", "\\|").strip()


def _render_schema_table(spec: dict, schema_name: str) -> str:
    """Render a Pydantic schema's properties as a markdown table.

    Returns an empty string for schemas without properties (e.g. simple
    string aliases) — the caller decides whether to skip the section.
    """
    schema = spec["components"]["schemas"].get(schema_name)
    if not schema or "properties" not in schema:
        return ""
    lines = ["| Field | Type | Required | Description |", "|---|---|---|---|"]
    required = set(schema.get("required", []))
    for name, prop in schema["properties"].items():
        type_str = _render_type(prop)
        req = "yes" if name in required else "no"
        desc = _escape_table_cell(prop.get("description", ""))
        # Honor Pydantic Field alias (we use it for "class" reserved-word case).
        alias = prop.get("alias") or name
        lines.append(f"| `{alias}` | {type_str} | {req} | {desc} |")
    return "\n".join(lines)


def _collect_referenced_schemas(operation: dict, found: set[str]) -> None:
    """Recurse into an operation dict collecting every $ref schema name.

    Used to populate the per-page "Schemas" section so each rendered
    page is self-contained: every schema mentioned in a request body,
    response, or parameter is shown as a field table at the bottom.
    """
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                name = node["$ref"].split("/")[-1]
                found.add(name)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(operation)


def _expand_transitive_schemas(spec: dict, found: set[str]) -> None:
    """Pull in schemas referenced by schemas already in `found`.

    A response that returns ErrorEnvelope only adds ErrorEnvelope to
    `found`; without this expansion the inner ErrorBody would not be
    rendered, leaving the reader with a dangling type reference. Walks
    the schema graph until no new names appear.
    """
    schemas = spec.get("components", {}).get("schemas", {})
    while True:
        before = set(found)
        for name in list(found):
            schema = schemas.get(name)
            if not schema:
                continue
            _collect_referenced_schemas(schema, found)
        if found == before:
            return


def _render_parameters_table(parameters: list) -> str:
    if not parameters:
        return ""
    lines = ["| Name | In | Type | Required | Description |", "|---|---|---|---|---|"]
    for p in parameters:
        name = p.get("name", "?")
        loc = p.get("in", "?")
        schema = p.get("schema", {})
        type_str = _render_type(schema)
        req = "yes" if p.get("required") else "no"
        desc = _escape_table_cell(p.get("description", ""))
        lines.append(f"| `{name}` | {loc} | {type_str} | {req} | {desc} |")
    return "\n".join(lines)


def _render_request_body(spec: dict, request_body: dict | None) -> list[str]:
    if not request_body:
        return []
    out: list[str] = ["### Request body"]
    if request_body.get("description"):
        out.append("")
        out.append(request_body["description"])
    out.append("")
    content = request_body.get("content", {})
    for ct, body in content.items():
        out.append(f"**Content type:** `{ct}`")
        out.append("")
        schema = body.get("schema", {})
        if "$ref" in schema:
            name = schema["$ref"].split("/")[-1]
            table = _render_schema_table(spec, name)
            if table:
                out.append(f"Schema: [`{name}`](#schema-{name.lower()})")
                out.append("")
                out.append(table)
            else:
                out.append(f"Schema: `{name}`")
        elif schema.get("type") == "object" and "properties" in schema:
            # Inline schema (e.g. multipart Body_*).
            lines = ["| Field | Type | Required | Description |", "|---|---|---|---|"]
            required = set(schema.get("required", []))
            for name, prop in schema["properties"].items():
                type_str = _render_type(prop)
                req = "yes" if name in required else "no"
                desc = _escape_table_cell(prop.get("description", ""))
                lines.append(f"| `{name}` | {type_str} | {req} | {desc} |")
            out.append("\n".join(lines))
        else:
            t = _render_type(schema)
            out.append(f"Type: {t}")
        out.append("")
    return out


def _render_responses(spec: dict, responses: dict) -> list[str]:
    out: list[str] = ["### Responses", ""]
    for status, resp in sorted(responses.items(), key=lambda kv: str(kv[0])):
        out.append(f"#### `{status}`")
        out.append("")
        if resp.get("description"):
            out.append(_escape_table_cell(resp["description"]))
            out.append("")
        content = resp.get("content", {})
        for ct, body in content.items():
            schema = body.get("schema", {})
            if "$ref" in schema:
                name = schema["$ref"].split("/")[-1]
                out.append(f"- `{ct}` → [`{name}`](#schema-{name.lower()})")
            elif schema:
                out.append(f"- `{ct}` → {_render_type(schema)}")
            else:
                out.append(f"- `{ct}`")
        out.append("")
    return out


def _render_endpoint(spec: dict, path: str, method: str, operation: dict) -> list[str]:
    out: list[str] = []
    summary = operation.get("summary") or "&nbsp;"
    out.append(f"## `{method.upper()} {path}`")
    out.append("")
    out.append(f"**{summary}**")
    out.append("")
    desc = operation.get("description") or ""
    if desc.strip():
        out.append(desc.strip())
        out.append("")
    if operation.get("parameters"):
        out.append("### Parameters")
        out.append("")
        out.append(_render_parameters_table(operation["parameters"]))
        out.append("")
    out.extend(_render_request_body(spec, operation.get("requestBody")))
    out.extend(_render_responses(spec, operation.get("responses", {})))
    return out


def _path_matches(path: str, group: dict) -> bool:
    if "paths" in group and path in group["paths"]:
        return True
    if "paths_prefix" in group and path.startswith(group["paths_prefix"]):
        return True
    return False


def _render_group(spec: dict, group: dict) -> str:
    """Render one Starlight markdown page for a route group."""
    parts: list[str] = []
    title = group["title"]
    description = group["description"]
    order = group["order"]
    parts.append("---")
    parts.append(f"title: {title}")
    parts.append(f"description: {description}")
    parts.append("sidebar:")
    parts.append(f"  order: {order}")
    parts.append("---")
    parts.append("")
    parts.append(
        "<!-- AUTO-GENERATED: do not edit. "
        "Source: aistack/api/* docstrings + Pydantic models in "
        "aistack/api/_schemas.py, rendered by scripts/gen_api_reference.py. -->"
    )
    parts.append("")

    referenced: set[str] = set()
    matched_any = False
    for path, methods in spec["paths"].items():
        if not _path_matches(path, group):
            continue
        for method, operation in methods.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue
            matched_any = True
            parts.extend(_render_endpoint(spec, path, method, operation))
            _collect_referenced_schemas(operation, referenced)

    if not matched_any:
        parts.append("(No endpoints in this group.)")
        parts.append("")

    _expand_transitive_schemas(spec, referenced)

    # Drop noisy auto-generated body schemas (FastAPI emits "Body_*"
    # for multipart/Form route bodies; the relevant info is already
    # rendered in the request body section).
    referenced = {n for n in referenced if not n.startswith("Body_")}
    referenced.discard("HTTPValidationError")
    referenced.discard("ValidationError")

    if referenced:
        parts.append("---")
        parts.append("")
        parts.append("## Schemas")
        parts.append("")
        for name in sorted(referenced):
            parts.append(f"### `{name}` {{#schema-{name.lower()}}}")
            schema = spec["components"]["schemas"].get(name, {})
            if schema.get("description"):
                parts.append("")
                parts.append(schema["description"].strip())
            parts.append("")
            table = _render_schema_table(spec, name)
            if table:
                parts.append(table)
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _render_index(groups: list[dict]) -> str:
    """Top-level index page for the reference section."""
    parts: list[str] = []
    parts.append("---")
    parts.append("title: Reference")
    parts.append("description: Auto-generated HTTP API reference for aistack — endpoints, schemas, error codes.")
    parts.append("sidebar:")
    parts.append("  order: 9")
    parts.append("---")
    parts.append("")
    parts.append(
        "<!-- AUTO-GENERATED: do not edit. "
        "Source: aistack/api/* docstrings + Pydantic models, rendered by "
        "scripts/gen_api_reference.py. -->"
    )
    parts.append("")
    parts.append("# HTTP API reference")
    parts.append("")
    parts.append(
        "These pages are generated from the live FastAPI app's OpenAPI "
        "spec on every build. The source of truth is the docstrings and "
        "Pydantic models in `aistack/api/`; editing the rendered "
        "markdown has no effect."
    )
    parts.append("")
    parts.append(
        "For the design rationale and integration journey (the *why*, "
        "not the *what*), start with the top-level "
        "[Integration Guide](../../integration/)."
    )
    parts.append("")
    parts.append("## Sections")
    parts.append("")
    for g in sorted(groups, key=lambda g: g["order"]):
        parts.append(f"- [{g['title']}](./{g['slug']}/) — {g['description']}")
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = app.openapi()

    # Index page.
    (OUT_DIR / "index.md").write_text(_render_index(GROUPS), encoding="utf-8")

    # Per-group pages.
    for group in GROUPS:
        page = _render_group(spec, group)
        (OUT_DIR / f"{group['slug']}.md").write_text(page, encoding="utf-8")

    print(f"[gen_api_reference] wrote {len(GROUPS) + 1} files into {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
