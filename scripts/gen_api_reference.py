"""Generate Starlight markdown API reference by AST-walking aistack source.

Reads aistack source code as text — no imports, no runtime, no
FastAPI / Pydantic / Starlette installed required. Extracts:

  - route decorators (@router.post / @app.get / @router.api_route) +
    their kwargs (summary / response_model / responses / etc.)
  - route handler docstrings (first Constant(str) statement of the body)
  - Form / File / Body / Query parameter declarations + their
    `description=` argument
  - Pydantic model classes in aistack/api/_schemas.py + their fields
    (annotation, Field(description=...), required vs Optional)
  - Enum-like Literal[...] type annotations
  - APIRouter(prefix="/admin", ...) so admin route paths get prefixed
    correctly

Renders one Starlight markdown page per group (5 groups: models,
asr, tts, llm, admin) plus an index page. Pydantic schemas referenced
by any rendered route appear as a per-page "Schemas" section with
field tables.

Run:

    python scripts/gen_api_reference.py

Pure-stdlib script. The site/scripts/sync-docs.mjs prebuild hook calls
it before astro build. CI installs only python, no other dependencies.

Source-of-truth principle: editing the rendered markdown has no
lasting effect — the next build overwrites it. Edit the docstrings,
Pydantic models, or Form() descriptions in aistack/api/* instead.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Source files to walk for route decorators.
ROUTE_FILES = [
    ROOT / "aistack" / "main.py",
    ROOT / "aistack" / "api" / "asr.py",
    ROOT / "aistack" / "api" / "tts.py",
    ROOT / "aistack" / "api" / "llm.py",
    ROOT / "aistack" / "admin" / "router.py",
]

# Source file for Pydantic schemas referenced from routes.
SCHEMA_FILE = ROOT / "aistack" / "api" / "_schemas.py"

OUT_DIR = ROOT / "docs" / "public" / "api" / "reference"

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
        "paths_prefix": "/v1/audio/{path",  # matches /v1/audio/{path:path} or {path}
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


# ── Data shapes ─────────────────────────────────────────────────────────────

@dataclass
class Param:
    name: str
    location: str        # 'form' | 'body' | 'query' | 'path' | 'header' | 'file'
    type_str: str
    required: bool
    description: str
    default: str | None = None  # rendered Python literal or None for required


@dataclass
class StatusResponse:
    status: str
    description: str
    schema_ref: str | None        # schema name if `model=Foo`
    content: dict[str, str]       # content-type → schema-name or "string" etc.


@dataclass
class Route:
    path: str
    method: str
    summary: str
    docstring: str
    response_model: str | None
    response_model_exclude_none: bool
    responses: list[StatusResponse]
    parameters: list[Param]
    request_body_form: list[Param]   # Form/File parameters → multipart body


@dataclass
class SchemaField:
    name: str
    alias: str | None
    type_str: str
    required: bool
    description: str


@dataclass
class Schema:
    name: str
    description: str
    fields: list[SchemaField]


# ── Type-annotation rendering ───────────────────────────────────────────────

PRIMITIVE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "bytes": "binary",
    "Path": "string (filesystem path)",
    "dict": "object",
    "list": "array",
    # FastAPI / framework types that are not Pydantic schemas. Render
    # as a primitive so we don't link to a non-existent schema page.
    "UploadFile": "binary (multipart upload)",
    "Request": "request",
    "Response": "response",
    "BackgroundTasks": "background-tasks",
}


# Populated by main() before any rendering. Used by _render_type to
# decide whether a capitalised name is actually a Pydantic schema (link)
# or just some other class (render bare). Without this gate, any
# CamelCase name becomes a dangling [`Foo`](#schema-foo) link.
KNOWN_SCHEMAS: set[str] = set()


def _render_type(node: ast.AST | None) -> str:
    if node is None:
        return "any"

    # bare name → primitive or schema link
    if isinstance(node, ast.Name):
        name = node.id
        if name in PRIMITIVE:
            return PRIMITIVE[name]
        if name in KNOWN_SCHEMAS:
            return f"[`{name}`](#schema-{name.lower()})"
        return name

    # Constant (e.g. None in BinOp)
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "null"
        return repr(node.value)

    # X | Y (PEP 604)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _render_type(node.left)
        right = _render_type(node.right)
        # Normalise Optional[X] = X | None
        if right == "null":
            return f"{left} \\| null"
        if left == "null":
            return f"{right} \\| null"
        return f"{left} \\| {right}"

    # Subscript: list[X], dict[K, V], Literal[...], Optional[X], Union[...]
    if isinstance(node, ast.Subscript):
        base = _render_subscript_base(node.value)
        slice_node = node.slice
        if base == "Literal":
            return _render_literal(slice_node)
        if base == "Optional":
            return f"{_render_type(slice_node)} \\| null"
        if base in ("list", "List", "Sequence"):
            return f"array of {_render_type(slice_node)}"
        if base in ("dict", "Dict", "Mapping"):
            if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
                k, v = slice_node.elts
                return f"object ({_render_type(k)} → {_render_type(v)})"
            return "object"
        if base in ("Union",):
            if isinstance(slice_node, ast.Tuple):
                return " \\| ".join(_render_type(e) for e in slice_node.elts)
            return _render_type(slice_node)
        if base == "Annotated":
            # First slice element is the actual type; rest are metadata.
            if isinstance(slice_node, ast.Tuple) and slice_node.elts:
                return _render_type(slice_node.elts[0])
            return "any"
        # Generic: Foo[X] -> Foo
        return base

    # Attribute (e.g. typing.List[X])
    if isinstance(node, ast.Attribute):
        return node.attr

    # Tuple of types in a union
    if isinstance(node, ast.Tuple):
        return " \\| ".join(_render_type(e) for e in node.elts)

    # Fallback: source-level rendering
    try:
        return ast.unparse(node)
    except Exception:
        return "any"


def _render_subscript_base(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _render_literal(slice_node: ast.AST) -> str:
    """Render Literal[...] as `enum (\"a\", \"b\", \"c\")`."""
    items: list[ast.AST]
    if isinstance(slice_node, ast.Tuple):
        items = list(slice_node.elts)
    else:
        items = [slice_node]
    rendered: list[str] = []
    for it in items:
        if isinstance(it, ast.Constant):
            rendered.append(f"`{it.value!r}`")
        else:
            rendered.append(f"`{ast.unparse(it)}`")
    return f"enum ({', '.join(rendered)})"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_value(node: ast.AST | None) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _format_default(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return f'"{node.value}"'
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return repr(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _escape_table(s: str) -> str:
    return (s or "").replace("\n", " ").replace("|", "\\|").strip()


# ── Route extraction ────────────────────────────────────────────────────────

# Extract router.prefix from `router = APIRouter(prefix="/admin", ...)`.
def _router_prefix(tree: ast.Module) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "router"):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        # APIRouter(prefix="/admin", ...) or just APIRouter()
        prefix_node = _kw(node.value, "prefix")
        s = _const_str(prefix_node)
        if s is not None:
            return s
    return ""


# Map decorator like `@router.post(...)` → method, or None for non-route.
HTTP_METHODS = {"get", "post", "put", "delete", "patch"}


def _route_decorator_info(dec: ast.Call) -> tuple[str | None, str | None, ast.Call]:
    """Return (method_uppercase_or_None, path_or_None, the_call)."""
    if not isinstance(dec, ast.Call):
        return None, None, dec
    func = dec.func
    if isinstance(func, ast.Attribute):
        attr = func.attr
        # router.post / app.get / etc.
        if attr in HTTP_METHODS:
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                return attr.upper(), dec.args[0].value, dec
        # router.api_route(path, methods=[...])
        if attr == "api_route":
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                methods_node = _kw(dec, "methods")
                if isinstance(methods_node, ast.List):
                    # Pick the first method as a representative; multi-method
                    # routes should be split into per-method decorators
                    # in the source for cleaner OpenAPI ids.
                    for elt in methods_node.elts:
                        m = _const_str(elt)
                        if m and m.lower() in HTTP_METHODS:
                            return m.upper(), dec.args[0].value, dec
    return None, None, dec


def _extract_responses(node: ast.AST | None) -> list[StatusResponse]:
    """Parse a `responses={...}` kwarg dict literal."""
    out: list[StatusResponse] = []
    if not isinstance(node, ast.Dict):
        return out
    for k, v in zip(node.keys, node.values):
        if k is None or not isinstance(v, ast.Dict):
            continue
        # Status key is usually int constant or str constant.
        if isinstance(k, ast.Constant):
            status = str(k.value)
        else:
            status = ast.unparse(k)
        desc = _const_str(_dict_get(v, "description")) or ""
        model_node = _dict_get(v, "model")
        schema_ref = None
        if isinstance(model_node, ast.Name):
            schema_ref = model_node.id
        content_node = _dict_get(v, "content")
        content: dict[str, str] = {}
        if isinstance(content_node, ast.Dict):
            for ck, cv in zip(content_node.keys, content_node.values):
                ct = _const_str(ck) or "?"
                if isinstance(cv, ast.Dict):
                    schema_node = _dict_get(cv, "schema")
                    if isinstance(schema_node, ast.Dict):
                        # Inline schema — try to summarise.
                        ref = _dict_get(schema_node, "$ref")
                        if isinstance(ref, ast.Constant) and isinstance(ref.value, str):
                            content[ct] = _ref_to_name(ref.value)
                        else:
                            tnode = _dict_get(schema_node, "type")
                            if isinstance(tnode, ast.Constant):
                                content[ct] = str(tnode.value)
                            else:
                                content[ct] = "object"
                    else:
                        content[ct] = ""
                else:
                    content[ct] = ""
        out.append(StatusResponse(
            status=status,
            description=desc,
            schema_ref=schema_ref,
            content=content,
        ))
    return out


def _dict_get(d: ast.Dict, key: str) -> ast.AST | None:
    for k, v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def _ref_to_name(ref: str) -> str:
    return ref.split("/")[-1]


def _extract_parameters(func: ast.FunctionDef) -> tuple[list[Param], list[Param]]:
    """Return (path/query parameters, form/file parameters).

    Skips `request: Request` and similar framework-injected parameters.
    """
    other: list[Param] = []
    form: list[Param] = []
    args = func.args
    pos_args = list(args.args)
    defaults = list(args.defaults)
    # Align defaults to the tail of args.
    pad = len(pos_args) - len(defaults)
    for i, arg in enumerate(pos_args):
        default = defaults[i - pad] if i >= pad else None
        if arg.arg in ("request", "self", "cls", "background_tasks"):
            continue
        type_str = _render_type(arg.annotation)
        name = arg.arg
        # Detect Form / File / Body / Query / Header from default.
        location = "query"
        description = ""
        required = True
        rendered_default: str | None = None
        if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
            fname = default.func.id
            location_map = {
                "Form": "form",
                "File": "file",
                "Body": "body",
                "Query": "query",
                "Path": "path",
                "Header": "header",
                "Cookie": "cookie",
            }
            if fname in location_map:
                location = location_map[fname]
                # Description kwarg.
                desc_node = _kw(default, "description")
                if desc_node is not None:
                    description = _const_str(desc_node) or ""
                # First positional: ... (required) or a literal default.
                if default.args:
                    first = default.args[0]
                    if isinstance(first, ast.Constant):
                        if first.value is Ellipsis:
                            required = True
                            rendered_default = None
                        else:
                            required = False
                            rendered_default = _format_default(first)
                    else:
                        required = False
                        rendered_default = _format_default(first)
        elif default is not None:
            required = False
            rendered_default = _format_default(default)

        param = Param(
            name=name,
            location=location,
            type_str=type_str,
            required=required,
            description=description,
            default=rendered_default,
        )
        if location in ("form", "file"):
            form.append(param)
        else:
            other.append(param)
    return other, form


def _module_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    """Map module-level variable name → the assigned AST value.

    Used to resolve `responses=_TTS_PROXY_RESPONSES` (a kwarg whose
    value is a Name pointing at a module-level Dict) and
    `func.__doc__ = _SOME_DOC` patterns.
    """
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
    return out


def _docstring_assignments(tree: ast.Module) -> dict[str, str]:
    """Map function-name → docstring set by `func.__doc__ = MODULE_VAR`."""
    assigns = _module_assignments(tree)
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if target.attr != "__doc__":
                continue
            if not isinstance(target.value, ast.Name):
                continue
            func_name = target.value.id
            # Resolve the value: either a string constant directly, or a
            # Name pointing at a module-level string assignment.
            value: ast.AST = node.value
            if isinstance(value, ast.Name) and value.id in assigns:
                value = assigns[value.id]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out[func_name] = value.value
    return out


def _path_template_params(route_path: str) -> list[Param]:
    """Parse `{name}` and `{name:converter}` segments from a route path.

    These are path parameters; FastAPI infers them from the URL
    template, so they don't appear as Form/Body in the function
    signature. Without this, my static parser would misclassify them
    as 'query' since the function arg has no Form() default.
    """
    params: list[Param] = []
    i = 0
    while i < len(route_path):
        if route_path[i] == "{":
            end = route_path.find("}", i)
            if end < 0:
                break
            name = route_path[i + 1 : end]
            if ":" in name:
                name = name.split(":", 1)[0]
            params.append(Param(
                name=name,
                location="path",
                type_str="string",
                required=True,
                description="",
                default=None,
            ))
            i = end + 1
        else:
            i += 1
    return params


def extract_routes_from_file(path: Path) -> list[Route]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    prefix = _router_prefix(tree)
    module_assigns = _module_assignments(tree)
    doc_assigns = _docstring_assignments(tree)
    routes: list[Route] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Skip private helpers; only decorated routes count.
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            method, route_path, call = _route_decorator_info(dec)
            if method is None or route_path is None:
                continue
            full_path = prefix + route_path if route_path.startswith("/") else prefix + "/" + route_path
            # main.py uses `app.get` / `app.post` — those have no router prefix.
            # Detect via the func.value of the decorator.
            if isinstance(dec.func, ast.Attribute) and isinstance(dec.func.value, ast.Name):
                if dec.func.value.id == "app":
                    full_path = route_path
            summary = _const_str(_kw(call, "summary")) or ""
            response_model_node = _kw(call, "response_model")
            response_model = None
            if isinstance(response_model_node, ast.Name):
                response_model = response_model_node.id
            # response_model=None explicitly -> keep None
            response_model_exclude_none = bool(_const_value(_kw(call, "response_model_exclude_none")))
            # Resolve `responses=_MODULE_VAR` indirection.
            responses_node = _kw(call, "responses")
            if isinstance(responses_node, ast.Name) and responses_node.id in module_assigns:
                responses_node = module_assigns[responses_node.id]
            responses = _extract_responses(responses_node)
            other, form = _extract_parameters(node)
            # Path parameters from URL template (FastAPI infers; we infer too).
            path_params = _path_template_params(route_path)
            # Avoid duplicating params already declared in the signature.
            existing = {p.name for p in other}
            other = [*path_params, *(p for p in other if p.name not in {pp.name for pp in path_params})] if path_params else other
            # Docstring: prefer the inline one; fall back to a module-level
            # `func.__doc__ = SHARED` assignment, used by tts.py to share
            # one doc across three method handlers.
            docstring = ast.get_docstring(node) or doc_assigns.get(node.name, "")
            routes.append(Route(
                path=full_path,
                method=method,
                summary=summary,
                docstring=docstring,
                response_model=response_model,
                response_model_exclude_none=response_model_exclude_none,
                responses=responses,
                parameters=other,
                request_body_form=form,
            ))
    return routes


# ── Schema extraction (Pydantic models) ─────────────────────────────────────

def _is_pydantic_class(cls: ast.ClassDef) -> bool:
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


def _field_metadata(default: ast.AST | None) -> tuple[bool, str, str | None]:
    """Inspect a field's default expression. Returns (required, description, alias)."""
    if default is None:
        return True, "", None  # No default → required (Pydantic 2 style)
    if isinstance(default, ast.Call) and isinstance(default.func, ast.Name) and default.func.id == "Field":
        # Required if first positional is `...` (Ellipsis); also required if no
        # default supplied via the `default=` kwarg and no positional.
        positional_default = default.args[0] if default.args else None
        if positional_default is not None and isinstance(positional_default, ast.Constant) and positional_default.value is Ellipsis:
            required = True
        elif positional_default is not None:
            required = False
        else:
            default_kw = _kw(default, "default")
            if default_kw is None:
                required = True
            else:
                required = False
        desc = _const_str(_kw(default, "description")) or ""
        alias = _const_str(_kw(default, "alias"))
        return required, desc, alias
    # A bare default → not required.
    return False, "", None


def extract_schemas() -> dict[str, Schema]:
    src = SCHEMA_FILE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: dict[str, Schema] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_pydantic_class(node):
            continue
        docstring = ast.get_docstring(node) or ""
        fields: list[SchemaField] = []
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue
            name = item.target.id
            # Skip Pydantic's `model_config = ConfigDict(...)` and similar.
            if name.startswith("model_config") or name.startswith("__"):
                continue
            type_str = _render_type(item.annotation)
            required, desc, alias = _field_metadata(item.value)
            fields.append(SchemaField(
                name=name,
                alias=alias,
                type_str=type_str,
                required=required,
                description=desc,
            ))
        out[node.name] = Schema(name=node.name, description=docstring.strip(), fields=fields)
    return out


# ── Markdown rendering ──────────────────────────────────────────────────────

def _render_param_table(params: list[Param]) -> str:
    if not params:
        return ""
    lines = ["| Name | In | Type | Required | Description |", "|---|---|---|---|---|"]
    for p in params:
        lines.append(
            f"| `{p.name}` | {p.location} | {p.type_str} | "
            f"{'yes' if p.required else 'no'} | {_escape_table(p.description)} |"
        )
    return "\n".join(lines)


def _render_form_body(form: list[Param]) -> list[str]:
    if not form:
        return []
    out = ["### Request body", "", "**Content type:** `multipart/form-data`", ""]
    out.append("| Field | Type | Required | Description |")
    out.append("|---|---|---|---|")
    for p in form:
        out.append(
            f"| `{p.name}` | {p.type_str} | "
            f"{'yes' if p.required else 'no'} | {_escape_table(p.description)} |"
        )
    out.append("")
    return out


def _render_responses(route: Route) -> list[str]:
    out = ["### Responses", ""]
    # Synthesise a 200 entry from response_model when not in the responses dict.
    explicit = {r.status for r in route.responses}
    items: list[StatusResponse] = list(route.responses)
    if "200" not in explicit and route.response_model:
        items.insert(0, StatusResponse(
            status="200",
            description="Successful response.",
            schema_ref=route.response_model,
            content={"application/json": route.response_model},
        ))

    for r in sorted(items, key=lambda x: x.status):
        out.append(f"#### `{r.status}`")
        out.append("")
        if r.description:
            out.append(_escape_table(r.description))
            out.append("")
        # Content lines. Empty entries on the 200 response inherit from
        # the route's response_model (FastAPI convention: an empty
        # `{"application/json": {}}` block leaves the auto-generated
        # schema in place but allows additional content types beside it).
        if r.content:
            for ct, schema in r.content.items():
                if not schema and r.status == "200" and route.response_model:
                    schema = route.response_model
                if schema and schema in KNOWN_SCHEMAS:
                    out.append(f"- `{ct}` → [`{schema}`](#schema-{schema.lower()})")
                elif schema:
                    out.append(f"- `{ct}` → {schema}")
                else:
                    out.append(f"- `{ct}`")
        elif r.schema_ref:
            out.append(f"- `application/json` → [`{r.schema_ref}`](#schema-{r.schema_ref.lower()})")
        out.append("")
    return out


def _render_route(route: Route) -> list[str]:
    out: list[str] = []
    out.append(f"## `{route.method} {route.path}`")
    out.append("")
    if route.summary:
        out.append(f"**{route.summary}**")
        out.append("")
    if route.docstring.strip():
        out.append(route.docstring.strip())
        out.append("")
    if route.parameters:
        out.append("### Parameters")
        out.append("")
        out.append(_render_param_table(route.parameters))
        out.append("")
    out.extend(_render_form_body(route.request_body_form))
    out.extend(_render_responses(route))
    return out


def _render_schema(schema: Schema) -> list[str]:
    out: list[str] = []
    out.append(f"### `{schema.name}` {{#schema-{schema.name.lower()}}}")
    out.append("")
    if schema.description:
        out.append(schema.description)
        out.append("")
    if not schema.fields:
        return out
    out.append("| Field | Type | Required | Description |")
    out.append("|---|---|---|---|")
    for f in schema.fields:
        display = f.alias or f.name
        out.append(
            f"| `{display}` | {f.type_str} | "
            f"{'yes' if f.required else 'no'} | {_escape_table(f.description)} |"
        )
    out.append("")
    return out


def _path_matches(path: str, group: dict) -> bool:
    if "paths" in group and path in group["paths"]:
        return True
    if "paths_prefix" in group and path.startswith(group["paths_prefix"]):
        return True
    return False


def _collect_referenced_schemas(routes: list[Route], all_schemas: dict[str, Schema]) -> set[str]:
    """Walk routes' schema refs + transitively into nested type strings."""
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name not in all_schemas:
            return
        seen.add(name)
        # Walk the schema's field type strings — look for `[\`Name\`]`-shaped
        # links, which is what _render_type produces for nested schema refs.
        for f in all_schemas[name].fields:
            for tok in _split_schema_links(f.type_str):
                visit(tok)

    for r in routes:
        if r.response_model:
            visit(r.response_model)
        for resp in r.responses:
            if resp.schema_ref:
                visit(resp.schema_ref)
            for v in resp.content.values():
                if v and v[:1].isupper():
                    visit(v)
    return seen


def _split_schema_links(type_str: str) -> list[str]:
    """Pull out schema names from a rendered type string like
    'array of [`TranscriptionWord`](#schema-transcriptionword) | null'."""
    out: list[str] = []
    i = 0
    while i < len(type_str):
        if type_str[i] == "[" and i + 1 < len(type_str) and type_str[i + 1] == "`":
            end = type_str.find("`", i + 2)
            if end > 0:
                out.append(type_str[i + 2 : end])
                i = end + 1
                continue
        i += 1
    return out


def render_group(group: dict, routes: list[Route], schemas: dict[str, Schema]) -> str:
    matched = [r for r in routes if _path_matches(r.path, group)]

    parts: list[str] = []
    parts.append("---")
    parts.append(f"title: {group['title']}")
    parts.append(f"description: {group['description']}")
    parts.append("sidebar:")
    parts.append(f"  order: {group['order']}")
    parts.append("---")
    parts.append("")
    parts.append(
        "<!-- AUTO-GENERATED: do not edit. "
        "Source: aistack/api/* docstrings + Pydantic models in "
        "aistack/api/_schemas.py, rendered by scripts/gen_api_reference.py. -->"
    )
    parts.append("")

    if not matched:
        parts.append("(No endpoints in this group.)")
        parts.append("")
    for r in sorted(matched, key=lambda x: (x.path, x.method)):
        parts.extend(_render_route(r))

    referenced = _collect_referenced_schemas(matched, schemas)
    if referenced:
        parts.append("---")
        parts.append("")
        parts.append("## Schemas")
        parts.append("")
        for name in sorted(referenced):
            parts.extend(_render_schema(schemas[name]))

    return "\n".join(parts).rstrip() + "\n"


def render_index(groups: list[dict]) -> str:
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
        "Source: aistack/api/* + aistack/api/_schemas.py, rendered by "
        "scripts/gen_api_reference.py. -->"
    )
    parts.append("")
    parts.append("# HTTP API reference")
    parts.append("")
    parts.append(
        "These pages are generated from the aistack source code by AST "
        "analysis on every build. The source of truth is the docstrings "
        "and Pydantic models in `aistack/api/`; editing the rendered "
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


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    schemas = extract_schemas()
    KNOWN_SCHEMAS.update(schemas.keys())
    routes: list[Route] = []
    for f in ROUTE_FILES:
        if not f.exists():
            continue
        routes.extend(extract_routes_from_file(f))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.md").write_text(render_index(GROUPS), encoding="utf-8")
    for group in GROUPS:
        page = render_group(group, routes, schemas)
        (OUT_DIR / f"{group['slug']}.md").write_text(page, encoding="utf-8")

    print(
        f"[gen_api_reference] AST-walked {len(ROUTE_FILES)} route files, "
        f"{SCHEMA_FILE.name} for schemas; wrote {len(GROUPS) + 1} files into {OUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
