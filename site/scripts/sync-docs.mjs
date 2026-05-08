// Sync docs/public/** → site/src/content/docs/
//
// Why this exists: aistack's published docs are authored in <repo>/docs/public/,
// but Starlight reads from site/src/content/docs/. This script mirrors the
// public tree into the Starlight content directory before each build.
//
// Two-stage flow:
//   1. Run scripts/gen_api_reference.py (Python) to refresh
//      docs/public/api/reference/*.md from the live FastAPI app's
//      OpenAPI spec. This is the "code is the document" mechanism —
//      the reference is derived from docstrings + Pydantic models, not
//      hand-written.
//   2. Mirror docs/public/ → site/src/content/docs/ for Starlight.
//
// Behavior:
//   - Wipes site/src/content/docs/ entirely (it is build output, gitignored)
//   - Recursively copies docs/public/ over
//   - Hardcoded skip: filenames starting with "_" (drafts/WIP) and "."
//
// Anything *not* under docs/public/ is never published, by construction.

import { readdir, mkdir, copyFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(here, '..', '..');
const SRC = resolve(REPO_ROOT, 'docs', 'public');
const DEST = resolve(here, '..', 'src', 'content', 'docs');
const GEN_SCRIPTS = [
  resolve(REPO_ROOT, 'scripts', 'gen_api_reference.py'),
  resolve(REPO_ROOT, 'scripts', 'gen_config_reference.py'),
];

async function copyTree(src, dest) {
  const entries = await readdir(src, { withFileTypes: true });
  await mkdir(dest, { recursive: true });
  for (const entry of entries) {
    if (entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
    const s = join(src, entry.name);
    const d = join(dest, entry.name);
    if (entry.isDirectory()) {
      await copyTree(s, d);
    } else if (entry.isFile()) {
      await copyFile(s, d);
    }
  }
}

function runOneGenerator(script) {
  // Try a sequence of Python interpreters; first one that runs (exit
  // 0) wins. Failure is fatal — auto-generated docs are not committed
  // to the repo, so the only way they reach the published site is via
  // these generators running successfully. CI installs Python +
  // aistack base before invoking astro build; local dev needs an
  // active venv with `pip install -e .` done. AISTACK_SKIP_API_GEN=1
  // is the explicit escape hatch for offline / quick-iter work.
  if (!existsSync(script)) {
    throw new Error(`[sync-docs] generator missing: ${script}`);
  }
  const candidates = process.env.AISTACK_PYTHON
    ? [process.env.AISTACK_PYTHON]
    : ['python', 'python3', 'py'];
  for (const exe of candidates) {
    const result = spawnSync(exe, [script], {
      cwd: REPO_ROOT,
      stdio: 'inherit',
    });
    if (result.error && result.error.code === 'ENOENT') continue;
    if (result.status === 0) return;
    throw new Error(`[sync-docs] ${exe} ${script} exited with status ${result.status}`);
  }
  throw new Error(
    `[sync-docs] no Python interpreter found for ${script} ` +
    '(tried AISTACK_PYTHON, python, python3, py). ' +
    'Install Python and run `pip install -e .` in the repo root, ' +
    'or set AISTACK_SKIP_API_GEN=1 to bypass (auto-generated reference will be missing).'
  );
}

function runGenerator() {
  if (process.env.AISTACK_SKIP_API_GEN === '1') {
    console.log('[sync-docs] AISTACK_SKIP_API_GEN=1 — skipping reference regen');
    return;
  }
  for (const script of GEN_SCRIPTS) runOneGenerator(script);
}

async function main() {
  if (!existsSync(SRC)) {
    console.error(`[sync-docs] Source directory missing: ${SRC}`);
    process.exit(1);
  }
  // Stage 1: regenerate auto-derived API reference markdown from code.
  runGenerator();
  // Stage 2: mirror docs/public into Starlight's content tree.
  if (existsSync(DEST)) {
    await rm(DEST, { recursive: true, force: true });
  }
  await copyTree(SRC, DEST);
  console.log(`[sync-docs] ${SRC} → ${DEST}`);
}

main().catch((err) => {
  console.error('[sync-docs] failed:', err);
  process.exit(1);
});
