# Research Notes (private working area)

> Drafts and works-in-progress for aistack research notes. Once a note is finalized it is **transferred** to `docs/public/` for bilingual publication on the dosmoon site; this folder is not published.

## What lives here

Notes that we plan to publish but are not yet ready: rough drafts, half-checked numbers, unfinished surveys, anything still being iterated on. Once a note is solid enough to publish, move it out of here (see "Publishing flow" below).

We collect "stable conclusions about the technology and decisions around aistack". Forms include but are not limited to:

- **Empirical notes** — numbers / configurations / reproducible phenomena from our own hardware
- **Landscape surveys** — industry overviews + selection recommendations
- **Capability boundaries** — clarifying what a tool can / cannot actually do
- **Positioning calls** — what the project does and explicitly does not do, and why

Common thread: **they keep being referenced later** — they remain valid across project evolution cycles, longer-lived than the progress chronicle.

What does *not* belong here (goes elsewhere):

- One-off trial-and-error logs → `docs/progress/`
- API contracts → `docs/public/api/`
- Product design or architecture decisions → `docs/design/`
- Pure transcription or translation of upstream docs → just read upstream

## Why a separate folder

Over the past year-plus aistack has been accumulating "ground truths upstream did not write down" — especially boundary configurations for running large models on consumer GPUs (the VRAM / PCIe / Windows shared-memory axes). These findings:

- **Do not belong** to the API contract — they are backend implementation details, but they are critical material for anyone wanting to self-deploy, tune, or reuse
- **Do not belong** to the progress chronicle — a log entry records "what happened that day"; a research note records "the stable conclusion we distilled after the fact"
- **Do not belong** to the AI-coding failure log — that is a meta-log of how AI works, recording "what AI got wrong"; research notes record "what we eventually figured out"

Promoting them to a first-class asset means in the future:

- Someone reads aistack source and asks "why is this API called this way" — the answer is in research-note
- We upgrade a dependency (NeMo 2.7 → 2.8) and want to know which workarounds still apply — the regression baseline lives here
- We publish what dosmoon has run into so the consumer-GPU deployment community has real material to work from

## Publishing flow

```
docs/research-note/{slug}.md         (private draft, English-first)
        │
        │  finalize: numbers checked, conclusions stable, ready to publish
        ▼
docs/public/research/{slug}.md       (English, published as-is)
docs/public/zh-cn/research/{slug}.md (Chinese translation, published)
```

After the move, the file no longer lives under `docs/research-note/` — it has graduated. This folder only contains drafts.

## Style conventions

- **English-first.** Author the canonical version in English. The Chinese version published under `docs/public/zh-cn/` is a translation of the English source, not an independent fork. Reason: aistack's audience is the global consumer-GPU deployment community; English is the lingua franca of the upstream projects we engage with (NVIDIA NeMo, OpenAI Whisper, FunASR, Hugging Face). Authoring in English keeps the note aligned with its sources and discoverable to the people most likely to act on it.
- Each note opens with a **TL;DR** of three lines or fewer answering "what do I take away from reading this".
- Quantities carry units and the machine ("8 GB VRAM + 30 GB shared RAM @ RTX 4060 Laptop", not just "twice the memory").
- Every reference to upstream docs and community posts must be a link. No "I heard that".
- End with an **Open questions** section listing what is not yet confirmed — research notes are not textbooks; admitting unknowns is more credible than pretending omniscience.
- Filenames are stable English slugs (e.g., `parakeet-on-consumer-gpu.md`). Titles in frontmatter are English in `docs/public/research/` and Chinese in `docs/public/zh-cn/research/`; slugs stay identical across both so the URLs line up.

## Currently in this folder

| File | Status |
|---|---|
| `_wip-parakeet-memory-dynamics.md` | Draft (underscore-prefixed; sync-docs ignores it) |

(All other notes have been transferred to `docs/public/research/` and `docs/public/zh-cn/research/`.)
