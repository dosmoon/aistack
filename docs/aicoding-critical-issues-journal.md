# AI Coding — Critical Issues Journal

> A standing log of systematic weaknesses observed when AI coding agents
> operate on non-trivial integration work. Each entry pins a concrete
> incident to the underlying behavioral pattern, so the same class of
> failure can be recognized faster the next time it appears.
>
> The intended audience is anyone (human or future AI session) who has
> to decide how much to trust an AI coding agent's output on deep
> systems work, and what guardrails to apply.

## The standing thesis

For an AI coding agent to claim it has "researched" a tool, library,
model, or runtime well enough to build on it, **three legs of
investigation must all agree on the same explanation**:

1. **Official documentation** — what the upstream maintainer says about
   the tool's design, capabilities, parameters, and known limits.
   Sources: model cards, API references, READMEs, design papers,
   official examples, release notes.

2. **Community experience** — what other practitioners have observed,
   debugged, and resolved. Sources: GitHub issues, forum threads, Stack
   Overflow, blog write-ups, mailing-list archives.

3. **Source-code / runtime root-cause analysis** — what the tool
   actually does in this specific environment. Sources: reading the
   library's source, dumping returned data structures, direct probes
   (`ctypes`, version queries, behavioral micro-tests), trace logs.

Each leg in isolation is a partial mirror. Official docs can be
aspirational or stale. Community wisdom can be cargo-culted or
context-mismatched. Runtime probes without grounding can mislead about
*why* something is the way it is.

**Self-consistency across all three is the bar.** When the three legs
disagree, the disagreement is itself diagnostic — it's pointing at a
flaw in the mental model that has to be resolved before any
implementation work can be trusted.

The systematic weakness of current AI coding agents is that they
**routinely operate on one leg, often the shallowest one**. They run a
keyword search, surface a few community posts, pattern-match a fix,
and ship. The other two legs are skipped. The output looks confident
and is plausible-sounding, but is built on a partial foundation that
cracks the moment the situation deviates from the searched-for shape.

## Why this matters for non-expert users

A senior engineer who suspects this can interrogate the AI:

- "Did you actually read the model card? Cite the relevant paragraph."
- "Did you probe the runtime to confirm? Show the probe and its output."
- "Did the community workaround you suggested actually solve a case
  that matches ours, or just one that *sounds* similar?"

Each interrogation can force a missing leg of the research to be
done. The senior knows what questions to ask and how to evaluate the
answers.

A non-expert user — the audience AI coding tools claim to serve —
cannot do this. They have no priors against which to judge whether
the AI's "research" is real or performed. So an AI agent with the
shallow-research default produces fundamentally untrustworthy output
in any deep system, and the user has no instrument to detect this.

This is not a "future improvement" item. It is a present-tense
defect that limits the class of systems AI coding can responsibly
build for non-engineering users. Until the three-legs discipline is
the default mode rather than something a sophisticated user has to
extract by repeated correction, deep systems work cannot be safely
delegated.

---

## Schema for entries

```
### YYYY-MM-DD — short title
- **Symptom**: what the AI saw and how it framed the problem.
- **AI's first instinct**: what the AI proposed before pushback.
- **What was missing**: which of the three legs were skipped.
- **The convergent answer**: what all three legs together pointed at.
- **Cost if shipped**: what the user would have lost.
- **Lesson for future sessions**: the durable takeaway.
```

Append in chronological order. Do not edit historical entries — add a
follow-up entry if a prior conclusion needs updating.

---

## Entries

### 2026-05-07 — Parakeet long-audio OOM, mistaken for routing problem

- **Symptom**: 17-minute English audio against `model=parakeet`
  crashed the worker with a Windows file-lock error
  (`[WinError 32] manifest.json`). Short audio (~1 min) worked fine.

- **AI's first instinct**: propose a duration-based auto router that
  silently downgrades long audio from Parakeet to faster-whisper
  ("safer fallback"). This would have permanently hidden Parakeet
  from any audio over a heuristic threshold.

- **What was missing**:
  - *Official docs*: never read. The HuggingFace model card states
    plainly that the model uses full attention by default (24 min on
    A100 80GB) and exposes a `change_attention_model("rel_pos_local_attn",
    att_context_size=...)` switch for long audio on smaller cards.
    The 24-min A100 number, scaled to 8 GB consumer hardware, is
    arithmetic that predicts the failure exactly.
  - *Community experience*: skimmed only for "WinError 32 manifest"
    keyword, which surfaced unrelated DataLoader race threads. Never
    queried for "Parakeet long audio" or "FastConformer attention
    memory" — both of which return immediate, on-topic results.
  - *Source / runtime*: never inspected NeMo's transcribe path or
    probed VRAM during a failing run. The OOM-vs-file-lock distinction
    was unverified speculation.

- **The convergent answer**: full self-attention is O(N²); 17 min on
  8 GB is OOM long before any file-system error becomes possible. The
  WinError 32 was a downstream cleanup race triggered by the OOM.
  Switching to local attention (`rel_pos_local_attn`,
  `att_context_size=[256,256]`) makes memory linear in audio length;
  17 min then fits comfortably and runs at GPU-native speed (~10 s end
  to end, RTF 0.01).

- **Cost if shipped**: Parakeet — the most accurate of the three ASR
  backends on English proper nouns — would have been silently disabled
  for any audio long enough to be interesting. The 27× GPU speedup
  would have been thrown away for no reason.

- **Lesson for future sessions**: when a model fails on long input,
  the *first* place to look is its own attention/memory architecture
  documentation. Routing-around is a reasonable fix only after the
  model's published memory model has been read and confirmed to not
  offer a switch.

### 2026-05-07 — `cudnnGetLibConfig` symbol error, three layers of misdiagnosis

- **Symptom**: after restarting the server, Parakeet model load
  printed `Could not load symbol cudnnGetLibConfig. Error code 127`
  on stderr and the worker process died.

- **AI's first instinct**: force `AISTACK_PARAKEET_DEVICE=cpu` as a
  default, accepting the loss of GPU acceleration. Wrote half of the
  CPU-fallback code before the user interrupted.

- **What was missing**:
  - *Official docs*: never checked cuDNN 9's released DLL split
    architecture. cuDNN 9 deliberately separates `cudnnGetLibConfig`
    into `cudnn_graph64_9.dll`; the symbol's presence depends on the
    minor version of the bundled library, not on the major version
    string in `cudnn64_9.dll`.
  - *Community experience*: searched, but mis-framed. The query
    "cudnnGetLibConfig NeMo Windows" surfaced one Colab thread
    without resolution; the correct framing "torch bundled cuDNN
    version where cudnnGetLibConfig was added" would have surfaced
    PyTorch issue threads about cuDNN 9.5+ requirements directly.
  - *Source / runtime*: not used until very late. A 5-line `ctypes`
    probe (`ctypes.WinDLL("cudnn64_9.dll")` plus `getattr(lib,
    "cudnnGetLibConfig", None)` across all eight cuDNN sub-DLLs)
    immediately confirms whether the symbol exists in the currently
    loaded environment. This probe was the first thing the user
    pushed for; it took 30 seconds to write and falsified two prior
    incorrect theories at once.

- **The convergent answer**: torch 2.5.1+cu121 ships cuDNN **9.1.0**.
  The `cudnnGetLibConfig` symbol entered cuDNN's exported API in a
  later minor version and is present in `cudnn_graph64_9.dll` only
  starting around 9.5+. NeMo 2.7.3 was built expecting the newer
  symbol. Upgrading to torch 2.7.1+cu126 (cuDNN 9.7.1) fixes the
  symbol resolution; no CPU fallback needed.

- **Cost if shipped**: Parakeet permanently relegated to CPU on this
  hardware class. The 100× real-time GPU path that ended up powering
  the headline benchmark would have been silently and permanently off.
  Future contributors would have inherited a "Parakeet doesn't work
  on GPU on Windows, that's just how it is" lore that was never true.

- **Lesson for future sessions**: for any DLL/symbol-resolution
  failure, a direct `ctypes` probe is the cheapest experiment in the
  diagnostic toolkit and should be the *first* probe, not the last.
  Speculating about ABI mismatches without the probe is a category
  error.

### 2026-05-07 — SenseVoice English run-on output, almost wrapped externally

- **Symptom**: SenseVoice English transcripts came back as
  `Msready,yessir.` — every word concatenated, no spaces. VideoCraft's
  cross-backend integration test caught this and proposed adding a
  `wordninja`-based post-processor to re-segment the words.

- **AI's first instinct**: accept VideoCraft's proposed fix at face
  value. Ready to introduce `wordninja` (or `wordsegment`) plus a
  truecase library as new dependencies, plus a language-detection
  branch, plus the inevitable maintenance of an English word
  dictionary.

- **What was missing**:
  - *Official docs*: never consulted. FunASR's README and demo scripts
    show `model.generate(...)` returning a `text` field that is, for
    the official `en.mp3` example, the perfectly-spaced string
    "The tribal chieftain called for the boy and presented him with
    50 pieces of gold." The model output is fine. The bug had to be
    elsewhere.
  - *Community experience*: zero relevant matches, *because the bug
    doesn't exist upstream*. The lack of community noise should have
    been a signal to look closer at the local pipeline rather than
    paper over with a downstream patch.
  - *Source / runtime*: not used. A 30-second probe — running
    `model.generate()` on the official `en.mp3` and printing
    `item.text` and `item.words` — would have shown that the `text`
    field is correctly spaced and the `words` array is one-token-
    per-word. This single observation falsifies the wordninja theory
    on the spot.

- **The convergent answer**: the bug was in our own code, in
  `aistack/asr/sensevoice.py`. We split each VAD chunk into subtitle-
  friendly sub-segments by walking the per-token `words[]` array
  (correct, for accurate timestamps), then rebuild text per sub-
  segment via `"".join(words)`. The empty-string join is right for
  Chinese (each token is one Han character) but wrong for English
  (each token is a word). A 20-line language-aware join helper
  resolves it cleanly with no new dependencies.

- **Cost if shipped**: a permanent extra dependency on a word-
  segmentation library and its dictionary, plus the ongoing
  maintenance of a post-processor that exists only to undo a bug
  we created and didn't notice. Worse, the post-processor would have
  been less accurate than the model's own native output (which we
  were ignoring) because word segmentation from concatenated text is
  inherently lossy.

- **Lesson for future sessions**: a bug report from a downstream
  consumer correctly describes the *symptom* but is often wrong
  about the *fix*. Before implementing any proposed downstream patch,
  run the upstream's own minimal example against the failing input
  and dump the result. If the upstream produces correct output, the
  bug is in the integration code, not in the model.

---

## Recurring patterns across entries

After three entries on a single day, the structural pattern is clear:

1. **Skipped leg #1: official docs.** Each failure had a directly-
   applicable answer in the upstream's official documentation that was
   not consulted. The model card / README / demo / API reference is
   almost always the cheapest and highest-information first stop, and
   it is the leg that current AI coding agents skip most frequently
   in favor of generic web search.

2. **Skipped leg #3: runtime probes.** Each failure had a sub-30-second
   runtime probe (`ctypes`, `repr(item)`, version dump, structure
   inspection) that would have settled the question definitively. The
   bias against probes — perhaps because they feel like "extra work"
   — is wildly miscalibrated relative to the cost of speculating
   without them.

3. **Over-reliance on shallow leg #2: keyword search.** When the AI
   does research, it tends to query a literal keyword from the error
   message and pattern-match the first plausible-looking thread. This
   surfaces *adjacent* solutions that may or may not apply. Reframing
   the search ("what *kind* of problem is this", "what does the
   library *call* this case in its own vocabulary") almost always
   produces better hits, and is a discipline the AI rarely applies on
   its own.

4. **Premature workaround proposal.** In each case the AI's first
   substantive output was a workaround (route around / disable / wrap
   in adapter), not a research plan. Workarounds proposed before the
   three legs converge are guesses dressed up as solutions.

These four patterns are linked. Together they describe a default
operating mode that is, charitably, *implementation-biased*. For
shallow CRUD work this default is harmless. For deep systems work —
where the difference between a real fix and a workaround is the
difference between a 100× speedup and a permanently disabled feature —
this default is dangerous.

## How a session should look when the discipline is applied

For any non-trivial library, model, or framework being integrated for
the first time, or any failure mode that does not yield to a five-
minute keyword search:

1. **Read first.** The model card, the upstream README, the most
   relevant API doc page. Take notes on parameters, limits, and
   default values.

2. **Run the upstream's own minimal example** on a clean input close
   to the target use case. Dump every field of the returned data
   structure. This is the single highest-yield activity per minute
   spent.

3. **Probe the runtime.** Version queries, structure inspection,
   `ctypes` for native deps, behavioral micro-tests. Confirm your
   mental model matches what the system actually does in *this*
   environment.

4. **Search the community** with the framing learned from steps 1–3,
   not with the raw error string. "Parakeet attention memory long
   audio" returns much more useful threads than "WinError 32 manifest
   parakeet".

5. **Only then propose a fix.** Cross-check the proposed fix against
   all three legs. If any leg disagrees with the proposal, the
   proposal is wrong.

For trivial work — well-documented APIs, mainstream patterns,
plenty of prior community resolution of identical failures — the full
discipline is overkill. The judgment of when to apply it is itself a
skill, but the safe default for unfamiliar deep-system territory is
"apply it always until proven otherwise."

---

## Open question

Whether AI coding agents can be made to *default* to the three-legs
discipline (rather than performing it only when prompted by a
sophisticated user) is an open methodology question. Until they can,
deploying AI coding to non-expert users for deep systems work is
premature in a load-bearing way that the marketing around these tools
does not currently acknowledge.
