# ai-mem

A long-term memory system for an AI assistant. It records everything that passes through the assistant's context, turns it into standalone memories, keeps track of what is true now, and puts the right memories in front of the model before it answers.

It is the first piece of a personal AI stack meant to run on open-weight models on my own hardware. Until then it runs inside Claude Code through hooks, and it is built so the model behind it can be swapped without touching the rest.

**Principle:** the model is responsible for understanding; memory is responsible for storing and recalling. Two steps call a model; everything else is plain code.

## How it works

```
 Claude Code session ──hooks──▶ 1. Store ──▶ queue ──▶ 2. Perceive (model) ──▶ memory graph
        ▲                                                     │                     ▲
        │                                                     ▼                     │
   5. Inject ◀── 4. Recall (code, ~1.5 s) ◀────────── 3. State (model) ───────────┘
```

| # | Piece | Job | Runs as |
|---|---|---|---|
| 1 | **Store** | Saves every transcript line and file version byte for byte. Add-only: the database refuses updates and deletes (SQLite triggers). Everything else can be rebuilt from it. | hooks + background worker |
| 2 | **Perceive** | A small model turns each input into short memories that stand alone (who said it, references resolved against the conversation so far), plus a note on what kind of moment it is. Memories are linked to the one before and to what they rely on. | background worker |
| 3 | **State** | A second model keeps one short page per subject: what holds now. Changes are new versions; old ones are kept with dates and the memories they rest on. Code checks the rules: every version rests on evidence, nothing is overwritten, a rule the model inferred about the user needs evidence from two separate conversations. | background worker |
| 4 | **Recall** | On every message: vector search (static embeddings, CPU) plus keyword search (SQLite FTS5), then spreading over links. A memory must stand out from the cue's own score distribution, not just pass a fixed cutoff. Behavior rules are matched by the kind of moment, not the topic. | `UserPromptSubmit` / `SessionStart` hooks |
| 5 | **Inject** | Puts the picks in front of the model with an id each, never repeating what the conversation already holds, within a size limit. Any memory can be traced down to the original raw text. | same hooks |

## Design choices worth noting

- **Resolve backwards, never wait.** Every step works from what the model already had in context; nothing waits for future input to decide anything.
- **Raw is the source of truth.** All derived data (memories, links, states, even the reading of each transcript line) is rebuildable from the raw store (`mem.py rederive`).
- **Explainable by default.** Every recall stores each candidate, its scores and why it got in or not (`mem.py why`); every model call stores its full prompt and answer (`mem.py calls`). A read-only dashboard visualizes the pipeline (`mem.py dashboard`).
- **Hooks never block the session.** Hooks stay under ~1.5 s, never wait on the database (records go to a spool if it is busy), and never fail the session.
- **Cost-aware.** Said content (user and assistant messages) is processed before tool output, messages are batched with their replies, and bulk history import is paced.

## Running it

Python 3.12+, with `numpy`, `tokenizers`, `safetensors` and the `minishlab/potion-base-32M` model in the Hugging Face cache. Model calls go through the Claude Code CLI (`claude -p`); set `AI_MEM_CLAUDE_EXE` if it isn't on `PATH`.

```
python mem.py sync        # store everything new
python mem.py live        # start queueing new inputs for perceiving
python mem.py work        # run the worker in this window
python mem.py status      # queue, memories, worker, recent errors
python mem.py search "…"  # run recall and show every decision
python mem.py dashboard   # http://localhost:8765
```

To turn it on in Claude Code, merge `hooks-settings.example.json` into `~/.claude/settings.json` with your path.

## Status

Version 1, in daily use since November 2026. Not yet built: learning which links pay off, recall on every model thought (today: every message), sources other than Claude Code, and local models for the two model steps. The prompts name the owner, since it is a personal system.

## Layout

```
mem.py              command line
aimem/store.py      raw store and incremental transcript reading
aimem/extract.py    what each transcript line is and who it came from
aimem/perceive.py   model step 1: memories
aimem/state.py      model step 2: what holds now, with rule checks
aimem/recall.py     search, thresholds, spreading over links
aimem/inject.py     formatting and records
aimem/hooks.py      Claude Code hook entry points
aimem/worker.py     background queue worker
aimem/db.py         schema and integrity triggers
aimem/dashboard.*   read-only dashboard
```
