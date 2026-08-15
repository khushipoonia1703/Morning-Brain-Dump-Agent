# Brain Dump Agent

## What this is

I speak one unstructured morning brain dump into a microphone. The system transcribes it, splits it into discrete items, researches the ones that need learning resources, verifies every link actually works, and publishes a single Notion page for the day.

Full architecture and rationale is in `DESIGN.md`. Read it before making structural changes.

## The one architectural idea you must not break

This system is a **deterministic pipeline with exactly one agent inside it.**

- Stages 1, 2, 4, 5 are **workflow steps**. The code decides what happens next. There is no decision for a model to make.
- Stage 3 is an **agent**. It has tools and a goal. *The model* decides which tool to call, in what order, and when it is finished. The code does not script its steps.

Do not turn stage 2 into an agent. Do not turn stage 3 into a fixed procedure. That boundary is the design.

## Stack

- Python 3.11+
- `openai` — transcription, classification, and the research agent
  - Transcription: `whisper-1`
  - Reasoning: `gpt-5.6-luna` for classification (stage 2), `gpt-5.6-terra` for the research agent (stage 3) — **verify the exact model strings against https://platform.openai.com/docs/models before building.** If either is not valid, use the current equivalent tier. Never silently fall back to an older model.
- `sounddevice` + `soundfile` — microphone capture
- `requests` — web search (Tavily), page fetching, and URL validation
- `notion-client` — publishing
- `rich` — terminal tables
- `python-dotenv` — secrets

Web search uses the **Tavily Search API** via a direct HTTPS call (`requests.post` to `https://api.tavily.com/search`), NOT a model call. `search_web` sends the query to Tavily and reads the result URLs straight from the JSON response — no model sits between the query and the URLs, which preserves the provenance guarantee and avoids paying for a nested model call on every search.

- `fetch_page` is a second source of submittable URLs: it harvests the anchor links off a page the agent fetched, so the agent can reach pages a search did not return directly.
- A `TAVILY_API_KEY` is required. Tavily has a free tier, so it need not be a paid dependency, but the key must be present.

**No frameworks.** No LangChain, no LangGraph, no CrewAI, no agent libraries. Plain Python and direct SDK calls. The agent loop in stage 3 is a `while` loop I write myself — that is deliberate, because I have to explain it live.

### Keys

`OPENAI_API_KEY` is the one **paid** key — it powers transcription, classification, and the research agent.

`TAVILY_API_KEY` is required for web search. Tavily has a free tier, so it need not be a paid dependency, but the key must be present — `research.py` exits with a clear message if it is missing.

Notion additionally needs a **free** internal integration token (`NOTION_API_KEY`) and a parent page ID. That is not a paid dependency, but the code does require it — do not write code that assumes Notion needs no auth.

## File layout

```
brain-dump-agent/
├── .env                  # secrets — NEVER commit, never print
├── .env.example
├── .gitignore
├── CLAUDE.md
├── DESIGN.md
├── requirements.txt
├── record.py             # stage 1: mic -> wav -> transcript
├── classify.py           # stage 2: transcript -> items.json
├── research.py           # stage 3: items -> verified resources  (THE AGENT)
├── publish.py            # stage 4+5: compose -> Notion page
├── main.py               # runs the full pipeline end to end
├── recordings/           # dump_YYYY-MM-DD_HHMM.wav        (gitignored)
├── transcripts/          # dump_YYYY-MM-DD_HHMM.txt        (gitignored)
├── items/                # items_YYYY-MM-DD_HHMM.json      (gitignored)
└── runs/                 # run_YYYY-MM-DD_HHMM.json        (gitignored)
```

**Every artifact filename carries a `_HHMM` timestamp.** Nothing is ever overwritten. When a stage is run standalone with no argument, it resolves the **most recent** file of the type it needs (`sorted(glob(...))[-1]`). Each stage also accepts an explicit path as `sys.argv[1]`.

## Build order — ONE STAGE AT A TIME

Build the current stage, then STOP. I test it, I commit it, then I tell you to continue. Do not build ahead.

1. **Stage 1 — `record.py`** ← START HERE
2. **Stage 2 — `classify.py`**
3. **Stage 3 — `research.py`**
4. **Stage 4/5 — `publish.py`**
5. **`main.py`** — wire the stages together, last

---

## Stage 1 — `record.py`

Records audio and transcribes it.

- Record from default mic via `sounddevice`, 16kHz mono. Start immediately, stop on Enter. Print elapsed seconds while recording.
- Save to `recordings/dump_YYYY-MM-DD_HHMM.wav` via `soundfile`.
- Transcribe with OpenAI Whisper (`whisper-1`).
- Save transcript to `transcripts/dump_YYYY-MM-DD_HHMM.txt` (same timestamp as the wav) and print it.

Failure handling:
- No microphone found → clear error, exit cleanly.
- Transcription API fails → **keep the .wav file** and print its path. Never lose a recording because a network call failed.

---

## Stage 2 — `classify.py`

A **workflow step, not an agent.** Reads the transcript, makes ONE structured LLM call, writes `items/items_YYYY-MM-DD_HHMM.json`. Use structured JSON output so the response parses without regex.

The single call does four jobs, in this order:

1. **Normalize** mistranscribed technical terms against a known-terms list.
2. **Segment** the ramble into discrete items.
3. **Classify** each item's type — including marking filler as `noise`.
4. **Tag** each item's remaining properties.

Note on ordering: segmentation comes **before** noise removal, because filler words are also boundary markers — "and", "also", "and then" are simultaneously filler and the cue that a new item has started. Stripping them first destroys the signal segmentation depends on. Noise is therefore a classification result, not a pre-filter.

### Item schema

```json
{
  "raw_text": "watch a claude court tutorial",
  "normalized": "Watch a Claude Code tutorial",
  "type": "study | task | reminder | noise",
  "needs_resources": true,
  "scope": "narrow | unbounded",
  "quantity": null,
  "time": null,
  "entities": ["Claude Code"]
}
```

| Field | Meaning |
|---|---|
| `raw_text` | What I actually said — kept for debugging normalization |
| `normalized` | Cleaned text, used everywhere downstream |
| `type` | `study`, `task`, `reminder`, or `noise` |
| `needs_resources` | Whether stage 3 should research it — **independent of `type`** |
| `scope` | `narrow`, or `unbounded` for topics too broad to research directly |
| `quantity` | Number spoken ("solve **two** problems" → 2), else null |
| `time` | Spoken clock time for reminders ("at 2 PM" → "14:00"), else null |
| `entities` | Technical terms, used to build better search queries |

### The four cases the prompt MUST handle explicitly

1. **Normalize mistranscribed technical terms** against this list. Keep it as one constant at the top of the file so it is easy to extend:

```python
KNOWN_TERMS = [
    "Claude Code", "RAG", "LangChain", "LangGraph", "CrewAI",
    "Emergent Labs", "system design", "CI/CD", "recursion",
    "Notion", "Whisper",
]
```

Real example: Whisper produced "Claude Court" — it must become "Claude Code". An unfixed term goes straight into a search query and returns nothing useful.

2. **Split compound items.** One spoken sentence often holds two separate intents joined by "and". Example: *"learn how to build with Emergent Labs and how to make improvements in my RAG research agent project"* is TWO items.

3. **Mark thinking-aloud filler as `noise`.** Examples: "and what else?", a stray "Ten", "umm". These are kept in the JSON and shown in the terminal table so I can see what was dropped, but they are **never published to Notion**.

4. **`needs_resources` is independent of `type`.** Never derive one field from the other:
   - "Solve 2 recursion problems" → `task`, `needs_resources: true`
   - "Get a haircut" → `task`, `needs_resources: false`
   - "Watch a Claude Code tutorial" → `study`, `needs_resources: true`

Also: `reminder` is used **only** when a time is actually spoken. "Call Sam at 6" is a `reminder` with `time: "18:00"`. "Call my professor" with no time is a plain `task`.

Set `scope: "unbounded"` for something like "learn system design" that is too broad to research meaningfully.

Print results as a `rich` table — including noise rows — so I can eyeball correctness.

### Known-good test case

This transcript must produce exactly these 8 non-noise items:

> Watch a Claude Court tutorial. Learn how to build with emergent labs and how to make improvements in my RAG research agent project. Call my college professor who asked me to inform her about what is going on in my job hunt process. And get a haircut. And what else? Solve two coding questions of recursion. Ten. Learn system design. And apply for five AI jobs.

| normalized | type | needs_resources | scope | quantity |
|---|---|---|---|---|
| Watch a Claude Code tutorial | study | true | narrow | null |
| Learn to build with Emergent Labs | study | true | narrow | null |
| Improve my RAG research agent project | study | true | narrow | null |
| Call college professor about job hunt progress | task | false | narrow | null |
| Get a haircut | task | false | narrow | null |
| Solve 2 recursion coding problems | task | true | narrow | 2 |
| Learn system design | study | true | unbounded | null |
| Apply to 5 AI jobs | task | false | narrow | 5 |

Plus `"and what else?"` and `"Ten"` classified as `noise`.

Note this transcript contains no reminders — that is correct and expected.

---

## Stage 3 — `research.py` — THE AGENT

Runs on **every item where `needs_resources` is true**, regardless of `type`. A task like "Solve 2 recursion problems" gets resources exactly like a study topic does.

### This stage is an agent, not a script

Do NOT write a fixed sequence of search → judge → validate. Give the model **tools and a goal**, and let it decide the sequence itself.

**Goal given to the model, per item:** find one YouTube video and two current articles or blog posts that would let me actually start on this item today.

**Tools exposed to the model:**

| Tool | Signature | Returns |
|---|---|---|
| `search_web` | `(query: str)` | Candidate results: title, URL, snippet |
| `fetch_page` | `(url: str)` | Page title, first ~2000 chars of text, and links found on the page (which become submittable) |
| `submit` | `(video_url, article_urls: list[str], notes: str)` | Ends the loop for this item |

The model chooses which tool to call and when. Different items will legitimately take different numbers of steps — a narrow topic might need one search; an unbounded one might fetch a roadmap page first to pick an entry point, then search that narrower thing.

**Budget per item (safety bounds, not a script):**
- Maximum 8 tool calls total
- Maximum 3 `search_web` calls
- If the budget is exhausted before `submit`, keep whatever validated resources were found and mark the item `incomplete: true`

### Getting URLs — this is the correctness backbone

**NEVER take a URL from the model's prose.** Submittable URLs come from exactly two code-controlled sources: the results `search_web` gets back from the Tavily API, and the anchor links `fetch_page` harvests off a page it fetched. Both are written into the `self.seen` whitelist, and `submit` rejects any URL that is not in it. If you find yourself pulling URLs out of the model's text, stop — that is model-generated output and defeats the entire point of this stage.

### URL validation

After `submit`, the code (not the model) validates every URL:

- `requests.get(url, headers={"User-Agent": <a real browser UA string>}, allow_redirects=True, timeout=10, stream=True)`
- Pass on any 2xx after redirects.
- **Do not use a bare HEAD request and do not use the default python-requests User-Agent.** YouTube and many docs sites return 403 or 405 to those, which would drop good links and corrupt the drop-rate metric.
- Record three outcomes separately: `ok`, `dead` (4xx/5xx/timeout), `unverified` (403 — likely bot-blocked, not actually dead). Only `ok` links are published. `unverified` is counted separately so the metric stays honest.

If validation kills a link and the model still has budget left, **hand the failure back to the model** so it can search again. Validation failure is a recovery path, not a dead end.

### Output

- Exactly 1 video + 2 articles per item where possible. If fewer survive, publish what survived and set `incomplete: true`. Never pad with an unvalidated link.
- For `scope: unbounded`, the agent narrows to ONE concrete entry point and records it in `narrowed_to` so the page can show a "this is broad — start here" note.
- **Log and print the validation drop rate**: URLs found vs. ok vs. dead vs. unverified. I need this number for my write-up.
- Write everything to `runs/run_YYYY-MM-DD_HHMM.json` so stage 4 can re-run without re-paying for search.

---

## Stage 4/5 — `publish.py`

Reads the run JSON, builds and publishes a Notion page. A workflow step — no model call here.

Page structure:

```
Brain Dump — {YYYY-MM-DD}

## Study
  ☐ Watch a Claude Code tutorial
       🎥 <video link>      one-line description
       📄 <article link>    one-line description
       📄 <article link>    one-line description

  ☐ Learn system design
       ⚠ Broad topic — start here: <narrowed_to>
       🎥 / 📄 / 📄 ...

## Tasks
  ☐ Solve 2 recursion coding problems
       🎥 / 📄 / 📄 ...          <- resources nest under the item that needs them
  ☐ Call college professor about job hunt progress
  ☐ Get a haircut
  ☐ Apply to 5 AI jobs

## Reminders
  ☐ 14:00 — Call Sam
  ☐ 22:00 — Mail Professor Sharma
```

Rules:

- Every item is a Notion `to_do` block so the whole page is a checklist.
- **Resources nest as child blocks under whichever item has them** — driven by `needs_resources`, not by `type`. A task with resources gets them; a study item without them doesn't. Each item appears exactly once on the page.
- `reminder` items go in their own section at the end, with the spoken time as plain text prefixing the item. **The time is text. Nothing fires.**
- Items marked `incomplete` get a short note saying fewer than three resources were verified.
- `noise` items are excluded from the page entirely.
- Quantities are preserved in the text ("Apply to 5 AI jobs").

Needs `NOTION_API_KEY` and `NOTION_PARENT_PAGE_ID` in `.env`. The parent page must be explicitly shared with the integration in Notion's UI, or the API returns 404 on a page that visibly exists — surface that clearly in the error message.

---

## Stage 5 — `main.py`

Runs stages 1 through 5 in order, passing file paths between them, printing progress and the final URL. Built last, once every stage works standalone.

---

## OUT OF SCOPE — do not build these

These were deliberately excluded. If you think one is needed, ask me first.

- **Schedulers, cron jobs, background daemons, push notifications, alarms.** This includes the `reminder` type — reminders are a labelled section in a document with a time written as text. Nothing is ever fired, queued, or scheduled. Do not import `schedule`, `apscheduler`, or anything similar.
- Any web UI, mobile app, or recording interface beyond the terminal.
- Cross-day memory, carryover of yesterday's items, or deduplication across runs.
- Any delete or destructive operation, on any surface.
- Databases, ORMs, migrations. JSON files on disk are the storage layer.
- Auth, user accounts, multi-user support. Single user, local machine.
- Agent frameworks of any kind.
- Tests beyond running each stage manually against the known-good case above.

## Rules

- **One stage at a time.** Build it, stop, wait for me to test and commit.
- **Never invent URLs.** They come from Tavily search results or links harvested off a fetched page, then pass HTTP validation.
- **Secrets live in `.env` only.** Never hardcode a key, never print one, never write one to a run file.
- Keep it simple and readable. I have to explain every line of this code on a live screen share — if a piece of code would take me more than a minute to explain, it's too clever.
- Prefer the standard library. Ask before adding a package not in `requirements.txt`.
- When something fails, fail loudly with a clear message. No silent fallbacks — especially not a silent fallback to a different model.
