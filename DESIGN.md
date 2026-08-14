# Brain Dump Agent — Design Document

*The problem, the architecture, every design decision, the tradeoffs, and the limitations.*

---

# Part I — The Problem

## 1.1 Where this came from

I have a lot going on in my head each morning. Things to study, people to call, jobs to apply to. I capture them inconsistently — some become phone reminders, some become sticky notes, most stay in my head and quietly disappear.

For a while I assumed the problem was capture. It isn't. Voice memos already solve capture, and my phone has plenty of them.

## 1.2 The real problem

**A captured thought is inert.**

"Learn RAG improvements" sitting in a notes app is exactly as useless as it was in my head, because acting on it still requires me to stop what I'm doing, go find where to start, wade through tutorials written against API versions that no longer exist, and pick something.

The friction was never remembering the intention. It's the gap between *having* an intention and having a *first actionable step*.

## 1.3 Why existing tools don't close it

| Tool | What it does | Where it stops |
|---|---|---|
| Voice memos | Records audio | Audio is unsearchable and unstructured |
| Notes apps | Stores text | Text stays inert; no next step |
| Reminders / to-do apps | Alerts you at a time | Tells you *when*, never *how to begin* |
| Asking an LLM directly | Gives an answer | Requires me to already be at my desk, focused, and asking — which is the state I wasn't in |

Every one of them closes the first half of the gap and leaves the second half to me. That second half is what this project is for.

---

# Part II — The Solution

## 2.1 In one sentence

I speak one unstructured brain dump in the morning; the system returns a single Notion checklist for the day where every item that needs a starting point already has verified resources attached.

## 2.2 The user experience

1. I run `python main.py`.
2. It starts recording. I talk — messily, out of order, with "umm"s and mid-sentence corrections.
3. I press Enter when I'm done.
4. A few minutes later, a Notion page URL is printed.
5. I open it on my phone and work through it.

That's the entire interface. No app to install, no UI to learn.

## 2.3 What is deliberately NOT built

Each of these was a decision, not an omission. Reasoning is in Part VI.

- **No scheduler, no notifications, no alarms.** Reminders are a section in a document.
- **No app, no web UI, no custom recording interface.** A terminal and Notion are enough.
- **No cross-day memory.** Each morning is processed independently.
- **No database.** JSON files on disk are the storage layer.
- **No agent framework.** Plain Python.
- **Nothing destructive.** The system's only write action is creating a page.

---

# Part III — Architecture

## 3.1 The shape of the system

**A deterministic pipeline with exactly one agent inside it.**

That phrasing is precise on purpose, and it rests on a distinction I want to be explicit about:

- A **workflow step** runs through a path my code defines. My code decides what happens next.
- An **agent** is given tools and a goal. *The model* decides what happens next, based on what it has found so far.

Stages 1, 2, 4 and 5 are workflow steps. Stage 3 is an agent.

Calling the whole system an agent would be overselling it. Calling it a script would miss what stage 3 does.

## 3.2 Data flow

```
  [ microphone ]
        │
        ▼
┌───────────────────────────┐
│ STAGE 1  record.py        │   sounddevice → .wav
│ Capture + Transcribe      │   Whisper      → text
└───────────────────────────┘   WORKFLOW
        │  transcripts/dump_*.txt   (raw string)
        ▼
┌───────────────────────────┐
│ STAGE 2  classify.py      │   ONE structured LLM call
│ Normalize · Segment ·     │   • fix mistranscribed terms
│ Classify · Tag            │   • split compound items
└───────────────────────────┘   WORKFLOW
        │  items/items_*.json
        ▼
┌───────────────────────────┐
│ STAGE 3  research.py      │   ← THE AGENT
│ tools: search_web,        │   model chooses tool + order
│ fetch_page, submit        │   code validates every URL
└───────────────────────────┘   AGENT
        │  runs/run_*.json
        ▼
┌───────────────────────────┐
│ STAGE 4+5  publish.py     │   build Notion blocks
│ Compose + Publish         │   → create page via API
└───────────────────────────┘   WORKFLOW
        │
        ▼
  [ Notion page URL ]
```

## 3.3 Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+, no framework | Readable; I have to explain every line live |
| Audio capture | `sounddevice` + `soundfile` | No compiler needed on Windows, unlike `pyaudio` |
| Transcription | OpenAI Whisper (`whisper-1`) | Accurate on technical speech; no local GPU needed |
| Reasoning | OpenAI GPT | Structured JSON output, server-side tools |
| Search | OpenAI `web_search` tool, Responses API | Built in — no second search provider or key |
| Validation | `requests` | Plain HTTP check; no library needed for this |
| Output | `notion-client` | Official SDK |
| Display | `rich` | Readable terminal tables while testing |
| Secrets | `python-dotenv` | Keys in `.env`, never in code |

**One paid API key powers the entire system.** Notion needs a free internal integration token, which is auth but not a cost.

## 3.4 State

Every stage reads a file and writes a file. Nothing is held in memory between stages. Every artifact filename carries a timestamp, so nothing is ever overwritten.

This matters for three reasons:

1. **Debuggability.** When the Notion page comes out wrong, I can look at `items.json` and know instantly whether the fault is upstream or downstream.
2. **Cost.** I can re-run the composer a dozen times without paying for transcription and search again.
3. **Replaceability.** Swapping the model behind any single stage is a contained change, because the contract between stages is a JSON file, not a function signature.

---

# Part IV — Module Reference

## 4.1 `record.py` — Capture and transcribe

Records at 16kHz mono, starts immediately, stops on Enter, saves the `.wav`, sends it to Whisper, writes and prints the transcript.

**Critical detail:** if transcription fails, the `.wav` is retained and its path printed. Losing a two-minute morning dump to a network timeout would destroy my trust in the tool immediately — and trust is the entire product. A tool I'm afraid to speak into has no value.

## 4.2 `classify.py` — Normalize, segment, classify

One structured LLM call. Four jobs:

**1. Normalize** technical terms that speech recognition mangles. Not cosmetic — a mistranscribed term goes straight into a search query and returns nothing useful. Whisper turned "Claude Code" into "Claude Court" on my real dump.

**2. Segment** the ramble into discrete items. Speech has no punctuation, and one spoken sentence frequently contains two unrelated intentions joined by "and."

**3. Classify** each item — including marking filler as `noise`.

**4. Tag** the remaining properties.

### Why segmentation comes before noise removal

My first instinct was to strip filler first, then split. That's backwards. The filler words *are* the boundary markers — "and", "also", "and then" are simultaneously noise and the cue that a new item started. Removing them first destroys the signal segmentation runs on. And a stray "Ten" is only identifiable as noise once you can see it isn't attached to "solve two problems."

So noise is a classification *result*, not a pre-filter. Noise items stay in the JSON and appear in the terminal table, so I can see what was thrown away and catch over-filtering. They never reach Notion.

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

**The field worth explaining is `needs_resources`, which is deliberately independent of `type`.** "Solve 2 recursion problems" is a task that benefits enormously from practice links. "Get a haircut" is a task that must never trigger a search. If you derive one field from the other, you break one of those two cases — guaranteed. This independence carries all the way through to the page: resources attach to whatever needs them, not to a category.

`scope` handles breadth. "Learn system design" is a real intention but too broad to research directly; searching it returns generic listicles. Marking it `unbounded` lets the agent narrow to one concrete entry point rather than pretending three links cover the topic.

`type: reminder` is used only when a time is actually spoken.

## 4.3 `research.py` — The agent

Runs on every item where `needs_resources` is true, regardless of type.

**This stage is not a script.** The model gets three tools — `search_web`, `fetch_page`, `submit` — and a goal: find one video and two current articles that would let me start on this today. It decides which tool to call, in what order, and when it's done.

Different items legitimately take different numbers of steps. A narrow topic might resolve in one search. "Learn system design" might fetch a roadmap page first to choose an entry point, then search that narrower thing. That variability is the point — my code doesn't know the step count in advance.

**Bounds, not a script:** 8 tool calls per item, at most 3 of them searches. A cap is a safety bound on an agent, not a procedure controlling it.

**Two rules the code enforces, not the model:**

1. URLs come from the search tool's citation annotations, never from the model's prose. This is where hallucinated links would otherwise re-enter.
2. Every URL is fetched and validated before it can be published. If validation kills a link and the model still has budget, the failure is handed back so it can search again — validation failure is a recovery path, not a dead end.

Validation records three outcomes separately: `ok`, `dead`, and `unverified` (403 — bot-blocked, not necessarily broken). Only `ok` links publish. Keeping `unverified` separate is what stops the drop-rate metric from lying to me.

## 4.4 `publish.py` — Compose and publish

Builds one Notion page. Every item is a checkbox. Resources nest as child blocks under whichever item has them. Reminders form their own section at the end with the spoken time as plain text. Noise never appears.

The parent page must be explicitly shared with the integration in Notion's UI, or the API returns 404 on a page that visibly exists.

---

# Part V — Worked Example

## 5.1 What I said

> "Watch a Claude Court tutorial. Learn how to build with emergent labs and how to make improvements in my RAG research agent project. Call my college professor who asked me to inform her about what is going on in my job hunt process. And get a haircut. And what else? Solve two coding questions of recursion. Ten. Learn system design. And apply for five AI jobs."

In there: a mistranscribed product name, a compound sentence, two pieces of verbal filler, a quantity spoken as a word, and a topic far too broad to research.

## 5.2 After stage 2

| normalized | type | needs_resources | scope | quantity |
|---|---|---|---|---|
| Watch a Claude Code tutorial | study | true | narrow | — |
| Learn to build with Emergent Labs | study | true | narrow | — |
| Improve my RAG research agent project | study | true | narrow | — |
| Call college professor about job hunt progress | task | false | narrow | — |
| Get a haircut | task | false | narrow | — |
| Solve 2 recursion coding problems | task | **true** | narrow | 2 |
| Learn system design | study | true | **unbounded** | — |
| Apply to 5 AI jobs | task | false | narrow | 5 |

Plus `"and what else?"` and `"Ten"` classified as `noise` and dropped from the page.

Four things happened here that a naive implementation gets wrong:

- "Claude Court" → "Claude Code" (normalization)
- One sentence became two items (segmentation on meaning, not punctuation)
- "Get a haircut" is a task with `needs_resources: false`
- "Solve 2 recursion problems" is a task with `needs_resources: true` — the same field, opposite direction, which is why it can't be derived from `type`

## 5.3 After stage 3

Five items go to the agent. Three are skipped entirely.

Note that the five are not the five `study` items — four study topics plus one task. That's the independence of `needs_resources` doing visible work.

"Learn system design" comes back narrowed to a single concrete entry point rather than three generic listicles.

*Measured on the first real run: URLs found ___, ok ___, dead ___, unverified ___. Segmentation errors ___ across ___ items.*

---

# Part VI — Design Decisions and Tradeoffs

## 6.1 One agent, not five

**Alternative:** make every stage an agent, or wrap the whole pipeline in an agent framework.

**Why not:** an agent is warranted where there is a decision to make. Transcription has none. Classification has none — one transcript in, one array out, nothing to retry against, no branch to choose. Making those agentic would buy unpredictable cost and harder debugging in exchange for flexibility the problem doesn't need.

Research is different. "Find good, current resources" has no fixed procedure. Whether the results are any good is a judgment, and what to do about bad results depends on why they were bad. That is a real decision, so that is where the agency goes.

**What would change my mind:** if the system had to handle open-ended requests, or hold a conversation about the items rather than process them in a batch.

## 6.2 No agent framework

**Alternative:** LangGraph or a similar orchestration library.

**Why not:** it would not have made the system more agentic. A framework re-expresses control flow; it doesn't add decisions. My pipeline is linear with one bounded loop, which is precisely the case where a graph engine is ceremony. It also works against a hard constraint of this project: I have to narrate this code live, and `StateGraph`, reducers and checkpointers are exactly the abstractions that are hard to explain under interruption.

**Tradeoff:** I hand-write the tool-dispatch loop and the retry bounds, which a framework would have given me.

**What would change my mind:** multiple agents needing to hand off to each other, or a genuine need for durable mid-run checkpointing.

## 6.3 One LLM call for normalize + segment + classify

**Alternative:** separate calls per stage, or one call per item.

**Why not:** these tasks are interdependent. Recognising that a stray "Ten" is noise requires seeing the sentence around it. Splitting a compound item requires the full context of both halves. Isolating the stages would remove exactly the context needed to do them correctly.

**What would change my mind:** transcripts longer than about five minutes, where context dilution starts to hurt. Then I'd chunk first.

## 6.4 Single vendor for the whole pipeline

**Alternative:** pick the strongest model for each stage across providers.

**Why not:** one API key, one SDK, one billing account, one set of error semantics. On a two-day build, integration surface is a bigger risk than marginal model quality. Transcription had to be OpenAI regardless — there is no Anthropic speech model — so going single-vendor removed a dependency rather than adding one.

**Tradeoff:** vendor lock-in, and I can't pick the best model per stage.

**What would change my mind:** classification accuracy plateauing below what I need. Because stages hand off through JSON files, swapping the model behind any single stage is contained.

## 6.5 Real search plus URL validation, never model recall

**Alternative:** ask the model directly for good resources.

**Why not:** language models produce URLs that are structurally perfect and completely dead. A single 404 forces me to verify every other link on the page, and at that point the system has saved me nothing. Validation isn't a feature, it's the correctness backbone.

This is also why URLs are pulled from citation annotations rather than the model's text. If I parsed links out of the prose, I'd have rebuilt the hallucination problem behind a validation step that only catches the ones that happen to be dead.

**What would change my mind:** nothing about validating. Only *how* — caching validated URLs so repeated topics aren't re-checked daily.

## 6.6 Three resources per item, in fixed roles

**Alternative:** return everything relevant that was found.

**Why not:** the problem I'm solving is overwhelm. Twenty links reproduce the paralysis I started with — the system would convert an unactionable thought into an unactionable reading list. One video and two articles is a starting path rather than a pile.

**What would change my mind:** if the three were consistently at the wrong level for me. The fix there is personalization, not volume.

## 6.7 Reminders are classified, not scheduled

**Alternative:** parse spoken times and fire notifications.

**Why not:** capturing the time costs one field. Acting on it costs a subsystem — cron, delivery, timezones, a process that has to be running. A scheduler is infrastructure, not intelligence, and it would have consumed the build time the research loop needed.

There's a second reason. When I recorded my actual brain dump, it contained **zero** time-bound items. I had designed around imagined usage and my own speech contradicted it. So reminders exist as a class because sometimes I do say "call Sam at six" — but they render as a line of text in a section, and nothing fires.

**What would change my mind:** evidence from real usage that items are missed at the *acting* stage rather than the *starting* stage.

## 6.8 Notion as the output surface

**Alternative:** a custom web app, or a daily email.

**Why not:** zero interface to build, already where I work, mobile access free. The output is a document I read, and a document is the right shape for that.

**Tradeoff:** coupled to Notion's block format, and the page is effectively read-only — checking something off doesn't feed back into the system.

**What would change my mind:** the moment the agent needs to *react* to my progress. Then I need real state, and Notion becomes a sync problem instead of a shortcut.

## 6.9 Autonomy matched to reversibility

Every action this system takes is additive: it creates a page. Nothing is overwritten, nothing deleted, and a bad run costs me one page I ignore. So it runs unsupervised, with no confirmation step.

That was a conscious rule, not a convenience. **How much autonomy an agent gets should scale with how cheap its mistakes are to undo.** When I considered building a screenshot-cleanup agent, deletion authority was the thing I would not have granted — losing one payment receipt costs far more than keeping a hundred junk images. I'd rather ship a narrower agent that never needs supervision than a broader one that does.

---

# Part VII — Failure Modes

| Failure | Cause | Handling |
|---|---|---|
| Technical terms mistranscribed | Speech recognition on domain vocabulary | Normalization against a known-terms list, before search |
| Compound items merged or over-split | "and" is ambiguous in spoken language | Segmentation prompted with examples of both error directions |
| Hallucinated resource links | Model generating URLs from recall | URLs taken only from citation annotations, then HTTP-verified |
| Good links wrongly dropped | Sites returning 403 to a default user agent | Browser UA, GET not HEAD, 403 recorded as `unverified` not `dead` |
| Agent loops without finishing | No natural stopping point on a vague item | Hard budget of 8 tool calls / 3 searches per item |
| Fewer than 3 resources survive | Validation drops candidates | Publish what survived, mark `incomplete` — never pad with unverified links |
| Unbounded topics | Scope too broad to research meaningfully | Agent narrows to one entry point, flagged on the page |
| Filler treated as a task | Thinking aloud mid-dump | Explicit `noise` type, excluded at publish |
| Recording lost to a failed API call | Network failure after capture | `.wav` retained on disk, path printed |
| Notion 404 on a page that exists | Integration not shared with the parent page | Documented in setup; explicit error message |

---

# Part VIII — Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

`.env` in the project root — see `.env.example`. For Notion: create an internal integration at notion.so/my-integrations, then open the parent page and share it with that integration. Without this the API returns 404 on a page that visibly exists.

```bash
python main.py          # full pipeline
python record.py        # capture only
python classify.py      # re-classify the latest transcript
python research.py      # re-run research from the latest items
python publish.py       # re-publish from the latest run
```

Each stage runs independently against files on disk, resolving the most recent file of the type it needs. That's how debugging works here.

---

# Part IX — Limitations

1. **No cross-day memory.** Each morning is independent, so a repeated intention gets re-researched daily and yesterday's unfinished work simply vanishes. The most obvious gap.
2. **No feedback signal.** The system never learns which resources I actually opened, so resource quality can't improve over time.
3. **Hand-maintained vocabulary.** The known-terms list is typed by me; it should be learned from my own history.
4. **Batch only.** No way to add a single thought at 3 PM — a once-a-morning tool by design.
5. **English only**, and untested against heavy accents or background noise.
6. **No evaluation harness.** Correctness is judged by reading the output against a known-good case, not by an automated test suite.
7. **The agent's judgment is unmeasured.** I validate that links *work*. I have no metric for whether they're *good*.

---

# Part X — What I'd Build Next

- **Resource feedback.** Track which links I open; weight future selection toward sources I actually use. This is the one that would most improve output quality, because right now the agent has no signal at all.
- **Learned vocabulary.** Build the known-terms list from my own transcript history instead of typing it.
- **Local transcription.** `faster-whisper` on-device — brain dumps are personal, and this removes the only step that sends my voice to a third party.
- **Evaluation harness.** Labelled transcripts with expected outputs, so prompt changes can be measured instead of eyeballed.
