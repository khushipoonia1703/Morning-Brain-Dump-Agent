"""Stage 2 — normalize, segment, classify, tag.

Reads a transcript, makes ONE structured LLM call, writes items/items_<stamp>.json
and prints a table so the result can be eyeballed.

    python classify.py                          # use the most recent transcript
    python classify.py transcripts/dump_*.txt   # use a specific transcript

This is a workflow step, NOT an agent. One transcript goes in, one array comes
out. There is no decision for the model to make about what happens next.
"""

import json
import os
import sys
from glob import glob
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

MODEL = "gpt-5.6"

TRANSCRIPTS_DIR = Path("transcripts")
ITEMS_DIR = Path("items")

# Terms speech recognition reliably mangles. Extend this list as new ones show up.
KNOWN_TERMS = [
    "Claude Code", "RAG", "LangChain", "LangGraph", "CrewAI",
    "Emergent Labs", "system design", "CI/CD", "recursion",
    "Notion", "Whisper", 
]

SYSTEM_PROMPT = f"""\
You turn a spoken morning brain dump into a list of discrete, structured items.

The input is a raw speech-to-text transcript. It has no reliable punctuation, it
rambles, it doubles back, and it contains thinking-aloud filler.

Do these four jobs, IN THIS ORDER:

1. NORMALIZE mistranscribed technical terms against this known-terms list:
{json.dumps(KNOWN_TERMS, indent=2)}
   Speech recognition mangles domain vocabulary. If a phrase is plausibly a
   garbled version of a known term, fix it. Real example: "Claude Court" is
   always "Claude Code". "Lang chain" is "LangChain". "emergent labs" is
   "Emergent Labs". A term left unfixed goes straight into a search query and
   returns nothing useful. Do NOT invent terms that were not spoken.

2. SEGMENT the transcript into discrete items — one intention per item.
   Do this BEFORE deciding what is filler. Words like "and", "also", "and then"
   are simultaneously filler AND the cue that a new item started, so they are
   your segmentation signal. Never strip them first.
   One spoken sentence often holds TWO separate intentions joined by "and".
   Example: "learn how to build with Emergent Labs and how to make improvements
   in my RAG research agent project" is TWO items, not one.
   Segment on meaning, not on punctuation. Do not merge unrelated intentions,
   and do not split a single intention into fragments.

3. CLASSIFY each item's type:
   - "study"    — something to learn, watch, read, or understand
   - "task"     — something to do
   - "reminder" — a task with a spoken clock time. ONLY use this when a time was
                  actually said. "Call Sam at 6" is a reminder with time "18:00".
                  "Call my professor" with no time is a plain task.
   - "noise"    — thinking-aloud filler that is not an intention at all.
                  Examples: "and what else?", "umm", a stray dangling number
                  like "Ten" that is not attached to any item.
   Noise is a classification RESULT, not a pre-filter. Keep noise items in the
   output — they are shown to the user so over-filtering is visible.

4. TAG the remaining fields on every item:
   - needs_resources: whether this item would benefit from learning resources
     (a video and articles) to get started today.
     THIS IS INDEPENDENT OF type. Never derive it from type. Judge the item.
       "Solve 2 recursion problems"    -> task,  needs_resources true
       "Get a haircut"                 -> task,  needs_resources false
       "Apply to 5 AI jobs"            -> task,  needs_resources false
       "Watch a Claude Code tutorial"  -> study, needs_resources true
     Personal errands and admin never need resources. Anything requiring a
     technical starting point does. noise is always false.
   - scope: "unbounded" if the topic is too broad to research meaningfully in
     three links (e.g. "learn system design"), otherwise "narrow".
   - quantity: the number spoken, as an integer ("solve two problems" -> 2,
     "apply for five AI jobs" -> 5). null if no number was spoken.
   - time: spoken clock time as 24-hour "HH:MM" ("at 2 PM" -> "14:00").
     null unless a time was actually spoken.
   - entities: the technical terms in the item, used later to build search
     queries. Empty list if there are none.
   - raw_text: the words as actually spoken, before normalization. Keep this
     verbatim so normalization can be debugged.
   - normalized: the cleaned, corrected, self-contained rewrite of the item.
     Write it as a short imperative line a person could act on. Preserve
     quantities in the text ("Apply to 5 AI jobs").

Return every item in the order it was spoken, including the noise ones.
"""

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string"},
                    "normalized": {"type": "string"},
                    "type": {"type": "string", "enum": ["study", "task", "reminder", "noise"]},
                    "needs_resources": {"type": "boolean"},
                    "scope": {"type": "string", "enum": ["narrow", "unbounded"]},
                    "quantity": {"type": ["integer", "null"]},
                    "time": {"type": ["string", "null"]},
                    "entities": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "raw_text", "normalized", "type", "needs_resources",
                    "scope", "quantity", "time", "entities",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def latest_transcript():
    """Resolve the most recent transcript when no path was given."""
    matches = sorted(glob(str(TRANSCRIPTS_DIR / "dump_*.txt")))
    if not matches:
        print(f"No transcripts found in {TRANSCRIPTS_DIR}/. Run record.py first.")
        sys.exit(1)
    return Path(matches[-1])


def classify(transcript):
    """One structured call. Returns the list of item dicts."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Add it to .env and try again.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "brain_dump_items",
                    "strict": True,
                    "schema": ITEM_SCHEMA,
                }
            },
        )
    except Exception as e:
        print(f"Classification failed ({MODEL}): {e}")
        sys.exit(1)

    return json.loads(response.output_text)["items"]


def save_items(items, stamp):
    """Write items/items_<stamp>.json and return the path."""
    ITEMS_DIR.mkdir(exist_ok=True)
    path = ITEMS_DIR / f"items_{stamp}.json"
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return path


def stamp_from_transcript(transcript_path):
    """Reuse the timestamp in dump_YYYY-MM-DD_HHMM.txt so the files stay paired."""
    name = Path(transcript_path).stem
    return name[len("dump_"):] if name.startswith("dump_") else name


def print_items(items):
    """Show every item, noise included, so over-filtering is visible."""
    console = Console()

    table = Table(title="Stage 2 - classified items", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("normalized")
    table.add_column("type")
    table.add_column("res", justify="center")
    table.add_column("scope")
    table.add_column("qty", justify="right")
    table.add_column("time")
    table.add_column("entities", style="dim")

    type_colour = {
        "study": "cyan",
        "task": "green",
        "reminder": "yellow",
        "noise": "bright_black",
    }

    for i, item in enumerate(items, 1):
        is_noise = item["type"] == "noise"
        row_style = "bright_black" if is_noise else None
        table.add_row(
            str(i),
            item["normalized"],
            f"[{type_colour[item['type']]}]{item['type']}[/]",
            "[bold]yes[/]" if item["needs_resources"] else "no",
            f"[magenta]{item['scope']}[/]" if item["scope"] == "unbounded" else item["scope"],
            "" if item["quantity"] is None else str(item["quantity"]),
            item["time"] or "",
            ", ".join(item["entities"]),
            style=row_style,
        )

    console.print(table)

    # The normalization fixes are the easiest thing to get silently wrong,
    # so show every text the model changed.
    changed = [it for it in items if it["raw_text"].strip() != it["normalized"].strip()]
    if changed:
        console.print("\n[bold]Normalized:[/]")
        for item in changed:
            console.print(f"  [dim]{item['raw_text']}[/]  ->  {item['normalized']}")

    real = [it for it in items if it["type"] != "noise"]
    noise = [it for it in items if it["type"] == "noise"]
    researched = [it for it in real if it["needs_resources"]]
    console.print(
        f"\n{len(real)} items, {len(noise)} noise, "
        f"{len(researched)} need resources."
    )


def main():
    transcript_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_transcript()
    if not transcript_path.exists():
        print(f"No such file: {transcript_path}")
        sys.exit(1)

    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        print(f"Transcript is empty: {transcript_path}")
        sys.exit(1)

    print(f"Transcript: {transcript_path}")
    print(f"Classifying with {MODEL} ...\n")

    items = classify(transcript)
    print_items(items)

    stamp = stamp_from_transcript(transcript_path)
    items_path = save_items(items, stamp)
    print(f"\nSaved items: {items_path}")


if __name__ == "__main__":
    main()
