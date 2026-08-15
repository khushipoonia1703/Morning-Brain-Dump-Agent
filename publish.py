"""Stage 4/5 — compose and publish.

Reads a run file and creates one Notion page for the day.

    python publish.py                    # use the most recent run file
    python publish.py runs/run_*.json    # use a specific run file

This is a workflow step, NOT an agent. There is no model call anywhere in this
file. It reads JSON off disk and calls the Notion API. The code decides every
block on the page.
"""

import json
import os
import re
import sys
from glob import glob
from pathlib import Path

from dotenv import load_dotenv
from notion_client import Client
from notion_client.errors import APIResponseError

RUNS_DIR = Path("runs")

# Section heading -> the item type that belongs under it, in page order.
SECTIONS = [
    ("Study", "study"),
    ("Tasks", "task"),
    ("Reminders", "reminder"),
]

# "noise" is deliberately absent from SECTIONS. Filler never reaches the page.

VIDEO_MARKER = "🎥"
ARTICLE_MARKER = "📄"


# ---------------------------------------------------------------------------
# Small block builders. Notion's block JSON is verbose; these keep the
# page-building code below readable.
# ---------------------------------------------------------------------------

def text(content, link=None):
    """One rich_text fragment, optionally a hyperlink."""
    return {
        "type": "text",
        "text": {"content": content, "link": {"url": link} if link else None},
    }


def heading(title):
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [text(title)]},
    }


def bullet(fragments):
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": fragments},
    }


def todo(content, children):
    """A checkbox. Resources and notes hang off it as child blocks."""
    block = {
        "object": "block",
        "type": "to_do",
        "to_do": {"rich_text": [text(content)], "checked": False},
    }
    if children:
        block["to_do"]["children"] = children
    return block


# ---------------------------------------------------------------------------
# Composing the page.
# ---------------------------------------------------------------------------

def children_for(item):
    """The blocks that nest under one item: notes first, then its resources."""
    children = []

    # A broad topic the agent narrowed down. Only unbounded items have this.
    if item.get("narrowed_to"):
        children.append(bullet([
            text(f"⚠ Broad topic — start here: {item['narrowed_to']}")
        ]))

    if item.get("incomplete"):
        children.append(bullet([
            text("Fewer resources than intended were verified for this item.")
        ]))

    # Driven by the item's own resources list, not by its type, and never by a
    # hardcoded count — whatever the agent verified is what gets rendered.
    for resource in item.get("resources", []):
        marker = VIDEO_MARKER if resource["kind"] == "video" else ARTICLE_MARKER
        children.append(bullet([
            text(f"{marker} "),
            text(resource["title"], link=resource["url"]),
        ]))

    return children


def item_text(item):
    """The checkbox label. Reminders carry their spoken time as plain text."""
    if item["type"] == "reminder" and item.get("time"):
        # The time is TEXT. Nothing is scheduled, queued, or fired.
        return f"{item['time']} — {item['normalized']}"
    return item["normalized"]


def build_blocks(items):
    """Turn the run file's items into the page's block list."""
    blocks = []
    for title, item_type in SECTIONS:
        section_items = [i for i in items if i.get("type") == item_type]
        if not section_items:
            continue
        blocks.append(heading(title))
        for item in section_items:
            blocks.append(todo(item_text(item), children_for(item)))
    return blocks


def page_date(run):
    """Date for the title, from the items filename, else generated_at."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", run.get("items_file", ""))
    if match:
        return match.group(0)
    return run.get("generated_at", "")[:10]


# ---------------------------------------------------------------------------
# Stage plumbing.
# ---------------------------------------------------------------------------

def latest_run():
    matches = sorted(glob(str(RUNS_DIR / "run_*.json")))
    if not matches:
        print(f"No run files in {RUNS_DIR}/. Run research.py first.")
        sys.exit(1)
    return Path(matches[-1])


def publish(run, blocks, title, api_key, parent_id):
    """Create the page and return its URL."""
    notion = Client(auth=api_key)
    try:
        page = notion.pages.create(
            parent={"page_id": parent_id},
            properties={"title": [text(title)]},
            children=blocks,
        )
    except APIResponseError as e:
        if e.status == 404:
            print("Notion returned 404 for the parent page.")
            print("The page ID may be wrong, but far more often this means the")
            print("parent page has not been shared with your integration.")
            print("Fix: open the page in Notion -> ... menu -> Connections ->")
            print("add your integration. Then run this again.")
            sys.exit(1)
        print(f"Notion API error ({e.status}): {e}")
        sys.exit(1)
    return page["url"]


def main():
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    if not run_path.exists():
        print(f"No such file: {run_path}")
        sys.exit(1)

    load_dotenv()
    api_key = os.getenv("NOTION_API_KEY")
    parent_id = os.getenv("NOTION_PARENT_PAGE_ID")
    if not api_key or not parent_id:
        missing = []
        if not api_key:
            missing.append("NOTION_API_KEY")
        if not parent_id:
            missing.append("NOTION_PARENT_PAGE_ID")
        print(f"Missing in .env: {', '.join(missing)}")
        print("NOTION_API_KEY is the internal integration token from")
        print("  notion.so/my-integrations (starts with 'ntn_' or 'secret_').")
        print("NOTION_PARENT_PAGE_ID is the 32-character id from the parent")
        print("  page's URL. The page must be shared with the integration.")
        sys.exit(1)

    run = json.loads(run_path.read_text(encoding="utf-8"))
    items = run.get("items", [])
    blocks = build_blocks(items)

    if not blocks:
        print(f"{run_path} has no publishable items (noise is excluded).")
        sys.exit(1)

    title = f"Brain Dump — {page_date(run)}"

    print(f"Run file: {run_path}")
    for section_title, item_type in SECTIONS:
        count = sum(1 for i in items if i.get("type") == item_type)
        if count:
            resources = sum(
                len(i.get("resources", []))
                for i in items if i.get("type") == item_type
            )
            print(f"  {section_title}: {count} items, {resources} resources")
    dropped = sum(1 for i in items if i.get("type") == "noise")
    if dropped:
        print(f"  (excluded {dropped} noise items)")

    url = publish(run, blocks, title, api_key, parent_id)
    print(f"\nPublished: {url}")


if __name__ == "__main__":
    main()
