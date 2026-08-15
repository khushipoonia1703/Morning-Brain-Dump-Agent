"""Stage 3 — THE AGENT.

For every item where needs_resources is true, find one video and one current
article that would let me start on it today.

    python research.py                      # use the most recent items file
    python research.py items/items_*.json   # use a specific items file

This stage is an AGENT, not a workflow step. The model is given three tools and
a goal. It decides which tool to call, in what order, and when it is finished.
The code does not script the sequence — it only enforces two things the model
is not allowed to decide:

  1. Where URLs come from. There are exactly two sources of submittable URLs,
     and the model is neither of them:
       - search_web, which returns them from a real search API
       - fetch_page, which harvests the anchor hrefs off a page it fetched
     Both write into self.seen, the whitelist submit() is checked against. A
     URL the model typed from memory is never in there, so it is rejected
     before it reaches validation.
  2. Whether a URL actually works. Every submitted URL is fetched by the code.
     If a link dies and the model still has budget, the failure is handed back
     so it can go find another one.
"""

import json
import os
import re
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

MODEL = "gpt-5.6-terra"

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 10

# Cap on how many anchor links one fetched page may contribute to the
# whitelist, so a link-heavy page cannot balloon self.seen.
MAX_LINKS_PER_PAGE = 10

# How many articles we are trying to keep per item, alongside one video.
ARTICLES_WANTED = 1

# Stop reading a fetched page after this much. A harvested link can point at a
# PDF or a video file, and response.text would otherwise pull all of it.
MAX_PAGE_BYTES = 500_000

# An anchor shorter than this is almost always chrome ("Home", "Log in", "Docs").
MIN_ANCHOR_TEXT = 15

# URL paths that are site furniture rather than content.
CHROME_PATHS = re.compile(
    r"/(login|signup|sign-up|pricing|about|contact|privacy|terms)(/|$)",
    re.IGNORECASE,
)

ITEMS_DIR = Path("items")
RUNS_DIR = Path("runs")

# Safety bounds on the agent. These are limits, not a procedure — the model is
# free to use them in any order and may finish well under them.
MAX_TOOL_CALLS = 8
MAX_SEARCHES = 3

# How many times a failed submit is handed back before the partial set is
# accepted. Stops an unfillable slot from consuming the whole call budget.
MAX_SUBMIT_RETRIES = 2

# A real browser UA. The default python-requests UA gets 403'd by YouTube and
# plenty of docs sites, which would drop good links and corrupt the drop rate.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

console = Console()


AGENT_SYSTEM = f"""\
You are finding practical starting resources for one item from someone's
morning brain dump.

You have three tools: search_web, fetch_page, and submit.

YOU decide which tools to use, in what order, and how many times. There is no
required sequence.

Most items are done in two calls: one search_web, then submit. The first result
set usually already contains a good video and a good article - prefer picking
both from it. Spend more than that only when the item actually needs it: a
second search when the first genuinely lacks a video OR an article, and
fetch_page when a result is too ambiguous to judge from its title and snippet,
or when a broad topic needs a look at what a page really covers before you can
narrow it.

Do NOT use fetch_page to check whether a link works. Every URL you submit is
fetched and checked by the system after you submit, so fetching to verify is
wasted effort. Judge relevance from the title and snippet.

Finishing early is a good outcome, not a lazy one. Unspent budget is not
wasted budget.

What counts as a good result:
- One video. Prefer YouTube. It should teach the thing, not advertise it.
- One article, blog post, or documentation page that is current and specific
  enough to act on today.
- Prefer official docs, well-known engineering blogs, and recent material.
- Avoid SEO listicles, content farms, and pages that are mostly ads.
- The two resources should complement each other, not repeat one source.

Hard rules:
- You may ONLY submit a URL if it came back from search_web, or if it was
  found on a page you opened with fetch_page. Those are the only two sources.
  A URL you type from memory is in neither set and will be rejected. If you
  need a URL, search for it, or fetch a page that links to it.
- Every URL you submit is fetched and checked by the system after you submit.
  If one is dead, you will be told which one, and you can search for a
  replacement if you have budget left.
- Call submit when you are done. If you cannot find both good resources,
  submit what you do have rather than submitting nothing.

Hard limits for this item: at most {MAX_TOOL_CALLS} tool calls, of which at
most {MAX_SEARCHES} may be search_web. These are ceilings for a difficult item,
not targets. A straightforward item should finish well under them.
"""

TOOLS = [
    {
        "type": "function",
        "name": "search_web",
        "description": (
            "Search the web and get back candidate results as title, url and a "
            "short snippet. URLs returned by this tool become submittable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "fetch_page",
        "description": (
            "Fetch a page you already have a URL for, to see what is actually "
            "on it. Returns the page title and the first ~2000 characters of "
            "text, and also reports any links found on that page - those links "
            "become submittable too, so this is a way to reach pages a search "
            "did not return directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "A URL from a previous search_web result, or one found "
                        "on a page you already fetched."
                    ),
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "submit",
        "description": (
            "Submit your chosen resources for this item. The system will then "
            "verify every URL actually loads."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": ["string", "null"],
                    "description": "The video URL, or null if you found none.",
                },
                "article_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One article or blog post URL, as a single-element list.",
                },
                "notes": {
                    "type": "string",
                    "description": "One short line on why these three are the right starting point.",
                },
                "narrowed_to": {
                    "type": ["string", "null"],
                    "description": (
                        "For a broad topic only: the one concrete entry point you "
                        "narrowed it down to. Otherwise null."
                    ),
                },
            },
            "required": ["video_url", "article_urls", "notes", "narrowed_to"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# ---------------------------------------------------------------------------
# URL validation — done by the code, never by the model.
# ---------------------------------------------------------------------------

def validate_url(url):
    """Return ('ok' | 'dead' | 'unverified', detail).

    GET not HEAD, browser UA, follows redirects. 403 is recorded as
    'unverified' rather than 'dead' because it usually means bot-blocked, not
    broken — counting it as dead would make the drop rate lie.
    """
    try:
        with requests.get(
            url,
            headers={"User-Agent": BROWSER_UA},
            allow_redirects=True,
            timeout=10,
            stream=True,
        ) as response:
            code = response.status_code
    except requests.RequestException as e:
        return "dead", type(e).__name__

    if 200 <= code < 300:
        return "ok", code
    if code == 403:
        return "unverified", code
    return "dead", code


def strip_html(html):
    """Crude tag strip. Good enough to let the model see what a page is about."""
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;?", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# The agent run for a single item.
# ---------------------------------------------------------------------------

class ItemRun:
    """Holds the budget, the tool implementations, and what survived validation."""

    def __init__(self, client, search_key, item):
        self.client = client        # OpenAI - the agent's brain
        self.search_key = search_key  # Tavily - the agent's eyes
        self.item = item
        self.calls_used = 0
        self.searches_used = 0
        self.submit_retries = 0
        self.done = False

        # url -> title. Populated ONLY by search_web (search API results) and
        # fetch_page (anchor hrefs off a fetched page). Never by the model.
        # This is the whitelist that submit is checked against.
        self.seen = {}

        # Counts ONLY the URLs search_web put into self.seen. Kept separate
        # because self.seen also holds harvested anchors, which would inflate
        # the "URLs found by search" metric.
        self.search_urls_found = 0

        # Resources that have already passed validation, kept across retries.
        self.kept_video = None
        self.kept_articles = []

        # url -> (status, detail). A resubmitted URL is answered from here
        # rather than re-fetched, so validation_log holds one row per distinct
        # URL and the drop rate is not inflated by retries.
        self.validated = {}

        self.validation_log = []
        self.notes = ""
        self.narrowed_to = None

    # -- tool 1 -------------------------------------------------------------

    def search_web(self, query):
        if self.searches_used >= MAX_SEARCHES:
            return "Search budget exhausted. Choose from the results you already have."
        self.searches_used += 1

        # A real search API. No model sits between the query and these URLs.
        try:
            response = requests.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {self.search_key}"},
                json={
                    "query": query,
                    "max_results": TAVILY_MAX_RESULTS,
                    "search_depth": "basic",
                },
                timeout=20,
            )
        except requests.RequestException as e:
            return f"Search failed: {type(e).__name__}. Try again or use what you have."

        if response.status_code != 200:
            # Include the body — a bare status code is not diagnosable when a
            # search fails mid-run.
            detail = response.text[:200].replace("\n", " ")
            console.print(
                f"      [red]search HTTP {response.status_code}: {detail}[/]"
            )
            return (
                f"Search failed: HTTP {response.status_code} ({detail}). "
                "Try again or use the results you already have."
            )

        results = []
        duplicates = 0
        for hit in response.json().get("results", []):
            url = hit.get("url")
            if not url:
                continue
            if url in self.seen:
                duplicates += 1
                continue
            title = hit.get("title") or url
            self.seen[url] = title
            self.search_urls_found += 1
            results.append({
                "title": title,
                "url": url,
                "snippet": (hit.get("content") or "")[:300],
            })

        if not results:
            if duplicates:
                # Telling the model "no results" here would send it rephrasing
                # a query that worked. It repeated itself.
                return (
                    f"All {duplicates} results were pages you already have. "
                    "Use those, or search for something more specific."
                )
            return "No results with citable URLs. Try a different query."
        return json.dumps(results, indent=2)

    # -- tool 2 -------------------------------------------------------------

    def fetch_page(self, url):
        # The URL must have come from a real source — a search result, or a link
        # found on a page already fetched. Both live in self.seen.
        if url not in self.seen:
            return (
                "That URL did not come from search_web or from a page you have "
                "already fetched, so it cannot be fetched. Search for it first."
            )
        try:
            with requests.get(
                url,
                headers={"User-Agent": BROWSER_UA},
                allow_redirects=True,
                timeout=10,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    return f"Fetch returned HTTP {response.status_code}."

                # A harvested link can point at a PDF, an image or a video.
                # Only read things that are actually text.
                content_type = response.headers.get("Content-Type", "")
                if "html" not in content_type.lower() and "text" not in content_type.lower():
                    return (
                        f"Not a readable page (Content-Type: "
                        f"{content_type or 'unknown'}). Choose a different URL."
                    )

                # Read at most MAX_PAGE_BYTES so one huge file cannot stall
                # the run.
                chunks, total = [], 0
                for chunk in response.iter_content(chunk_size=8192):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= MAX_PAGE_BYTES:
                        break
                html = b"".join(chunks).decode(
                    response.encoding or "utf-8", errors="replace"
                )
                final_url = response.url
        except requests.RequestException as e:
            return f"Fetch failed: {type(e).__name__}."

        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
        title = strip_html(match.group(1)) if match else self.seen[url]

        discovered = self.harvest_links(html, final_url)

        payload = {"title": title, "text": strip_html(html)[:2000]}
        note = ""
        if discovered:
            note = (
                f"\n\n{len(discovered)} new links on this page were added to your "
                "candidates and can now be submitted:\n"
                + "\n".join(f"  {u}  {t}" for u, t in discovered)
            )
        return json.dumps(payload, indent=2) + note

    def harvest_links(self, html, base_url):
        """Add this page's most content-looking anchor hrefs to the whitelist.

        The URLs come off the page itself, so they carry the same provenance
        guarantee as a search result: the model did not author them.

        Taking the first N anchors would collect the logo, the nav bar and the
        cookie banner. So every anchor is considered, obvious site chrome is
        dropped, and the survivors are ranked by anchor-text length — a rough
        proxy for "this is a link to real content", which is good enough here.
        """
        candidates = []
        for href, anchor_text in re.findall(
            r"(?is)<a\b[^>]*\bhref\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html
        ):
            href = href.strip()
            if href.startswith("#"):
                continue

            absolute = urljoin(base_url, href)
            if urlparse(absolute).scheme not in ("http", "https"):
                continue
            if absolute in self.seen:
                continue

            text = strip_html(anchor_text)[:120]
            if len(text) < MIN_ANCHOR_TEXT:
                continue
            if CHROME_PATHS.search(urlparse(absolute).path):
                continue

            candidates.append((absolute, text))

        # Longest anchor text first, then keep the cap.
        candidates.sort(key=lambda pair: len(pair[1]), reverse=True)

        discovered = []
        for absolute, text in candidates[:MAX_LINKS_PER_PAGE]:
            self.seen[absolute] = text
            discovered.append((absolute, text))
        return discovered

    # -- tool 3 -------------------------------------------------------------

    def submit(self, video_url, article_urls, notes, narrowed_to):
        candidates = ([video_url] if video_url else []) + list(article_urls or [])

        # Rule 1: a URL the model did not get from search cannot be published.
        # Checked before anything is recorded, so a rejected submit leaves no
        # trace of itself on the item.
        invented = [u for u in candidates if u not in self.seen]
        if invented:
            return (
                "Rejected - these URLs did not come from search_web and were "
                "not found on a page you fetched: "
                + ", ".join(invented)
                + ". Only submit URLs from a search result or from a fetched page."
            )

        self.notes = notes or self.notes
        # narrowed_to drives the "broad topic - start here" note on the page, so
        # it is only meaningful for an unbounded item. The model will sometimes
        # volunteer one anyway; ignore it.
        if self.item.get("scope") == "unbounded":
            self.narrowed_to = narrowed_to or self.narrowed_to

        failures = []
        for url in candidates:
            if url in self.validated:
                # Already checked on an earlier submit. Reuse the verdict rather
                # than re-fetching it and logging it twice.
                status, detail = self.validated[url]
            else:
                status, detail = validate_url(url)
                self.validated[url] = (status, detail)
                self.validation_log.append(
                    {"url": url, "status": status, "detail": str(detail)}
                )
                console.print(f"      validate {status:<10} {url}", style="dim")

            # Only a dead link forces a replacement. A 403 means bot-blocked,
            # not broken - keep it, and warn. The run file still records it as
            # "unverified" in validation, so the metric stays honest.
            if status == "dead":
                failures.append(f"{url} ({status}, {detail})")
                continue
            if status == "unverified":
                console.print(f"      [yellow]keeping unverified (403) {url}[/]")

            resource = {"url": url, "title": self.seen[url]}
            if url == video_url and self.kept_video is None:
                self.kept_video = resource
            elif (
                url != video_url
                and url not in [a["url"] for a in self.kept_articles]
                and len(self.kept_articles) < ARTICLES_WANTED
            ):
                self.kept_articles.append(resource)

        if self.is_complete():
            self.done = True
            return "Accepted. All resources verified. You are finished with this item."

        # Rule 2: validation failure is a recovery path, not a dead end.
        # Hand the failure back while there is still budget to act on it.
        if self.calls_used < MAX_TOOL_CALLS:
            self.submit_retries += 1

            # Some slots are genuinely unfillable. Without this, the item burns
            # every remaining call re-submitting the same impossible set.
            if self.submit_retries >= MAX_SUBMIT_RETRIES:
                self.done = True
                return (
                    "Accepted with what was verified - no retries left for this "
                    "item. You are finished with it."
                )

            message = ["Some resources did not survive validation."]
            if failures:
                message.append("Failed: " + "; ".join(failures))
            message.append(
                f"Verified so far: {'1' if self.kept_video else '0'} video, "
                f"{len(self.kept_articles)} of {ARTICLES_WANTED} article."
            )
            message.append(
                f"Find replacements for what is missing, then submit again "
                f"({MAX_SUBMIT_RETRIES - self.submit_retries} retries left)."
            )
            return " ".join(message)

        self.done = True
        return "Budget exhausted. Keeping what was verified."

    # -- plumbing -----------------------------------------------------------

    def is_complete(self):
        return self.kept_video is not None and len(self.kept_articles) >= ARTICLES_WANTED

    def dispatch(self, call):
        """Run whichever tool the model chose and return its output as a string."""
        args = json.loads(call.arguments)
        self.calls_used += 1

        if call.name == "search_web":
            console.print(f"    [{self.calls_used}] search_web  {args['query']!r}")
            output = self.search_web(args["query"])
        elif call.name == "fetch_page":
            console.print(f"    [{self.calls_used}] fetch_page  {args['url']}")
            output = self.fetch_page(args["url"])
        elif call.name == "submit":
            console.print(f"    [{self.calls_used}] submit")
            output = self.submit(
                args.get("video_url"),
                args.get("article_urls"),
                args.get("notes", ""),
                args.get("narrowed_to"),
            )
        else:
            output = f"Unknown tool: {call.name}"

        # Only mention the budget once it is nearly gone. Reporting what is left
        # after every call reads as a target to spend - in the first run all 5
        # items used exactly 3 of 3 searches.
        remaining = MAX_TOOL_CALLS - self.calls_used
        if remaining <= 2:
            return f"{output}\n\n[only {remaining} tool calls left - submit what you have]"
        return output

    def result(self):
        resources = []
        if self.kept_video:
            resources.append({"kind": "video", **self.kept_video})
        for article in self.kept_articles[:ARTICLES_WANTED]:
            resources.append({"kind": "article", **article})

        return {
            **self.item,
            "researched": True,
            "incomplete": not self.is_complete(),
            "narrowed_to": self.narrowed_to,
            "notes": self.notes,
            "resources": resources,
            "validation": self.validation_log,
            "urls_found": self.search_urls_found,
            "candidates_seen": len(self.seen),
            "tool_calls_used": self.calls_used,
            "searches_used": self.searches_used,
        }


def goal_for(item):
    """The goal handed to the model for one item."""
    lines = [
        f"Item: {item['normalized']}",
        f"Type: {item['type']}",
    ]
    if item.get("entities"):
        lines.append(f"Technical terms: {', '.join(item['entities'])}")
    lines.append("")
    lines.append(
        "Find one video and one current article or blog post that would let "
        "me actually start on this item today."
    )
    if item.get("scope") == "unbounded":
        lines.append("")
        lines.append(
            "This topic is too broad to cover in two links. Narrow it down to "
            "ONE concrete entry point worth starting with, find resources for "
            "that, and pass the entry point as narrowed_to."
        )
    return "\n".join(lines)


def research_item(client, search_key, item):
    """Run the agent loop for one item. The model drives; the code bounds it."""
    run = ItemRun(client, search_key, item)
    conversation = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": goal_for(item)},
    ]
    silent_turns = 0

    while run.calls_used < MAX_TOOL_CALLS and not run.done:
        response = client.responses.create(
            model=MODEL,
            input=conversation,
            tools=TOOLS,
        )
        conversation += response.output

        calls = [o for o in response.output if getattr(o, "type", None) == "function_call"]
        if not calls:
            # The model answered in prose instead of acting. Nudge once, then stop.
            silent_turns += 1
            if silent_turns >= 2:
                break
            conversation.append({
                "role": "user",
                "content": "Use a tool. Call submit once you have your resources.",
            })
            continue

        silent_turns = 0
        for call in calls:
            if run.calls_used >= MAX_TOOL_CALLS:
                output = "Tool budget exhausted for this item."
            else:
                output = run.dispatch(call)
            conversation.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            })
            if run.done:
                break

    return run.result()


# ---------------------------------------------------------------------------
# Stage plumbing.
# ---------------------------------------------------------------------------

def latest_items():
    matches = sorted(glob(str(ITEMS_DIR / "items_*.json")))
    if not matches:
        print(f"No items files in {ITEMS_DIR}/. Run classify.py first.")
        sys.exit(1)
    return Path(matches[-1])


def stamp_from_items(items_path):
    name = Path(items_path).stem
    return name[len("items_"):] if name.startswith("items_") else name


def save_run(payload, stamp):
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"run_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_report(results, stats):
    table = Table(title="Stage 3 - researched items", title_style="bold")
    table.add_column("item")
    table.add_column("calls", justify="right")
    table.add_column("searches", justify="right")
    table.add_column("kept", justify="center")
    table.add_column("status")

    for item in results:
        if not item.get("researched"):
            continue
        kept = len(item["resources"])
        status = "[yellow]incomplete[/]" if item["incomplete"] else "[green]complete[/]"
        table.add_row(
            item["normalized"],
            str(item["tool_calls_used"]),
            str(item["searches_used"]),
            f"{kept}/2",
            status,
        )
    console.print(table)

    for item in results:
        if not item.get("researched") or not item["resources"]:
            continue
        console.print(f"\n[bold]{item['normalized']}[/]")
        if item["narrowed_to"]:
            console.print(f"  [magenta]broad topic - start here: {item['narrowed_to']}[/]")
        for resource in item["resources"]:
            icon = "video  " if resource["kind"] == "video" else "article"
            console.print(f"  {icon}  {resource['url']}")
            console.print(f"           [dim]{resource['title']}[/]")

    validated = stats["ok"] + stats["dead"] + stats["unverified"]
    drop_rate = (stats["dead"] / validated * 100) if validated else 0.0
    console.print("\n[bold]Validation drop rate[/]")
    console.print(f"  URLs found by search : {stats['found']}")
    console.print(f"  candidates seen      : {stats.get('candidates_seen', 0)}")
    console.print(f"  URLs validated       : {validated}")
    console.print(f"  ok                   : {stats['ok']}")
    console.print(f"  dead                 : {stats['dead']}")
    console.print(f"  unverified (403)     : {stats['unverified']}")
    console.print(f"  drop rate            : {drop_rate:.1f}%  (dead / validated)")


def main():
    items_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_items()
    if not items_path.exists():
        print(f"No such file: {items_path}")
        sys.exit(1)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Add it to .env and try again.")
        sys.exit(1)
    search_key = os.getenv("TAVILY_API_KEY")
    if not search_key:
        print("TAVILY_API_KEY is not set. Add it to .env and try again.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    items = json.loads(items_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        print(f"{items_path} is not an items file.")
        print("Expected the JSON array written by classify.py, e.g. items/items_*.json.")
        sys.exit(1)

    to_research = [i for i in items if i.get("needs_resources")]
    noise = [i for i in items if i.get("type") == "noise"]
    skipped = [i for i in items if not i.get("needs_resources") and i.get("type") != "noise"]

    console.print(f"Items file: {items_path}")
    console.print(
        f"{len(to_research)} need resources, {len(skipped)} skipped, "
        f"{len(noise)} noise.\n"
    )

    stats = {"found": 0, "ok": 0, "dead": 0, "unverified": 0}
    results = []
    failure = None

    for index, item in enumerate(items):
        if not item.get("needs_resources"):
            results.append({**item, "researched": False, "resources": []})
            continue

        console.print(f"[bold]{item['normalized']}[/]")
        try:
            researched = research_item(client, search_key, item)
        except Exception as e:
            # Searching costs money. Do not throw away the items that already
            # succeeded because the API failed partway down the list — keep
            # them, record what is missing, and still write the run file.
            console.print(f"  [red]Research failed: {e}[/]")
            failure = e
            for remaining in items[index:]:
                if remaining.get("needs_resources"):
                    results.append({
                        **remaining,
                        "researched": False,
                        "incomplete": True,
                        "resources": [],
                        "error": str(e),
                    })
                else:
                    results.append({**remaining, "researched": False, "resources": []})
            break

        for entry in researched["validation"]:
            stats[entry["status"]] += 1
        results.append(researched)
        console.print()

    # "found" counts every distinct URL search surfaced, whether the model
    # chose to submit it or not.
    stats["found"] = sum(i.get("urls_found", 0) for i in results)
    stats["candidates_seen"] = sum(i.get("candidates_seen", 0) for i in results)

    print_report(results, stats)

    stamp = stamp_from_items(items_path)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "items_file": str(items_path),
        "model": MODEL,
        "stats": stats,
        "partial": failure is not None,
        "items": results,
    }
    run_path = save_run(payload, stamp)
    console.print(f"\nSaved run: {run_path}")

    if failure is not None:
        console.print(
            f"\n[red]Run is PARTIAL - stopped at '{items[index]['normalized']}'.[/]"
        )
        console.print(f"[red]{type(failure).__name__}: {failure}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
