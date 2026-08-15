"""Runs the whole pipeline: microphone in, Notion page out.

    python main.py

Each stage is run exactly the way you would run it by hand, with the previous
stage's output passed in as an explicit path. Nothing is held in memory between
stages — the contract between them is a file on disk, which is what makes any
stage re-runnable on its own afterwards.

    record.py   ->  transcripts/dump_<stamp>.txt
    classify.py ->  items/items_<stamp>.json
    research.py ->  runs/run_<stamp>.json
    publish.py  ->  a Notion page URL

This is a workflow step. No model call happens here; the stages make their own.
"""

import re
import subprocess
import sys
from glob import glob
from pathlib import Path

TRANSCRIPTS_DIR = Path("transcripts")
ITEMS_DIR = Path("items")
RUNS_DIR = Path("runs")


def banner(number, title):
    print()
    print("=" * 62)
    print(f"  STAGE {number} - {title}")
    print("=" * 62)


def run_stage(script, arg=None, capture=False):
    """Run one stage as its own process. Returns its stdout when captured.

    Not captured by default, so the recording timer and the rich tables appear
    live, and so stage 1 can still read Enter from the keyboard.
    """
    command = [sys.executable, script]
    if arg is not None:
        command.append(str(arg))

    # The child writes straight to the console while our own prints sit in
    # Python's buffer, so flush first or the banners land out of order.
    sys.stdout.flush()

    if capture:
        result = subprocess.run(command, text=True, capture_output=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    else:
        result = subprocess.run(command)

    return result


def require(path, script):
    """Stop with a clear message if a stage did not produce what comes next."""
    if not Path(path).exists():
        print(f"\n{script} finished but {path} is missing. Stopping.")
        sys.exit(1)
    return Path(path)


def newest_transcript():
    matches = sorted(glob(str(TRANSCRIPTS_DIR / "dump_*.txt")))
    if not matches:
        print("\nrecord.py produced no transcript. Stopping.")
        sys.exit(1)
    return Path(matches[-1])


def main():
    # -- stage 1: microphone -> wav -> transcript ---------------------------
    banner(1, "Record and transcribe")
    result = run_stage("record.py")
    if result.returncode != 0:
        print("\nrecord.py failed. Stopping — nothing downstream can run.")
        sys.exit(result.returncode)

    # record.py picks its own timestamp, so read it back off the filename it
    # just wrote. Every later path is derived from this one stamp.
    transcript_path = newest_transcript()
    stamp = transcript_path.stem[len("dump_"):]
    print(f"\n-> transcript: {transcript_path}")

    # -- stage 2: transcript -> items --------------------------------------
    banner(2, "Normalize, segment, classify")
    result = run_stage("classify.py", transcript_path)
    if result.returncode != 0:
        print(f"\nclassify.py failed. Your transcript is safe at {transcript_path}.")
        sys.exit(result.returncode)
    items_path = require(ITEMS_DIR / f"items_{stamp}.json", "classify.py")
    print(f"\n-> items: {items_path}")

    # -- stage 3: items -> verified resources (THE AGENT) -------------------
    banner(3, "Research (the agent)")
    result = run_stage("research.py", items_path)
    run_path = RUNS_DIR / f"run_{stamp}.json"

    if result.returncode != 0:
        # research.py exits 1 on a partial run but still writes the run file.
        # Searching costs money, so publish what it managed to verify rather
        # than throwing the whole morning away.
        if run_path.exists():
            print("\nresearch.py stopped early, but wrote a run file.")
            print("Publishing what it verified.")
        else:
            print(f"\nresearch.py failed. Your items are safe at {items_path}.")
            sys.exit(result.returncode)
    require(run_path, "research.py")
    print(f"\n-> run: {run_path}")

    # -- stage 4/5: run file -> Notion page ---------------------------------
    banner(4, "Compose and publish")
    result = run_stage("publish.py", run_path, capture=True)
    if result.returncode != 0:
        print(f"\npublish.py failed. Re-publish without re-paying for research:")
        print(f"  python publish.py {run_path}")
        sys.exit(result.returncode)

    print()
    print("=" * 62)
    match = re.search(r"https://\S+", result.stdout)
    if match:
        print(f"  Done. Your page: {match.group(0)}")
    else:
        print("  Done.")
    print("=" * 62)


if __name__ == "__main__":
    main()
