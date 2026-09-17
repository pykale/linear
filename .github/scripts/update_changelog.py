#!/usr/bin/env python3
"""Insert a generated release section into .github/CHANGELOG.md.

Usage: update_changelog.py <version> <generated-file> <changelog-file>

The generated file holds the label-grouped list of pull requests produced by
release-changelog-builder-action. It is turned into a section

    # Version <version>

    #### <group>

    - <pull request>

which is written to the top of the changelog. Group headings without any
entries below them are dropped, so an empty "Other Changes" group does not
leave a dangling heading behind.

The generated file is rewritten to hold the final section, so the same text
can be reused as the pull request comment body.

Running the workflow more than once for the same version is safe: a section
that is already present is replaced in place instead of being duplicated, and
identical content leaves the file untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECTION_RE = re.compile(r"^#\s+Version\s+(?P<version>\S+)\s*$")
GROUP_RE = re.compile(r"^#{2,6}\s")


def normalise_version(version: str) -> str:
    """Drop a leading ``v`` so tag names and headers compare equal."""
    return version.strip().lstrip("vV")


def tidy(text: str) -> str:
    """Trim trailing whitespace and collapse runs of blank lines."""
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if not line and collapsed and not collapsed[-1]:
            continue
        collapsed.append(line)
    while collapsed and not collapsed[0]:
        collapsed.pop(0)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return "\n".join(collapsed)


def drop_empty_groups(text: str) -> str:
    """Remove group headings that have no entries below them."""
    lines = text.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if GROUP_RE.match(lines[index]):
            end = index + 1
            while end < len(lines) and not GROUP_RE.match(lines[end]):
                end += 1
            if any(line.strip() for line in lines[index + 1 : end]):
                kept.extend(lines[index:end])
            index = end
        else:
            kept.append(lines[index])
            index += 1
    return tidy("\n".join(kept))


def replace_section(changelog: str, version: str, section: str) -> str:
    """Replace the section of ``version``, or prepend it when absent."""
    lines = changelog.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = SECTION_RE.match(line)
        if match and normalise_version(match.group("version")) == version:
            start = index
            break

    if start is None:
        return tidy(f"{section}\n\n{changelog}") + "\n"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if SECTION_RE.match(lines[index]):
            end = index
            break

    merged = "\n\n".join(["\n".join(lines[:start]), section, "\n".join(lines[end:])])
    return tidy(merged) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="release version, e.g. 0.2.0b1")
    parser.add_argument("generated", type=Path, help="changelog built by the action")
    parser.add_argument("changelog", type=Path, help="changelog file to update")
    args = parser.parse_args()

    if not args.generated.is_file():
        sys.stderr.write(f"error: {args.generated} was not generated\n")
        return 1

    body = drop_empty_groups(args.generated.read_text(encoding="utf-8"))
    if not body:
        sys.stderr.write(f"error: {args.generated} is empty\n")
        return 1

    version = normalise_version(args.version)
    section = f"# Version {version}\n\n{body}"

    existing = args.changelog.read_text(encoding="utf-8") if args.changelog.is_file() else ""
    args.changelog.write_text(replace_section(existing, version, section), encoding="utf-8")
    # Reuse the section as the pull request comment body.
    args.generated.write_text(section + "\n", encoding="utf-8")

    sys.stdout.write(f"updated {args.changelog} for version {version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
