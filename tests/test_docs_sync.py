"""Local design notes (kept out of git): the excerpt must track the design doc's revision.

docs/DESIGN.md and docs/DESIGN_EXCERPT.md live only in working copies; in a fresh clone
this check skips.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _revision(name: str) -> tuple[int, str]:
    text = (DOCS / name).read_text()
    match = re.search(r"Revision: (\d+) \((\d{4}-\d{2}-\d{2})\)", text)
    assert match, f"{name} has no 'Revision: N (YYYY-MM-DD)' line"
    return int(match.group(1)), match.group(2)


def test_excerpt_tracks_design_revision():
    if not (DOCS / "DESIGN.md").exists():
        pytest.skip("design notes are local-only")
    assert _revision("DESIGN.md") == _revision("DESIGN_EXCERPT.md")
