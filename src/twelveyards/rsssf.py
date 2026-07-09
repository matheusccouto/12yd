"""Parse the RSSSF penalty-shootouts oracle page for completeness verification.

PRD: RSSSF is a verification oracle, never a data source. The scraper asserts
its count of in-scope shootouts matches the count RSSSF lists, and writes
discrepancies to `discrepancies.json` if they diverge.

The page lives at `https://www.rsssf.org/miscellaneous/penaltiestour.html`.
It has six sections (one per major confederation tournament), each a
`<h4>` heading followed by a `<pre>` block of fixed-width rows. Each row
starts with a 4-digit year and is one penalty shootout.

This parser is intentionally tolerant: it skips blank lines, header lines, and
non-data lines (e.g. the `Year Round Teams ... Penalties` header). The output
is a list of `RSSSFShootout` records in document order, which makes the test
of "is the count correct for this date window" trivial.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .leagues import LEAGUE_BY_ID
from .tournaments import RSSSF_TO_LEAGUE_NAME

# The Confederations Cup is on the page but out of scope for v1.
RSSSF_OUT_OF_SCOPE: frozenset[str] = frozenset({"Confederations Cup"})

# Each data row starts with a 4-digit year. Anything else is a header/blank.
_YEAR_LINE_RE = re.compile(r"^(\d{4})\s")


@dataclass(frozen=True)
class RSSSFShootout:
    """One penalty shootout as listed on the RSSSF page.

    `tournament` is the FotMob league name (e.g. "World Cup"), so the
    record joins cleanly with the scraper's league constants. `year` is the
    tournament year (e.g. 2022 for the 2022 World Cup). `raw` is the full
    line as it appears on the page, for debugging.
    """

    tournament: str
    year: int
    round_label: str
    raw: str


def parse_rsssf_html(html: str) -> list[RSSSFShootout]:
    """Parse the full RSSSF penaltiestour page into a flat list of shootouts.

    The page is HTML, but the structure is shallow: `<h4>` headings introduce
    tournament sections; each section is followed by a single `<pre>` block.
    We split on the headings and then walk each block line by line. Header
    lines (the column labels) and blank lines are skipped.

    Out-of-scope sections (currently: the Confederations Cup) are skipped.
    """
    out: list[RSSSFShootout] = []
    # Walk the headings in document order, then for each heading take the
    # block up to the next heading.
    sections = _split_into_sections(html)
    for heading, body in sections:
        league_name = RSSSF_TO_LEAGUE_NAME.get(heading)
        if league_name is None:
            # Out of scope (e.g. Confederations Cup). Skip.
            continue
        for raw in _iter_data_lines(body):
            match = _YEAR_LINE_RE.match(raw)
            if match is None:
                continue
            year = int(match.group(1))
            round_label = _extract_round_label(raw)
            out.append(
                RSSSFShootout(
                    tournament=league_name,
                    year=year,
                    round_label=round_label,
                    raw=raw,
                )
            )
    return out


def load_rsssf_html(path: str | Path) -> str:
    """Load an RSSSF HTML file from disk, handling the page's latin-1 encoding.

    The page declares `<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">`
    but uses extended-ASCII bytes (e.g. `é` in `Copa América`) that are not
    valid UTF-8. We default to latin-1, which is the de-facto encoding for
    these old RSSSF pages.
    """
    return Path(path).read_text(encoding="latin-1")


def count_shootouts_by_pairs(
    shootouts: Iterable[RSSSFShootout],
    league_seasons: Iterable[tuple[int, int]],
) -> int:
    """Count shootouts matching the given set of (FotMob league_id, season) pairs.

    Both the `shootouts` records and the `league_seasons` tuples are
    keyed by FotMob league name. The function does NOT do an RSSSF-heading
    join — it relies on the fact that `RSSSFShootout.tournament` is already
    the FotMob league name (the heading map is applied at parse time).

    This is the right way to count the RSSSF shootouts inside the current
    Prediction Window, because it correctly handles tournaments that span
    calendar years (Euro 2020, AFCON 2021, AFCON 2023, etc.).

    A naive year-range count is wrong for this purpose: e.g. the Euro 2020
    is listed under year 2020 on RSSSF even though every match was played
    in 2021. We anchor on the (FotMob league_id, season) pair the scraper
    uses as the source of truth.
    """
    target: set[tuple[str, int]] = set()
    for league_id, season in league_seasons:
        league = LEAGUE_BY_ID[league_id]
        target.add((league.name, season))
    return sum(1 for s in shootouts if (s.tournament, s.year) in target)


def _split_into_sections(html: str) -> list[tuple[str, str]]:
    """Return `(heading, body)` pairs for each `<h4>` section of the page.

    The body is the text between this heading and the next `<h4>` (or end
    of document). We use a simple string scan rather than an HTML parser
    because the page is shallow and we want to be tolerant of small markup
    variations.
    """
    sections: list[tuple[str, str]] = []
    cursor = 0
    while True:
        h_open = html.find("<h4>", cursor)
        if h_open == -1:
            break
        h_close = html.find("</h4>", h_open)
        if h_close == -1:
            break
        heading = html[h_open + len("<h4>") : h_close].strip()
        next_h = html.find("<h4>", h_close)
        body_end = next_h if next_h != -1 else len(html)
        body = html[h_close + len("</h4>") : body_end]
        sections.append((heading, body))
        cursor = h_close + len("</h4>")
    return sections


def _iter_data_lines(body: str) -> Iterable[str]:
    """Yield the non-empty, non-header lines of a `<pre>` block.

    The first non-empty line in each block is the column header (`Year Round
    Teams ... Penalties`); we detect and skip it. Every other non-empty line
    is a data row (one shootout).
    """
    header_seen = False
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not header_seen:
            # The first non-empty line is the column header.
            header_seen = True
            continue
        yield line


def _extract_round_label(line: str) -> str:
    """Extract the round label from a RSSSF row (e.g. "2R", "QF", "F", "SF").

    The label is the second whitespace-separated token in the line, after
    the 4-digit year. Returns "" if the line is malformed.
    """
    parts = line.split()
    if len(parts) < 2:
        return ""
    return parts[1]
