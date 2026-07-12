"""Parse FotMob season-fixture entries into MatchRef dataclasses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .fotmob_parsing import coerce_int

if TYPE_CHECKING:
    from collections.abc import Mapping

# FotMob match pageUrl shape: `/matches/{seo}/{h2h}#{match_id}`.
# The `seo` is kebab-case (e.g. `argentina-vs-france`); `h2h` is a 6-char
# alphanumeric (e.g. `1hox8a`). The match_id is parsed from the URL anchor.
_PAGE_URL_RE = re.compile(r"^/matches/(?P<seo>[^/]+)/(?P<h2h>[^/#?]+)(?:#\d+)?$")


def parse_page_url(page_url: str) -> tuple[int, str, str]:
    """
    Parse a FotMob match `pageUrl` into (match_id, seo, h2h).

    Format: `/matches/{seo}/{h2h}#{match_id}`. The `seo` is kebab-case
    (e.g. `argentina-vs-france`); `h2h` is a 6-char alphanumeric (e.g.
    `1hox8a`). The match_id is parsed from the URL anchor (after `#`).
    """
    anchor_idx = page_url.find("#")
    if anchor_idx == -1:
        msg = f"pageUrl missing '#{{match_id}}' anchor: {page_url!r}"
        raise ValueError(msg)
    match_id = coerce_int(page_url[anchor_idx + 1 :])
    if not match_id:
        msg = f"pageUrl anchor did not yield an int match_id: {page_url!r}"
        raise ValueError(msg)
    path = page_url[:anchor_idx]
    match = _PAGE_URL_RE.match(path)
    if match is None:
        msg = f"pageUrl path did not match /matches/{{seo}}/{{h2h}}: {page_url!r}"
        raise ValueError(msg)
    return match_id, match.group("seo"), match.group("h2h")


@dataclass(frozen=True)
class MatchRef:
    """
    A reference to one match, parsed from a season fixture entry.

    Carries the union of fields every downstream consumer needs; fields a
    given consumer does not need are left at their defaults. The defaults
    are `0` for ints (the same sentinel `coerce_int` returns for missing
    data) and `""` for strings.
    """

    match_id: int
    seo: str
    h2h: str
    home_team_id: int = 0
    home_team_name: str = ""
    away_team_id: int = 0
    away_team_name: str = ""
    round_name: str = ""
    match_date: str = ""
    score_str: str = ""

    @classmethod
    def from_fixture(cls, fixture: Mapping[str, Any]) -> MatchRef | None:
        """
        Build a `MatchRef` from a season fixture entry, or `None` if malformed.

        Returns `None` when the entry's `pageUrl` is missing/unparseable —
        callers that need to filter rather than crash (e.g. the roster
        orchestrator) iterate the source list and drop `None`s. Returns a
        fully-populated `MatchRef` when the entry parses cleanly; the
        team id fields are `0` for fixtures without a numeric `home.id` /
        `away.id` (defensive; the live payload always has them).

        The fixture's `id` and `pageUrl` are the source of truth for the
        match identity. The pageUrl format is `/matches/{seo}/{h2h}#{match_id}`
        — the trailing `#...` is the visible URL anchor, not part of h2h.
        """
        page_url = str(fixture.get("pageUrl") or "")
        if not page_url:
            return None
        try:
            match_id, seo, h2h = parse_page_url(page_url)
        except ValueError:
            return None

        home = fixture.get("home") or {}
        away = fixture.get("away") or {}
        status = fixture.get("status") or {}
        return cls(
            match_id=match_id,
            seo=seo,
            h2h=h2h,
            home_team_id=coerce_int(home.get("id")),
            home_team_name=str(home.get("name", "")),
            away_team_id=coerce_int(away.get("id")),
            away_team_name=str(away.get("name", "")),
            round_name=str(fixture.get("roundName") or fixture.get("round") or ""),
            match_date=str(status.get("utcTime") or ""),
            score_str=str(status.get("scoreStr") or ""),
        )
