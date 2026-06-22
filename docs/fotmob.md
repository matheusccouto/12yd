# FotMob API — efficient access

Public data access for FotMob via Next.js `__next/data` JSON routes. No auth, no rate limit, no anti-bot. CloudFront edge cache, 1 hour TTL, `gzip` supported, `ETag` supported.

## The pattern

```http
GET https://www.fotmob.com/_next/data/{buildId}/{path}.json[?query]
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36
Accept-Encoding: gzip
```

### Compression and cache (mandatory for efficiency)

| Header | Behavior |
|---|---|
| `Accept-Encoding: gzip` | Server returns gzip-compressed JSON. **~5.8× smaller.** |
| `If-None-Match: "<etag>"` | Server returns `304 Not Modified` with empty body. **Zero bytes on cache hit.** |
| `Cache-Control: public, max-age=3600, s-maxage=3600, stale-while-revalidate=1200, stale-if-error=86400` | CloudFront edge cache, 1h fresh, 20 min revalidation, 24h stale-on-error. |
| `vary: Accept-Encoding` | Gzipped and identity responses cached separately. |

### Response shape

| Field | Type | Notes |
|---|---|---|
| `content-type` | `application/json` | Always, when using `__next/data`. |
| `content-encoding` | `gzip` or absent | Per `Accept-Encoding`. |
| `etag` | `W/"<hash>"` | Weak ETag. Send on next request for 304. |
| `x-cache` | `Hit from cloudfront` or `Miss from cloudfront` | Edge hit indicator. |
| `age` | seconds since edge cached | Subtract from `max-age` for `max-age` remaining. |

## Endpoints

### Build ID discovery (one-time per deploy)

```http
GET https://www.fotmob.com/
```

Parse `<script id="__NEXT_DATA__" type="application/json">{...}</script>` and read `buildId`. Stable across the lifetime of one deploy. Same buildId for match/league/player pages.

```python
import re, json, httpx
r = httpx.get("https://www.fotmob.com/", headers={"User-Agent": UA}, timeout=15)
build_id = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', r.text, re.DOTALL).group(1))["buildId"]
```

### Match

```http
GET https://www.fotmob.com/_next/data/{buildId}/matches/{slug}.json
```

- `slug`: 6-char SEO segment from the match URL. Discover from league fixture list (`pageUrl` field).
- Data path: `pageProps.content.matchFacts.events.penaltyShootoutEvents`
- Also: `pageProps.content.shotmap` (in-game shots), `pageProps.content.stats`, `pageProps.content.lineup`, `pageProps.content.momentum`, `pageProps.content.playerStats`.

### League season fixture list

```http
GET https://www.fotmob.com/_next/data/{buildId}/leagues/{leagueId}/{tab}/{slug}.json?season={year}
```

- `tab`: `overview` (default), `fixtures`, `results`, `table`.
- `slug`: SEO segment (e.g. `world-cup`, `ucl`, `euro`, `grp`).
- `season`: 4-digit year.
- Data path: `pageProps.fixtures.allMatches` or `pageProps.overview.matches.allMatches`.
- Per-entry: `id`, `pageUrl`, `home.{id,name}`, `away.{id,name}`, `status.{utcTime,finished,scoreStr,reason.{shortKey,longKey}}`.

### League seasons index

```http
GET https://www.fotmob.com/_next/data/{buildId}/leagues/{leagueId}.json
```

- Data path: `pageProps.seasons[]` — list of `{seasonName, winner, loser, ...}`.

### Player

```http
GET https://www.fotmob.com/_next/data/{buildId}/players/{playerId}/{slug}.json
```

- `slug`: URL-friendly name (e.g. `kylian-mbappe`).
- Data path: `pageProps.data` (note: `data`, not `content`).
- Top keys: `careerHistory`, `recentMatches`, `statSeasons`, `mainLeagueStats`, `primaryTeam`, `marketValues`, `trophies`, `playerInformation`, `traits`, `injuryInformation`.

## Shootout discovery

Filter the season fixture list at the API level, no per-match fetch required:

```python
shootouts = [m for m in fixtures if m["status"]["reason"]["shortKey"] == "penalties_short"]
```

`shortKey` values:

| Value | Meaning |
|---|---|
| `fulltime_short` | Normal full-time |
| `penalties_short` | **Shootout match** |
| `extratime_short` | After extra time, no shootout |
| `postponed_short` | Cancelled/postponed |

## Per-kick data shape

`pageProps.content.matchFacts.events.penaltyShootoutEvents` is an array (length 0 for non-shootout matches). Each entry:

| Field | Type | Notes |
|---|---|---|
| `eventId` | int | Stable per-kick id. |
| `time` | int | Match minute. |
| `isHome` | bool | Kicking team. |
| `type` | string | `Goal` / `MissedPenalty` / `SavedPenalty`. |
| `player.id` | int | Stable kicker id. |
| `player.name` | string | |
| `homeScore`, `awayScore` | int | Running regular-time score. |
| `penShootoutScore` | [int, int] | Running shootout score. |
| `newScore` | [int, int] | Same as `penShootoutScore` for the post-kick state. |
| `shotmapEvent.expectedGoals` | float | Pre-kick xG. |
| `shotmapEvent.expectedGoalsOnTarget` | float | Conditional on being on-target. |
| `shotmapEvent.shotType` | string | `RightFoot` / `LeftFoot`. |
| `shotmapEvent.goalCrossedY` | float | Goal-mouth Y in meters. |
| `shotmapEvent.goalCrossedZ` | float | Goal-mouth Z in meters (height). |
| `shotmapEvent.onGoalShot.x` | float | **[0, 2] — kicker's perspective: 0 = left post, 1 = center, 2 = right post.** |
| `shotmapEvent.onGoalShot.y` | float | [0, 1] — height within goal frame. |
| `shotmapEvent.isOnTarget` | bool | |

**Missed kicks have no `shotmapEvent`.** `onGoalShot.x` is `None` (not 0) for misses.

## L/C/R bucketing

```python
def side(x):
    if x is None: return None  # miss
    if x < 0.667: return "L"
    if x > 1.333: return "R"
    return "C"
```

Verified on Argentina vs France (2022 WC Final, 8 kicks): 4L / 0C / 2C / 0R for the 6 goals (4L, 2C, 0R).

## Caching strategy

```python
import json, gzip, httpx
from pathlib import Path

CACHE = Path("data/fotmob_cache")
CACHE.mkdir(parents=True, exist_ok=True)

def get(url, headers=None):
    cache_file = CACHE / (url.replace("/", "_").replace("?", "_q_")[:200] + ".json.gz")
    etag_file = cache_file.with_suffix(".etag")
    h = {"User-Agent": UA, "Accept-Encoding": "gzip", **({} if headers is None else headers)}
    if etag_file.exists():
        h["If-None-Match"] = etag_file.read_text().strip()
    r = httpx.get(url, headers=h, timeout=15, follow_redirects=True)
    if r.status_code == 304:
        return json.loads(gzip.decompress(cache_file.read_bytes()))
    r.raise_for_status()
    body = gzip.decompress(r.content) if r.headers.get("content-encoding") == "gzip" else r.content
    cache_file.write_bytes(gzip.compress(body))
    if "etag" in r.headers:
        etag_file.write_text(r.headers["etag"])
    return json.loads(body)
```

Persistent local cache. 304 hits are zero-bandwidth. Disk hits are sub-millisecond. Bypasses the 1h CloudFront TTL.

## Anti-patterns

| Pattern | Cost | Replace with |
|---|---|---|
| `GET /matches/{slug}/{hash}` (HTML) | 1.27 MB | `GET /_next/data/{buildId}/matches/{slug}.json` (473 KB) |
| No `Accept-Encoding: gzip` | 473 KB | With gzip (81 KB) |
| No `If-None-Match` | Always 473 KB | ETag (304 → 0 bytes) |
| `?tab=facts` on the URL | Same 1.27 MB (tab is hash-only) | `__next/data` is the same regardless of tab |
| `GET /api/matchDetails?matchId=...` | 404 (removed) | `__next/data` |
| `?season=...` on the HTML page | Same 1.27 MB | `?season=...` on `__next/data` (returns the requested season) |

## Quick reference

```text
# Match (replace buildId, slug)
GET /_next/data/{buildId}/matches/{slug}.json

# League season
GET /_next/data/{buildId}/leagues/{leagueId}/overview/{slug}.json?season={year}

# League seasons
GET /_next/data/{buildId}/leagues/{leagueId}.json

# Player
GET /_next/data/{buildId}/players/{playerId}/{slug}.json

# Headers (always)
User-Agent: Mozilla/5.0 ... Chrome/131.0.0.0 Safari/537.36
Accept-Encoding: gzip
If-None-Match: "<etag>"     # optional, for 304

# Shootout filter
m["status"]["reason"]["shortKey"] == "penalties_short"

# Per-kick side
m["shotmapEvent"]["onGoalShot"]["x"]  # 0..2, 1 = center
```

## Sample response (excerpt)

Argentina vs France, 2022 FIFA World Cup Final. matchId `3370572`, slug `1hox8a`, buildId `5JrXFqDcvBep-L0Qv6mBO`. Full 81 KB gzipped JSON saved to `docs/samples/match_3370572.json.gz`.

Eight kicks, all with full `shotmapEvent` populated:

```json
[
  {
    "time": 121, "type": "Goal", "isHome": false,
    "player": {"id": 701154, "name": "Kylian Mbappé"},
    "penShootoutScore": [0, 1],
    "shotmapEvent": {
      "shotType": "RightFoot",
      "expectedGoals": 0.7884,
      "goalCrossedY": 37.2025, "goalCrossedZ": 0.9375,
      "onGoalShot": {"x": 0.153, "y": 0.248, "zoomRatio": 1},
      "isOnTarget": true
    }
  },
  {
    "time": 121, "type": "Goal", "isHome": true,
    "player": {"id": 687008, "name": "Gonzalo Montiel"},
    "penShootoutScore": [4, 2],
    "shotmapEvent": {
      "shotType": "RightFoot",
      "expectedGoals": 0.7884,
      "goalCrossedY": 36.44, "goalCrossedZ": 0.1605,
      "onGoalShot": {"x": 0.354, "y": 0.042, "zoomRatio": 1},
      "isOnTarget": true
    }
  }
]
```

Missed kicks (Coman, Tchouaméni) have `type: "MissedPenalty"` and **no `shotmapEvent`** field.
