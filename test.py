# %%

from twelveyards.fotmob.client import FotMob

client = FotMob()

# %%
response = client.get(
    "leagues/77",
    params={"season": "2026"},
)

response["pageProps"]["fixtures"]["allMatches"]

# %% [markdown]
# ```json
# {
#     "round": "1/8",
#     "roundName": "Round of 16",
#     "pageUrl": "/matches/switzerland-vs-colombia/1urkle#4653849",
#     "id": "4653849",
#     "home": {"name": "Switzerland", "shortName": "Switzerland", "id": "6717"},
#     "away": {"name": "Colombia", "shortName": "Colombia", "id": "8258"},
#     "status": {
#         "utcTime": "2026-07-07T20:00:00Z",
#         "timezone": "UTC",
#         "finished": true,
#         "started": true,
#         "cancelled": false,
#         "awarded": false,
#         "scoreStr": "0 - 0",
#         "reason": {
#             "short": "Pen",
#             "shortKey": "penalties_short",
#             "long": "After penalties",
#             "longKey": "afterpenalties"
#         }
#     }
# }
# ```

# %%
response["pageProps"]["fixtures"]["allMatches"][95]["status"]["finished"]

# %%
# Finished matches have a reason for why they finished.
# Options are: FT (full time), AET (after extra time), Pen (after penalties)
response["pageProps"]["fixtures"]["allMatches"][95]["status"]["reason"]["short"]

# %%
response = client.get(
    "matches/_/1urkle",  # accepts anything before the match id
    params={"season": "2026"},
)


# %%
(
    response["pageProps"]["content"]["lineup"]["homeTeam"]["starters"]
    + response["pageProps"]["content"]["lineup"]["homeTeam"]["subs"]
)

# %%
(
    response["pageProps"]["content"]["lineup"]["awayTeam"]["starters"]
    + response["pageProps"]["content"]["lineup"]["awayTeam"]["subs"]
)

# %%
# Look for x["eventType"] == "Goal" to confirm if it was a goal or not.
[
    x
    for x in response["pageProps"]["content"]["shotmap"]["Periods"]["All"]
    if x["period"] == "PenaltyShootout"
]

# %%
response = client.get("players/207617")


# %%
response["pageProps"]["data"]["positionDescription"]["primaryPosition"]["key"]

# %%
next(
    iter(
        [
            x
            for x in response["pageProps"]["data"]["playerInformation"]
            if x["translationKey"] == "preferred_foot"
        ],
    ),
)["value"]["key"]

# %%
response = client.get("leagues/77")


# %%
tm = client.get("leagues")["pageProps"]["fallback"]["/api/translationmapping?locale=leagues"]
leagues = {int(k): v for k, v in tm["TournamentPrefixes"].items()}
leagues |= {int(k): v for k, v in tm["TournamentTemplates"].items()}

leagues  # 363 known IDs — some e.g. 47 (Premier League) are missing


# %%
[
    (lid, s["seasonName"])
    for lid in [77, 47, 87, 42, 50]
    for s in client.get(f"leagues/{lid}")["pageProps"].get("seasons", [])
][:10]


# %%
data = client.get("leagues/77", params={"season": "2022"})
fixtures = data["pageProps"]["fixtures"]["allMatches"]

[
    m for m in fixtures
    if m["status"]["reason"]["shortKey"] == "penalties_short"
]



