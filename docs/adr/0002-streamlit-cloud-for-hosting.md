# Streamlit Cloud for dashboard hosting

The dashboard (PRD Phase 2) needs a public URL so a viewer can see the per-kicker predictions during a live shootout. We chose Streamlit Cloud — the same vendor as the framework — because the deployment is a one-line `app.py`-at-the-root convention, the dependency manifest is the existing `pyproject.toml` (no separate `requirements.txt`), and the user already uses Streamlit. Considered: Hugging Face Spaces (would have unified the model + data + hosting under HF, but Streamlit-on-HF-Spaces has its own conventions and the user already chose HF for the data, not the hosting), Heroku (paid, more setup), self-hosting (operational burden, no benefit at v2's expected traffic). Streamlit Cloud wins on the fit: the dashboard is a thin layer over the library, the entry point is conventional, and the cost is zero at v2's scale.

## Consequences

- The repo's root must contain an `app.py`. This is a hard convention; v2's `app.py` is a thin ~30-line entry point that imports from `src/penalty_pred/dashboard.py`, so the dashboard's logic stays in the library and remains unit-testable without Streamlit.
- The deployment is tied to the `matheusccouto/12yd` GitHub repo. Any rename or move on GitHub breaks the deployment until Streamlit Cloud is re-pointed.
- Streamlit Cloud's free tier has cold-start latency and a small memory budget. The dashboard's per-load work (HF model download + FotMob fixture fetch + per-kicker re-score) fits comfortably; a future scaling concern is out of scope for v2.
