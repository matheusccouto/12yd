# Hugging Face for model and data persistence

The v1 implementation kept the model (`output/lightgbm.pkl`) and the data (`output/*.jsonl`) on disk, both gitignored, with no remote copy. Phase 2 (Streamlit dashboard) needs to read the model and the predictions from a reachable location; the alternative (re-train on every dashboard load) is not viable. We chose Hugging Face (`couto/12yd`) as the persistence layer — a file store with a model-card surface that the dashboard reads from via `huggingface_hub.hf_hub_download`, with no new server, no CI, no scheduled job. Manual `huggingface-cli upload` after the slice pipeline re-runs. Considered: S3 (more setup, no model card), git-LFS (no model card, no version control on JSONL data), DVC (extra layer of tooling). HF wins on simplicity: it's a single `huggingface-cli` command to push, a single `hf_hub_download` to pull, and the model card is a Markdown README at the root.

## Consequences

- The HF account (`couto`) is the canonical source of the deployed model. A future retrain that changes the LightGBM must also push to HF; nothing deploys without an HF upload.
- The push is manual, so a stale model on HF is a real risk. Mitigated by the smoke tests pinning the artifact bytes (the test fails if the live `.pkl` doesn't match the expected MD5), but the push itself is not in the test path.
- The model card is a minimum-viable README (name, description, metrics, 5-line usage). A future iteration can add training-data summary, limitations, and version history; v2 does not.
