#!/bin/bash
set -e

prompt='
1. Pick the task you decide has the highest priority, not necessarily the first in the list, while respecting "Blocked by" relationships.
2. Run any feedback loops that exist for this repo, such as tests, lint, or typecheck.
3. Do not commit if any feedback loop fails.
4. Append a structured entry to progress.txt covering the task completed and PRD item reference, key decisions and reasoning, files changed, and blockers or notes for the next iteration.
5. If the issue has acceptance criteria in docs/PRD.md, mark them done if applicable.
6. Commit and push the change.

Work on a single task per iteration. Output <promise>COMPLETE</promise> if all work in the PRD is complete.

The following files are loaded into your context: docs/PRD.md is the PRD that drives the work, progress.txt is the structured log of completed work, AGENTS.md is the agent-facing config, and CONTEXT.md is the domain glossary.
'

while true; do
  result=$(opencode run "$prompt" \
    --dangerously-skip-permissions \
    --model opencode-go/minimax-m3 \
    --variant thinking \
    --file docs/PRD.md \
    --file progress.txt \
    --file AGENTS.md \
    --file CONTEXT.md)

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    exit 0
  fi
done
