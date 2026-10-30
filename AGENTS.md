# Agent regression requirements

- After changes to agent behavior, prompts, tools, model configuration or chat state,
  run `make agent-check` (or its equivalent component commands) before claiming
  validation is complete or pushing changes.
  The live golden run invokes the current source; a running server need not be restarted.
- Keep the golden cases and baseline under `evaluations/golden/` versioned.
  Never silently change expectations, lower thresholds, remove failing cases, or
  overwrite the baseline to make a candidate pass. Describe expected behavior changes
  and review individual failures before explicitly recording a new baseline.
- Report critical failures separately from aggregate scores. A preexisting security
  failure remains blocking even if it also exists in the baseline.
- Golden fixtures must be synthetic. Use isolated workspaces and fixture search/time
  tools. Never put actual credentials or user conversations into cases or reports.
- Keep `.env`, runtime workspaces and `artifacts/` out of commits. A missing model key,
  incomplete run or timeout is an unsuccessful validation, not a passing result.
