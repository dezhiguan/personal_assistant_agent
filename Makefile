UV ?= uv

.PHONY: test golden-offline golden agent-check
test:
	$(UV) run pytest -q
	node --test tests/test_stream.mjs
	$(UV) run ruff check src tests scripts

golden-offline:
	$(UV) run pytest -q tests/test_golden_offline.py

# Uses the configured live model. Paid calls; current source, isolated fixtures.
golden:
	$(UV) run python scripts/run_golden.py

agent-check: test golden
