"""Add tiered USD cost estimates to native LangChain LLM spans."""

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.observability.costs import estimate_cost


class CostTracer(LangChainTracer):
    def __init__(self, *, settings: Settings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings or Settings()

    def copy_with_metadata_defaults(self, **kwargs):
        copied = super().copy_with_metadata_defaults(**kwargs)
        copied.settings = self.settings
        return copied

    def _update_run_single(self, run: Run) -> None:
        # Native _on_llm_end has already extracted and aggregated provider usage.
        metadata = run.extra.get("metadata", {})
        usage = metadata.get("usage_metadata")
        if run.run_type == "llm" and usage:
            estimate = estimate_cost(usage, self.settings)
            if estimate:
                run.set(
                    usage_metadata={
                        **usage,
                        **{k: estimate[k] for k in ("input_cost", "output_cost", "total_cost")},
                    },
                    metadata={"cost_estimate": estimate},
                )
        super()._update_run_single(run)
