"""One versioned scenario source synchronized to a LangSmith dataset."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from personal_assistant_agent.evaluation.golden import load_cases

DEFAULT_CASES = Path(__file__).resolve().parents[3] / "evaluations/golden/cases.json"
DEFAULT_DATASET = "personal-assistant-golden-v2"


def examples(path=DEFAULT_CASES):
    cases, version = load_cases(path)
    return [
        {
            "inputs": {"case": case.model_dump()},
            "outputs": {
                "rubric": case.rubric,
                "checks": [t.checks.model_dump() for t in case.turns],
            },
            "metadata": {
                "case_id": case.id,
                "dimension": case.dimension,
                "critical": case.critical,
                "synthetic": True,
                "source_hash": version,
            },
        }
        for case in cases
    ], version


def sync_dataset(client, path=DEFAULT_CASES, name=DEFAULT_DATASET):
    records, version = examples(path)
    dataset = next(client.list_datasets(dataset_name=name, limit=1), None)
    if dataset is None:
        dataset = client.create_dataset(
            name,
            description="Versioned synthetic Agent scenarios; no user conversations",
            metadata={"synthetic": True},
        )
    existing = {str(e.id): e for e in client.list_examples(dataset_id=dataset.id)}
    desired = {
        uuid5(NAMESPACE_URL, "personal-assistant:" + name + ":" + r["metadata"]["case_id"]): r
        for r in records
    }
    if set(existing) - {str(identifier) for identifier in desired}:
        raise ValueError(
            "Dataset contains extra examples; review them instead of silently deleting"
        )
    create = []
    for identifier, record in desired.items():
        previous = existing.get(str(identifier))
        if previous is None:
            create.append({"id": identifier, **record})
            continue
        metadata = previous.metadata or {}
        if (
            previous.inputs != record["inputs"]
            or previous.outputs != record["outputs"]
            or any(metadata.get(k) != v for k, v in record["metadata"].items())
        ):
            client.update_example(
                identifier, **{**record, "metadata": {**metadata, **record["metadata"]}}
            )
    if create:
        client.create_examples(dataset_id=dataset.id, examples=create)
    return dataset, version, len(records)
