"""Synchronize repository-owned synthetic cases without duplicating examples."""

import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langsmith import Client

DEFAULT_DATASET = "personal-assistant-regression-v1"
DEFAULT_CASES = Path("evaluations/assistant-v1.json")


def sync_dataset(client: Client, path: Path = DEFAULT_CASES, name: str = DEFAULT_DATASET):
    raw = path.read_bytes()
    cases = json.loads(raw)
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Evaluation cases must have unique identifiers")
    version = hashlib.sha256(raw).hexdigest()
    dataset = next(client.list_datasets(dataset_name=name, limit=1), None)
    if dataset is None:
        dataset = client.create_dataset(
            name,
            description="个人智能助理回归评测：合成样例，不包含用户聊天记录。",
            metadata={"source": "repository", "synthetic": True},
        )
    existing = {str(e.id): e for e in client.list_examples(dataset_id=dataset.id)}
    create = []
    for case in cases:
        identifier = uuid5(NAMESPACE_URL, "personal-assistant:" + name + ":" + case["id"])
        example = {
            "id": identifier,
            "inputs": case["inputs"],
            "outputs": case["reference"],
            "metadata": {
                "case_id": case["id"],
                "category": case["category"],
                "source_hash": version,
            },
        }
        previous = existing.get(str(identifier))
        if previous is None:
            create.append(example)
        elif previous.inputs != example["inputs"] or previous.outputs != example["outputs"]:
            client.update_example(
                identifier,
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
            )
    if create:
        client.create_examples(dataset_id=dataset.id, examples=create)
    return dataset, version, len(cases)
