"""Update selected local configuration fields without exposing existing secrets."""

import os
import tempfile
from pathlib import Path


def update_env(values: dict[str, str], path: Path | None = None) -> None:
    path = path or Path(os.getenv("ASSISTANT_ENV_FILE", ".env"))
    for key, value in values.items():
        if not key.replace("_", "").isalnum() or any(c in value for c in "\r\n"):
            raise ValueError("Invalid environment variable update")
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if line.partition("=")[0] not in values]
    lines.extend(f"{key}={value}" for key, value in values.items())
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        output.write("\n".join(lines) + "\n")
        temporary = Path(output.name)
    temporary.chmod(0o600)
    temporary.replace(path)
