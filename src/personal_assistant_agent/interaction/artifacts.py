"""Task-local immutable result references; revisions import copies, never shared writers."""

import hashlib

from personal_assistant_agent.agents.workspace import WorkspaceBackend

INTERNAL = {"conversation_history", "large_tool_results"}


def index_artifacts(settings):
    backend = WorkspaceBackend(settings.workspace_dir)
    files = []
    for candidate in sorted(backend.cwd.rglob("*")):
        relative = candidate.relative_to(backend.cwd).as_posix()
        if relative.split("/", 1)[0] in INTERNAL or candidate.name.startswith(".artifact-import-"):
            continue
        checked = backend._resolve_path("/" + relative)
        if not checked.is_file():
            continue
        if checked.stat().st_size > backend.max_file_size_bytes:
            raise ValueError("成果文件超过工作区大小上限。")
        content = checked.read_bytes()
        files.append(
            {
                "path": "/" + relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
        if len(files) > 64:
            raise ValueError("成果文件超过64个，请整理后再交付。")
    return files


def import_artifacts(source, source_settings, target_settings):
    import os
    import tempfile

    old = WorkspaceBackend(source_settings.workspace_dir)
    new = WorkspaceBackend(target_settings.workspace_dir)
    imported = []
    for artifact in source.get("artifacts", []):
        original = old._resolve_path(artifact["path"])
        if original.stat().st_size > old.max_file_size_bytes:
            raise ValueError("原成果文件超过工作区大小上限。")
        raw = original.read_bytes()
        if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValueError("原成果已被外部修改，需要核对版本。")
        path = f"/sources/{source['id']}{artifact['path']}"
        target = new._resolve_path(path)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("修订任务的导入副本已改变，需要核对。")
        else:
            # A crash before rename leaves only an unreferenced temp file. A replay
            # can safely finish the same import without changing the original.
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".artifact-import-", delete=False
            ) as handle:
                temporary = handle.name
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        imported.append(path)
    return imported
