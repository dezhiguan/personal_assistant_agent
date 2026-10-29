"""Disk-backed Deep Agents tools with a deliberately small command interpreter.

No command is passed to an OS shell, subprocess, Python evaluator or executable.
This is a single-user application boundary, not an OS sandbox for untrusted code.
"""

import shlex
import stat
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import DeleteResult, ExecuteResponse, SandboxBackendProtocol

EXECUTE_DESCRIPTION = """Execute one allowed command in the workspace virtual root (/).
Supported: pwd; ls [path]; cat file...; head/tail [-n 1..1000] file;
wc [-l|-w|-c] file; mkdir [-p] path; touch file; echo text...
Paths are relative to /, which maps to the configured workspace directory.
No shell expansion, pipes, redirection, chaining, environment variables, scripts,
interpreters, network commands or other executables. Use write_file/edit_file
to write content, and glob/grep for searching. Errors have a nonzero exit code.
"""


class WorkspaceBackend(FilesystemBackend, SandboxBackendProtocol):
    """Use the same validated virtual paths for files and command arguments."""

    def __init__(self, directory: str, *, project_root: Path | None = None):
        project = (project_root or Path.cwd()).resolve()
        relative = Path(directory)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(p == ".." or p.startswith(".") for p in relative.parts)
        ):
            raise ValueError("ASSISTANT_WORKSPACE_DIR 必须是项目内非隐藏的相对子目录。")
        root = project / relative
        if not root.resolve().is_relative_to(project) or root.resolve() == project:
            raise ValueError("工作目录不能越出项目或使用项目根目录。")
        for parent in [root, *root.parents]:
            if parent == project:
                break
            if parent.is_symlink():
                raise ValueError("工作目录不能经过符号链接。")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        super().__init__(root_dir=root, virtual_mode=True, max_file_size_mb=1)

    @property
    def id(self) -> str:
        return "local-restricted-workspace"

    def _resolve_path(self, key: str) -> Path:
        try:
            resolved = super()._resolve_path(key)
        except ValueError:
            raise PermissionError("路径必须位于 workspace 虚拟根目录内。") from None
        # The agent cannot create symlinks; also reject preexisting links, special
        # files and hardlinks supplied by a local user. No untrusted local writers
        # may mutate this directory concurrently (see README).
        path = self.cwd
        for part in Path(key.lstrip("/")).parts:
            path /= part
            if path.is_symlink():
                raise PermissionError("工作区不允许符号链接。")
        if resolved.exists():
            info = resolved.stat()
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise PermissionError("工作区仅支持普通文件和目录。")
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                raise PermissionError("工作区不允许硬链接文件。")
        return resolved

    def delete(self, file_path: str) -> DeleteResult:
        try:
            if self._resolve_path(file_path) == self.cwd:
                return DeleteResult(error="不能删除 workspace 根目录。")
        except OSError:
            return DeleteResult(error="无法删除：路径不在允许的工作区内。")
        return super().delete(file_path)

    def _to_virtual_path(self, path: Path) -> str:
        # ls/glob/grep enumerate files without going through _resolve_path.
        # Apply the same link/special-file checks before exposing their results.
        self._resolve_path(path.relative_to(self.cwd).as_posix())
        return super()._to_virtual_path(path)

    def _text(self, key: str) -> str:
        path = self._resolve_path(key)
        if not path.is_file():
            raise ValueError("需要已存在的普通文件。")
        with path.open("rb") as handle:
            raw = handle.read(self.max_file_size_bytes + 1)
        if len(raw) > self.max_file_size_bytes:
            raise ValueError("命令最多读取 1 MiB 文件，请使用 read_file 分页。")
        return raw.decode("utf-8")

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            if timeout is not None and not 1 <= timeout <= 30:
                raise ValueError("timeout 必须为 1–30 秒；命令不支持后台任务。")
            if not command.strip() or len(command) > 4096:
                raise ValueError("命令必须为 1–4096 字符。")
            if any(c in command for c in "\n\r\x00;&|<>`$"):
                raise ValueError("不支持 Shell 连接、重定向、变量或替换。")
            args = shlex.split(command)
            if not args or len(args) > 64:
                raise ValueError("命令参数过多或为空。")
            output = self._command(args)
            return ExecuteResponse(
                output=output[:16000], exit_code=0, truncated=len(output) > 16000
            )
        except (OSError, ValueError) as error:
            # OSError can embed host paths. Only expose path-free errors.
            detail = (
                (error.strerror or type(error).__name__)
                if isinstance(error, OSError)
                else str(error)
            )
            return ExecuteResponse(output=f"执行失败：{detail}", exit_code=1)

    def _command(self, args: list[str]) -> str:
        name, *values = args
        if name == "pwd" and not values:
            return "/\n"
        if name == "echo":
            return " ".join(values) + "\n"
        if name == "ls" and len(values) <= 1:
            result = self.ls(values[0] if values else "/")
            if result.error:
                raise ValueError(result.error)
            return "\n".join(item["path"] for item in result.entries or []) + "\n"
        if name == "cat" and values:
            return "".join(self._text(value) for value in values)
        if name in {"head", "tail"}:
            count = 10
            if len(values) == 3 and values[0] == "-n":
                count = int(values[1])
                values = values[2:]
            if len(values) != 1 or not 1 <= count <= 1000:
                raise ValueError("用法：head/tail [-n 1..1000] file")
            lines = self._text(values[0]).splitlines(keepends=True)
            return "".join(lines[:count] if name == "head" else lines[-count:])
        if name == "wc":
            flag = values.pop(0) if values and values[0] in {"-l", "-w", "-c"} else ""
            if len(values) != 1:
                raise ValueError("用法：wc [-l|-w|-c] file")
            text = self._text(values[0])
            counts = {"-l": text.count("\n"), "-w": len(text.split()), "-c": len(text.encode())}
            return str(counts[flag]) + "\n" if flag else " ".join(map(str, counts.values())) + "\n"
        if name in {"mkdir", "touch"}:
            parents = name == "mkdir" and values[:1] == ["-p"]
            if parents:
                values = values[1:]
            if len(values) != 1 or values[0].startswith("-"):
                raise ValueError("用法：mkdir [-p] path 或 touch file")
            path = self._resolve_path(values[0])
            if name == "mkdir":
                path.mkdir(mode=0o700, parents=parents, exist_ok=parents)
            else:
                path.touch(mode=0o600)
            return ""
        raise ValueError("命令不在允许列表内或参数无效。请参考 execute 工具说明。")
