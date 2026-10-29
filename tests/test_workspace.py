"""Actual disk operations and command boundary without host shell access."""

import pytest

from personal_assistant_agent.agents.workspace import WorkspaceBackend


@pytest.fixture
def backend(tmp_path):
    return WorkspaceBackend("workspace", project_root=tmp_path)


def test_file_lifecycle_and_commands_share_disk(backend, tmp_path):
    assert backend.execute("mkdir -p /notes").exit_code == 0
    assert backend.write("/notes/a.txt", "one\ntwo\n").error is None
    assert backend.execute("cat /notes/a.txt").output == "one\ntwo\n"
    assert backend.execute("wc -l notes/a.txt").output == "2\n"
    assert backend.execute("head -n 1 notes/a.txt").output == "one\n"
    assert backend.execute("tail -n 1 notes/a.txt").output == "two\n"
    assert backend.execute("pwd").output == "/\n"
    assert "/notes/a.txt" in backend.execute("ls /notes").output
    assert backend.execute("touch /empty.txt").exit_code == 0
    assert (tmp_path / "workspace/empty.txt").exists()
    assert backend.edit("/notes/a.txt", "one", "three").error is None
    assert "three" in backend.read("/notes/a.txt").file_data["content"]
    assert backend.delete("/notes/a.txt").error is None
    assert not (tmp_path / "workspace/notes/a.txt").exists()
    assert backend.delete("/").error


@pytest.mark.parametrize(
    "command",
    [
        "cat ../.env",
        "cat /../../etc/passwd",
        "cat /etc/passwd",
        "cat ~/.ssh/id_rsa",
        "ls; pwd",
        "echo $(id)",
        "echo `id`",
        "echo $DASHSCOPE_API_KEY",
        "cat a | wc",
        "echo x > a",
        "pwd\nls",
        "python -c 'print(1)'",
        "sh -c pwd",
        "curl https://baidu.com",
        "env",
        "rm -rf /",
        "ln -s /etc/passwd secret",
        "ls --help",
        "head -n 999999 a",
    ],
)
def test_execute_refuses_escapes_and_unapproved_commands(backend, command):
    result = backend.execute(command)
    assert result.exit_code != 0
    assert str(backend.cwd) not in result.output


def test_file_tools_reject_escape_and_external_symlinks(backend, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("DO_NOT_READ")
    (backend.cwd / "escape").symlink_to(tmp_path, target_is_directory=True)
    for path in ["../.env", "/../.env", "/escape/.env"]:
        assert backend.read(path).error
        assert backend.write(path, "bad").error
        assert backend.edit(path, "DO_NOT_READ", "bad").error
        assert backend.delete(path).error
        assert backend.execute(f"cat {path}").exit_code != 0
    assert "DO_NOT_READ" not in str(backend.grep("DO_NOT_READ"))
    assert "DO_NOT_READ" not in str(backend.glob("**/*"))
    assert secret.read_text() == "DO_NOT_READ"


@pytest.mark.parametrize("directory", [".", "", "..", "../other", "/tmp/workspace", ".hidden"])
def test_workspace_config_cannot_use_system_or_project_root(tmp_path, directory):
    with pytest.raises(ValueError):
        WorkspaceBackend(directory, project_root=tmp_path)


def test_workspace_root_cannot_follow_symlink(tmp_path):
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        WorkspaceBackend("alias/workspace", project_root=tmp_path)


def test_search_and_file_tools_do_not_expose_external_hardlinks(backend, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("hardlink-private-content")
    (backend.cwd / "link.txt").hardlink_to(secret)
    assert backend.read("/link.txt").error
    assert backend.execute("cat /link.txt").exit_code == 1
    assert not backend.glob("**/*").matches
    assert "hardlink-private-content" not in str(backend.grep("hardlink-private-content"))


def test_command_output_is_bounded(backend):
    backend.write("/large.txt", "x" * 17000)
    result = backend.execute("cat large.txt")
    assert result.truncated
    assert len(result.output) == 16000
    assert backend.execute("pwd", timeout=0).exit_code == 1
