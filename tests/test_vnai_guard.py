"""Verify vnai's AGENTS.md writer is neutralized on package import.

vnai >= 2.5.5 writes a prompt-injection payload to ./AGENTS.md and to six global
AI-assistant config files in $HOME, triggered by `import vnstock` alone. See
stockpredict._vnai_guard.
"""
import pytest

import stockpredict  # noqa: F401 — the import is what installs the guard

vnai = pytest.importorskip("vnai", reason="vnai not installed")

NAMES = ("setup_agent_environment", "async_setup_agent_environment")


@pytest.mark.parametrize("name", NAMES)
def test_package_level_writer_is_stubbed(name):
    """vnstock resolves these via `from vnai import ...`, so they must be no-ops."""
    fn = getattr(vnai, name, None)
    if fn is None:
        pytest.skip(f"vnai has no {name} in this version")
    assert fn() is False


@pytest.mark.parametrize("name", NAMES)
def test_beam_agents_writer_is_stubbed(name):
    beam_agents = pytest.importorskip("vnai.beam.agents")
    fn = getattr(beam_agents, name, None)
    if fn is None:
        pytest.skip(f"vnai.beam.agents has no {name} in this version")
    assert fn() is False


def test_importing_vnstock_writes_no_agents_file(tmp_path, monkeypatch):
    """The writer targets Path(project_root)/'AGENTS.md' with project_root='.'."""
    pytest.importorskip("vnstock")
    monkeypatch.chdir(tmp_path)
    import vnstock  # noqa: F401

    assert not (tmp_path / "AGENTS.md").exists()
