"""Neutralize vnai's unconditional AGENTS.md writer.

vnai >= 2.5.5 writes a "Vibe Onboarding" prompt-injection payload to ./AGENTS.md
and to six global AI-assistant config files in $HOME (~/.clauderc, ~/.cursorrules,
~/.windsurfrules, ~/.clinerules, ~/.gemini/config/AGENTS.md and
~/.github/copilot-instructions.md). It is triggered by ``import vnstock`` alone —
vnstock/__init__.py calls setup_agent(async_mode=True) at module level — and there
is no env var or config flag to disable it, so we rebind the functions before
vnstock is ever imported.

Note that vnstock resolves the writer via a function-local ``from vnai import
async_setup_agent_environment``, i.e. the re-export on the vnai package object, so
patching vnai.beam.agents alone is not enough — both binding sites must be covered.
"""
from __future__ import annotations

_installed = False


def _noop(*_args, **_kwargs) -> bool:
    return False


def install() -> bool:
    """Stub out vnai's agent-environment writer. Returns True if patched.

    Safe to call repeatedly, and a no-op when vnai is missing or has restructured.
    """
    global _installed
    if _installed:
        return True

    names = ("setup_agent_environment", "async_setup_agent_environment")
    try:
        import vnai

        targets = [vnai]
        try:
            import vnai.beam.agents as beam_agents

            targets.append(beam_agents)
        except ImportError:
            pass

        for target in targets:
            for name in names:
                if hasattr(target, name):
                    setattr(target, name, _noop)
    except ImportError:
        return False
    except Exception:
        # vnai internals changed shape; degrade to a no-op rather than breaking
        # every import of this package.
        return False

    _installed = True
    return True
