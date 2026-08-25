"""Skill library loader.

The flow skill library directory is set in the config file
(``skills.flow_dir``, default ``skills/skills_openroad_flow``). Each stage
topic is a directory that can hold:

  SKILL.md      — stage knowledge shared by every agent role
  DEBUGGER.md   — the Debugger's stage-specific checklist
  OPTIMIZER.md  — the Optimizer's stage-specific knob map
  template/*.tcl — executable templates (e.g. image_snapshot)

plus generic role skills at ``debugger/SKILL.md`` / ``optimizer/SKILL.md``.
Adding a new stage = adding a directory; nothing else to register.
"""

from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT

DEFAULT_ROOT = PROJECT_ROOT / "skills" / "skills_openroad_flow"


def _root(base_dir=None) -> Path:
    if base_dir is None:
        return DEFAULT_ROOT
    p = Path(base_dir)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def load_skill(name: str, base_dir=None) -> str:
    """Load ``<topic>`` (its SKILL.md) or ``<topic>/<FILE>.md``."""
    root = _root(base_dir)
    for c in (root / name, root / name / "SKILL.md", root / f"{name}.md"):
        if c.is_file():
            return _read(c)
    return ""


def load_role_skills(role: str, topics: list[str], base_dir=None,
                     max_chars: int = 12000) -> str:
    """Skills for one agent role on one stage: the generic role skill, then
    each topic's shared SKILL.md and its role-specific <ROLE>.md."""
    root = _root(base_dir)
    parts: list[str] = []
    generic = root / role / "SKILL.md"
    if generic.is_file():
        parts.append(f"### skill: {role}\n{_read(generic).strip()}")
    for t in topics:
        shared = root / t / "SKILL.md"
        if shared.is_file():
            parts.append(f"### skill: {t}\n{_read(shared).strip()}")
        rolef = root / t / f"{role.upper()}.md"
        if rolef.is_file():
            parts.append(f"### skill: {t}/{role}\n{_read(rolef).strip()}")
    return "\n\n".join(parts)[:max_chars]


def template_path(topic: str, filename: str, base_dir=None) -> Path | None:
    """Path of an executable template inside a skill, if present."""
    p = _root(base_dir) / topic / "template" / filename
    return p if p.is_file() else None
