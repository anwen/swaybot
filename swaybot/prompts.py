from pathlib import Path

try:
    from jinja2 import Template
except ImportError:  # pragma: no cover
    Template = None  # type: ignore[misc, assignment]


DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, prompts_dir: Path | str | None = None) -> str:
    """Load a prompt template from disk."""
    directory = Path(prompts_dir) if prompts_dir else DEFAULT_PROMPTS_DIR
    path = directory / f"{name}.j2"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render(template_str: str, **kwargs) -> str:
    """Render a template string using Jinja2 if available."""
    if Template is None:
        raise ImportError(
            "Prompt templates require the 'jinja2' package. "
            "Install it with: pip install 'swaybot[llm]'"
        )
    return Template(template_str).render(**kwargs)


def render_prompt(name: str, prompts_dir: Path | str | None = None, **kwargs) -> str:
    """Load and render a prompt template."""
    template_str = load_prompt(name, prompts_dir=prompts_dir)
    return _render(template_str, **kwargs)
