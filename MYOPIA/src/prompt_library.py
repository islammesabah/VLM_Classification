"""Single access point for every prompt the harness sends.

No prompt wording lives in the code: each template is a named entry in
``prompts/templates.json`` and each per-dataset asset (zero-shot paraphrases,
class descriptions, CHiLS superclass captions, the image-description prompt)
is a file under ``prompts/``. Call sites name a template and supply its
placeholders, so a prompt can be read or changed without touching Python.

The prompts directory is located relative to this file, so scripts work from
any working directory. Set MYOPIA_PROMPTS_DIR to point somewhere else.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

ENV_VAR = "MYOPIA_PROMPTS_DIR"


@lru_cache(maxsize=1)
def prompts_dir() -> Path:
    """Locate the prompts directory: env override, then alongside or above src/."""
    override = os.getenv(ENV_VAR)
    if override:
        path = Path(override)
        if not path.is_dir():
            raise FileNotFoundError(f"{ENV_VAR} does not point at a directory: {path}")
        return path

    here = Path(__file__).resolve().parent
    for candidate in (here / "prompts", here.parent / "prompts", Path("prompts")):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find a prompts/ directory near {here}; set {ENV_VAR} to its path."
    )


@lru_cache(maxsize=1)
def _templates() -> dict:
    return json.loads((prompts_dir() / "templates.json").read_text())


def template(name: str) -> str:
    """Return a raw template string by name."""
    templates = _templates()
    if name not in templates:
        available = ", ".join(sorted(k for k in templates if not k.startswith("_")))
        raise KeyError(f"No prompt template named {name!r}. Available: {available}")
    return templates[name]


def render(name: str, **fields) -> str:
    """Return a template with its {placeholders} filled in."""
    try:
        return template(name).format(**fields)
    except KeyError as missing:
        raise KeyError(f"Prompt template {name!r} needs field {missing}") from None


def read_text(relative_path: str) -> str:
    """Read a plain-text prompt file, e.g. 'describe_image.txt'."""
    return (prompts_dir() / relative_path).read_text()


def load_json(relative_path: str):
    """Read a JSON prompt asset, e.g. 'GTSRB/zs_prompts.json'."""
    return json.loads((prompts_dir() / relative_path).read_text())


def zero_shot_prompts(dataset_name: str) -> list:
    """The ten zero-shot paraphrases for a dataset, in RunId order."""
    return load_json(f"{dataset_name}/zs_prompts.json")["prompts"]


def class_descriptions(dataset_name: str) -> dict:
    """Per-class descriptions used by the with-descriptions conditions."""
    return load_json(f"{dataset_name}/classes_description.json")


def clip_superclass_prompts(dataset_name: str) -> dict:
    """CHiLS root-branch captions for a dataset; empty if it has none."""
    data = load_json("clip_superclass_prompts.json")
    return data.get(dataset_name, {})
