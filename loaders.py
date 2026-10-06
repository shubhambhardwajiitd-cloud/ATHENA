"""Safe YAML loaders for ATHENA objectives and attack cards."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError
import yaml

from core import AttackCard, Objective


class LoadError(Exception):
    """Raised when an ATHENA data file cannot be loaded or validated."""


ModelT = TypeVar("ModelT", bound=BaseModel)


class _DuplicateKeyError(yaml.YAMLError):
    """Raised internally when a YAML mapping repeats a key."""

    def __init__(self, key: object) -> None:
        self.key = key
        super().__init__()


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen:
                    raise _DuplicateKeyError(key)
                seen.add(key)
            except TypeError:
                return super().construct_mapping(node, deep=deep)
        return super().construct_mapping(node, deep=deep)


def _location(file_path: Path, record_index: int) -> str:
    return f"{file_path.name}[{record_index}]"


def _validation_fields(exc: ValidationError) -> str:
    fields: list[str] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        fields.append(location or "record")
    return ", ".join(dict.fromkeys(fields))


def _read_records(file_path: Path) -> list[Mapping]:
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LoadError(f"{file_path.name}: could not read file") from exc

    try:
        data = yaml.load(raw_text, Loader=_UniqueKeySafeLoader)
    except _DuplicateKeyError as exc:
        raise LoadError(f"{file_path.name}: duplicate key {exc.key!r}") from exc
    except yaml.YAMLError as exc:
        raise LoadError(f"{file_path.name}: invalid YAML") from exc

    if data is None:
        return []
    if isinstance(data, Mapping):
        return [data]
    if isinstance(data, list):
        for index, record in enumerate(data):
            if not isinstance(record, Mapping):
                raise LoadError(
                    f"{_location(file_path, index)}: record must be a mapping"
                )
        return data
    raise LoadError(f"{file_path.name}: top level must be a mapping or list")


def _validate_objective(
    objective: Objective, file_path: Path, record_index: int
) -> None:
    location = _location(file_path, record_index)
    if not objective.success_checks:
        raise LoadError(f"{location}: success_checks must be non-empty")
    if objective.max_turns < 1:
        raise LoadError(f"{location}: max_turns must be at least 1")
    if objective.max_turns_per_card < 1:
        raise LoadError(f"{location}: max_turns_per_card must be at least 1")
    if any(check.level not in (1, 2) for check in objective.success_checks):
        raise LoadError(f"{location}: success check level must be 1 or 2")


def _validate_card(card: AttackCard, file_path: Path, record_index: int) -> None:
    if not card.objective_types:
        raise LoadError(
            f"{_location(file_path, record_index)}: objective_types must be non-empty"
        )


def _load_directory(
    path: str | Path,
    model: type[ModelT],
    extra_validation: Callable[[ModelT, Path, int], None],
) -> list[ModelT]:
    directory = Path(path)
    if not directory.is_dir():
        raise LoadError(f"{directory.name or directory}: directory not found")

    try:
        files = sorted(
            (
                entry
                for entry in directory.iterdir()
                if entry.is_file() and entry.suffix in {".yaml", ".yml"}
            ),
            key=lambda entry: entry.name,
        )
    except OSError as exc:
        raise LoadError(f"{directory.name}: could not list directory") from exc

    loaded: list[ModelT] = []
    seen_ids: dict[str, str] = {}
    allowed_keys = set(model.model_fields)

    for file_path in files:
        for index, record in enumerate(_read_records(file_path)):
            location = _location(file_path, index)
            extra_keys = [key for key in record if key not in allowed_keys]
            if extra_keys:
                key = sorted(extra_keys, key=str)[0]
                raise LoadError(f"{location}: unknown key {key!r}")

            try:
                item = model.model_validate(record)
            except ValidationError as exc:
                fields = _validation_fields(exc)
                raise LoadError(f"{location}: validation failed for {fields}") from exc

            extra_validation(item, file_path, index)
            if item.id in seen_ids:
                first_location = seen_ids[item.id]
                raise LoadError(
                    f"{location}: duplicate id {item.id!r}; "
                    f"first defined at {first_location}"
                )
            seen_ids[item.id] = location
            loaded.append(item)

    return loaded


def load_objectives(path: str | Path) -> list[Objective]:
    """Load and validate objectives from a non-recursive YAML directory."""
    return _load_directory(path, Objective, _validate_objective)


def load_cards(path: str | Path) -> list[AttackCard]:
    """Load and validate attack cards from a non-recursive YAML directory."""
    return _load_directory(path, AttackCard, _validate_card)
