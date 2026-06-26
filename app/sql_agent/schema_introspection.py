from dataclasses import dataclass

from app.shared.exceptions import ValidationError

ALLOWED_TYPES = {"string", "number", "boolean", "date"}


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    type: str


def normalize_columns(raw: list[dict]) -> list[ColumnSchema]:
    """Validates/normalizes the caller-supplied column schema (backend's
    DataTable.schemaJson, already trusted as this table's real columns).
    This service never introspects a database directly — it only ever
    reasons about the schema description it's given, so it needs no DB
    credentials at all."""
    if not raw:
        raise ValidationError("columns must not be empty")

    seen: set[str] = set()
    columns: list[ColumnSchema] = []
    for entry in raw:
        name = str(entry.get("name", "")).strip()
        col_type = str(entry.get("type", "")).strip().lower()

        if not name:
            raise ValidationError("column name must not be empty")
        if col_type not in ALLOWED_TYPES:
            raise ValidationError(f'column "{name}" has unsupported type "{col_type}"')
        if name in seen:
            continue

        seen.add(name)
        columns.append(ColumnSchema(name=name, type=col_type))

    return columns
