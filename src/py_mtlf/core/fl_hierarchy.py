from __future__ import annotations

from uuid import UUID


def normalize_plan_id(value: str) -> str:
    normalized = _normalize_uuid(value, "plan_id")
    if UUID(normalized).version != 4:
        raise ValueError("plan_id must be a UUIDv4")
    return normalized


def normalize_nf_instance_id(value: str) -> str:
    return _normalize_uuid(value, "NF instance ID")


def _normalize_uuid(value: str, name: str) -> str:
    try:
        return str(UUID(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a UUID") from error
