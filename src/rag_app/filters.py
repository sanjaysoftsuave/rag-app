"""Backend-neutral metadata filter.

Both the numpy store and Qdrant accept the same `MetaFilter`, which is the
whole point: filtering is a retrieval concept, not a vendor feature. Each
backend translates it into whatever it natively speaks.

Deliberately small — equality and match-any only. Real systems add ranges and
negation; the translation shape does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetaFilter:
    """field -> allowed value, or list of allowed values (match-any)."""

    must: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.must)

    def matches(self, metadata: dict[str, Any]) -> bool:
        for key, expected in self.must.items():
            actual = metadata.get(key)
            allowed = expected if isinstance(expected, (list, tuple, set)) else [expected]
            # A list-valued field (tags) matches if any element is allowed.
            if isinstance(actual, (list, tuple)):
                if not any(a in allowed for a in actual):
                    return False
            elif actual not in allowed:
                return False
        return True

    @classmethod
    def parse(cls, pairs: list[str] | None) -> MetaFilter:
        """Build from CLI `key=value` strings. Repeat a key to allow several values."""
        if not pairs:
            return cls()
        must: dict[str, Any] = {}
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(f"Filter {pair!r} must look like field=value")
            key, _, value = pair.partition("=")
            key, value = key.strip(), value.strip()
            if not key or not value:
                raise ValueError(f"Filter {pair!r} must look like field=value")
            if key in must:
                existing = must[key]
                must[key] = (existing if isinstance(existing, list) else [existing]) + [value]
            else:
                must[key] = value
        return cls(must=must)

    def describe(self) -> str:
        if not self.must:
            return "(none)"
        parts = []
        for key, value in sorted(self.must.items()):
            rendered = "|".join(map(str, value)) if isinstance(value, list) else str(value)
            parts.append(f"{key}={rendered}")
        return " AND ".join(parts)
