"""Anonymization utilities for exported conversation JSONL records."""

import re
import secrets
import string

DEFAULT_EMAIL_LOCAL_PART_LENGTH = 16

# Intentionally pragmatic (not RFC-complete): good enough for user-provided and model-generated text.
EMAIL_REGEX = re.compile(
    r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b"
)

_ANON_LOCAL_PART_ALPHABET = string.ascii_lowercase + string.digits


def is_anonymized_email(email: str, *, local_part_length: int = DEFAULT_EMAIL_LOCAL_PART_LENGTH) -> bool:
    """Return True if email looks like an anonymized replacement.

    Args:
        email: Email string to validate.
        local_part_length: Expected length of the local-part.
    """
    if "@" not in email:
        return False

    local_part, _domain = email.split("@", 1)
    if len(local_part) != local_part_length:
        return False

    return bool(re.fullmatch(rf"[a-z0-9]{{{local_part_length}}}", local_part))


class EmailAnonymizer:
    """Replace emails with randomized, fixed-length equivalents within a session."""

    def __init__(self, *, local_part_length: int = DEFAULT_EMAIL_LOCAL_PART_LENGTH) -> None:
        if local_part_length <= 0:
            raise ValueError("local_part_length must be > 0")
        self._local_part_length = local_part_length
        self._mapping: dict[str, str] = {}
        self._used: set[str] = set()

    def scrub(self, value: object) -> object:
        """Recursively replace emails in arbitrary JSON-like payloads.

        Args:
            value: JSON-like structure (dict/list/str/etc).

        Returns:
            A structure with emails replaced in all strings.
        """
        match value:
            case str():
                return self._scrub_text(value)
            case list():
                return [self.scrub(item) for item in value]
            case dict():
                return {key: self.scrub(item) for key, item in value.items()}
            case _:
                return value

    def _scrub_text(self, text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            original = match.group(0)
            normalized = original.lower()
            existing = self._mapping.get(normalized)
            if existing is not None:
                return existing

            _local, domain = normalized.split("@", 1)
            anonymized_local = self._random_local_part()
            anonymized = f"{anonymized_local}@{domain}"
            self._mapping[normalized] = anonymized
            return anonymized

        return EMAIL_REGEX.sub(_replace, text)

    def _random_local_part(self) -> str:
        while True:
            candidate = "".join(
                secrets.choice(_ANON_LOCAL_PART_ALPHABET)
                for _ in range(self._local_part_length)
            )
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate

