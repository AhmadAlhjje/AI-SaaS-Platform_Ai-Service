CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough English heuristic (~4 chars/token). Only used as a fallback when
    a provider response doesn't include real usage counts."""
    return max(1, len(text) // CHARS_PER_TOKEN)
