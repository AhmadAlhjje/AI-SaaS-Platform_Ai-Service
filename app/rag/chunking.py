# Mirrors backend's generate-chunks.job.ts (CHUNK_SIZE/CHUNK_OVERLAP) so a
# document chunked by either path produces comparable chunk sizes.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    trimmed = text.strip()
    if not trimmed:
        return []

    chunks: list[str] = []
    start = 0
    length = len(trimmed)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(trimmed[start:end])
        if end == length:
            break
        start = end - overlap

    return chunks
