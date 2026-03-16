"""Search engine — Vector search + keyword boost + result formatting."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .db import DejavuDB
from .embedder import OllamaEmbedder, prepare_query_text


# ---------------------------------------------------------------------------
# Temporal parsing
# ---------------------------------------------------------------------------

def _build_temporal_patterns() -> list[tuple[str, callable]]:
    """Build regex patterns for temporal hint parsing.

    Computes 'now' fresh each call so results stay correct in long-running
    processes (e.g. MCP server).
    """
    now = time.time()

    def _ago(days: int):
        return now - (days * 86400), now

    def _year(y: int):
        start = datetime(y, 1, 1).timestamp()
        end = datetime(y, 12, 31, 23, 59, 59).timestamp()
        return start, end

    def _season(name: str):
        """Approximate seasons for the most recent occurrence."""
        current_year = datetime.now().year
        seasons = {
            "spring": (3, 5),
            "summer": (6, 8),
            "fall": (9, 11),
            "autumn": (9, 11),
            "winter": (12, 2),
        }
        if name not in seasons:
            return None
        start_m, end_m = seasons[name]
        for y in (current_year, current_year - 1):
            if name == "winter":
                start = datetime(y - 1, 12, 1).timestamp()
                end = datetime(y, 2, 28, 23, 59, 59).timestamp()
            else:
                start = datetime(y, start_m, 1).timestamp()
                end_day = 30 if end_m in (4, 6, 9, 11) else 31
                end = datetime(y, end_m, end_day, 23, 59, 59).timestamp()
            if start < now:
                return start, end
        return None

    return [
        # Explicit years
        (r"\b(20\d{2})\b", lambda m: _year(int(m.group(1)))),
        # Relative
        (r"last\s+week", lambda m: _ago(7)),
        (r"last\s+month", lambda m: _ago(30)),
        (r"last\s+year", lambda m: _ago(365)),
        (r"a\s+few\s+months?\s+ago", lambda m: _ago(120)),
        (r"a\s+couple\s+months?\s+ago", lambda m: _ago(60)),
        (r"recently", lambda m: _ago(30)),
        (r"a\s+while\s+ago", lambda m: _ago(180)),
        # Seasons
        (r"last\s+(spring|summer|fall|autumn|winter)", lambda m: _season(m.group(1))),
        (r"this\s+(spring|summer|fall|autumn|winter)", lambda m: _season(m.group(1))),
    ]

# Static list of pattern strings for strip_temporal_hint (doesn't need fresh `now`)
_TEMPORAL_PATTERN_STRINGS = [p[0] for p in _build_temporal_patterns()]


def parse_temporal_hint(text: str) -> Optional[tuple[float, float]]:
    """
    Parse temporal hints from query text.
    Returns (min_mtime, max_mtime) or None.

    Rebuilds patterns each call so 'now' is always fresh.
    """
    text_lower = text.lower()
    for pattern, handler in _build_temporal_patterns():
        m = re.search(pattern, text_lower)
        if m:
            result = handler(m)
            if result:
                return result
    return None


def strip_temporal_hint(text: str) -> str:
    """Remove temporal hints from query text so they don't pollute the embedding."""
    cleaned = text
    for pattern in _TEMPORAL_PATTERN_STRINGS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    # Clean up leftover artifacts
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


# ---------------------------------------------------------------------------
# Language parsing
# ---------------------------------------------------------------------------

LANGUAGE_ALIASES = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "tsx": "tsx",
    "rust": "rust", "rs": "rust",
    "go": "go", "golang": "go",
    "ruby": "ruby", "rb": "ruby",
    "java": "java",
    "kotlin": "kotlin", "kt": "kotlin",
    "c": "c",
    "cpp": "cpp", "c++": "cpp",
    "php": "php",
    "bash": "bash", "shell": "bash", "sh": "bash",
    "swift": "swift",
    "html": "html",
    "css": "css",
    "sql": "sql",
}


def parse_language_hint(text: str) -> Optional[str]:
    """Extract language hint from query text.

    Uses pattern matching with context to avoid false positives on
    ambiguous words like 'go', 'c', 'r', 'rust' that appear in normal English.
    """
    text_lower = text.lower()

    # Short/ambiguous aliases that need strong contextual signals to match.
    # These will ONLY match via the explicit patterns below, never via bare word scan.
    _AMBIGUOUS_ALIASES = {"go", "c", "r", "v", "rs", "rb", "kt", "ts", "js", "py", "sh"}

    # "in python", "the python one", "a python script"
    for pattern in [
        r"(?:in|the|a|my)\s+(\w+)\s+(?:one|script|file|code|component|function|class|thing|project|repo)",
        r"(?:written\s+in|coded\s+in|built\s+with|using)\s+(\w+)$",
        r"(?:in|using|with)\s+(\w+)$",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            lang = m.group(1)
            if lang in LANGUAGE_ALIASES:
                return LANGUAGE_ALIASES[lang]

    # Direct mention — but SKIP ambiguous short aliases that are common English words
    for alias, lang in LANGUAGE_ALIASES.items():
        if alias in _AMBIGUOUS_ALIASES:
            continue
        # For aliases with special chars (e.g. "c++"), use substring match
        if "+" in alias or not alias.isalnum():
            if alias in text_lower:
                return lang
        elif alias in text_lower.split():
            return lang

    return None


def strip_language_hint(text: str) -> str:
    """Remove language hints from query text."""
    escaped_aliases = [re.escape(k) for k in LANGUAGE_ALIASES.keys()]
    aliases_pattern = "|".join(escaped_aliases)
    patterns = [
        r"(?:in|the|a|my)\s+(?:" + aliases_pattern + r")\s+(?:one|script|file|code|component|function|class|thing)",
        r"(?:in|using|with)\s+(?:" + aliases_pattern + r")\s*$",
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", cleaned).strip() or text


# ---------------------------------------------------------------------------
# Keyword boost
# ---------------------------------------------------------------------------

def compute_keyword_boost(query: str, chunk: dict, boost_weight: float = 0.15) -> float:
    """
    Compute a keyword match bonus for a chunk.

    Returns a value [0, boost_weight] based on how many query terms
    appear in the chunk's name, signature, or docstring.
    """
    query_terms = set(re.findall(r"\w{3,}", query.lower()))
    if not query_terms:
        return 0.0

    searchable = " ".join(filter(None, [
        (chunk.get("name") or "").lower(),
        (chunk.get("signature") or "").lower(),
        (chunk.get("docstring") or "").lower(),
    ]))

    matches = sum(1 for term in query_terms if term in searchable)
    ratio = matches / len(query_terms)
    return ratio * boost_weight


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result with metadata."""
    rank: int
    file_path: str
    name: Optional[str]
    chunk_type: str
    language: str
    start_line: int
    end_line: int
    source_preview: str
    similarity: float
    file_mtime: float
    # Score breakdown for --explain mode
    vector_score: float = 0.0
    keyword_boost_score: float = 0.0

    def format_markdown(self) -> str:
        """Format as markdown for MCP/CLI output."""
        mod_date = datetime.fromtimestamp(self.file_mtime).strftime("%Y-%m-%d")
        name_display = self.name or "(unnamed)"
        type_display = self.chunk_type.title()
        pct = f"{self.similarity * 100:.0f}%"

        lines = [
            f"### {self.rank}. {name_display} ({type_display}) — {pct} match",
            f"**File:** `{self.file_path}`",
            f"**Language:** {self.language} | **Modified:** {mod_date} | **Lines:** {self.start_line}-{self.end_line}",
            "",
            "```" + self.language,
            self.source_preview,
            "```",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "rank": self.rank,
            "file_path": self.file_path,
            "name": self.name,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "similarity": round(self.similarity, 4),
            "vector_score": round(self.vector_score, 4),
            "keyword_boost": round(self.keyword_boost_score, 4),
            "modified": datetime.fromtimestamp(self.file_mtime).strftime("%Y-%m-%d"),
            "source_preview": self.source_preview,
        }


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

async def search(
    db: DejavuDB,
    embedder: OllamaEmbedder,
    query: str,
    language: Optional[str] = None,
    when: Optional[str] = None,
    path_contains: Optional[str] = None,
    limit: int = 10,
    keyword_boost: float = 0.15,
) -> list[SearchResult]:
    """
    Full search pipeline:
    1. Parse hints from query
    2. Embed cleaned query
    3. Vector KNN search with filters
    4. Apply keyword boost
    5. Return ranked results
    """
    # Parse hints
    if not language:
        language = parse_language_hint(query)

    temporal = None
    if when:
        temporal = parse_temporal_hint(when)
    elif parse_temporal_hint(query):
        temporal = parse_temporal_hint(query)

    # Clean query for embedding (remove filter hints)
    clean_query = strip_temporal_hint(strip_language_hint(query))
    query_text = prepare_query_text(clean_query)

    # Embed query
    try:
        query_embedding = await embedder.embed_one(query_text)
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate query embedding — is Ollama running? Error: {e}"
        ) from e

    # Vector search
    min_mtime = temporal[0] if temporal else None
    max_mtime = temporal[1] if temporal else None

    raw_results = db.vector_search(
        query_embedding=query_embedding,
        limit=limit * 2,  # fetch extra for re-ranking
        language=language,
        min_mtime=min_mtime,
        max_mtime=max_mtime,
        path_contains=path_contains,
    )

    if not raw_results:
        return []

    # Score with keyword boost
    scored = []
    for r in raw_results:
        # sqlite-vec distance is L2; convert to similarity [0,1]
        # Lower distance = more similar
        distance = r["distance"]
        base_similarity = max(0.0, 1.0 - (distance / 2.0))
        boost = compute_keyword_boost(query, r, keyword_boost)
        final_score = base_similarity + boost
        scored.append((final_score, r))

    # Sort by final score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build results — deduplicate by file_path + start_line
    results = []
    seen_chunks: set[tuple[str, int]] = set()
    for score, r in scored:
        # Deduplicate overlapping chunks (same file + overlapping line ranges)
        chunk_key = (r["file_path"], r["start_line"])
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)

        rank = len(results) + 1

        # Preview: first 30 lines of source
        source_lines = (r["source"] or "").split("\n")
        preview = "\n".join(source_lines[:30])
        if len(source_lines) > 30:
            preview += f"\n// ... ({len(source_lines) - 30} more lines)"

        distance = r["distance"]
        vec_score = max(0.0, 1.0 - (distance / 2.0))
        kw_boost = compute_keyword_boost(query, r, keyword_boost)

        results.append(SearchResult(
            rank=rank,
            file_path=r["file_path"],
            name=r.get("name"),
            chunk_type=r["chunk_type"],
            language=r["language"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            source_preview=preview,
            similarity=min(score, 1.0),
            file_mtime=r["file_mtime"],
            vector_score=vec_score,
            keyword_boost_score=kw_boost,
        ))

        if len(results) >= limit:
            break

    return results
