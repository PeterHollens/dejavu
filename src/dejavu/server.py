"""MCP Server — Exposes Déjà Vu search and indexing as MCP tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, ConfigDict, Field

from .config import DejavuConfig
from .db import DejavuDB
from .embedder import OllamaEmbedder
from .indexer import IndexProgress, index_all, index_path
from .search import SearchResult, search

# ---------------------------------------------------------------------------
# Lifespan — initialize DB and embedder once
# ---------------------------------------------------------------------------

def _build_instructions(db: DejavuDB) -> str:
    """Build dynamic server instructions from the current index state."""
    try:
        stats = db.stats()
    except Exception:
        return "Déjà Vu — semantic code search. Index may not be initialized yet."

    if stats["chunks"] == 0:
        return (
            "Déjà Vu — semantic code search across local codebases.\n"
            "The index is currently empty. Use dejavu_reindex to index code directories first."
        )

    lang_list = ", ".join(
        f"{lang} ({count})" for lang, count in list(stats["languages"].items())[:10]
    )
    return (
        f"Déjà Vu — semantic code search across local codebases.\n"
        f"Index contains {stats['chunks']} code chunks from {stats['repos']} repos.\n"
        f"Languages: {lang_list}.\n"
        f"Search with natural language descriptions. You can include temporal hints "
        f"('last summer', '2024') and language hints ('in python') directly in the query. "
        f"Use dejavu_search for finding code, dejavu_reindex to update the index."
    )


@asynccontextmanager
async def app_lifespan():
    config = DejavuConfig.load()
    db = DejavuDB(config.db_path)
    db.init_schema()
    embedder = OllamaEmbedder(
        base_url=config.ollama_base_url,
        model=config.embedding_model,
        fallback_model=config.embedding_fallback_model,
    )
    # Update server instructions based on index state
    mcp.instructions = _build_instructions(db)
    yield {"config": config, "db": db, "embedder": embedder}
    db.close()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP("dejavu_mcp", lifespan=app_lifespan)


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class SearchInput(BaseModel):
    """Input for searching indexed code with natural language."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        description=(
            "Natural language description of the code you're looking for. "
            "Be as vague or specific as you want: "
            "'that function that parsed CSV files and grouped them by date', "
            "'the React component with the animated sidebar', "
            "'some bash script that deployed to AWS'. "
            "You can include temporal hints like 'from last summer' or "
            "language hints like 'in python' directly in the query."
        ),
        min_length=3,
        max_length=1000,
    )
    language: Optional[str] = Field(
        default=None,
        description="Filter by programming language: python, javascript, typescript, rust, go, etc.",
    )
    when: Optional[str] = Field(
        default=None,
        description=(
            "When you roughly built/modified it: "
            "'last summer', '2024', 'a few months ago', 'last week', 'recently'"
        ),
    )
    path_contains: Optional[str] = Field(
        default=None,
        description="Filter to file paths containing this string: 'work', 'personal', 'client-x'",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )


class ReindexInput(BaseModel):
    """Input for reindexing code directories."""
    model_config = ConfigDict(str_strip_whitespace=True)

    path: Optional[str] = Field(
        default=None,
        description=(
            "Specific directory path to index. "
            "If omitted, indexes all configured root paths."
        ),
    )


class ForgetInput(BaseModel):
    """Input for removing a path from the index."""
    model_config = ConfigDict(str_strip_whitespace=True)

    path: str = Field(
        ...,
        description="The repository/directory path to remove from the index.",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="dejavu_search",
    annotations={
        "title": "Search Code by Description",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dejavu_search(params: SearchInput, ctx: Context) -> str:
    """Search your indexed codebase using natural language descriptions.

    Finds code you've written before even when you don't remember the filename,
    function name, or exact keywords. Describe what it did and Déjà Vu will
    find it using semantic vector search.

    Returns ranked results with file paths, code previews, and similarity scores.
    """
    db: DejavuDB = ctx.request_context.lifespan_state["db"]
    embedder: OllamaEmbedder = ctx.request_context.lifespan_state["embedder"]
    config: DejavuConfig = ctx.request_context.lifespan_state["config"]

    try:
        results = await search(
            db=db,
            embedder=embedder,
            query=params.query,
            language=params.language,
            when=params.when,
            path_contains=params.path_contains,
            limit=params.limit,
            keyword_boost=config.keyword_boost,
        )
    except RuntimeError as e:
        return f"Search failed: {e}"

    if not results:
        stats = db.stats()
        if stats["chunks"] == 0:
            return (
                "No results found — the index is empty. "
                "Run `dejavu_reindex` first to index your code directories."
            )
        return (
            f"No matches found for: \"{params.query}\"\n\n"
            f"Index contains {stats['chunks']} chunks from {stats['repos']} repos. "
            "Try a different description or broaden your search."
        )

    # Format results
    lines = [f"## Found {len(results)} matches for \"{params.query}\"\n"]
    for r in results:
        lines.append(r.format_markdown())
        lines.append("\n---\n")

    return "\n".join(lines)


@mcp.tool(
    name="dejavu_reindex",
    annotations={
        "title": "Index/Reindex Code Directories",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dejavu_reindex(params: ReindexInput, ctx: Context) -> str:
    """Index or reindex code directories to make them searchable.

    Scans directories for code files, extracts functions/classes/components
    using AST parsing, generates semantic embeddings, and stores everything
    in a local SQLite database.

    Incremental: only processes files modified since the last scan.
    """
    db: DejavuDB = ctx.request_context.lifespan_state["db"]
    embedder: OllamaEmbedder = ctx.request_context.lifespan_state["embedder"]
    config: DejavuConfig = ctx.request_context.lifespan_state["config"]

    progress = IndexProgress()

    if params.path:
        await index_path(params.path, config, db, embedder, progress)
    else:
        await index_all(config, db, embedder, progress)

    # Format report
    lines = ["## Indexing Complete\n"]
    lines.append(f"- **Repos found:** {progress.repos_found}")
    lines.append(f"- **Repos indexed:** {progress.repos_indexed}")
    lines.append(f"- **Files indexed:** {progress.files_indexed}")
    lines.append(f"- **Chunks extracted:** {progress.chunks_extracted}")
    lines.append(f"- **Chunks embedded:** {progress.chunks_embedded}")

    if progress.errors:
        lines.append(f"\n### Errors ({len(progress.errors)})")
        for err in progress.errors[:10]:
            lines.append(f"- {err}")
        if len(progress.errors) > 10:
            lines.append(f"- ... and {len(progress.errors) - 10} more")

    return "\n".join(lines)


@mcp.tool(
    name="dejavu_status",
    annotations={
        "title": "Index Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dejavu_status(ctx: Context) -> str:
    """Show the current state of the Déjà Vu index.

    Returns statistics about indexed repos, chunks, embeddings,
    and language distribution.
    """
    db: DejavuDB = ctx.request_context.lifespan_state["db"]
    config: DejavuConfig = ctx.request_context.lifespan_state["config"]

    stats = db.stats()
    repos = db.list_repos()

    lines = ["## Déjà Vu Index Status\n"]
    lines.append(f"- **Total repos:** {stats['repos']}")
    lines.append(f"- **Total chunks:** {stats['chunks']}")
    lines.append(f"- **Total embeddings:** {stats['embeddings']}")
    lines.append(f"- **DB path:** `{config.db_path}`")

    if stats["languages"]:
        lines.append("\n### Languages")
        for lang, count in stats["languages"].items():
            lines.append(f"- {lang}: {count} chunks")

    if repos:
        lines.append("\n### Indexed Repos")
        for repo in repos[:20]:
            lines.append(
                f"- `{repo['path']}` — {repo['chunk_count']} chunks, "
                f"{repo['file_count']} files"
            )
        if len(repos) > 20:
            lines.append(f"- ... and {len(repos) - 20} more")

    lines.append(f"\n### Configured Root Paths")
    for p in config.root_paths:
        lines.append(f"- `{p}`")

    return "\n".join(lines)


@mcp.tool(
    name="dejavu_forget",
    annotations={
        "title": "Remove Path from Index",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dejavu_forget(params: ForgetInput, ctx: Context) -> str:
    """Remove a repository/directory and all its chunks from the index.

    This deletes all indexed data for the specified path. The original
    source files are NOT affected — only the search index is modified.
    """
    db: DejavuDB = ctx.request_context.lifespan_state["db"]

    repo = db.get_repo(params.path)
    if not repo:
        return f"Path not found in index: `{params.path}`"

    # Delete all chunks for this repo
    c = db.conn
    chunk_ids = c.execute(
        "SELECT id FROM chunks WHERE repo_id=?", (repo["id"],)
    ).fetchall()
    ids = [r["id"] for r in chunk_ids]

    if ids:
        placeholders = ",".join("?" * len(ids))
        if db.use_vec:
            c.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", ids)
        c.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", ids)
        c.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)

    c.execute("DELETE FROM repos WHERE id=?", (repo["id"],))
    c.commit()

    return (
        f"Removed `{params.path}` from index.\n"
        f"Deleted {len(ids)} chunks. Source files were not modified."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for the dejavu-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
