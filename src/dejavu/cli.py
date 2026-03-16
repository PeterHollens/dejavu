"""CLI interface — Click-based command line tool for Déjà Vu."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .config import DejavuConfig
from .db import DejavuDB
from .embedder import OllamaEmbedder
from .indexer import IndexProgress, index_all, index_path
from .search import search

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_components(config: DejavuConfig):
    """Initialize DB and embedder from config."""
    db = DejavuDB(config.db_path)
    db.init_schema()
    embedder = OllamaEmbedder(
        base_url=config.ollama_base_url,
        model=config.embedding_model,
        fallback_model=config.embedding_fallback_model,
    )
    return db, embedder


def _run_async(coro):
    """Run an async function from sync context."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.argument("query", required=False)
@click.option("--lang", "-l", help="Filter by language (python, javascript, etc.)")
@click.option("--when", "-w", help="Temporal hint ('last summer', '2024', etc.)")
@click.option("--path", "-p", "path_filter", help="Filter paths containing this string")
@click.option("--limit", "-n", default=10, help="Max results (default: 10)")
@click.option("--json-output", "--json", "json_output", is_flag=True, help="Output results as JSON")
@click.option("--explain", is_flag=True, help="Show score breakdown for each result")
@click.pass_context
def main(ctx, query, lang, when, path_filter, limit, json_output, explain):
    """Déjà Vu — Find code you forgot, by describing what it did.

    \b
    Examples:
      dejavu "that drag and drop kanban board"
      dejavu "CSV parser that grouped by date" --lang python
      dejavu "animated sidebar component" --when "last summer"
      dejavu "auth middleware" --explain
      dejavu "deployment script" --json
      dejavu index ~/projects/my-app
      dejavu status
    """
    if ctx.invoked_subcommand is not None:
        return

    if not query:
        click.echo(ctx.get_help())
        return

    # Direct search mode
    config = DejavuConfig.load()
    db, embedder = _get_components(config)

    try:
        # Check if embedder is available
        if not _run_async(embedder.is_available()):
            console.print(
                Panel(
                    "[red]Ollama is not running or no embedding model found.[/red]\n\n"
                    f"Start Ollama and pull the model:\n"
                    f"  [bold]ollama pull {config.embedding_model}[/bold]",
                    title="Connection Error",
                )
            )
            sys.exit(1)

        stats = db.stats()
        if stats["chunks"] == 0:
            console.print(
                Panel(
                    "[yellow]Index is empty.[/yellow] Run [bold]dejavu index[/bold] first to index your code.",
                    title="No Data",
                )
            )
            sys.exit(1)

        with console.status("[bold green]Searching...", spinner="dots"):
            results = _run_async(search(
                db=db,
                embedder=embedder,
                query=query,
                language=lang,
                when=when,
                path_contains=path_filter,
                limit=limit,
                keyword_boost=config.keyword_boost,
            ))

        if not results:
            if json_output:
                click.echo(json.dumps({"query": query, "results": []}, indent=2))
            else:
                console.print(f"\n[dim]No matches found for:[/dim] \"{query}\"\n")
                console.print("[dim]Try a different description or broaden your search.[/dim]")
            return

        # JSON output mode
        if json_output:
            output = {
                "query": query,
                "result_count": len(results),
                "results": [r.to_dict() for r in results],
            }
            click.echo(json.dumps(output, indent=2))
            return

        console.print(f"\n[bold]Found {len(results)} matches[/bold] for \"{query}\"\n")

        for r in results:
            # Header
            pct = f"{r.similarity * 100:.0f}%"
            name = r.name or "(unnamed)"
            header = Text()
            header.append(f"#{r.rank} ", style="dim")
            header.append(name, style="bold")
            header.append(f" ({r.chunk_type})", style="dim")
            header.append(f" — {pct}", style="bold green" if r.similarity > 0.7 else "yellow")

            from datetime import datetime
            mod_date = datetime.fromtimestamp(r.file_mtime).strftime("%Y-%m-%d")

            console.print(header)
            console.print(f"  [dim]{r.file_path}[/dim]")
            console.print(f"  [dim]{r.language} | {mod_date} | lines {r.start_line}-{r.end_line}[/dim]")

            # Explain mode: show score breakdown
            if explain:
                vec_pct = f"{r.vector_score * 100:.1f}%"
                kw_pct = f"{r.keyword_boost_score * 100:.1f}%"
                console.print(
                    f"  [dim cyan]scores: vector={vec_pct}  keyword_boost=+{kw_pct}  "
                    f"combined={pct}[/dim cyan]"
                )

            # Code preview (first 15 lines)
            preview_lines = r.source_preview.split("\n")[:15]
            preview = "\n".join(preview_lines)
            if len(r.source_preview.split("\n")) > 15:
                preview += f"\n  ... ({len(r.source_preview.split(chr(10))) - 15} more lines)"

            syntax = Syntax(preview, r.language, theme="monokai", line_numbers=False)
            console.print(Panel(syntax, border_style="dim", padding=(0, 1)))
            console.print()

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

@main.command()
@click.argument("path", required=False)
def index(path):
    """Index code directories to make them searchable.

    \b
    Examples:
      dejavu index                    # Index all configured root paths
      dejavu index ~/projects/my-app  # Index a specific directory
    """
    config = DejavuConfig.load()
    db, embedder = _get_components(config)

    try:
        # Check embedder
        if not _run_async(embedder.is_available()):
            console.print(
                Panel(
                    "[red]Ollama is not running or no embedding model found.[/red]\n\n"
                    f"Start Ollama and pull the model:\n"
                    f"  [bold]ollama pull {config.embedding_model}[/bold]",
                    title="Connection Error",
                )
            )
            sys.exit(1)

        progress = IndexProgress()

        if path:
            console.print(f"[bold]Indexing:[/bold] {path}\n")
            with console.status("[bold green]Indexing...", spinner="dots"):
                _run_async(index_path(path, config, db, embedder, progress))
        else:
            console.print(f"[bold]Indexing all configured root paths:[/bold]")
            for p in config.root_paths:
                console.print(f"  [dim]{p}[/dim]")
            console.print()

            with console.status("[bold green]Discovering and indexing repos...", spinner="dots"):
                _run_async(index_all(config, db, embedder, progress))

        # Report
        table = Table(show_header=False, border_style="dim")
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")
        table.add_row("Repos found", str(progress.repos_found))
        table.add_row("Repos indexed", str(progress.repos_indexed))
        table.add_row("Files indexed", str(progress.files_indexed))
        table.add_row("Chunks extracted", str(progress.chunks_extracted))
        table.add_row("Chunks embedded", str(progress.chunks_embedded))
        console.print(Panel(table, title="[bold green]Indexing Complete", border_style="green"))

        if progress.errors:
            console.print(f"\n[yellow]Errors ({len(progress.errors)}):[/yellow]")
            for err in progress.errors[:10]:
                console.print(f"  [dim]{err}[/dim]")

    finally:
        db.close()


@main.command()
def status():
    """Show index statistics."""
    config = DejavuConfig.load()
    db, _ = _get_components(config)

    try:
        stats = db.stats()
        repos = db.list_repos()

        table = Table(show_header=False, border_style="dim")
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")
        table.add_row("Total repos", str(stats["repos"]))
        table.add_row("Total chunks", str(stats["chunks"]))
        table.add_row("Total embeddings", str(stats["embeddings"]))
        table.add_row("DB path", str(config.db_path))
        console.print(Panel(table, title="[bold]Déjà Vu Index", border_style="blue"))

        if stats["languages"]:
            lang_table = Table(title="Languages", border_style="dim")
            lang_table.add_column("Language")
            lang_table.add_column("Chunks", justify="right")
            for lang, count in stats["languages"].items():
                lang_table.add_row(lang, str(count))
            console.print(lang_table)

        if repos:
            repo_table = Table(title="Indexed Repos", border_style="dim")
            repo_table.add_column("Path")
            repo_table.add_column("Files", justify="right")
            repo_table.add_column("Chunks", justify="right")
            for repo in repos[:20]:
                repo_table.add_row(
                    repo["path"],
                    str(repo["file_count"]),
                    str(repo["chunk_count"]),
                )
            if len(repos) > 20:
                repo_table.add_row(f"... and {len(repos) - 20} more", "", "")
            console.print(repo_table)

    finally:
        db.close()


@main.command()
def config():
    """Show current configuration."""
    cfg = DejavuConfig.load()

    console.print(Panel(
        f"[bold]Root paths:[/bold]\n"
        + "\n".join(f"  {p}" for p in cfg.root_paths)
        + f"\n\n[bold]DB path:[/bold] {cfg.db_path}"
        + f"\n[bold]Embedding model:[/bold] {cfg.embedding_model}"
        + f"\n[bold]Ollama URL:[/bold] {cfg.ollama_base_url}"
        + f"\n[bold]Batch size:[/bold] {cfg.embedding_batch_size}"
        + f"\n[bold]Default limit:[/bold] {cfg.default_limit}",
        title="[bold]Déjà Vu Config",
        border_style="blue",
    ))

    console.print(f"\n[dim]Config file: {cfg.db_path.parent / 'config.toml'}[/dim]")
    console.print("[dim]Edit this file to customize root paths, aliases, and settings.[/dim]")


@main.command()
def init():
    """Initialize Déjà Vu with default config."""
    cfg = DejavuConfig()
    cfg.save()
    db = DejavuDB(cfg.db_path)
    db.init_schema()
    db.close()

    console.print(Panel(
        f"[green]Created config at:[/green] {cfg.db_path.parent / 'config.toml'}\n"
        f"[green]Created database at:[/green] {cfg.db_path}\n\n"
        "Next steps:\n"
        f"  1. [bold]ollama pull {cfg.embedding_model}[/bold]\n"
        "  2. Edit ~/.dejavu/config.toml to set your root paths\n"
        "  3. [bold]dejavu index[/bold] to build the index\n"
        "  4. [bold]dejavu \"describe what you're looking for\"[/bold]",
        title="[bold green]Déjà Vu Initialized",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
