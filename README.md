# Dejavu

**Find code you forgot, by describing what it did.**

Dejavu is a semantic code search tool that lets you find code across your projects using natural language descriptions. Instead of remembering filenames, function names, or exact keywords, just describe what the code did:

```
dejavu "that drag and drop kanban board"
dejavu "CSV parser that grouped by date" --lang python
dejavu "animated sidebar component" --when "last summer"
```

## How it works

1. **Index** your code directories -- Dejavu uses tree-sitter AST parsing to extract functions, classes, and methods from 20+ languages
2. **Embed** each code chunk using local vector embeddings via [Ollama](https://ollama.com) (no data leaves your machine)
3. **Search** with natural language -- your query is embedded and matched against your code using vector similarity

Everything runs locally. Your code never leaves your machine.

## Install

```bash
pip install dejavu-code
```

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally

Pull the embedding model:

```bash
ollama pull nomic-embed-code
```

### Optional: faster vector search

For large codebases, install the sqlite-vec extension for hardware-accelerated KNN search:

```bash
pip install "dejavu-code[vec]"
```

Without it, Dejavu falls back to numpy-based cosine similarity (works fine for most codebases).

## Quick start

```bash
# 1. Initialize config
dejavu init

# 2. Edit ~/.dejavu/config.toml to set your code directories
#    (defaults: ~/code, ~/projects, ~/dev, ~/src, ~/repos, ~/work)

# 3. Index your code
dejavu index

# 4. Search!
dejavu "that function that parsed CSV files and grouped them by date"
```

### Index a specific directory

```bash
dejavu index ~/projects/my-app
```

### Filter by language or time

```bash
dejavu "auth middleware" --lang python
dejavu "React component with tabs" --when "last summer"
dejavu "deployment script" --path work
```

### JSON output (for scripts and agents)

```bash
dejavu "auth middleware" --json
```

Returns structured JSON with all result metadata -- useful for piping into other tools or agent workflows.

### Explain mode (score breakdown)

```bash
dejavu "CSV parser" --explain
```

Shows how each result was scored:
```
#1 parse_csv (Function) — 87%
  /home/user/projects/etl/parsers.py
  python | 2025-08-14 | lines 42-78
  scores: vector=82.3%  keyword_boost=+4.5%  combined=87%
```

### Check index status

```bash
dejavu status
```

## Claude Code integration (MCP server)

Dejavu includes an [MCP](https://modelcontextprotocol.io) server that gives Claude direct access to your code search index. This is the primary way to use Dejavu -- Claude can find code you've written before without you needing to remember where it lives.

### Setup with Claude Code

Run this from your terminal:

```bash
claude mcp add dejavu -- dejavu-mcp
```

That's it. Claude Code will now have access to the `dejavu_search`, `dejavu_reindex`, `dejavu_status`, and `dejavu_forget` tools.

### Setup with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "dejavu": {
      "command": "dejavu-mcp"
    }
  }
}
```

### What Claude can do with Dejavu

Once connected, you can ask Claude things like:

- "Search my code for that CSV parser I wrote last year"
- "Find the React component that had the animated sidebar"
- "Look for any auth middleware I wrote in Python"
- "Reindex my projects directory"

### Available MCP tools

| Tool | Description |
|------|-------------|
| `dejavu_search` | Search indexed code by natural language description. Supports language filters, temporal hints, and path filters. |
| `dejavu_reindex` | Index or re-index code directories. Incremental -- only processes modified files. |
| `dejavu_status` | Show index statistics: repo count, chunk count, languages, and configured paths. |
| `dejavu_forget` | Remove a repository/directory from the index. Source files are never modified. |

## Configuration

Config lives at `~/.dejavu/config.toml`. Created by `dejavu init`.

```toml
[paths]
roots = ["~/code", "~/projects"]

[index]
db_path = "~/.dejavu/index.db"
max_file_size_kb = 500

[embedding]
provider = "ollama"
model = "nomic-embed-code"
fallback_model = "nomic-embed-text"
batch_size = 32

[embedding.ollama]
base_url = "http://localhost:11434"

[search]
default_limit = 10
keyword_boost = 0.15
```

### Environment variable overrides

| Variable | Description |
|----------|-------------|
| `DEJAVU_DB` | Override database path |
| `OLLAMA_HOST` | Override Ollama URL |

## Supported languages

Tree-sitter AST parsing (extracts functions, classes, methods):

Python, JavaScript, TypeScript, TSX, Rust, Go, Ruby, Java, Kotlin, C, C++, PHP, Bash, Swift

Sliding-window fallback (indexes file contents in chunks):

SQL, HTML, CSS, SCSS, Svelte, Vue, TOML, YAML, JSON, Protobuf, Lua, Julia, Scala, Zig, Elixir, and more.

## Architecture

```
                         ┌─────────────┐
                         │  Claude Code │
                         │  (MCP client)│
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                    ┌────┤  server.py   ├────┐
                    │    │  (MCP tools) │    │
                    │    └──────────────┘    │
               ┌────▼─────┐          ┌──────▼──────┐
               │  cli.py   │          │  search.py   │
               │  (Click)  │          │  (pipeline)  │
               └────┬──────┘          └──────┬───────┘
                    │                        │
         ┌──────────▼──────────┐    ┌────────▼────────┐
         │    indexer.py       │    │   embedder.py    │
         │  (orchestrator)     │    │ (Ollama client)  │
         └──┬──────────┬───┘  │    └────────┬─────────┘
            │          │      │             │
   ┌────────▼──┐  ┌────▼─────┐    ┌────────▼─────────┐
   │discovery.py│  │extractor │    │   Ollama (local)  │
   │(find repos)│  │(tree-sit)│    │ nomic-embed-code  │
   └────────────┘  └──────────┘    └──────────────────┘
                        │
                 ┌──────▼───────┐
                 │    db.py     │
                 │   (SQLite +  │
                 │  sqlite-vec) │
                 └──────────────┘
```

### Search pipeline

1. **Parse query** -- extract language hints ("in python"), temporal hints ("last summer"), path filters
2. **Clean & embed** -- strip hints from query text, generate vector embedding via Ollama
3. **Vector search** -- KNN lookup in sqlite-vec (or numpy fallback) with filters applied
4. **Keyword boost** -- bonus score for results whose name/signature/docstring match query terms
5. **Rank & deduplicate** -- sort by combined score, remove overlapping chunks from same file

### Indexing pipeline

1. **Discover** -- walk configured root paths, find repos by project markers (.git, package.json, etc.)
2. **Filter** -- skip binary files, node_modules, .gitignore'd paths, files over 500KB
3. **Extract** -- tree-sitter AST parsing pulls out functions, classes, methods with names and docstrings
4. **Embed** -- batch-generate vector embeddings via Ollama's local API
5. **Store** -- write chunks and embeddings to SQLite (incremental: only re-processes modified files)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
