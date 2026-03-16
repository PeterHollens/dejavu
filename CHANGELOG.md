# Changelog

## 0.2.0 - Search improvements and developer experience

- `--json` output flag for structured results (scripts, agents, pipelines)
- `--explain` mode showing score breakdown (vector score, keyword boost, combined)
- Dynamic MCP server instructions (tells Claude what's in the index)
- Architecture diagram and expanded documentation in README
- Score breakdown fields on SearchResult for programmatic access

## 0.1.0 - Initial release

- Semantic code search via natural language descriptions
- Tree-sitter AST parsing for 20+ languages
- Local embeddings via Ollama (nomic-embed-code)
- SQLite storage with sqlite-vec KNN search (numpy fallback)
- CLI with search, index, status, config, and init commands
- MCP server for Claude integration
- Temporal hints ("last summer", "2024") and language filters
- Keyword boosting for improved relevance
- Incremental indexing (only re-processes modified files)
- .gitignore-aware file discovery
