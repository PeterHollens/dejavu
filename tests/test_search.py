"""Tests for the search engine layer."""

import time

from dejavu.search import (
    SearchResult,
    compute_keyword_boost,
    parse_language_hint,
    parse_temporal_hint,
    strip_language_hint,
    strip_temporal_hint,
)


class TestParseTemporalHint:
    def test_explicit_year(self):
        result = parse_temporal_hint("something from 2024")
        assert result is not None
        start, end = result
        assert start < end

    def test_last_week(self):
        result = parse_temporal_hint("last week")
        assert result is not None
        start, end = result
        assert end > start

    def test_last_month(self):
        result = parse_temporal_hint("last month")
        assert result is not None

    def test_last_summer(self):
        result = parse_temporal_hint("last summer")
        assert result is not None

    def test_recently(self):
        result = parse_temporal_hint("recently")
        assert result is not None

    def test_no_temporal_hint(self):
        result = parse_temporal_hint("some random query about code")
        assert result is None


class TestStripTemporalHint:
    def test_strips_year(self):
        result = strip_temporal_hint("that parser from 2024")
        assert "2024" not in result
        assert "parser" in result

    def test_strips_last_week(self):
        result = strip_temporal_hint("auth middleware last week")
        assert "last week" not in result.lower()
        assert "auth" in result

    def test_preserves_non_temporal(self):
        text = "drag and drop kanban board"
        assert strip_temporal_hint(text) == text


class TestParseLanguageHint:
    def test_in_python(self):
        assert parse_language_hint("that CSV parser in python") == "python"

    def test_the_javascript_one(self):
        assert parse_language_hint("the javascript component") == "javascript"

    def test_written_in_rust(self):
        assert parse_language_hint("written in rust") == "rust"

    def test_using_typescript(self):
        assert parse_language_hint("using typescript") == "typescript"

    def test_no_language_hint(self):
        assert parse_language_hint("that function I wrote") is None

    def test_ambiguous_go_not_matched_without_context(self):
        # "go" is ambiguous -- should not match without "in go" context
        assert parse_language_hint("go ahead and search") is None

    def test_python_alias_py(self):
        assert parse_language_hint("my py script") == "python"


class TestStripLanguageHint:
    def test_strips_in_python(self):
        result = strip_language_hint("CSV parser in python")
        assert "python" not in result.lower()
        assert "CSV" in result

    def test_preserves_non_language(self):
        text = "drag and drop board"
        assert strip_language_hint(text) == text


class TestKeywordBoost:
    def test_full_match(self):
        chunk = {
            "name": "parse_csv",
            "signature": "def parse_csv(path: str)",
            "docstring": "Parse a CSV file and return rows.",
        }
        boost = compute_keyword_boost("parse csv file", chunk, boost_weight=0.15)
        assert boost > 0

    def test_no_match(self):
        chunk = {
            "name": "render_button",
            "signature": "function renderButton(props)",
            "docstring": "Render a button component.",
        }
        boost = compute_keyword_boost("database migration", chunk, boost_weight=0.15)
        assert boost == 0.0

    def test_partial_match(self):
        chunk = {
            "name": "parse_csv",
            "signature": "def parse_csv(path)",
            "docstring": None,
        }
        boost = compute_keyword_boost("parse json data", chunk, boost_weight=0.15)
        assert 0 < boost < 0.15

    def test_empty_query(self):
        chunk = {"name": "foo", "signature": None, "docstring": None}
        assert compute_keyword_boost("", chunk) == 0.0


class TestSearchResultToDict:
    def _make_result(self, **overrides) -> SearchResult:
        defaults = dict(
            rank=1,
            file_path="/repo/main.py",
            name="hello",
            chunk_type="function",
            language="python",
            start_line=1,
            end_line=10,
            source_preview="def hello():\n    pass",
            similarity=0.85,
            file_mtime=1700000000.0,
            vector_score=0.80,
            keyword_boost_score=0.05,
        )
        defaults.update(overrides)
        return SearchResult(**defaults)

    def test_to_dict_has_all_keys(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["rank"] == 1
        assert d["file_path"] == "/repo/main.py"
        assert d["name"] == "hello"
        assert d["chunk_type"] == "function"
        assert d["language"] == "python"
        assert d["start_line"] == 1
        assert d["end_line"] == 10
        assert d["similarity"] == 0.85
        assert d["vector_score"] == 0.80
        assert d["keyword_boost"] == 0.05
        assert d["modified"] == "2023-11-14"
        assert "def hello" in d["source_preview"]

    def test_to_dict_rounds_scores(self):
        r = self._make_result(similarity=0.87654321, vector_score=0.8234567)
        d = r.to_dict()
        assert d["similarity"] == 0.8765
        assert d["vector_score"] == 0.8235

    def test_to_dict_handles_none_name(self):
        r = self._make_result(name=None)
        d = r.to_dict()
        assert d["name"] is None

    def test_format_markdown_contains_key_info(self):
        r = self._make_result()
        md = r.format_markdown()
        assert "hello" in md
        assert "Function" in md
        assert "85%" in md
        assert "/repo/main.py" in md
        assert "```python" in md

    def test_score_breakdown_defaults_to_zero(self):
        r = SearchResult(
            rank=1, file_path="/a.py", name="f", chunk_type="function",
            language="python", start_line=1, end_line=3,
            source_preview="x", similarity=0.5, file_mtime=1000.0,
        )
        assert r.vector_score == 0.0
        assert r.keyword_boost_score == 0.0


class TestParseLanguageHintEdgeCases:
    def test_cpp_detected(self):
        assert parse_language_hint("that sorting algorithm in c++") == "cpp"

    def test_cpp_bare_mention(self):
        assert parse_language_hint("c++ linked list") == "cpp"

    def test_golang_alias(self):
        assert parse_language_hint("written in golang") == "go"

    def test_shell_alias(self):
        assert parse_language_hint("my shell script") == "bash"
