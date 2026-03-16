"""Tests for the MCP server layer."""

import pytest

from dejavu.db import DejavuDB

try:
    from dejavu.server import _build_instructions
except ImportError:
    pytest.skip("MCP dependencies not available", allow_module_level=True)


@pytest.fixture
def db(tmp_path):
    db = DejavuDB(tmp_path / "test.db")
    db.init_schema()
    yield db
    db.close()


class TestBuildInstructions:
    def test_empty_index(self, db):
        instructions = _build_instructions(db)
        assert "empty" in instructions.lower()
        assert "dejavu_reindex" in instructions

    def test_populated_index(self, db):
        repo_id = db.upsert_repo("/repo")
        db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/main.py",
            chunk_type="function",
            name="hello",
            signature="def hello():",
            docstring=None,
            source="def hello():\n    pass\n    return True",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        instructions = _build_instructions(db)
        assert "1 code chunks" in instructions
        assert "1 repos" in instructions
        assert "python" in instructions.lower()
        assert "dejavu_search" in instructions

    def test_multiple_languages(self, db):
        repo_id = db.upsert_repo("/repo")
        db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/main.py",
            chunk_type="function",
            name="py_func",
            signature=None,
            docstring=None,
            source="def f():\n    pass\n    pass",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/app.js",
            chunk_type="function",
            name="js_func",
            signature=None,
            docstring=None,
            source="function f() {\n    return 1;\n}",
            language="javascript",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        instructions = _build_instructions(db)
        assert "python" in instructions.lower()
        assert "javascript" in instructions.lower()
        assert "2 code chunks" in instructions
