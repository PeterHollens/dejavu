"""Tests for CLI output modes (--json, --explain)."""

import json

import pytest

try:
    from click.testing import CliRunner
    from dejavu.cli import main
except ImportError:
    pytest.skip("CLI dependencies not available", allow_module_level=True)


class TestCliHelp:
    def test_no_args_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "Déjà Vu" in result.output

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "--explain" in result.output
        assert "--lang" in result.output
        assert "--when" in result.output
        assert "--path" in result.output
        assert "--limit" in result.output

    def test_init_subcommand_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output

    def test_index_subcommand_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["index", "--help"])
        assert result.exit_code == 0
        assert "Index" in result.output

    def test_status_subcommand_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0

    def test_config_subcommand_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
