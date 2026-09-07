"""Tests for configuration loading.

The contract: every problem is found at startup and reported together. Nothing
here is allowed to surface partway through a send loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outreach_agent.config import (
    DEFAULT_MODEL,
    STRUCTURED_OUTPUT_MODELS,
    ConfigError,
    load_config,
    load_cv,
)

MINIMAL = {"ANTHROPIC_API_KEY": "sk-test", "APOLLO_API_KEY": "apollo-test"}


def load(**overrides):
    env = dict(MINIMAL)
    env.update(overrides)
    return load_config(env=env, load_dotenv_file=False)


# ─────────────────────────── required keys ───────────────────────────


def test_a_dry_run_needs_only_the_two_api_keys():
    config = load()
    assert config.anthropic_api_key == "sk-test"
    assert config.apollo_api_key == "apollo-test"


def test_every_missing_key_is_reported_at_once():
    """One pass to fix the .env, not five."""
    with pytest.raises(ConfigError) as exc:
        load_config(env={}, load_dotenv_file=False)

    assert len(exc.value.problems) == 2
    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "APOLLO_API_KEY" in message


def test_an_empty_value_counts_as_missing():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load(ANTHROPIC_API_KEY="   ")


def test_sending_credentials_are_only_required_for_live():
    """A dry run must work before Gmail is ever set up."""
    load()  # no Gmail values at all — fine

    with pytest.raises(ConfigError) as exc:
        load_config(env=dict(MINIMAL), require_send=True, load_dotenv_file=False)

    problems = " ".join(exc.value.problems)
    assert "GMAIL_ADDRESS" in problems
    assert "GMAIL_APP_PASSWORD" in problems
    assert "SENDER_NAME" in problems


def test_live_config_passes_with_gmail_credentials():
    config = load_config(
        env={
            **MINIMAL,
            "GMAIL_ADDRESS": "jane@gmail.com",
            "GMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
            "SENDER_NAME": "Jane",
        },
        require_send=True,
        load_dotenv_file=False,
    )
    assert config.gmail_address == "jane@gmail.com"


def test_a_malformed_gmail_address_is_rejected():
    with pytest.raises(ConfigError, match="not an email address"):
        load(GMAIL_ADDRESS="not-an-address")


# ─────────────────────────── the model allowlist ───────────────────────────


def test_the_default_model_supports_structured_outputs():
    """Both LLM calls depend on it, so the default must be on the list."""
    assert DEFAULT_MODEL in STRUCTURED_OUTPUT_MODELS
    assert load().anthropic_model == DEFAULT_MODEL


def test_a_model_without_structured_outputs_fails_at_startup():
    """Not with a 400 at contact seventeen."""
    with pytest.raises(ConfigError, match="structured outputs"):
        load(ANTHROPIC_MODEL="claude-2.1")


def test_a_supported_alternative_model_is_accepted():
    assert load(ANTHROPIC_MODEL="claude-opus-5").anthropic_model == "claude-opus-5"


# ─────────────────────────── numbers and ranges ───────────────────────────


def test_defaults_are_sensible():
    config = load()
    assert config.daily_cap == 25
    assert config.min_delay_seconds == 30
    assert config.max_delay_seconds == 180
    assert config.database_url.startswith("sqlite")


def test_a_non_numeric_cap_is_rejected():
    with pytest.raises(ConfigError, match="whole number"):
        load(DAILY_CAP="lots")


def test_a_zero_or_negative_cap_is_rejected():
    with pytest.raises(ConfigError, match="greater than zero"):
        load(DAILY_CAP="0")


def test_an_inverted_delay_window_is_rejected():
    with pytest.raises(ConfigError, match="greater than"):
        load(MIN_DELAY_SECONDS="300", MAX_DELAY_SECONDS="60")


def test_a_cap_above_the_hard_ceiling_is_accepted_here_and_clamped_later():
    """config.py validates; runner.py enforces the ceiling."""
    assert load(DAILY_CAP="1000").daily_cap == 1000


def test_a_malformed_database_url_is_rejected():
    with pytest.raises(ConfigError, match="SQLAlchemy URL"):
        load(DATABASE_URL="just-a-filename.db")


def test_a_postgres_url_is_accepted_without_code_changes():
    config = load(DATABASE_URL="postgresql+psycopg://user:pw@host:5432/db")
    assert config.database_url.startswith("postgresql")


# ─────────────────────────── enumerated values ───────────────────────────


def test_an_unknown_sender_backend_is_rejected():
    with pytest.raises(ConfigError, match="sender backend"):
        load(SENDER_BACKEND="carrier-pigeon")


def test_an_unknown_log_level_is_rejected():
    with pytest.raises(ConfigError, match="log level"):
        load(LOG_LEVEL="chatty")


def test_several_problems_are_reported_together():
    with pytest.raises(ConfigError) as exc:
        load(DAILY_CAP="nope", SENDER_BACKEND="pigeon", ANTHROPIC_MODEL="claude-1")

    assert len(exc.value.problems) == 3


# ─────────────────────────── the CV ───────────────────────────


def test_the_cv_is_read(tmp_path: Path):
    path = tmp_path / "cv.txt"
    path.write_text("Jane Engineer. Builds things.")
    assert load_cv(path) == "Jane Engineer. Builds things."


def test_a_missing_cv_fails_with_an_actionable_message(tmp_path: Path):
    with pytest.raises(ConfigError, match="CV file not found"):
        load_cv(tmp_path / "nope.txt")


def test_an_empty_cv_is_rejected(tmp_path: Path):
    """Claude may only use facts from this file, so an empty one is useless."""
    path = tmp_path / "cv.txt"
    path.write_text("   \n  ")
    with pytest.raises(ConfigError, match="empty"):
        load_cv(path)


# ─────────────────────────── isolation ───────────────────────────


def test_the_real_environment_is_not_consulted_when_env_is_given(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leaked-real-key")
    config = load(ANTHROPIC_API_KEY="sk-test")
    assert config.anthropic_api_key == "sk-test"
