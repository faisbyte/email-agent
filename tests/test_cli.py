"""Tests for the command line surface.

The property worth pinning down here: `run` without `--live` cannot send, and
`--live` without confirmation cannot send either.
"""

from __future__ import annotations

import pytest

from outreach_agent.cli import build_parser, main

MINIMAL_ENV = {
    "ANTHROPIC_API_KEY": "sk-test",
    "APOLLO_API_KEY": "apollo-test",
    "DATABASE_URL": "sqlite://",
}


@pytest.fixture
def clean_env(monkeypatch):
    """A predictable environment: no .env, no inherited real keys."""
    for key in list(MINIMAL_ENV) + [
        "GMAIL_ADDRESS",
        "GMAIL_APP_PASSWORD",
        "SENDER_NAME",
        "ANTHROPIC_MODEL",
        "CAMPAIGN",
        "CV_PATH",
        "DAILY_CAP",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("outreach_agent.config.load_dotenv", lambda **kwargs: None)
    return monkeypatch


# ─────────────────────────── parsing ───────────────────────────


def test_run_defaults_to_dry_run():
    args = build_parser().parse_args(["run"])
    assert args.live is False


def test_live_must_be_explicit():
    args = build_parser().parse_args(["run", "--live"])
    assert args.live is True


def test_search_has_no_live_flag():
    """Discovery cannot send, so it must not offer the option."""
    args = build_parser().parse_args(["search"])
    assert not hasattr(args, "live")


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# ─────────────────────────── configuration failures ───────────────────────────


def test_a_missing_key_stops_the_command_before_anything_happens(clean_env, capsys):
    exit_code = main(["run"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
    assert "APOLLO_API_KEY" in err


def test_live_without_gmail_credentials_fails_at_startup(clean_env, capsys):
    """Not at contact seventeen."""
    for key, value in MINIMAL_ENV.items():
        clean_env.setenv(key, value)

    exit_code = main(["run", "--live"])

    assert exit_code == 2
    assert "GMAIL_APP_PASSWORD" in capsys.readouterr().err


def test_a_dry_run_does_not_need_gmail_credentials(clean_env, tmp_path, capsys):
    for key, value in MINIMAL_ENV.items():
        clean_env.setenv(key, value)
    clean_env.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    clean_env.setenv("CAMPAIGN", "campaigns/job_search.toml")

    cv = tmp_path / "cv.txt"
    cv.write_text("Jane Engineer. Builds things.")
    clean_env.setenv("CV_PATH", str(cv))

    main(["init-db"])
    capsys.readouterr()
    exit_code = main(["run"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "dry run" in out.lower()


# ─────────────────────────── the confirmation gate ───────────────────────────


def test_a_live_run_is_cancelled_without_typed_confirmation(clean_env, tmp_path, capsys):
    for key, value in MINIMAL_ENV.items():
        clean_env.setenv(key, value)
    clean_env.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'live.db'}")
    clean_env.setenv("GMAIL_ADDRESS", "jane@gmail.com")
    clean_env.setenv("GMAIL_APP_PASSWORD", "app-password")
    clean_env.setenv("SENDER_NAME", "Jane")
    clean_env.setenv("CAMPAIGN", "campaigns/job_search.toml")

    cv = tmp_path / "cv.txt"
    cv.write_text("Jane Engineer.")
    clean_env.setenv("CV_PATH", str(cv))

    clean_env.setattr("builtins.input", lambda _="": "no")

    exit_code = main(["run", "--live"])

    assert exit_code == 0
    assert "Cancelled" in capsys.readouterr().out


def test_running_before_init_db_says_so(clean_env, tmp_path, capsys):
    """A raw SQLAlchemy traceback is a bad first experience."""
    for key, value in MINIMAL_ENV.items():
        clean_env.setenv(key, value)
    clean_env.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}")
    clean_env.setenv("CAMPAIGN", "campaigns/job_search.toml")

    cv = tmp_path / "cv.txt"
    cv.write_text("Jane Engineer.")
    clean_env.setenv("CV_PATH", str(cv))

    exit_code = main(["run"])

    assert exit_code == 2
    assert "outreach init-db" in capsys.readouterr().err


def test_init_db_creates_the_schema(clean_env, tmp_path, capsys):
    db = tmp_path / "test.db"
    for key, value in MINIMAL_ENV.items():
        clean_env.setenv(key, value)
    clean_env.setenv("DATABASE_URL", f"sqlite:///{db}")

    assert main(["init-db"]) == 0
    assert db.exists()
    assert "Schema created" in capsys.readouterr().out


def test_suppress_records_an_address(clean_env, tmp_path, capsys):
    db = tmp_path / "test.db"
    for key, value in MINIMAL_ENV.items():
        clean_env.setenv(key, value)
    clean_env.setenv("DATABASE_URL", f"sqlite:///{db}")

    main(["init-db"])
    assert main(["suppress", "bob@corp.com", "--reason", "asked"]) == 0

    from outreach_agent.store import is_suppressed, make_engine, make_session_factory

    with make_session_factory(make_engine(f"sqlite:///{db}"))() as session:
        assert is_suppressed(session, "bob@corp.com") is True


def test_stats_runs_on_an_empty_database(clean_env, tmp_path, capsys):
    db = tmp_path / "test.db"
    for key, value in MINIMAL_ENV.items():
        clean_env.setenv(key, value)
    clean_env.setenv("DATABASE_URL", f"sqlite:///{db}")

    main(["init-db"])
    assert main(["stats"]) == 0

    out = capsys.readouterr().out
    assert "nothing yet" in out
    assert "hard ceiling 50" in out
