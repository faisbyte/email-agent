"""Environment loading and validation.

The contract of this module: if it returns a Config, the run has everything it
needs. Every problem is found here, at startup, and reported together. Nothing
in this codebase discovers a missing key at contact seventeen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Models documented as supporting structured outputs. Both LLM calls in this
# project (compose, classify) depend on structured outputs, so a model outside
# this set cannot work and is rejected at startup rather than at request time.
#
# Checked 2026-09-07 against:
#   https://platform.claude.com/docs/en/build-with-claude/structured-outputs
# Refreshing this is a one-line edit.
STRUCTURED_OUTPUT_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
    }
)

SENDER_BACKENDS = frozenset({"smtp", "gmail_api"})
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_DATABASE_URL = "sqlite:///outreach.db"


class ConfigError(Exception):
    """Raised when the environment is not fit to run.

    Carries every problem found, not just the first one, so a misconfigured
    .env is fixed in a single pass.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        joined = "\n".join(f"  - {p}" for p in problems)
        super().__init__(
            f"Configuration is not usable ({len(problems)} problem(s)):\n{joined}\n\n"
            "Copy .env.example to .env and fill in the values."
        )


@dataclass(frozen=True)
class Config:
    """Validated runtime configuration. Constructed only by load_config()."""

    anthropic_api_key: str
    apollo_api_key: str
    anthropic_model: str
    database_url: str

    gmail_address: str
    gmail_app_password: str
    sender_name: str

    daily_cap: int
    min_delay_seconds: int
    max_delay_seconds: int
    apollo_enrichment_budget: int

    campaign_path: Path
    cv_path: Path

    smtp_host: str
    smtp_port: int
    imap_host: str
    imap_port: int
    imap_folder: str
    sender_backend: str

    log_level: str

    @property
    def unsubscribe_mailto(self) -> str:
        """Address a recipient replies to in order to be removed."""
        return self.gmail_address


class _Collector:
    """Accumulates problems so all of them are reported at once."""

    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.problems: list[str] = []

    def required_str(self, key: str, *, why: str) -> str:
        value = (self.env.get(key) or "").strip()
        if not value:
            self.problems.append(f"{key} is missing or empty — needed {why}")
        return value

    def optional_str(self, key: str, default: str) -> str:
        value = (self.env.get(key) or "").strip()
        return value or default

    def positive_int(self, key: str, default: int) -> int:
        raw = (self.env.get(key) or "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            self.problems.append(f"{key} must be a whole number, got {raw!r}")
            return default
        if value <= 0:
            self.problems.append(f"{key} must be greater than zero, got {value}")
            return default
        return value

    def one_of(self, key: str, default: str, allowed: frozenset[str], *, label: str) -> str:
        value = self.optional_str(key, default)
        if value not in allowed:
            options = ", ".join(sorted(allowed))
            self.problems.append(f"{key}={value!r} is not a valid {label}. Options: {options}")
        return value


def load_config(
    *,
    require_send: bool = False,
    env: dict[str, str] | None = None,
    load_dotenv_file: bool = True,
) -> Config:
    """Read and validate configuration.

    Args:
        require_send: True when the caller intends to actually deliver mail
            (`--live`). Gmail credentials are only demanded in that case, so a
            dry run works with just the Anthropic and Apollo keys.
        env: Environment mapping to read. Defaults to os.environ. Injected by
            tests so they never depend on the developer's real shell.
        load_dotenv_file: Load a .env file if present. The real environment
            always wins over the file.

    Raises:
        ConfigError: with every problem found, never just the first.
    """
    if env is None:
        if load_dotenv_file:
            load_dotenv(override=False)
        env = dict(os.environ)

    c = _Collector(env)

    anthropic_api_key = c.required_str("ANTHROPIC_API_KEY", why="to compose and classify email")
    apollo_api_key = c.required_str("APOLLO_API_KEY", why="to search for and enrich contacts")

    model = c.optional_str("ANTHROPIC_MODEL", DEFAULT_MODEL)
    if model not in STRUCTURED_OUTPUT_MODELS:
        c.problems.append(
            f"ANTHROPIC_MODEL={model!r} does not support structured outputs, which both the "
            "composer and the reply classifier require. Supported: "
            + ", ".join(sorted(STRUCTURED_OUTPUT_MODELS))
        )

    database_url = c.optional_str("DATABASE_URL", DEFAULT_DATABASE_URL)
    if "://" not in database_url:
        c.problems.append(
            f"DATABASE_URL={database_url!r} is not a SQLAlchemy URL. "
            f"Example: {DEFAULT_DATABASE_URL}"
        )

    # Sending credentials: only mandatory when the caller means to send.
    if require_send:
        gmail_address = c.required_str("GMAIL_ADDRESS", why="as the From address for live sending")
        gmail_app_password = c.required_str(
            "GMAIL_APP_PASSWORD", why="to authenticate with Gmail for live sending"
        )
        sender_name = c.required_str("SENDER_NAME", why="as the display name for live sending")
    else:
        gmail_address = c.optional_str("GMAIL_ADDRESS", "")
        gmail_app_password = c.optional_str("GMAIL_APP_PASSWORD", "")
        sender_name = c.optional_str("SENDER_NAME", "")

    if gmail_address and "@" not in gmail_address:
        c.problems.append(f"GMAIL_ADDRESS={gmail_address!r} is not an email address")

    daily_cap = c.positive_int("DAILY_CAP", 25)
    min_delay = c.positive_int("MIN_DELAY_SECONDS", 30)
    max_delay = c.positive_int("MAX_DELAY_SECONDS", 180)
    if min_delay > max_delay:
        c.problems.append(
            f"MIN_DELAY_SECONDS ({min_delay}) is greater than MAX_DELAY_SECONDS ({max_delay})"
        )

    enrichment_budget = c.positive_int("APOLLO_ENRICHMENT_BUDGET", 50)

    campaign_path = Path(c.optional_str("CAMPAIGN", "campaigns/job_search.toml"))
    cv_path = Path(c.optional_str("CV_PATH", "data/cv.txt"))

    smtp_host = c.optional_str("SMTP_HOST", "smtp.gmail.com")
    smtp_port = c.positive_int("SMTP_PORT", 465)
    imap_host = c.optional_str("IMAP_HOST", "imap.gmail.com")
    imap_port = c.positive_int("IMAP_PORT", 993)
    imap_folder = c.optional_str("IMAP_FOLDER", "INBOX")
    sender_backend = c.one_of("SENDER_BACKEND", "smtp", SENDER_BACKENDS, label="sender backend")
    log_level = c.one_of("LOG_LEVEL", "INFO", LOG_LEVELS, label="log level")

    if c.problems:
        raise ConfigError(c.problems)

    return Config(
        anthropic_api_key=anthropic_api_key,
        apollo_api_key=apollo_api_key,
        anthropic_model=model,
        database_url=database_url,
        gmail_address=gmail_address,
        gmail_app_password=gmail_app_password,
        sender_name=sender_name,
        daily_cap=daily_cap,
        min_delay_seconds=min_delay,
        max_delay_seconds=max_delay,
        apollo_enrichment_budget=enrichment_budget,
        campaign_path=campaign_path,
        cv_path=cv_path,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        imap_host=imap_host,
        imap_port=imap_port,
        imap_folder=imap_folder,
        sender_backend=sender_backend,
        log_level=log_level,
    )


def load_cv(path: Path | str) -> str:
    """Read the plain-text CV the composer is allowed to draw facts from.

    Read at startup for the same reason as everything else here: an unreadable
    CV should stop the run before it begins, not halfway through a list.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ConfigError(
            [
                f"CV file not found at {p}. Set CV_PATH, or copy data/cv.example.txt "
                f"to {p} and put your real background in it."
            ]
        ) from exc
    except OSError as exc:
        raise ConfigError([f"CV file at {p} could not be read: {exc}"]) from exc

    if not text:
        raise ConfigError([f"CV file at {p} is empty. Claude may only use facts it finds here."])
    return text
