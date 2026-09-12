"""Shared fixtures.

Two guarantees enforced here rather than remembered:

1. Every test runs against a fresh in-memory SQLite database.
2. No test can open a network connection. If one tries, it fails loudly
   instead of quietly making a real call to Apollo, Anthropic, or Gmail.
"""

from __future__ import annotations

import socket
import time
from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from outreach_agent import store
from outreach_agent.campaigns import ApolloFilters, Campaign, FollowUpPolicy


class NetworkAccessDenied(RuntimeError):
    """Raised when a test tries to open a socket."""


class RealSleepDenied(RuntimeError):
    """Raised when a test would actually wait."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make outbound connections impossible for the whole suite.

    This is the mechanical guarantee behind "no real API calls". It does not
    depend on anyone remembering to mock the right client: if a code path
    reaches for a socket, the test fails with a clear message.

    Sockets may still be constructed — some libraries do that at import time —
    but nothing can connect.
    """

    def deny(*args: object, **kwargs: object) -> None:
        raise NetworkAccessDenied(
            "A test attempted to open a network connection. Every external "
            "service (Apollo, Anthropic, SMTP, IMAP) must be injected as a fake."
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly instead of actually waiting.

    The runner pauses 30-180 seconds between live sends. A test that forgets to
    inject a fake sleep would otherwise pass while taking hours, which is
    indistinguishable from a hang. Short sleeps are left alone so unrelated
    library internals are not disturbed.
    """
    real_sleep = time.sleep

    def guarded(seconds: float) -> None:
        if seconds >= 1.0:
            raise RealSleepDenied(
                f"A test tried to sleep for {seconds:.1f}s. Inject a fake sleep "
                "(tests.fakes.RecordingSleep) instead of using the real clock."
            )
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", guarded)


@pytest.fixture
def engine():
    """A private in-memory database, shared across sessions within one test.

    StaticPool keeps every session on the same connection, which is what makes
    ``sqlite://`` behave like a real database for the duration of a test.
    """
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    store.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return store.make_session_factory(engine)


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        yield s


@pytest.fixture
def campaign() -> Campaign:
    """A minimal campaign, independent of the shipped TOML files."""
    return Campaign(
        slug="test_campaign",
        name="Test Campaign",
        sender_persona="An engineer who writes plainly.",
        goal="Start a short conversation.",
        tone=["Direct.", "No markdown."],
        subject_guidance="Under nine words.",
        max_words=120,
        max_sentences=5,
        removal_line="Reply 'no thanks' and I won't write again.",
        apollo=ApolloFilters(person_titles=["Technical Recruiter"]),
        follow_up=FollowUpPolicy(max_follow_ups=1, days_between=5, goal="One short nudge."),
        source_path=None,
    )


# ─────────────────────────── data helpers ───────────────────────────


@pytest.fixture
def make_contact(session):
    """Create a contact with sensible defaults. Returns a factory."""
    counter = {"n": 0}

    def _make(email: str | None = None, **kwargs) -> store.Contact:
        counter["n"] += 1
        n = counter["n"]
        contact = store.upsert_contact(
            session,
            email=email or f"person{n}@example.com",
            name=kwargs.pop("name", f"Person {n}"),
            title=kwargs.pop("title", "Technical Recruiter"),
            **kwargs,
        )
        session.commit()
        return contact

    return _make


@pytest.fixture
def make_sent(session):
    """Record a delivered message, optionally backdated.

    Backdating goes through the ORM after the write because ``sent_at`` is
    stamped by record_send; tests that exercise the daily cap and follow-up
    windows need to place messages in the past.
    """

    def _make(
        contact: store.Contact,
        *,
        days_ago: float = 0,
        message_id: str | None = None,
        follow_up_count: int = 0,
        campaign: str = "test_campaign",
        status: store.OutreachStatus = store.OutreachStatus.SENT,
    ) -> store.Outreach:
        row = store.record_send(
            session,
            contact=contact,
            campaign=campaign,
            subject="Subject",
            body="Body",
            status=status,
            message_id=message_id or f"<msg-{contact.id}-{follow_up_count}@test>",
            follow_up_count=follow_up_count,
        )
        if days_ago and row.sent_at is not None:
            row.sent_at = store.utcnow() - timedelta(days=days_ago)
        session.commit()
        return row

    return _make
