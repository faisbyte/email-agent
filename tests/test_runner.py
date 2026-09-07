"""Tests for the orchestration loop.

This file is where the safety properties are pinned down: the cap cannot be
raised by configuration, a dry run cannot deliver, nobody is written to twice,
and a suppression takes effect on the very next send.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from tests.fakes import (
    ExplodingSender,
    FakeAnthropic,
    FakeHttp,
    FakeResponse,
    RecordingSleep,
    person_payload,
)

from outreach_agent import store
from outreach_agent.apollo_client import ApolloClient
from outreach_agent.composer import Composer, EmailDraft
from outreach_agent.runner import (
    HARD_DAILY_CAP,
    MAX_CONSECUTIVE_FAILURES,
    discover,
    effective_daily_cap,
    run,
    run_follow_ups,
)
from outreach_agent.sender import DryRunSender, OutboundEmail, Sender, SendError, build_sender
from outreach_agent.store import OutreachStatus, ReplyClassification

CV = "Jane Engineer. Built a payments system. Python, Postgres."


# ─────────────────────────── harness ───────────────────────────


def make_config(**overrides):
    base = dict(
        daily_cap=25,
        min_delay_seconds=30,
        max_delay_seconds=180,
        gmail_address="jane@gmail.com",
        gmail_app_password="app-password",
        sender_name="Jane",
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        sender_backend="smtp",
        anthropic_model="claude-sonnet-5",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_composer(count: int = 100, *, failures: int = 0):
    """A composer whose model always returns a usable draft."""
    outputs: list = [RuntimeError("model exploded")] * failures
    outputs += [
        EmailDraft(subject="a quick question", body="Hi there,\n\nShort note.\n\nJane")
        for _ in range(count)
    ]
    client = FakeAnthropic(outputs=outputs)
    return Composer(client, model="claude-sonnet-5"), client


class RecordingSender(Sender):
    """A sender that succeeds and remembers everything it was given."""

    def __init__(self, *, fail_times: int = 0, fail_with: Exception | None = None) -> None:
        self.sent: list[OutboundEmail] = []
        self.fail_times = fail_times
        self.fail_with = fail_with or SendError("smtp said no")
        self.attempts = 0

    def send(self, message: OutboundEmail):
        self.attempts += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_with
        self.sent.append(message)
        from outreach_agent.sender import SendResult

        return SendResult(message_id=f"<sent-{len(self.sent)}@gmail.com>", delivered=True)


@pytest.fixture
def contacts(make_contact):
    return [make_contact(f"person{i}@corp{i}.com") for i in range(10)]


def do_run(session, campaign, *, config=None, sender=None, composer=None, **kwargs):
    config = config or make_config()
    sender = sender if sender is not None else RecordingSender()
    composer = composer or make_composer()[0]
    # Never fall through to the real time.sleep: a live run pauses 30-180
    # seconds between sends, which would make this suite take hours.
    kwargs.setdefault("sleep", RecordingSleep())
    return run(config, campaign, session, composer, sender, CV, **kwargs), sender


# ─────────────────────────── the hard ceiling ───────────────────────────


def test_effective_cap_clamps_a_configured_value_above_the_ceiling(caplog):
    with caplog.at_level("WARNING"):
        assert effective_daily_cap(1000) == HARD_DAILY_CAP
    assert "hard ceiling" in caplog.text


def test_effective_cap_allows_a_lower_value():
    assert effective_daily_cap(5) == 5


def test_configuration_cannot_raise_the_cap(session, campaign, make_contact):
    """The headline safety property: .env cannot turn this into a spam cannon."""
    for i in range(HARD_DAILY_CAP + 10):
        make_contact(f"p{i}@corp.com")

    config = make_config(daily_cap=1000)
    composer, _ = make_composer(count=200)
    result, sender = do_run(session, campaign, config=config, composer=composer, live=True)

    assert result.sent == HARD_DAILY_CAP
    assert len(sender.sent) == HARD_DAILY_CAP
    assert store.sent_today(session) == HARD_DAILY_CAP
    assert "daily cap" in result.stopped_reason


def test_the_cap_counts_sends_already_made_today(session, campaign, make_contact, make_sent):
    """Counted from the database, so it survives a restart mid-day."""
    for i in range(10):
        make_contact(f"p{i}@corp.com")

    burner = make_contact("burner@corp.com")
    for i in range(3):
        make_sent(burner, message_id=f"<earlier-{i}@gmail.com>")

    result, sender = do_run(session, campaign, config=make_config(daily_cap=5), live=True)

    assert result.sent == 2, "three of the five were already spent today"
    assert store.sent_today(session) == 5


def test_yesterdays_sends_do_not_count_against_todays_cap(
    session, campaign, make_contact, make_sent
):
    for i in range(5):
        make_contact(f"p{i}@corp.com")
    burner = make_contact("burner@corp.com")
    for i in range(4):
        make_sent(burner, days_ago=1.2, message_id=f"<yesterday-{i}@gmail.com>")

    result, _ = do_run(session, campaign, config=make_config(daily_cap=3), live=True)
    assert result.sent == 3


def test_the_run_limit_is_respected(session, campaign, contacts):
    result, sender = do_run(session, campaign, live=True, limit=2)

    assert result.sent == 2
    assert len(sender.sent) == 2


# ─────────────────────────── dry run ───────────────────────────


def test_a_dry_run_never_reaches_a_live_sender(session, campaign, contacts):
    """Structural, not conditional: the loop is never handed a live sender."""
    exploding = ExplodingSender()
    config = make_config()
    composer, _ = make_composer()

    sender = build_sender(config, live=False)
    assert isinstance(sender, DryRunSender)

    result = run(config, campaign, session, composer, sender, CV, live=False)

    assert result.sent == 10
    assert exploding.attempts == 0
    assert len(sender.outbox) == 10


def test_a_dry_run_records_dry_run_rows_only(session, campaign, contacts):
    do_run(session, campaign, sender=DryRunSender(), live=False)

    assert store.counts_by_status(session) == {"dry_run": 10}
    assert store.sent_today(session) == 0


def test_a_dry_run_does_not_count_as_contact(session, campaign, contacts):
    """A rehearsal must not block the real send."""
    do_run(session, campaign, sender=DryRunSender(), live=False)
    assert store.already_contacted(session, contacts[0].email) is False

    result, sender = do_run(session, campaign, live=True)
    assert result.sent == 10


def test_a_dry_run_does_not_wait(session, campaign, contacts):
    """Nobody rehearses a run that takes three hours."""
    sleep = RecordingSleep()
    do_run(session, campaign, sender=DryRunSender(), live=False, sleep=sleep)

    assert sleep.count == 0


# ─────────────────────────── deduplication ───────────────────────────


def test_an_already_contacted_person_is_skipped(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    make_sent(contact)

    result, sender = do_run(session, campaign, live=True)

    assert result.sent == 0
    assert sender.attempts == 0


def test_running_twice_writes_to_nobody_twice(session, campaign, contacts):
    first, _ = do_run(session, campaign, live=True)
    second, sender = do_run(session, campaign, live=True)

    assert first.sent == 10
    assert second.sent == 0
    assert sender.attempts == 0


def test_a_failed_send_still_counts_as_contacted(session, campaign, make_contact):
    """An SMTP error can be raised after handoff; a duplicate is worse."""
    make_contact("bob@corp.com")
    failing = RecordingSender(fail_times=1)

    do_run(session, campaign, sender=failing, live=True)
    assert store.already_contacted(session, "bob@corp.com") is True

    result, retry_sender = do_run(session, campaign, live=True)
    assert result.sent == 0
    assert retry_sender.attempts == 0


def test_a_contact_who_replied_is_skipped(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    row = make_sent(contact, message_id="<m1@gmail.com>")
    store.record_reply(
        session, message_id=row.message_id, classification=ReplyClassification.INTERESTED
    )
    session.commit()

    result, _ = do_run(session, campaign, live=True)
    assert result.sent == 0


# ─────────────────────────── suppression ───────────────────────────


def test_a_suppressed_address_is_skipped(session, campaign, make_contact):
    make_contact("bob@corp.com")
    store.add_suppression(session, "bob@corp.com", reason="opt_out")
    session.commit()

    result, sender = do_run(session, campaign, live=True)

    assert result.sent == 0
    assert sender.attempts == 0


def test_a_domain_suppression_blocks_every_address_on_it(session, campaign, make_contact):
    make_contact("a@blocked.com")
    make_contact("b@blocked.com")
    make_contact("c@allowed.com")
    store.add_suppression(session, "blocked.com", reason="manual")
    session.commit()

    result, sender = do_run(session, campaign, live=True)

    assert result.sent == 1
    assert sender.sent[0].to_email == "c@allowed.com"


def test_a_suppression_added_mid_run_takes_effect_immediately(session, campaign, make_contact):
    """The check runs before every send, not once at selection time.

    This is the case that a select-time-only check would get wrong: the reply
    poller writes a suppression while the run is paused between sends.
    """
    make_contact("first@corp.com")
    make_contact("second@corp.com")
    make_contact("third@corp.com")

    config = make_config()
    composer, _ = make_composer()
    sender = RecordingSender()

    def suppress_after_first(_seconds):
        if not any(s.to_email == "second@corp.com" for s in sender.sent):
            store.add_suppression(session, "second@corp.com", reason="opt_out")
            session.commit()

    result = run(
        config, campaign, session, composer, sender, CV, live=True, sleep=suppress_after_first
    )

    recipients = [m.to_email for m in sender.sent]
    assert "second@corp.com" not in recipients
    assert recipients == ["first@corp.com", "third@corp.com"]
    assert result.skips.get("suppressed") == 1


def test_the_per_send_gate_rejects_a_suppressed_contact(session, campaign, make_contact):
    """The gate is tested directly, because selection normally filters first.

    Both layers matter: selection is an optimisation, the gate is the guarantee.
    """
    from outreach_agent.runner import RunResult, _clear_to_send, _final_gate

    contact = make_contact("bob@corp.com")
    result = RunResult()
    assert _clear_to_send(session, contact, result) is True
    assert _final_gate(session, contact, result) is True

    store.add_suppression(session, "bob@corp.com", reason="opt_out")
    session.commit()

    assert _clear_to_send(session, contact, result) is False
    assert _final_gate(session, contact, result) is False
    assert result.skips["suppressed"] == 2


def test_the_per_send_gate_rejects_an_already_contacted_contact(
    session, campaign, make_contact, make_sent
):
    from outreach_agent.runner import RunResult, _final_gate

    contact = make_contact("bob@corp.com")
    result = RunResult()
    assert _final_gate(session, contact, result) is True

    make_sent(contact)
    assert _final_gate(session, contact, result) is False
    assert result.skips["already_contacted"] == 1


def test_the_follow_up_gate_does_not_reject_on_prior_contact(
    session, campaign, make_contact, make_sent
):
    """Prior contact is the precondition for a follow-up, not a blocker."""
    from outreach_agent.runner import RunResult, _final_gate

    contact = make_contact("bob@corp.com")
    make_sent(contact)
    result = RunResult()

    assert _final_gate(session, contact, result, check_contacted=False) is True


# ─────────────────────────── the removal line ───────────────────────────


def test_every_sent_body_offers_removal(session, campaign, contacts):
    _, sender = do_run(session, campaign, live=True)

    assert len(sender.sent) == 10
    for message in sender.sent:
        assert message.body.endswith(campaign.removal_line)


def test_a_body_without_a_removal_line_is_never_sent(session, campaign, make_contact, monkeypatch):
    """The last gate. A future change to the composer cannot slip past it."""
    make_contact("bob@corp.com")
    composer, _ = make_composer()
    sender = RecordingSender()

    from outreach_agent.composer import ComposedEmail

    monkeypatch.setattr(
        composer,
        "compose",
        lambda *a, **k: ComposedEmail(subject="s", body="No way out of this list."),
    )

    result = run(make_config(), campaign, session, composer, sender, CV, live=True)

    assert sender.attempts == 0
    assert result.sent == 0
    assert result.failed == 1
    assert "removal line" in result.errors[0]


# ─────────────────────────── the delay ───────────────────────────


def test_the_delay_falls_inside_the_configured_window(session, campaign, contacts):
    sleep = RecordingSleep()
    do_run(session, campaign, live=True, sleep=sleep, rng=random.Random(7))

    assert sleep.count > 0
    for delay in sleep.calls:
        assert 30 <= delay <= 180


def test_there_is_no_delay_after_the_last_send(session, campaign, make_contact):
    """A run must not end by sitting idle for three minutes."""
    for i in range(3):
        make_contact(f"p{i}@corp.com")
    sleep = RecordingSleep()

    result, _ = do_run(session, campaign, live=True, sleep=sleep)

    assert result.sent == 3
    assert sleep.count == 2, "one gap between each pair of sends, none trailing"


def test_a_single_send_never_waits(session, campaign, make_contact):
    make_contact("only@corp.com")
    sleep = RecordingSleep()

    do_run(session, campaign, live=True, sleep=sleep)
    assert sleep.count == 0


def test_the_delay_window_is_configurable(session, campaign, make_contact):
    for i in range(3):
        make_contact(f"p{i}@corp.com")
    sleep = RecordingSleep()

    do_run(
        session,
        campaign,
        config=make_config(min_delay_seconds=5, max_delay_seconds=6),
        live=True,
        sleep=sleep,
    )

    for delay in sleep.calls:
        assert 5 <= delay <= 6


# ─────────────────────────── failures ───────────────────────────


def test_a_send_failure_is_recorded_and_the_run_continues(session, campaign, make_contact):
    for i in range(4):
        make_contact(f"p{i}@corp.com")
    sender = RecordingSender(fail_times=1)

    result, _ = do_run(session, campaign, sender=sender, live=True)

    assert result.failed == 1
    assert result.sent == 3
    counts = store.counts_by_status(session)
    assert counts["failed"] == 1
    assert counts["sent"] == 3


def test_the_failure_reason_is_stored(session, campaign, make_contact):
    make_contact("bob@corp.com")
    sender = RecordingSender(fail_times=1, fail_with=SendError("mailbox full"))

    do_run(session, campaign, sender=sender, live=True)

    row = session.scalars(store.select(store.Outreach)).first()
    assert row.status == OutreachStatus.FAILED.value
    assert "mailbox full" in row.error


def test_consecutive_failures_abort_the_run(session, campaign, make_contact):
    """A wrong app password should cost five attempts, not the whole list."""
    for i in range(30):
        make_contact(f"p{i}@corp.com")
    sender = RecordingSender(fail_times=99)

    result, _ = do_run(session, campaign, sender=sender, live=True)

    assert sender.attempts == MAX_CONSECUTIVE_FAILURES
    assert result.failed == MAX_CONSECUTIVE_FAILURES
    assert "consecutive failures" in result.stopped_reason


def test_the_failure_streak_resets_after_a_success(session, campaign, make_contact):
    for i in range(12):
        make_contact(f"p{i}@corp.com")

    class Alternating(RecordingSender):
        def send(self, message):
            self.attempts += 1
            if self.attempts % 2 == 1:
                raise SendError("transient")
            self.sent.append(message)
            from outreach_agent.sender import SendResult

            return SendResult(message_id=f"<x{self.attempts}@g.com>", delivered=True)

    sender = Alternating()
    result, _ = do_run(session, campaign, sender=sender, live=True)

    assert result.stopped_reason is None or "consecutive" not in result.stopped_reason
    assert result.sent == 6


def test_a_composition_failure_is_recorded_and_skipped(session, campaign, make_contact):
    for i in range(3):
        make_contact(f"p{i}@corp.com")
    composer, _ = make_composer(count=10, failures=1)

    result, sender = do_run(session, campaign, composer=composer, live=True)

    assert result.failed == 1
    assert result.sent == 2
    assert len(sender.sent) == 2


def test_an_unexpected_sender_fault_does_not_kill_the_run(session, campaign, make_contact):
    for i in range(3):
        make_contact(f"p{i}@corp.com")
    sender = RecordingSender(fail_times=1, fail_with=ValueError("something odd"))

    result, _ = do_run(session, campaign, sender=sender, live=True)

    assert result.failed == 1
    assert result.sent == 2


# ─────────────────────────── run bookkeeping ───────────────────────────


def test_a_run_row_is_written_and_closed(session, campaign, contacts):
    result, _ = do_run(session, campaign, live=True)

    run_row = session.get(store.Run, result.run_id)
    assert run_row.finished_at is not None
    assert run_row.sent_count == result.sent
    assert run_row.mode == "live"
    assert run_row.campaign == campaign.slug


def test_the_run_is_closed_even_when_the_loop_dies(session, campaign, make_contact, monkeypatch):
    """Sends already made are real and must be recorded."""
    make_contact("bob@corp.com")
    composer, _ = make_composer()
    sender = RecordingSender()

    monkeypatch.setattr(
        "outreach_agent.runner.candidates",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("database went away")),
    )

    with pytest.raises(RuntimeError):
        run(make_config(), campaign, session, composer, sender, CV, live=True)

    run_row = session.scalars(store.select(store.Run)).first()
    assert run_row.finished_at is not None


def test_a_dry_run_is_labelled_as_one(session, campaign, contacts):
    result, _ = do_run(session, campaign, sender=DryRunSender(), live=False)
    assert session.get(store.Run, result.run_id).mode == "dry_run"


# ─────────────────────────── follow-ups ───────────────────────────


def test_follow_ups_actually_send(session, campaign, make_contact, make_sent):
    """The regression test for the inverted check.

    Every follow-up target is already contacted. If this pass inherited
    already_contacted(), it would skip 100% of its candidates and silently do
    nothing forever.
    """
    contact = make_contact("bob@corp.com")
    make_sent(contact, days_ago=10, message_id="<first@gmail.com>")

    composer, _ = make_composer()
    sender = RecordingSender()
    result = run_follow_ups(
        make_config(), campaign, session, composer, sender, CV, live=True, sleep=RecordingSleep()
    )

    assert result.sent == 1, "a contacted, unreplied person must receive a follow-up"
    assert len(sender.sent) == 1


def test_a_follow_up_writes_a_new_row_and_leaves_the_original_alone(
    session, campaign, make_contact, make_sent
):
    contact = make_contact("bob@corp.com")
    original = make_sent(contact, days_ago=10, message_id="<first@gmail.com>")
    original_subject = original.subject

    composer, _ = make_composer()
    run_follow_ups(
        make_config(), campaign, session, composer, RecordingSender(), CV,
        live=True, sleep=RecordingSleep()
    )

    rows = list(session.scalars(store.select(store.Outreach).order_by(store.Outreach.id)))
    assert len(rows) == 2
    assert rows[0].subject == original_subject, "sent history is never rewritten"
    assert rows[0].follow_up_count == 0
    assert rows[1].follow_up_count == 1
    assert rows[1].message_id != rows[0].message_id


def test_a_follow_up_is_threaded_onto_the_previous_message(
    session, campaign, make_contact, make_sent
):
    contact = make_contact("bob@corp.com")
    make_sent(contact, days_ago=10, message_id="<first@gmail.com>")

    composer, _ = make_composer()
    sender = RecordingSender()
    run_follow_ups(
        make_config(), campaign, session, composer, sender, CV, live=True, sleep=RecordingSleep()
    )

    assert sender.sent[0].in_reply_to == "<first@gmail.com>"
    assert "<first@gmail.com>" in sender.sent[0].references


def test_a_human_reply_stops_all_follow_ups(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    row = make_sent(contact, days_ago=10, message_id="<first@gmail.com>")
    store.record_reply(
        session, message_id=row.message_id, classification=ReplyClassification.REJECTION
    )
    session.commit()

    composer, _ = make_composer()
    sender = RecordingSender()
    result = run_follow_ups(
        make_config(), campaign, session, composer, sender, CV, live=True, sleep=RecordingSleep()
    )

    assert result.sent == 0
    assert sender.attempts == 0


def test_an_auto_reply_does_not_stop_follow_ups(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    row = make_sent(contact, days_ago=10, message_id="<first@gmail.com>")
    store.record_reply(
        session, message_id=row.message_id, classification=ReplyClassification.AUTO_REPLY
    )
    session.commit()

    composer, _ = make_composer()
    result = run_follow_ups(
        make_config(), campaign, session, composer, RecordingSender(), CV,
        live=True, sleep=RecordingSleep()
    )

    assert result.sent == 1


def test_follow_ups_respect_max_follow_ups(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    make_sent(contact, days_ago=20, follow_up_count=0, message_id="<a@gmail.com>")
    make_sent(contact, days_ago=10, follow_up_count=1, message_id="<b@gmail.com>")

    composer, _ = make_composer()
    result = run_follow_ups(
        make_config(), campaign, session, composer, RecordingSender(), CV,
        live=True, sleep=RecordingSleep()
    )

    assert campaign.follow_up.max_follow_ups == 1
    assert result.sent == 0


def test_follow_ups_respect_the_waiting_period(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    make_sent(contact, days_ago=1, message_id="<recent@gmail.com>")

    composer, _ = make_composer()
    result = run_follow_ups(
        make_config(), campaign, session, composer, RecordingSender(), CV,
        live=True, sleep=RecordingSleep()
    )

    assert result.sent == 0


def test_a_suppressed_contact_gets_no_follow_up(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    make_sent(contact, days_ago=10, message_id="<first@gmail.com>")
    store.add_suppression(session, "bob@corp.com", reason="opt_out")
    session.commit()

    composer, _ = make_composer()
    sender = RecordingSender()
    result = run_follow_ups(
        make_config(), campaign, session, composer, sender, CV, live=True, sleep=RecordingSleep()
    )

    assert result.sent == 0
    assert sender.attempts == 0


def test_follow_ups_obey_the_daily_cap(session, campaign, make_contact, make_sent):
    for i in range(6):
        contact = make_contact(f"p{i}@corp.com")
        make_sent(contact, days_ago=10, message_id=f"<old-{i}@gmail.com>")

    composer, _ = make_composer()
    result = run_follow_ups(
        make_config(daily_cap=2),
        campaign,
        session,
        composer,
        RecordingSender(),
        CV,
        live=True,
        sleep=RecordingSleep(),
    )

    assert result.sent == 2


def test_a_dry_run_follow_up_delivers_nothing(session, campaign, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    make_sent(contact, days_ago=10, message_id="<first@gmail.com>")

    composer, _ = make_composer()
    sender = DryRunSender()
    result = run_follow_ups(
        make_config(), campaign, session, composer, sender, CV, live=False, sleep=RecordingSleep()
    )

    assert result.sent == 1
    assert len(sender.outbox) == 1
    assert store.sent_today(session) == 0


# ─────────────────────────── discovery ───────────────────────────


def make_apollo(responses, *, budget: int = 50):
    return ApolloClient(
        "key",
        http=FakeHttp(responses),
        sleep=RecordingSleep(),
        rng=random.Random(1),
        enrichment_budget=budget,
    )


def test_discover_stores_people_with_usable_emails(session, campaign):
    apollo = make_apollo(
        [FakeResponse(200, {"people": [person_payload("p1", email="bob@corp.com")]})]
    )
    stats = discover(make_config(), campaign, session, apollo, limit=1)

    assert stats["stored"] == 1
    contact = session.scalars(store.select(store.Contact)).first()
    assert contact.email == "bob@corp.com"
    assert contact.organization.domain == "corp.com"


def test_discover_never_spends_a_credit_on_someone_already_contacted(
    session, campaign, make_contact, make_sent
):
    """The most avoidable waste in the system, and it gets a test."""
    existing = make_contact("bob@corp.com")
    make_sent(existing)

    apollo = make_apollo(
        [
            FakeResponse(200, {"people": [person_payload("p1", email="bob@corp.com")]}),
            FakeResponse(200, {"people": []}),
        ]
    )
    stats = discover(make_config(), campaign, session, apollo, limit=5)

    assert apollo.credits_used == 0
    assert stats["skipped"] == 1


def test_discover_never_spends_a_credit_on_a_suppressed_address(session, campaign):
    store.add_suppression(session, "corp.com", reason="manual")
    session.commit()

    apollo = make_apollo(
        [
            FakeResponse(200, {"people": [person_payload("p1", email="bob@corp.com")]}),
            FakeResponse(200, {"people": []}),
        ]
    )
    discover(make_config(), campaign, session, apollo, limit=5)

    assert apollo.credits_used == 0


def test_discover_enriches_a_locked_email(session, campaign):
    locked = person_payload("p1", email="email_not_unlocked@domain.com")
    apollo = make_apollo(
        [
            FakeResponse(200, {"people": [locked]}),
            FakeResponse(200, {"person": person_payload("p1", email="bob@corp.com")}),
        ]
    )
    stats = discover(make_config(), campaign, session, apollo, limit=1)

    assert apollo.credits_used == 1
    assert stats["stored"] == 1


def test_discover_stops_cleanly_when_credits_run_out(session, campaign):
    locked = person_payload("p1", email="email_not_unlocked@domain.com")
    apollo = make_apollo(
        [
            FakeResponse(200, {"people": [locked]}),
            FakeResponse(402, {"error": "no credits"}),
        ]
    )
    stats = discover(make_config(), campaign, session, apollo, limit=5)

    assert stats.get("stopped_out_of_credits") == 1
    assert stats["stored"] == 0


def test_discover_respects_the_enrichment_budget(session, campaign):
    locked = [person_payload(f"p{i}", email="email_not_unlocked@domain.com") for i in range(5)]
    apollo = make_apollo(
        [FakeResponse(200, {"people": locked})]
        + [FakeResponse(200, {"person": person_payload(f"px{i}", email=f"x{i}@corp.com")})
           for i in range(5)],
        budget=2,
    )
    stats = discover(make_config(), campaign, session, apollo, limit=5)

    assert apollo.credits_used == 2
    assert stats.get("stopped_out_of_credits") == 1


def test_discover_sends_nothing(session, campaign):
    """It is a search command. It has no sender and cannot acquire one."""
    apollo = make_apollo(
        [FakeResponse(200, {"people": [person_payload("p1", email="bob@corp.com")]})]
    )
    discover(make_config(), campaign, session, apollo, limit=1)

    assert store.counts_by_status(session) == {}
