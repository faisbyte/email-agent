"""Tests for the persistence layer.

The two invariants — unique email, suppression check — get the most attention
here, because everything upstream trusts them.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from outreach_agent import store
from outreach_agent.store import (
    Contact,
    OutreachStatus,
    ReplyClassification,
    RunMode,
    Suppression,
)

# ─────────────────────────── normalisation ───────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bob@Corp.com", "bob@corp.com"),
        ("  bob@corp.com  ", "bob@corp.com"),
        ("BOB@CORP.COM", "bob@corp.com"),
        ("", ""),
    ],
)
def test_normalize_email(raw, expected):
    assert store.normalize_email(raw) == expected


def test_email_domain():
    assert store.email_domain("Bob@Corp.com") == "corp.com"
    assert store.email_domain("nonsense") == ""


# ─────────────────────────── unique email ───────────────────────────


def test_upsert_contact_returns_existing_row_for_same_email(session):
    first = store.upsert_contact(session, email="bob@corp.com", name="Bob")
    session.commit()
    second = store.upsert_contact(session, email="bob@corp.com", name="Robert")

    assert second.id == first.id
    assert session.scalar(select(store.func.count()).select_from(Contact)) == 1


def test_casing_and_whitespace_cannot_create_a_duplicate(session):
    store.upsert_contact(session, email="bob@corp.com")
    session.commit()
    store.upsert_contact(session, email="  BOB@Corp.COM ")
    session.commit()

    assert session.scalar(select(store.func.count()).select_from(Contact)) == 1


def test_unique_constraint_is_enforced_by_the_database_not_just_the_helper(session):
    """Bypassing upsert_contact must still fail. The constraint is the guarantee."""
    from sqlalchemy.exc import IntegrityError

    session.add(Contact(email="bob@corp.com"))
    session.commit()
    session.add(Contact(email="bob@corp.com"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_upsert_contact_survives_a_duplicate_and_leaves_the_transaction_usable(
    session, monkeypatch
):
    """The savepoint test.

    A duplicate insert must not poison the enclosing transaction. If it did,
    every later write in the run would fail — which is exactly the failure mode
    a bare ``try/except IntegrityError`` on the session introduces.

    The race is forced by blinding the pre-flight SELECT once, so the INSERT
    actually collides with the unique constraint rather than being avoided.
    """
    first = store.upsert_contact(session, email="bob@corp.com")
    session.commit()

    real_scalar = session.scalar
    calls = {"n": 0}

    def blind_once(statement, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # the row exists, but this lookup does not see it
        return real_scalar(statement, *args, **kwargs)

    monkeypatch.setattr(session, "scalar", blind_once)
    recovered = store.upsert_contact(session, email="bob@corp.com", name="Robert")
    monkeypatch.undo()

    assert recovered.id == first.id, "the collision must resolve to the existing row"

    # The real assertion: the session is still usable afterwards.
    later = store.upsert_contact(session, email="someone@else.com", name="Someone")
    session.commit()

    assert later.id is not None
    assert session.scalar(select(store.func.count()).select_from(Contact)) == 2


def test_upsert_contact_rejects_a_malformed_address(session):
    with pytest.raises(ValueError):
        store.upsert_contact(session, email="not-an-email")


def test_upsert_contact_backfills_but_never_overwrites(session):
    store.upsert_contact(session, email="bob@corp.com", name="Bob", title=None)
    session.commit()
    updated = store.upsert_contact(
        session, email="bob@corp.com", name="Robert", title="Recruiter"
    )
    session.commit()

    assert updated.name == "Bob"          # known value kept
    assert updated.title == "Recruiter"   # unknown value filled in


# ─────────────────────────── suppression ───────────────────────────


def test_is_suppressed_matches_the_exact_address(session, make_contact):
    contact = make_contact("bob@corp.com")
    assert store.is_suppressed(session, contact.email) is False

    store.add_suppression(session, "bob@corp.com", reason="opt_out")
    session.commit()
    assert store.is_suppressed(session, "bob@corp.com") is True


def test_is_suppressed_matches_a_whole_domain(session):
    store.add_suppression(session, "corp.com", reason="manual")
    session.commit()

    assert store.is_suppressed(session, "bob@corp.com") is True
    assert store.is_suppressed(session, "anyone@corp.com") is True
    assert store.is_suppressed(session, "bob@other.com") is False


def test_is_suppressed_ignores_casing(session):
    store.add_suppression(session, "Bob@Corp.com")
    session.commit()
    assert store.is_suppressed(session, "BOB@CORP.COM") is True


def test_an_unreadable_address_is_treated_as_suppressed(session):
    """Fail closed. An address we cannot parse is one we must not mail."""
    assert store.is_suppressed(session, "") is True


def test_add_suppression_is_idempotent(session):
    first = store.add_suppression(session, "bob@corp.com", reason="opt_out")
    session.commit()
    second = store.add_suppression(session, "BOB@corp.com", reason="again")
    session.commit()

    assert second.id == first.id
    assert second.reason == "opt_out"  # first reason wins
    assert session.scalar(select(store.func.count()).select_from(Suppression)) == 1


# ─────────────────────────── prior contact ───────────────────────────


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (OutreachStatus.SENT, True),
        (OutreachStatus.FAILED, True),
        (OutreachStatus.BOUNCED, True),
        (OutreachStatus.DRY_RUN, False),
    ],
)
def test_already_contacted_by_status(session, make_contact, status, expected):
    """FAILED counts as contacted: an SMTP error can be raised after handoff,
    and a duplicate email is worse than a missed one."""
    contact = make_contact("bob@corp.com")
    store.record_send(
        session,
        contact=contact,
        campaign="c",
        subject="s",
        body="b",
        status=status,
    )
    session.commit()

    assert store.already_contacted(session, "bob@corp.com") is expected


def test_already_contacted_normalises_the_address(session, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    make_sent(contact)
    assert store.already_contacted(session, "BOB@Corp.com ") is True


def test_already_contacted_is_global_across_campaigns(session, make_contact, make_sent):
    """One address, one email, ever — regardless of which campaign sent it."""
    contact = make_contact("bob@corp.com")
    make_sent(contact, campaign="job_search")
    assert store.already_contacted(session, "bob@corp.com") is True


# ─────────────────────────── daily cap counting ───────────────────────────


def test_sent_today_counts_only_sent_rows(session, make_contact):
    contact = make_contact()
    for status in (OutreachStatus.SENT, OutreachStatus.DRY_RUN, OutreachStatus.FAILED):
        store.record_send(
            session, contact=contact, campaign="c", subject="s", body="b", status=status
        )
    session.commit()

    assert store.sent_today(session) == 1


def test_sent_today_excludes_yesterday(session, make_contact, make_sent):
    contact = make_contact()
    make_sent(contact, days_ago=1.5, message_id="<old@test>")
    make_sent(contact, message_id="<new@test>")

    assert store.sent_today(session) == 1


def test_sent_today_handles_naive_datetimes_from_sqlite(session, make_contact, make_sent):
    """SQLite returns naive datetimes.

    If the comparison is done against a naive value without attaching UTC, the
    daily cap resets at the wrong hour — or raises. A message sent one minute
    after midnight UTC must be counted today.
    """
    contact = make_contact()
    row = make_sent(contact)
    row.sent_at = store.utc_day_start() + timedelta(minutes=1)
    session.commit()

    # expire_on_commit is off, so force a genuine reload from the database.
    session.expire_all()
    stored = session.scalar(select(store.Outreach).where(store.Outreach.id == row.id))
    assert stored.sent_at.tzinfo is None, "SQLite is expected to return naive datetimes"
    assert store.as_utc(stored.sent_at).tzinfo is not None

    assert store.sent_today(session) == 1


def test_sent_today_excludes_a_message_one_minute_before_midnight(
    session, make_contact, make_sent
):
    contact = make_contact()
    row = make_sent(contact)
    row.sent_at = store.utc_day_start() - timedelta(minutes=1)
    session.commit()

    assert store.sent_today(session) == 0


# ─────────────────────────── replies ───────────────────────────


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        (ReplyClassification.INTERESTED, True),
        (ReplyClassification.REJECTION, True),
        (ReplyClassification.OPT_OUT, True),
        (ReplyClassification.AUTO_REPLY, False),
    ],
)
def test_has_human_reply(session, make_contact, make_sent, classification, expected):
    """An out-of-office is not a human saying no, so it must not stop follow-ups."""
    contact = make_contact()
    row = make_sent(contact, message_id="<m1@test>")
    store.record_reply(
        session, message_id=row.message_id, classification=classification, body="text"
    )
    session.commit()

    assert store.has_human_reply(session, contact.id) is expected


def test_record_reply_stores_the_body(session, make_contact, make_sent):
    contact = make_contact()
    row = make_sent(contact, message_id="<m1@test>")
    store.record_reply(
        session,
        message_id="<m1@test>",
        classification=ReplyClassification.INTERESTED,
        body="Sure, send your CV.",
    )
    session.commit()

    assert row.reply_body == "Sure, send your CV."
    assert row.replied_at is not None


def test_record_reply_ignores_unknown_message_ids(session):
    """Unrelated mail in the inbox is normal and must not raise."""
    assert store.record_reply(
        session, message_id="<never-sent@test>", classification=ReplyClassification.INTERESTED
    ) is None


def test_record_bounce_sets_the_status(session, make_contact, make_sent):
    contact = make_contact()
    row = make_sent(contact, message_id="<m1@test>")

    bounced = store.record_bounce(session, message_id="<m1@test>")
    session.commit()

    assert bounced is not None
    assert row.status == OutreachStatus.BOUNCED.value
    assert store.already_contacted(session, contact.email) is True


# ─────────────────────────── candidate selection ───────────────────────────


def test_candidates_excludes_contacted_and_suppressed(session, make_contact, make_sent):
    fresh = make_contact("fresh@corp.com")
    contacted = make_contact("contacted@corp.com")
    make_contact("suppressed@corp.com")

    make_sent(contacted)
    store.add_suppression(session, "suppressed@corp.com", reason="opt_out")
    session.commit()

    picked = store.candidates(session, limit=10)
    assert [c.id for c in picked] == [fresh.id]


def test_candidates_excludes_a_domain_suppression(session, make_contact):
    make_contact("a@blocked.com")
    keep = make_contact("b@allowed.com")
    store.add_suppression(session, "blocked.com", reason="manual")
    session.commit()

    picked = store.candidates(session, limit=10)
    assert [c.id for c in picked] == [keep.id]


def test_candidates_includes_a_contact_with_only_a_dry_run_row(session, make_contact):
    """A rehearsal is not contact."""
    contact = make_contact()
    store.record_send(
        session,
        contact=contact,
        campaign="c",
        subject="s",
        body="b",
        status=OutreachStatus.DRY_RUN,
    )
    session.commit()

    assert [c.id for c in store.candidates(session, limit=10)] == [contact.id]


def test_candidates_respects_the_limit(session, make_contact):
    for _ in range(5):
        make_contact()
    assert len(store.candidates(session, limit=2)) == 2
    assert store.candidates(session, limit=0) == []


# ─────────────────────────── follow-up selection ───────────────────────────


def test_follow_up_candidates_requires_a_sent_row(session, make_contact, make_sent):
    """The regression test for the inverted check.

    follow_up_candidates selects on the presence of a sent message, not its
    absence. A never-contacted person is not a follow-up target; a contacted
    one is.
    """
    make_contact("never@corp.com")  # no outreach at all
    contacted = make_contact("contacted@corp.com")
    make_sent(contacted, days_ago=10)

    due = store.follow_up_candidates(session, limit=10, max_follow_ups=1, days_between=5)
    assert [row.contact_id for row in due] == [contacted.id]


def test_follow_up_candidates_respects_the_waiting_period(session, make_contact, make_sent):
    recent = make_contact("recent@corp.com")
    make_sent(recent, days_ago=1)

    assert store.follow_up_candidates(session, limit=10, max_follow_ups=1, days_between=5) == []


def test_follow_up_candidates_respects_max_follow_ups(session, make_contact, make_sent):
    contact = make_contact()
    make_sent(contact, days_ago=20, follow_up_count=0, message_id="<m0@test>")
    make_sent(contact, days_ago=10, follow_up_count=1, message_id="<m1@test>")

    assert store.follow_up_candidates(session, limit=10, max_follow_ups=1, days_between=5) == []

    due = store.follow_up_candidates(session, limit=10, max_follow_ups=2, days_between=5)
    assert [row.follow_up_count for row in due] == [1]


def test_follow_up_candidates_excludes_human_replies(session, make_contact, make_sent):
    replied = make_contact("replied@corp.com")
    row = make_sent(replied, days_ago=10, message_id="<r1@test>")
    store.record_reply(
        session, message_id=row.message_id, classification=ReplyClassification.REJECTION
    )
    session.commit()

    assert store.follow_up_candidates(session, limit=10, max_follow_ups=2, days_between=5) == []


def test_follow_up_candidates_keeps_going_after_an_auto_reply(session, make_contact, make_sent):
    contact = make_contact()
    row = make_sent(contact, days_ago=10, message_id="<a1@test>")
    store.record_reply(
        session, message_id=row.message_id, classification=ReplyClassification.AUTO_REPLY
    )
    session.commit()

    due = store.follow_up_candidates(session, limit=10, max_follow_ups=2, days_between=5)
    assert [r.contact_id for r in due] == [contact.id]


def test_follow_up_candidates_excludes_suppressed_and_bounced(session, make_contact, make_sent):
    suppressed = make_contact("gone@corp.com")
    make_sent(suppressed, days_ago=10, message_id="<s1@test>")
    store.add_suppression(session, "gone@corp.com", reason="opt_out")

    bounced = make_contact("dead@corp.com")
    row = make_sent(bounced, days_ago=10, message_id="<b1@test>")
    store.record_bounce(session, message_id=row.message_id)
    session.commit()

    assert store.follow_up_candidates(session, limit=10, max_follow_ups=2, days_between=5) == []


# ─────────────────────────── organizations and runs ───────────────────────────


def test_upsert_organization_matches_on_apollo_id_then_domain(session):
    first = store.upsert_organization(session, name="Corp", domain="corp.com", apollo_org_id="o1")
    session.commit()

    same_by_id = store.upsert_organization(session, name="Corp Inc", apollo_org_id="o1")
    same_by_domain = store.upsert_organization(session, domain="corp.com")
    session.commit()

    assert same_by_id.id == first.id
    assert same_by_domain.id == first.id
    assert first.name == "Corp"  # not overwritten


def test_upsert_organization_returns_none_when_there_is_nothing_to_store(session):
    assert store.upsert_organization(session) is None


def test_run_bookkeeping(session):
    run = store.start_run(session, mode=RunMode.DRY_RUN, campaign="job_search")
    session.commit()
    assert run.finished_at is None

    store.finish_run(session, run, sent_count=3, error_count=1)
    session.commit()

    assert run.finished_at is not None
    assert (run.sent_count, run.error_count) == (3, 1)
    assert run.mode == "dry_run"


def test_counts_by_status(session, make_contact):
    contact = make_contact()
    for status in (OutreachStatus.SENT, OutreachStatus.SENT, OutreachStatus.FAILED):
        store.record_send(
            session, contact=contact, campaign="c", subject="s", body="b", status=status
        )
    session.commit()

    assert store.counts_by_status(session) == {"sent": 2, "failed": 1}


# ─────────────────────────── the kill switch itself ───────────────────────────


def test_network_access_is_denied():
    """The guarantee behind 'no real API calls' is itself tested.

    Matched on the message rather than the class: conftest is imported under a
    different module name than the one a test would import it by, so identity
    comparison on the exception type is unreliable.
    """
    import socket as socket_module

    with pytest.raises(RuntimeError, match="attempted to open a network connection"):
        socket_module.create_connection(("example.com", 443))

    with pytest.raises(RuntimeError, match="attempted to open a network connection"):
        socket_module.socket().connect(("example.com", 443))
