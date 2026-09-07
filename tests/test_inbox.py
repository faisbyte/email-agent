"""Tests for reading and acting on replies.

The consequential assertions: a bounce never reaches the model, an opt-out is
suppressed in the same transaction as the reply, and an auto-reply does not
count as a person saying no.
"""

from __future__ import annotations

import pytest
from tests.fakes import FakeAnthropic

from outreach_agent import store
from outreach_agent.inbox import (
    ReplyClassifier,
    ReplyVerdict,
    looks_like_bounce,
    looks_like_opt_out,
    parse_message,
    process_replies,
)
from outreach_agent.store import OutreachStatus, ReplyClassification


def raw_email(
    *,
    sender: str = "Bob Roberts <bob@corp.com>",
    subject: str = "Re: a quick question",
    body: str = "Thanks, send your CV.",
    in_reply_to: str | None = "<original@gmail.com>",
    references: str | None = None,
    extra_headers: str = "",
    content_type: str = "text/plain; charset=utf-8",
) -> bytes:
    headers = [
        f"From: {sender}",
        "To: jane@gmail.com",
        f"Subject: {subject}",
        "Message-ID: <reply-1@corp.com>",
        f"Content-Type: {content_type}",
    ]
    if in_reply_to:
        headers.append(f"In-Reply-To: {in_reply_to}")
    if references:
        headers.append(f"References: {references}")
    if extra_headers:
        headers.append(extra_headers)
    return ("\r\n".join(headers) + "\r\n\r\n" + body).encode("utf-8")


def classifier_returning(*classifications):
    client = FakeAnthropic(
        outputs=[ReplyVerdict(classification=c, reason="because") for c in classifications]
    )
    return ReplyClassifier(client, model="claude-sonnet-5"), client


# ─────────────────────────── parsing ───────────────────────────


def test_parse_message_extracts_headers_and_body():
    message = parse_message(raw_email())

    assert message.from_address == "bob@corp.com"
    assert message.subject == "Re: a quick question"
    assert message.in_reply_to == "<original@gmail.com>"
    assert message.body == "Thanks, send your CV."


def test_parse_message_collects_references():
    message = parse_message(raw_email(references="<a@x.com> <b@x.com>"))
    assert message.references == ["<a@x.com>", "<b@x.com>"]


def test_candidate_ids_prefer_in_reply_to_then_references():
    message = parse_message(
        raw_email(in_reply_to="<second@x.com>", references="<first@x.com> <second@x.com>")
    )
    candidates = message.candidate_message_ids()

    assert candidates[0] == "<second@x.com>"
    assert "<first@x.com>" in candidates


def test_candidate_ids_fall_back_to_ids_found_in_the_raw_text():
    """A bounce usually carries the original only inside the attached report."""
    message = parse_message(
        raw_email(
            in_reply_to=None,
            body="Delivery failed for message <original@gmail.com>.",
        )
    )
    assert "<original@gmail.com>" in message.candidate_message_ids()


# ─────────────────────────── bounce detection ───────────────────────────


@pytest.mark.parametrize(
    "sender",
    [
        "Mail Delivery Subsystem <MAILER-DAEMON@googlemail.com>",
        "postmaster@corp.com",
        "noreply@corp.com",
    ],
)
def test_bounce_senders_are_detected(sender):
    assert looks_like_bounce(parse_message(raw_email(sender=sender))) is True


def test_delivery_status_content_type_is_detected():
    raw = raw_email(
        sender="someone@corp.com",
        content_type='multipart/report; report-type=delivery-status; boundary="b"',
    )
    assert looks_like_bounce(parse_message(raw)) is True


def test_a_normal_reply_is_not_a_bounce():
    assert looks_like_bounce(parse_message(raw_email())) is False


# ─────────────────────────── opt-out detection ───────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Please unsubscribe me.",
        "Remove me from your list.",
        "take me off this list",
        "Do not contact me again.",
        "Please opt-out my address",
        "stop emailing me",
        "no thanks",
        "DON'T EMAIL ME",
    ],
)
def test_explicit_removal_requests_are_matched(text):
    assert looks_like_opt_out(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Thanks, send your CV.",
        "We have no openings right now, try again in six months.",
        "I am out of the office until Monday.",
    ],
)
def test_ordinary_replies_are_not_opt_outs(text):
    assert looks_like_opt_out(text) is False


# ─────────────────────────── classification ───────────────────────────


def test_an_explicit_opt_out_never_reaches_the_model():
    """Free, deterministic, and it cannot be talked out of the answer."""
    classifier, client = classifier_returning("interested")

    result = classifier.classify("Please remove me from your list.")

    assert result is ReplyClassification.OPT_OUT
    assert client.call_count == 0


def test_an_opt_out_in_the_subject_also_short_circuits():
    classifier, client = classifier_returning("interested")
    result = classifier.classify("(no body)", subject="unsubscribe")

    assert result is ReplyClassification.OPT_OUT
    assert client.call_count == 0


@pytest.mark.parametrize(
    "verdict",
    ["interested", "rejection", "auto_reply", "opt_out"],
)
def test_the_model_verdict_is_returned(verdict):
    classifier, client = classifier_returning(verdict)
    result = classifier.classify("Some ambiguous reply about hiring.")

    assert result is ReplyClassification(verdict)
    assert client.call_count == 1


def test_the_classifier_requests_structured_output():
    classifier, client = classifier_returning("rejection")
    classifier.classify("No openings.")

    assert client.calls[0]["output_format"] is ReplyVerdict
    assert client.calls[0]["model"] == "claude-sonnet-5"


def test_an_unreadable_verdict_defaults_to_rejection(caplog):
    """Safe direction: it stops follow-ups rather than continuing hopefully."""

    class BlankMessages:
        def parse(self, **kwargs):
            return type("R", (), {"parsed_output": None, "stop_reason": "end_turn"})()

    client = type("C", (), {"messages": BlankMessages()})()
    classifier = ReplyClassifier(client, model="claude-sonnet-5")

    with caplog.at_level("WARNING"):
        assert classifier.classify("???") is ReplyClassification.REJECTION


# ─────────────────────────── consequences ───────────────────────────


@pytest.fixture
def sent_message(session, make_contact, make_sent):
    contact = make_contact("bob@corp.com")
    row = make_sent(contact, message_id="<original@gmail.com>")
    return contact, row


def test_an_opt_out_writes_a_suppression_immediately(session, sent_message):
    contact, row = sent_message
    classifier, client = classifier_returning()
    message = parse_message(raw_email(body="Please remove me from your list."))

    report = process_replies(session, [message], classifier)

    assert store.is_suppressed(session, "bob@corp.com") is True
    assert row.reply_classification == ReplyClassification.OPT_OUT.value
    assert report.suppressions == 1
    assert client.call_count == 0


def test_an_opt_out_suppression_and_reply_land_together(session, sent_message):
    """One commit covers both, so a crash cannot leave a person mailable."""
    contact, row = sent_message
    classifier, _ = classifier_returning()
    message = parse_message(raw_email(body="unsubscribe"))

    process_replies(session, [message], classifier)
    session.expire_all()

    assert store.is_suppressed(session, "bob@corp.com") is True
    assert store.has_human_reply(session, contact.id) is True


def test_an_interested_reply_stops_follow_ups(session, sent_message):
    contact, row = sent_message
    classifier, _ = classifier_returning("interested")
    message = parse_message(raw_email(body="Sure, send it over."))

    process_replies(session, [message], classifier)

    assert store.has_human_reply(session, contact.id) is True
    assert row.reply_body == "Sure, send it over."
    assert store.is_suppressed(session, "bob@corp.com") is False


def test_an_auto_reply_does_not_stop_follow_ups(session, sent_message):
    """An out-of-office is not a human saying no."""
    contact, row = sent_message
    classifier, _ = classifier_returning("auto_reply")
    message = parse_message(raw_email(body="I am out of the office until Monday."))

    process_replies(session, [message], classifier)

    assert row.reply_classification == ReplyClassification.AUTO_REPLY.value
    assert store.has_human_reply(session, contact.id) is False
    assert store.is_suppressed(session, "bob@corp.com") is False


def test_a_rejection_stops_follow_ups_without_suppressing(session, sent_message):
    contact, _ = sent_message
    classifier, _ = classifier_returning("rejection")
    message = parse_message(raw_email(body="No openings at the moment."))

    process_replies(session, [message], classifier)

    assert store.has_human_reply(session, contact.id) is True
    assert store.is_suppressed(session, "bob@corp.com") is False


# ─────────────────────────── bounces ───────────────────────────


def test_a_bounce_is_recorded_and_never_classified(session, sent_message):
    contact, row = sent_message
    classifier, client = classifier_returning("interested")
    message = parse_message(
        raw_email(
            sender="MAILER-DAEMON@googlemail.com",
            subject="Delivery Status Notification (Failure)",
            in_reply_to=None,
            body="Your message to bob@corp.com could not be delivered.\n<original@gmail.com>",
        )
    )

    report = process_replies(session, [message], classifier)

    assert row.status == OutreachStatus.BOUNCED.value
    assert row.reply_classification is None
    assert client.call_count == 0, "a bounce must never reach the model"
    assert report.bounces == 1


def test_a_bounce_does_not_count_as_a_human_reply(session, sent_message):
    contact, _ = sent_message
    classifier, _ = classifier_returning()
    message = parse_message(
        raw_email(
            sender="MAILER-DAEMON@googlemail.com",
            in_reply_to=None,
            body="failed: <original@gmail.com>",
        )
    )

    process_replies(session, [message], classifier)
    assert store.has_human_reply(session, contact.id) is False


# ─────────────────────────── matching ───────────────────────────


def test_unrelated_inbox_mail_is_ignored(session, sent_message):
    classifier, client = classifier_returning("interested")
    message = parse_message(
        raw_email(sender="newsletter@elsewhere.com", in_reply_to="<nothing@we.sent>")
    )

    report = process_replies(session, [message], classifier)

    assert report.unmatched == 1
    assert report.outcomes == []
    assert client.call_count == 0


def test_a_reply_matched_via_references(session, sent_message):
    contact, row = sent_message
    classifier, _ = classifier_returning("interested")
    message = parse_message(
        raw_email(in_reply_to=None, references="<original@gmail.com> <other@x.com>")
    )

    process_replies(session, [message], classifier)
    assert row.reply_classification == ReplyClassification.INTERESTED.value


def test_several_messages_are_processed_independently(session, make_contact, make_sent):
    alice = make_contact("alice@corp.com")
    bob = make_contact("bob@other.com")
    row_a = make_sent(alice, message_id="<a@gmail.com>")
    row_b = make_sent(bob, message_id="<b@gmail.com>")

    classifier, _ = classifier_returning("interested")
    messages = [
        parse_message(raw_email(sender="alice@corp.com", in_reply_to="<a@gmail.com>",
                                body="Yes please.")),
        parse_message(raw_email(sender="bob@other.com", in_reply_to="<b@gmail.com>",
                                body="Please remove me.")),
    ]

    report = process_replies(session, messages, classifier)

    assert row_a.reply_classification == ReplyClassification.INTERESTED.value
    assert row_b.reply_classification == ReplyClassification.OPT_OUT.value
    assert store.is_suppressed(session, "bob@other.com") is True
    assert store.is_suppressed(session, "alice@corp.com") is False
    assert len(report.outcomes) == 2
