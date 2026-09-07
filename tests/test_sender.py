"""Tests for delivery.

Header correctness matters here for a reason beyond tidiness: Message-ID is
what threads a reply back to the right contact, and List-Unsubscribe is what
keeps the sending address alive.
"""

from __future__ import annotations

import smtplib
from types import SimpleNamespace

import pytest
from tests.fakes import FakeSMTPFactory

from outreach_agent.sender import (
    DryRunSender,
    GmailApiSender,
    OutboundEmail,
    SendError,
    SmtpSender,
    build_sender,
)


@pytest.fixture
def message() -> OutboundEmail:
    return OutboundEmail(
        to_email="bob@corp.com",
        to_name="Bob Roberts",
        subject="a quick question",
        body="Hi Bob,\n\nShort note.\n\nReply 'no thanks' and I won't write again.",
    )


def make_smtp_sender(factory=None, **kwargs):
    factory = factory or FakeSMTPFactory()
    sender = SmtpSender(
        host="smtp.gmail.com",
        port=465,
        username="jane@gmail.com",
        password="app-password",
        from_address="jane@gmail.com",
        from_name="Jane Engineer",
        smtp_factory=factory,
        **kwargs,
    )
    return sender, factory


# ─────────────────────────── SMTP delivery ───────────────────────────


def test_send_logs_in_and_delivers(message):
    sender, factory = make_smtp_sender()
    result = sender.send(message)

    smtp = factory.built[0]
    assert smtp.logged_in_as == "jane@gmail.com"
    assert len(smtp.sent_messages) == 1
    assert result.delivered is True
    assert result.message_id.startswith("<")


def test_the_connection_is_closed_after_every_send(message):
    sender, factory = make_smtp_sender()
    sender.send(message)
    assert factory.built[0].quit_called is True


def test_a_fresh_connection_per_message(message):
    """Sends are minutes apart; a socket held across a run gets closed for you."""
    sender, factory = make_smtp_sender()
    sender.send(message)
    sender.send(message)
    assert len(factory.built) == 2


def test_missing_credentials_are_refused_at_construction():
    with pytest.raises(SendError):
        SmtpSender(
            host="h", port=465, username="", password="",
            from_address="a@b.com", from_name="A",
        )


# ─────────────────────────── headers ───────────────────────────


def test_message_id_is_real_and_unique(message):
    sender, _ = make_smtp_sender()
    first, id_a = sender.build_message(message)
    _, id_b = sender.build_message(message)

    assert first["Message-ID"] == id_a
    assert id_a != id_b
    assert "gmail.com" in id_a


def test_from_and_to_headers_carry_display_names(message):
    sender, _ = make_smtp_sender()
    built, _ = sender.build_message(message)

    assert built["From"] == "Jane Engineer <jane@gmail.com>"
    assert built["To"] == "Bob Roberts <bob@corp.com>"


def test_list_unsubscribe_is_a_mailto(message):
    sender, _ = make_smtp_sender()
    built, _ = sender.build_message(message)

    assert built["List-Unsubscribe"] == "<mailto:jane@gmail.com?subject=unsubscribe>"


def test_no_one_click_unsubscribe_header(message):
    """List-Unsubscribe-Post promises a POST endpoint this project does not serve.

    Pairing it with a mailto is invalid and penalised by some receivers.
    """
    sender, _ = make_smtp_sender()
    built, _ = sender.build_message(message)

    assert built["List-Unsubscribe-Post"] is None


def test_a_follow_up_is_threaded_onto_the_original(message):
    message.in_reply_to = "<original@gmail.com>"
    sender, _ = make_smtp_sender()
    built, _ = sender.build_message(message)

    assert built["In-Reply-To"] == "<original@gmail.com>"
    assert built["References"] == "<original@gmail.com>"


def test_references_accumulate_across_a_thread(message):
    message.in_reply_to = "<second@gmail.com>"
    message.references = ["<first@gmail.com>", "<second@gmail.com>"]
    sender, _ = make_smtp_sender()
    built, _ = sender.build_message(message)

    assert built["References"] == "<first@gmail.com> <second@gmail.com>"


def test_a_first_email_has_no_threading_headers(message):
    sender, _ = make_smtp_sender()
    built, _ = sender.build_message(message)

    assert built["In-Reply-To"] is None
    assert built["References"] is None


def test_the_body_survives_intact(message):
    sender, _ = make_smtp_sender()
    built, _ = sender.build_message(message)
    assert built.get_content().strip() == message.body.strip()


# ─────────────────────────── failures ───────────────────────────


def test_a_bad_app_password_gives_an_actionable_error(message):
    factory = FakeSMTPFactory(
        fail_on_login=smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")
    )
    sender, _ = make_smtp_sender(factory)

    with pytest.raises(SendError, match="app password"):
        sender.send(message)


def test_a_transport_failure_becomes_a_send_error(message):
    factory = FakeSMTPFactory(fail_on_send=smtplib.SMTPServerDisconnected("gone"))
    sender, _ = make_smtp_sender(factory)

    with pytest.raises(SendError):
        sender.send(message)


def test_an_os_error_becomes_a_send_error(message):
    factory = FakeSMTPFactory(fail_on_send=OSError("network unreachable"))
    sender, _ = make_smtp_sender(factory)

    with pytest.raises(SendError):
        sender.send(message)


# ─────────────────────────── the other two backends ───────────────────────────


def test_gmail_api_sender_raises_not_implemented(message):
    with pytest.raises(NotImplementedError, match="not implemented"):
        GmailApiSender().send(message)


def test_dry_run_sender_records_without_delivering(message):
    sender = DryRunSender()
    result = sender.send(message)

    assert result.delivered is False
    assert sender.outbox == [message]
    assert result.message_id.startswith("<")


# ─────────────────────────── the structural guarantee ───────────────────────────


def _config(**overrides):
    base = dict(
        gmail_address="jane@gmail.com",
        gmail_app_password="app-password",
        sender_name="Jane",
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        sender_backend="smtp",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_sender_returns_a_dry_run_sender_without_live():
    """The heart of it: a run that did not ask to send cannot send.

    Not 'skips the send' — never holds an object capable of sending.
    """
    sender = build_sender(_config(), live=False)
    assert isinstance(sender, DryRunSender)
    assert not isinstance(sender, SmtpSender)


def test_build_sender_returns_smtp_only_with_live():
    sender = build_sender(_config(), live=True)
    assert isinstance(sender, SmtpSender)


def test_build_sender_honours_the_gmail_api_backend_when_live():
    sender = build_sender(_config(sender_backend="gmail_api"), live=True)
    assert isinstance(sender, GmailApiSender)


def test_dry_run_ignores_the_backend_setting():
    """Even a misconfigured backend cannot produce a live sender in a dry run."""
    sender = build_sender(_config(sender_backend="gmail_api"), live=False)
    assert isinstance(sender, DryRunSender)


def test_dry_run_works_without_any_credentials():
    """A dry run must not require the Gmail app password."""
    sender = build_sender(_config(gmail_address="", gmail_app_password=""), live=False)
    assert isinstance(sender, DryRunSender)
