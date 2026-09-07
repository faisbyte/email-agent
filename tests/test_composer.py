"""Tests for email composition.

The load-bearing assertion here is the footer: every body that leaves this
module offers removal, regardless of what the model wrote.
"""

from __future__ import annotations

import pytest
from tests.fakes import FakeAnthropic

from outreach_agent.composer import (
    ComposedEmail,
    Composer,
    CompositionError,
    EmailDraft,
    ensure_footer,
    has_footer,
)
from outreach_agent.store import Contact


@pytest.fixture
def contact() -> Contact:
    return Contact(
        id=1,
        email="bob@corp.com",
        name="Bob Roberts",
        title="Technical Recruiter",
        location="Sydney, NSW, Australia",
    )


@pytest.fixture
def cv_text() -> str:
    return "Jane Engineer. Built a payments system at Corp. Python, Postgres."


def make_composer(outputs):
    client = FakeAnthropic(outputs=outputs)
    return Composer(client, model="claude-sonnet-5"), client


def draft(subject="a quick question", body="Hi Bob,\n\nShort note.\n\nJane"):
    return EmailDraft(subject=subject, body=body)


# ─────────────────────────── the footer guarantee ───────────────────────────


def test_every_body_ends_with_the_removal_line(contact, campaign, cv_text):
    composer, _ = make_composer([draft()])
    email = composer.compose(contact, campaign, cv_text)

    assert email.body.endswith(campaign.removal_line)
    assert has_footer(email.body, campaign.removal_line)


def test_the_footer_is_added_even_when_the_model_ignores_every_instruction(
    contact, campaign, cv_text
):
    composer, _ = make_composer([draft(body="No footer here at all.")])
    email = composer.compose(contact, campaign, cv_text)

    assert campaign.removal_line in email.body


def test_a_model_written_footer_is_not_duplicated(contact, campaign, cv_text):
    body = f"Hi Bob,\n\nShort note.\n\n{campaign.removal_line}"
    composer, _ = make_composer([draft(body=body)])
    email = composer.compose(contact, campaign, cv_text)

    assert email.body.count(campaign.removal_line) == 1


def test_a_near_duplicate_footer_is_removed(campaign):
    """Whitespace and casing differences are still duplicates."""
    body = f"Hello.\n\n  {campaign.removal_line.upper()}  "
    result = ensure_footer(body, campaign.removal_line)

    assert result.count(campaign.removal_line) == 1
    assert result.upper().count(campaign.removal_line.upper()) == 1


def test_ensure_footer_refuses_an_empty_removal_line():
    with pytest.raises(CompositionError):
        ensure_footer("Hello.", "   ")


def test_ensure_footer_refuses_a_body_that_is_only_a_footer(campaign):
    with pytest.raises(CompositionError):
        ensure_footer(campaign.removal_line, campaign.removal_line)


# ─────────────────────────── sanitising ───────────────────────────


def test_subject_newlines_are_collapsed(contact, campaign, cv_text):
    composer, _ = make_composer([draft(subject="a question\nabout   roles")])
    email = composer.compose(contact, campaign, cv_text)

    assert email.subject == "a question about roles"
    assert "\n" not in email.subject


def test_a_re_prefix_is_stripped_from_a_first_email(contact, campaign, cv_text):
    composer, _ = make_composer([draft(subject="Re: engineering roles")])
    email = composer.compose(contact, campaign, cv_text)
    assert email.subject == "engineering roles"


def test_an_empty_subject_is_rejected(contact, campaign, cv_text):
    composer, _ = make_composer([draft(subject="   ")])
    with pytest.raises(CompositionError):
        composer.compose(contact, campaign, cv_text)


def test_an_empty_body_is_rejected(contact, campaign, cv_text):
    composer, _ = make_composer([draft(body="  \n ")])
    with pytest.raises(CompositionError):
        composer.compose(contact, campaign, cv_text)


def test_an_absurdly_long_body_is_rejected(contact, campaign, cv_text):
    composer, _ = make_composer([draft(body="word " * 5000)])
    with pytest.raises(CompositionError):
        composer.compose(contact, campaign, cv_text)


def test_a_long_but_plausible_body_is_allowed_with_a_warning(
    contact, campaign, cv_text, caplog
):
    """Truncating an email mid-sentence is worse than sending a long one."""
    composer, _ = make_composer([draft(body="word " * 400)])
    with caplog.at_level("WARNING"):
        email = composer.compose(contact, campaign, cv_text)

    assert "over the campaign limit" in caplog.text
    assert email.body.endswith(campaign.removal_line)


# ─────────────────────────── what the model is told ───────────────────────────


def test_the_cv_and_persona_go_in_the_system_prompt(contact, campaign, cv_text):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, cv_text)

    system = client.last_system()
    assert cv_text in system
    assert campaign.sender_persona.strip() in system
    assert str(campaign.max_words) in system


def test_the_contact_facts_go_in_the_user_turn(contact, campaign, cv_text):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, cv_text)

    user = client.last_user_text()
    assert "Bob Roberts" in user
    assert "Technical Recruiter" in user


def test_the_system_prompt_is_identical_between_contacts(campaign, cv_text):
    """Byte-identical prefixes are what make the cache breakpoint worth setting."""
    composer, client = make_composer([draft(), draft()])
    a = Contact(id=1, email="a@corp.com", name="A", title="T")
    b = Contact(id=2, email="b@corp.com", name="B", title="T")

    composer.compose(a, campaign, cv_text)
    first = client.last_system()
    composer.compose(b, campaign, cv_text)
    second = client.last_system()

    assert first == second


def test_the_model_is_told_not_to_write_its_own_opt_out(contact, campaign, cv_text):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, cv_text)
    assert "unsubscribe or opt-out line" in client.last_system()


def test_the_configured_model_is_used(contact, campaign, cv_text):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, cv_text)
    assert client.calls[-1]["model"] == "claude-sonnet-5"


def test_structured_output_is_requested(contact, campaign, cv_text):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, cv_text)
    assert client.calls[-1]["output_format"] is EmailDraft


def test_a_cache_breakpoint_is_set_on_the_system_prompt(contact, campaign, cv_text):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, cv_text)

    system = client.calls[-1]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


# ─────────────────────────── refusals and follow-ups ───────────────────────────


def test_a_refusal_is_reported_as_a_composition_error(contact, campaign, cv_text, monkeypatch):
    client = FakeAnthropic(outputs=[draft()])
    composer = Composer(client, model="claude-sonnet-5")

    real_parse = client.messages.parse

    def refusing_parse(**kwargs):
        result = real_parse(**kwargs)
        object.__setattr__(result, "stop_reason", "refusal")
        return result

    monkeypatch.setattr(client.messages, "parse", refusing_parse)

    with pytest.raises(CompositionError, match="declined"):
        composer.compose(contact, campaign, cv_text)


def test_api_errors_are_not_swallowed(contact, campaign, cv_text):
    """The runner decides what a transport failure means, not the composer."""
    composer, _ = make_composer([RuntimeError("api exploded")])
    with pytest.raises(RuntimeError):
        composer.compose(contact, campaign, cv_text)


def test_follow_up_includes_the_previous_email(contact, campaign, cv_text):
    composer, client = make_composer([draft(subject="following up")])
    email = composer.compose_follow_up(
        contact,
        campaign,
        cv_text,
        previous_subject="engineering roles",
        previous_body="Hi Bob, short note.",
    )

    user = client.last_user_text()
    assert "engineering roles" in user
    assert "Hi Bob, short note." in user
    assert campaign.follow_up.goal in user
    assert isinstance(email, ComposedEmail)
    assert email.body.endswith(campaign.removal_line)
