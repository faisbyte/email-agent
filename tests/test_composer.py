"""Tests for email composition.

Two things carry the weight here. The five-sentence cap is enforced in Python
and survives exactly one corrective retry. And the parts of the email that must
always be present — the greeting, the sign-off, the website, the removal line —
are assembled in code, so they are tested as structure rather than as something
we asked a model for.
"""

from __future__ import annotations

import pytest
from tests.fakes import FakeAnthropic

from outreach_agent.campaigns import ApolloFilters, Campaign, FollowUpPolicy
from outreach_agent.composer import (
    ComposedEmail,
    Composer,
    CompositionError,
    EmailDraft,
    count_sentences,
    first_name,
    has_footer,
    render_message,
    validate_draft,
)
from outreach_agent.store import Contact

SENDER = "Jane Engineer"
WEBSITE = "https://jane.dev"
CV = "Jane Engineer. Built a payments system at Corp. Python, Postgres."

FIVE = (
    "I am a software engineer based in Sydney. "
    "I am finishing a payments system built on Python and Postgres. "
    "Your team posted two backend roles last month. "
    "The second one lines up closely with the work I have been doing. "
    "Are there openings worth talking about?"
)
SIX = FIVE + " I would be glad to send more detail."


@pytest.fixture
def contact() -> Contact:
    return Contact(
        id=1,
        email="bob@corp.com",
        name="Bob Roberts",
        title="Technical Recruiter",
        location="Sydney, NSW, Australia",
    )


def make_composer(outputs, *, website: str = ""):
    client = FakeAnthropic(outputs=outputs)
    composer = Composer(
        client,
        model="claude-sonnet-5",
        sender_name=SENDER,
        personal_website=website,
    )
    return composer, client


def draft(subject: str = "backend roles", body: str = FIVE) -> EmailDraft:
    return EmailDraft(subject=subject, body=body)


# ─────────────────────────── first_name ───────────────────────────


@pytest.mark.parametrize(
    ("full", "expected"),
    [
        ("Sarah Chen", "Sarah"),
        ("Sarah", "Sarah"),
        ("Dr. Sarah Chen", "Sarah"),
        ("Dr Sarah Chen", "Sarah"),
        ("Prof. Sarah Chen", "Sarah"),
        ("Mr Bob Roberts", "Bob"),
        ("Ms. Alice O'Neill", "Alice"),
        ("  Sarah   Chen  ", "Sarah"),
        ("", ""),
        (None, ""),
    ],
)
def test_first_name(full, expected):
    assert first_name(full) == expected


def test_a_nameless_contact_still_gets_a_greeting():
    message = render_message(
        FIVE, contact_name=None, sender_name=SENDER, removal_line="Reply 'no thanks'."
    )
    assert message.startswith("Hi there,")


# ─────────────────────────── count_sentences ───────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One sentence here.", 1),
        ("One. Two. Three.", 3),
        (FIVE, 5),
        (SIX, 6),
        ("A question? An exclamation! A statement.", 3),
        ("No terminal punctuation", 1),
        ("", 0),
        ("   ", 0),
    ],
)
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected


def test_count_sentences_ignores_trailing_whitespace():
    assert count_sentences(FIVE + "   \n") == 5


# ─────────────────────────── render_message ───────────────────────────


def test_the_rendered_message_has_the_expected_shape():
    message = render_message(
        FIVE,
        contact_name="Bob Roberts",
        sender_name=SENDER,
        removal_line="Reply 'no thanks' and I won't write again.",
        personal_website=WEBSITE,
    )

    assert message == (
        "Hi Bob,\n"
        "\n"
        f"{FIVE}\n"
        "\n"
        "Thanks,\n"
        f"{SENDER}\n"
        f"{WEBSITE}\n"
        "Reply 'no thanks' and I won't write again."
    )


def test_the_website_line_appears_when_set():
    message = render_message(
        FIVE,
        contact_name="Bob",
        sender_name=SENDER,
        removal_line="Removal line.",
        personal_website=WEBSITE,
    )
    assert WEBSITE in message.splitlines()


def test_an_unset_website_leaves_no_blank_gap():
    """The line disappears, rather than becoming an empty one."""
    message = render_message(
        FIVE,
        contact_name="Bob",
        sender_name=SENDER,
        removal_line="Removal line.",
        personal_website="",
    )
    lines = message.splitlines()

    assert lines[-3:] == ["Thanks,", SENDER, "Removal line."]
    assert "" not in lines[-3:]
    assert "https://" not in message


def test_a_whitespace_only_website_is_treated_as_unset():
    message = render_message(
        FIVE,
        contact_name="Bob",
        sender_name=SENDER,
        removal_line="Removal line.",
        personal_website="   ",
    )
    assert message.splitlines()[-2:] == [SENDER, "Removal line."]


def test_every_rendered_message_ends_with_the_removal_line(campaign):
    for website in ("", WEBSITE):
        message = render_message(
            FIVE,
            contact_name="Bob",
            sender_name=SENDER,
            removal_line=campaign.removal_line,
            personal_website=website,
        )
        assert message.endswith(campaign.removal_line)
        assert has_footer(message, campaign.removal_line)


# ─────────────────────────── the cap counts only the body ───────────────────────────


def test_the_scaffolding_does_not_count_toward_the_five(campaign):
    """The greeting, sign-off, website and removal line are all sentences by any
    naive count. Only the model's body is measured against the cap."""
    body_count = count_sentences(FIVE)
    message = render_message(
        FIVE,
        contact_name="Bob Roberts",
        sender_name=SENDER,
        removal_line=campaign.removal_line,
        personal_website=WEBSITE,
    )

    assert body_count == 5
    assert count_sentences(message) > 5, "the whole message is longer, as expected"

    # The validator only ever sees the body, which is the point.
    assert validate_draft(draft(body=FIVE), campaign) == []


def test_a_five_sentence_body_passes_through_unchanged(contact, campaign):
    composer, client = make_composer([draft()])
    email = composer.compose(contact, campaign, CV)

    assert client.call_count == 1
    assert FIVE in email.body
    assert isinstance(email, ComposedEmail)


# ─────────────────────────── the retry ───────────────────────────


def test_a_six_sentence_body_triggers_exactly_one_retry(contact, campaign):
    composer, client = make_composer([draft(body=SIX), draft(body=FIVE)])
    email = composer.compose(contact, campaign, CV)

    assert client.call_count == 2, "one original call, one retry"
    assert FIVE in email.body
    assert "I would be glad to send more detail." not in email.body


def test_the_retry_states_the_actual_count_and_the_limit(contact, campaign):
    composer, client = make_composer([draft(body=SIX), draft(body=FIVE)])
    composer.compose(contact, campaign, CV)

    correction = client.calls[-1]["messages"][-1]["content"]
    assert "6 sentences" in correction
    assert "limit is 5" in correction


def test_the_retry_shows_the_model_its_own_draft(contact, campaign):
    composer, client = make_composer([draft(body=SIX), draft(body=FIVE)])
    composer.compose(contact, campaign, CV)

    messages = client.calls[-1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert SIX in messages[1]["content"]


def test_a_six_sentence_retry_raises(contact, campaign):
    """No truncation. The ask is the last sentence, so cutting the body short
    produces an email that stops before its own point."""
    composer, client = make_composer([draft(body=SIX), draft(body=SIX)])

    with pytest.raises(CompositionError, match="after one retry"):
        composer.compose(contact, campaign, CV)

    assert client.call_count == 2, "one retry only, never two"


def test_the_cap_comes_from_the_campaign(contact):
    """max_sentences is campaign configuration, not a constant in the composer."""
    generous = Campaign(
        slug="generous",
        name="Generous",
        sender_persona="An engineer.",
        goal="Talk.",
        tone=["Direct."],
        subject_guidance="Short.",
        max_words=200,
        max_sentences=6,
        removal_line="Reply 'no thanks'.",
        apollo=ApolloFilters(person_titles=["Recruiter"]),
        follow_up=FollowUpPolicy(),
    )
    composer, client = make_composer([draft(body=SIX)])
    composer.compose(contact, generous, CV)

    assert client.call_count == 1, "six sentences is fine when the campaign allows six"


# ─────────────────────────── other rejections ───────────────────────────


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("", "empty"),
        ("Hi Bob, I am an engineer. Any openings?", "greeting"),
        ("I am an engineer. Any openings? Thanks,", "sign-off"),
        ("I am an engineer with **real** experience. Any openings?", "bold"),
        ("I am an engineer. Any openings? \U0001f600", "emoji"),
        ("- I am an engineer.\n- Any openings?", "bullet"),
        ("I am an engineer. See [my site](https://x.com). Any openings?", "markdown links"),
        ("# Heading\n\nI am an engineer.", "heading"),
        ("`code`. Any openings?", "backticks"),
    ],
)
def test_bad_bodies_are_rejected(body, fragment, campaign):
    problems = validate_draft(draft(body=body), campaign)
    assert problems, f"expected {body!r} to be rejected"
    assert any(fragment in p.lower() for p in problems)


def test_each_rejection_triggers_a_retry(contact, campaign):
    composer, client = make_composer(
        [draft(body="Hi Bob, I am an engineer. Any openings?"), draft(body=FIVE)]
    )
    email = composer.compose(contact, campaign, CV)

    assert client.call_count == 2
    assert email.body.count("Hi Bob,") == 1, "only the assembled greeting survives"


def test_a_newline_in_the_subject_is_rejected(campaign):
    problems = validate_draft(draft(subject="backend\nroles"), campaign)
    assert any("line break" in p for p in problems)


def test_an_empty_subject_is_rejected(campaign):
    assert validate_draft(draft(subject="   "), campaign)


def test_a_subject_is_tidied_after_validation(contact, campaign):
    composer, _ = make_composer([draft(subject="Re: backend roles")])
    email = composer.compose(contact, campaign, CV)
    assert email.subject == "backend roles"


def test_a_compliant_draft_has_no_problems(campaign):
    assert validate_draft(draft(), campaign) == []


# ─────────────────────────── what the model is told ───────────────────────────


def test_the_cv_goes_in_the_system_prompt_not_the_user_turn(contact, campaign):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)

    assert CV in client.last_system()
    assert CV not in client.last_user_text()


def test_the_contact_facts_go_in_the_user_turn(contact, campaign):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)

    user = client.last_user_text()
    assert "Bob Roberts" in user
    assert "Technical Recruiter" in user


def test_the_system_prompt_is_identical_between_contacts(campaign):
    """Byte-identical prefixes are what make the cache breakpoint worth setting."""
    composer, client = make_composer([draft(), draft()])
    a = Contact(id=1, email="a@corp.com", name="A", title="T")
    b = Contact(id=2, email="b@corp.com", name="B", title="T")

    composer.compose(a, campaign, CV)
    first = client.last_system()
    composer.compose(b, campaign, CV)

    assert first == client.last_system()


def test_the_model_is_told_not_to_write_the_scaffolding(contact, campaign):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)
    system = client.last_system()

    assert "Do NOT open with" in system
    assert "Do NOT close with" in system
    assert "Do NOT include any URL" in system


def test_the_model_is_told_every_claim_must_trace_to_the_cv(contact, campaign):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)
    system = client.last_system()

    assert "traceable to the BACKGROUND" in system
    assert "long admired" in system


def test_the_model_is_told_to_avoid_what_would_fool_the_splitter(contact, campaign):
    """Abbreviations, ellipses and decimals are what a regex splitter gets wrong."""
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)
    system = client.last_system()

    assert "e.g." in system
    assert "ellipses" in system
    assert "decimal numbers" in system


def test_the_sentence_budget_is_in_the_prompt(contact, campaign):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)
    system = client.last_system()

    assert "at most 5 sentences" in system
    assert "Who the sender is" in system
    assert "The ask" in system


def test_structured_output_is_requested(contact, campaign):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)

    assert client.calls[-1]["output_format"] is EmailDraft
    assert client.calls[-1]["model"] == "claude-sonnet-5"


def test_a_cache_breakpoint_is_set_on_the_system_prompt(contact, campaign):
    composer, client = make_composer([draft()])
    composer.compose(contact, campaign, CV)

    assert client.calls[-1]["system"][0]["cache_control"] == {"type": "ephemeral"}


# ─────────────────────────── assembly in context ───────────────────────────


def test_the_composed_email_carries_the_full_scaffolding(contact, campaign):
    composer, _ = make_composer([draft()], website=WEBSITE)
    email = composer.compose(contact, campaign, CV)

    assert email.body.startswith("Hi Bob,")
    assert "Thanks,\nJane Engineer" in email.body
    assert WEBSITE in email.body
    assert email.body.endswith(campaign.removal_line)


def test_a_composed_email_without_a_website_is_still_well_formed(contact, campaign):
    composer, _ = make_composer([draft()])
    email = composer.compose(contact, campaign, CV)

    assert email.body.endswith(f"Jane Engineer\n{campaign.removal_line}")


# ─────────────────────────── refusals and follow-ups ───────────────────────────


def test_a_refusal_is_reported_as_a_composition_error(contact, campaign, monkeypatch):
    client = FakeAnthropic(outputs=[draft()])
    composer = Composer(client, model="claude-sonnet-5", sender_name=SENDER)
    real_parse = client.messages.parse

    def refusing_parse(**kwargs):
        result = real_parse(**kwargs)
        object.__setattr__(result, "stop_reason", "refusal")
        return result

    monkeypatch.setattr(client.messages, "parse", refusing_parse)

    with pytest.raises(CompositionError, match="declined"):
        composer.compose(contact, campaign, CV)


def test_api_errors_are_not_swallowed(contact, campaign):
    """The runner decides what a transport failure means, not the composer."""
    composer, _ = make_composer([RuntimeError("api exploded")])
    with pytest.raises(RuntimeError):
        composer.compose(contact, campaign, CV)


def test_follow_up_includes_the_previous_email(contact, campaign):
    composer, client = make_composer([draft(subject="following up")])
    email = composer.compose_follow_up(
        contact,
        campaign,
        CV,
        previous_subject="backend roles",
        previous_body="Hi Bob, short note.",
    )

    user = client.last_user_text()
    assert "backend roles" in user
    assert campaign.follow_up.goal in user
    assert email.body.endswith(campaign.removal_line)
    assert email.body.startswith("Hi Bob,")


def test_a_follow_up_obeys_the_same_cap(contact, campaign):
    composer, client = make_composer([draft(body=SIX), draft(body=FIVE)])
    composer.compose_follow_up(
        contact, campaign, CV, previous_subject="s", previous_body="b"
    )
    assert client.call_count == 2
