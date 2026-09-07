"""Email composition via the Anthropic API.

This is one of exactly two places Claude is called. It takes a contact, a
campaign and your CV, and returns a subject and a body. It does not decide who
to write to, whether to send, or what happens next — that is the runner's job.

The removal line is appended by Python, not requested from the model. A
guarantee that depends on a language model following an instruction is not a
guarantee.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .campaigns import Campaign
from .store import Contact

log = logging.getLogger(__name__)

MAX_SUBJECT_CHARS = 200
MAX_BODY_CHARS = 8000


class CompositionError(Exception):
    """The model returned something unusable."""


class EmailDraft(BaseModel):
    """The structured output contract with the model.

    Passed to ``client.messages.parse(output_format=...)``, which constrains
    the response to this schema. Verified against anthropic==1.4.0: ``parse``
    lives on the stable ``client.messages`` namespace and needs no beta header.
    """

    subject: str = Field(description="Email subject line. Plain text, no prefixes.")
    body: str = Field(description="Email body. Plain text, no markdown, no signature block.")


@dataclass(frozen=True)
class ComposedEmail:
    subject: str
    body: str


SYSTEM_TEMPLATE = """\
You write short, plain outreach emails on behalf of one person. You are given \
that person's background and a description of who they are writing to. You \
return a subject line and a body.

WHO YOU ARE WRITING AS
{persona}

WHAT THIS EMAIL IS FOR
{goal}

STYLE
{tone}

SUBJECT LINE
{subject_guidance}

RULES — these are absolute
- Use only facts that appear in the BACKGROUND section below. If a fact is not \
there, it does not exist. Never invent a project, a company, a metric, or a date.
- Never claim a prior relationship, a prior conversation, or having used their \
product, unless the background says so.
- Never write a placeholder such as [Company] or [your name]. If you do not \
know something, write the sentence without it.
- Plain text only. No markdown, no bullet points, no emoji, no headers.
- Stay under {max_words} words in the body.
- Do not write an unsubscribe or opt-out line. One is added automatically \
after you finish, and a second would be redundant.
- Write the body as if it will be read by a busy person in ten seconds.

BACKGROUND — the only facts you may use
{cv}
"""

USER_TEMPLATE = """\
Write the email to this person.

Name: {name}
Title: {title}
Company: {company}
Location: {location}

Use their first name if you have it. Give one specific, checkable reason you \
are writing to them in particular.
"""

FOLLOW_UP_USER_TEMPLATE = """\
Write a short follow-up to this person. They have not replied.

Name: {name}
Title: {title}
Company: {company}
Location: {location}

The first email had the subject "{previous_subject}" and said:
---
{previous_body}
---

FOLLOW-UP INSTRUCTIONS
{follow_up_goal}

Reference the earlier email in a single clause. Do not repeat it. Be shorter \
than it was. Make it easy to say no.
"""


class Composer:
    """Turns a contact into an email. Stateless between calls."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_tokens: int = 2000,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def compose(
        self,
        contact: Contact,
        campaign: Campaign,
        cv_text: str,
    ) -> ComposedEmail:
        """Compose a first-contact email."""
        user_prompt = USER_TEMPLATE.format(
            name=contact.name or "(unknown)",
            title=contact.title or "(unknown)",
            company=_company_of(contact),
            location=contact.location or "(unknown)",
        )
        return self._call(campaign, cv_text, user_prompt)

    def compose_follow_up(
        self,
        contact: Contact,
        campaign: Campaign,
        cv_text: str,
        *,
        previous_subject: str,
        previous_body: str,
    ) -> ComposedEmail:
        """Compose a follow-up to a message that got no reply."""
        user_prompt = FOLLOW_UP_USER_TEMPLATE.format(
            name=contact.name or "(unknown)",
            title=contact.title or "(unknown)",
            company=_company_of(contact),
            location=contact.location or "(unknown)",
            previous_subject=previous_subject,
            previous_body=previous_body,
            follow_up_goal=campaign.follow_up.goal or "One short nudge. Restate the ask briefly.",
        )
        return self._call(campaign, cv_text, user_prompt)

    # ── internals ─────────────────────────────────────────────

    def _call(self, campaign: Campaign, cv_text: str, user_prompt: str) -> ComposedEmail:
        system = SYSTEM_TEMPLATE.format(
            persona=campaign.sender_persona.strip(),
            goal=campaign.goal.strip(),
            tone="\n".join(f"- {t}" for t in campaign.tone) or "- Plain and direct.",
            subject_guidance=campaign.subject_guidance.strip(),
            max_words=campaign.max_words,
            cv=cv_text.strip(),
        )

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt is the stable half — persona, rules, CV — and is
            # byte-identical for every contact in a run, so it is worth a cache
            # breakpoint. Short CVs may fall under the minimum cacheable prefix,
            # in which case this is simply a no-op.
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
            output_format=EmailDraft,
        )

        if getattr(response, "stop_reason", None) == "refusal":
            raise CompositionError(
                "the model declined to write this email; check the campaign persona and CV"
            )

        draft = getattr(response, "parsed_output", None)
        if draft is None:
            raise CompositionError("the model returned no parsed output")

        subject = _clean_subject(getattr(draft, "subject", ""))
        body = _clean_body(getattr(draft, "body", ""), campaign)

        return ComposedEmail(subject=subject, body=body)


# ─────────────────────────── sanitising ───────────────────────────


def _company_of(contact: Contact) -> str:
    org = getattr(contact, "organization", None)
    if org is not None and getattr(org, "name", None):
        return org.name
    return "(unknown)"


def _clean_subject(raw: str) -> str:
    """A subject is one line. Enforce that rather than trusting it."""
    subject = " ".join((raw or "").split())
    subject = re.sub(r"^(re|fwd|subject)\s*:\s*", "", subject, flags=re.IGNORECASE).strip()
    if not subject:
        raise CompositionError("the model returned an empty subject line")
    return subject[:MAX_SUBJECT_CHARS]


def _clean_body(raw: str, campaign: Campaign) -> str:
    body = (raw or "").strip()
    if not body:
        raise CompositionError("the model returned an empty body")

    if len(body) > MAX_BODY_CHARS:
        raise CompositionError(
            f"the model returned {len(body)} characters, over the {MAX_BODY_CHARS} limit"
        )

    word_count = len(body.split())
    if word_count > campaign.max_words * 1.5:
        # Worth knowing about, but truncating an email mid-sentence is worse
        # than sending a long one.
        log.warning(
            "composed body is %d words, well over the campaign limit of %d",
            word_count,
            campaign.max_words,
        )

    return ensure_footer(body, campaign.removal_line)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def ensure_footer(body: str, removal_line: str) -> str:
    """Guarantee exactly one removal line at the bottom of the body.

    Called on every composed email, without exception. Any near-duplicate the
    model wrote on its own is removed first, so the recipient never sees the
    offer twice.
    """
    removal_line = removal_line.strip()
    if not removal_line:
        raise CompositionError("campaign has no removal_line; every email must offer removal")

    target = _normalise(removal_line)
    kept = [line for line in body.splitlines() if _normalise(line) != target]

    cleaned = "\n".join(kept).rstrip()
    if not cleaned:
        raise CompositionError("the body was empty once the removal line was removed")

    return f"{cleaned}\n\n{removal_line}"


def has_footer(body: str, removal_line: str) -> bool:
    """True if the body ends with the campaign's removal line."""
    return _normalise(body).endswith(_normalise(removal_line))
