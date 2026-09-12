"""Email composition via the Anthropic API.

This is one of exactly two places Claude is called. It returns a subject and
five sentences. It does not decide who to write to, whether to send, or what
happens next — that is the runner's job.

**The model writes only the middle of the email.** The greeting, the sign-off,
the website line and the removal line are assembled by ``render_message`` in
Python. That split is deliberate: it turns the sentence count, the link and the
way out of the list into mechanical guarantees rather than instructions we hand
to a language model and hope it follows. Only the model's ``body`` is ever
counted against the five-sentence cap.
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
MAX_BODY_CHARS = 4000

#: Sentence boundary. Deliberately crude, and kept honest by the system prompt,
#: which forbids the three things that would fool it: abbreviations, ellipses
#: and decimal numbers. Five sentences of plain prose do not need a parser, and
#: reaching for nltk or spacy to count to five would be a poor trade.
SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?:\s|$)")

#: Titles stripped before taking a first name, so "Dr. Sarah Chen" greets Sarah.
TITLE_PATTERN = re.compile(r"^(dr|mr|ms|mrs|miss|prof|professor)\.?\s+", re.IGNORECASE)

GREETING_PATTERN = re.compile(r"^\s*(hi|hello|dear|hey|greetings)\b[^\n]{0,40}[,:]", re.IGNORECASE)
SIGN_OFF_PATTERN = re.compile(
    r"\b(thanks|thank you|best|best regards|regards|sincerely|cheers|kind regards)\s*,?\s*$",
    re.IGNORECASE,
)

#: Markdown the model should never emit in a plain-text email.
MARKDOWN_PATTERNS = (
    (re.compile(r"\*\*"), "bold markers (**)"),
    (re.compile(r"`"), "backticks"),
    (re.compile(r"\[[^\]]+\]\([^)]+\)"), "markdown links"),
    (re.compile(r"^\s*#{1,6}\s", re.MULTILINE), "markdown headings"),
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), "bullet points"),
    (re.compile(r"^\s*>\s", re.MULTILINE), "block quotes"),
)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002700-\U000027bf"
    "\U0001f000-\U0001f0ff"
    "\U00002600-\U000026ff"
    "\U0000fe00-\U0000fe0f"
    "\U00002190-\U000021ff"
    "]+"
)


class CompositionError(Exception):
    """The model could not produce a usable email.

    Raised after one corrective retry has already failed. The runner treats
    this as "no email exists for this contact yet" — emphatically not as a
    failed send.
    """


class EmailDraft(BaseModel):
    """The structured output contract with the model.

    Passed to ``client.messages.parse(output_format=...)``, which constrains
    the response to this schema. Verified against anthropic==1.4.0: ``parse``
    lives on the stable ``client.messages`` namespace and needs no beta header.

    ``body`` is the middle of the email only — no greeting, no sign-off, no
    links, no opt-out line. Those are added afterwards, in Python.
    """

    subject: str = Field(description="Subject line. Plain text, one line, no 'Re:' prefix.")
    body: str = Field(
        description=(
            "The five sentences of the email only. No greeting, no sign-off, no "
            "name, no links, no unsubscribe line."
        )
    )


@dataclass(frozen=True)
class ComposedEmail:
    """A finished email: subject, and the fully assembled body."""

    subject: str
    body: str


SYSTEM_TEMPLATE = """\
You write the middle of a short outreach email on behalf of one person. You \
return a subject line and a body of at most {max_sentences} sentences.

WHO YOU ARE WRITING AS
{persona}

WHAT THIS EMAIL IS FOR
{goal}

STYLE
{tone}

SUBJECT LINE
{subject_guidance}

WHAT THE BODY MUST CONTAIN — {max_sentences} sentences, in this order
1. Who the sender is.
2. What the sender is doing at the moment.
3-4. One specific, true reason for writing to THIS person or THIS organisation.
5. The ask: whether there are opportunities worth talking about.

If a sentence is not doing one of those four jobs, delete it. Sentences 3 and 4 \
may be a single sentence if one is enough; never more than {max_sentences} in total.

WHAT YOU MUST NOT WRITE
The body is the middle of the email only. The greeting, the sign-off, the \
sender's name, any website link and the opt-out line are added automatically \
after you finish. Writing them yourself produces duplicates.
- Do NOT open with "Hi", "Hello", "Dear" or any greeting.
- Do NOT close with "Thanks", "Best", "Regards" or any sign-off or name.
- Do NOT include any URL, link or unsubscribe line.
- Do NOT write "I hope this finds you well" or any variation of it.
- Do NOT spend a sentence praising their company or its mission.

RULES — these are absolute
- Every factual claim about the sender must be traceable to the BACKGROUND \
text below. If a fact is not there, it does not exist. Do not infer seniority, \
do not invent years of experience, do not claim familiarity with their company, \
and never write "I have long admired" or anything like it.
- Never claim a prior relationship, a prior conversation, or having used their \
product, unless the background says so.
- Never write a placeholder such as [Company] or [your name]. If you do not \
know something, write the sentence without it.
- Plain text only. No markdown, no bullet points, no emoji, no headers.
- Use no abbreviations with full stops (no "e.g.", "i.e.", "Inc.", "Ph.D."), no \
ellipses, and no decimal numbers. Each full stop must end a sentence.
- Aim for under {max_words} words. Brevity beats completeness.
- Write as if it will be read by a busy person in ten seconds.

BACKGROUND — the only facts you may use about the sender
{cv}
"""

USER_TEMPLATE = """\
Write the email to this person.

Name: {name}
Title: {title}
Company: {company}
Location: {location}

Give one specific, checkable reason you are writing to them in particular.
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

CORRECTION_TEMPLATE = """\
That draft cannot be sent. {problems}

Rewrite it. Return at most {max_sentences} sentences in the body, covering who \
the sender is, what they are doing now, one specific reason for writing to this \
person, and the ask. No greeting, no sign-off, no name, no links.
"""


# ─────────────────────────── text helpers ───────────────────────────


def first_name(full_name: str | None) -> str:
    """The name to greet someone by.

    Takes the first whitespace-separated token, after stripping a leading
    title, and falls back to the whole string when there is only one token.
    Returns "" when there is nothing usable, which the greeting handles.
    """
    name = (full_name or "").strip()
    if not name:
        return ""

    name = TITLE_PATTERN.sub("", name).strip()
    if not name:
        return ""

    return name.split()[0]


def count_sentences(text: str) -> int:
    """How many sentences the body contains.

    Splits on terminal punctuation. The system prompt forbids abbreviations,
    ellipses and decimals, which is what keeps this honest for text this short.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len([part for part in SENTENCE_BOUNDARY.split(stripped) if part.strip()])


def render_message(
    body: str,
    *,
    contact_name: str | None,
    sender_name: str,
    removal_line: str,
    personal_website: str = "",
) -> str:
    """Assemble the finished email around the model's five sentences.

        Hi {first name},

        {body}

        Thanks,
        {sender name}
        {website}          <- omitted entirely when unset
        {removal line}

    Everything except ``body`` is written here rather than requested from the
    model, which is what makes the removal line and the website unconditional.
    """
    greeting_name = first_name(contact_name)
    greeting = f"Hi {greeting_name}," if greeting_name else "Hi there,"

    lines = [greeting, "", body.strip(), "", "Thanks,", sender_name.strip()]

    # Appended, not interpolated, so an unset website leaves no blank gap.
    if personal_website.strip():
        lines.append(personal_website.strip())

    lines.append(removal_line.strip())
    return "\n".join(lines)


def has_footer(body: str, removal_line: str) -> bool:
    """True if the rendered message ends with the campaign's removal line."""
    return _normalise(body).endswith(_normalise(removal_line))


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


# ─────────────────────────── validation ───────────────────────────


def validate_draft(draft: EmailDraft, campaign: Campaign) -> list[str]:
    """Everything wrong with a draft, in language the model can act on.

    An empty list means it ships. Anything else triggers one corrective retry.
    """
    problems: list[str] = []

    subject = (draft.subject or "").strip()
    body = (draft.body or "").strip()

    if not subject:
        problems.append("The subject line was empty.")
    elif "\n" in draft.subject.strip():
        problems.append("The subject line contained a line break; it must be a single line.")

    if not body:
        problems.append("The body was empty.")
        return problems

    if len(body) > MAX_BODY_CHARS:
        problems.append(
            f"The body was {len(body)} characters, far past anything sendable."
        )
        return problems

    sentences = count_sentences(body)
    if sentences > campaign.max_sentences:
        problems.append(
            f"The body was {sentences} sentences; the limit is {campaign.max_sentences}."
        )

    if GREETING_PATTERN.search(body):
        problems.append(
            "The body opened with a greeting. The greeting is added automatically, "
            "so yours would be a duplicate."
        )

    if SIGN_OFF_PATTERN.search(body):
        problems.append(
            "The body ended with a sign-off. The sign-off and name are added "
            "automatically, so yours would be a duplicate."
        )

    for pattern, label in MARKDOWN_PATTERNS:
        if pattern.search(body):
            problems.append(f"The body contained {label}; this is a plain-text email.")
            break

    if EMOJI_PATTERN.search(body):
        problems.append("The body contained an emoji; this is a plain-text email.")

    return problems


# ─────────────────────────── the composer ───────────────────────────


class Composer:
    """Turns a contact into an email. Stateless between calls."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        sender_name: str,
        personal_website: str = "",
        max_tokens: int = 1500,
    ) -> None:
        self.client = client
        self.model = model
        self.sender_name = sender_name
        self.personal_website = personal_website
        self.max_tokens = max_tokens

    def compose(self, contact: Contact, campaign: Campaign, cv_text: str) -> ComposedEmail:
        """Compose a first-contact email."""
        user_prompt = USER_TEMPLATE.format(
            name=contact.name or "(unknown)",
            title=contact.title or "(unknown)",
            company=_company_of(contact),
            location=contact.location or "(unknown)",
        )
        return self._call(contact, campaign, cv_text, user_prompt)

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
        return self._call(contact, campaign, cv_text, user_prompt)

    # ── internals ─────────────────────────────────────────────

    def _call(
        self,
        contact: Contact,
        campaign: Campaign,
        cv_text: str,
        user_prompt: str,
    ) -> ComposedEmail:
        system = self._system_prompt(campaign, cv_text)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        draft = self._request(system, messages)
        problems = validate_draft(draft, campaign)

        if problems:
            # Exactly one corrective retry. Truncating instead is not an option:
            # the ask is the last sentence, so cutting the body short produces
            # an email that stops before its own point.
            log.info("retrying composition for %s: %s", contact.email, " ".join(problems))
            messages = messages + [
                {"role": "assistant", "content": draft.model_dump_json()},
                {
                    "role": "user",
                    "content": CORRECTION_TEMPLATE.format(
                        problems=" ".join(problems),
                        max_sentences=campaign.max_sentences,
                    ),
                },
            ]
            draft = self._request(system, messages)
            problems = validate_draft(draft, campaign)

            if problems:
                raise CompositionError(
                    "the model could not produce a usable email after one retry: "
                    + " ".join(problems)
                )

        return ComposedEmail(
            subject=_clean_subject(draft.subject),
            body=render_message(
                draft.body,
                contact_name=contact.name,
                sender_name=self.sender_name,
                removal_line=campaign.removal_line,
                personal_website=self.personal_website,
            ),
        )

    def _system_prompt(self, campaign: Campaign, cv_text: str) -> str:
        return SYSTEM_TEMPLATE.format(
            persona=campaign.sender_persona.strip(),
            goal=campaign.goal.strip(),
            tone="\n".join(f"- {t}" for t in campaign.tone) or "- Plain and direct.",
            subject_guidance=campaign.subject_guidance.strip(),
            max_words=campaign.max_words,
            max_sentences=campaign.max_sentences,
            cv=cv_text.strip(),
        )

    def _request(self, system: str, messages: list[dict[str, Any]]) -> EmailDraft:
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
            messages=messages,
            output_format=EmailDraft,
        )

        if getattr(response, "stop_reason", None) == "refusal":
            raise CompositionError(
                "the model declined to write this email; check the campaign persona and CV"
            )

        draft = getattr(response, "parsed_output", None)
        if draft is None:
            raise CompositionError("the model returned no parsed output")
        return draft


def _company_of(contact: Contact) -> str:
    org = getattr(contact, "organization", None)
    if org is not None and getattr(org, "name", None):
        return org.name
    return "(unknown)"


def _clean_subject(raw: str) -> str:
    """A subject is one line.

    A line break is rejected by validate_draft before this runs; this only
    tidies whitespace, strips a stray reply prefix, and caps the length.
    """
    subject = " ".join((raw or "").split())
    subject = re.sub(r"^(re|fwd|subject)\s*:\s*", "", subject, flags=re.IGNORECASE).strip()
    if not subject:
        raise CompositionError("the model returned an empty subject line")
    return subject[:MAX_SUBJECT_CHARS]
