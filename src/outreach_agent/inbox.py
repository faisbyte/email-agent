"""Reading replies.

This is the second and last place Claude is called, and it is called as late as
possible. Two deterministic checks run first:

* **Bounces** are detected from headers. A delivery failure is not a reply and
  must never be classified as one.
* **Explicit opt-outs** are matched by regex. Someone who wrote "remove me"
  gets removed whether or not a model agrees, and it costs nothing.

Only what survives both reaches the classifier.
"""

from __future__ import annotations

import email
import imaplib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from email.message import Message
from email.utils import parseaddr
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .store import (
    Outreach,
    ReplyClassification,
    add_suppression,
    email_domain,
    record_bounce,
    record_reply,
    utcnow,
)

log = logging.getLogger(__name__)

MESSAGE_ID_PATTERN = re.compile(r"<[^<>@\s]+@[^<>\s]+>")

#: Senders that mean "this is a delivery report", not "a person replied".
BOUNCE_SENDERS = ("mailer-daemon", "postmaster", "no-reply@", "noreply@")

#: Content types used by delivery status notifications.
BOUNCE_CONTENT_TYPES = ("multipart/report", "message/delivery-status")

#: Unambiguous removal requests. Matched before any model call: a person who
#: wrote this does not need a second opinion, and the check is free.
OPT_OUT_PATTERNS = (
    r"\bunsubscribe\b",
    r"\bremove me\b",
    r"\btake me off\b",
    r"\bdo not (?:contact|email|write)\b",
    r"\bdon'?t (?:contact|email|write) me\b",
    r"\bopt[- ]?out\b",
    r"\bstop (?:emailing|contacting)\b",
    r"\bno thanks\b",
)
_OPT_OUT_RE = re.compile("|".join(OPT_OUT_PATTERNS), re.IGNORECASE)

CLASSIFIER_SYSTEM = """\
You classify replies to a cold outreach email into exactly one category.

interested  — any openness at all: curiosity, a question, a referral to a \
colleague, a request for more information, or a yes.
rejection   — a person declining: no openings, not a fit, not interested, \
already filled, now is not a good time.
auto_reply  — machine-generated: out of office, vacation autoresponder, ticket \
acknowledgement, "we received your message".
opt_out     — an explicit demand to stop contacting them or to be removed.

Rules:
- A human saying "not right now, try in six months" is a rejection, not an \
opt_out. Reserve opt_out for an actual demand to stop.
- An out-of-office that also says "contact my colleague" is still auto_reply.
- When genuinely torn between rejection and opt_out, choose opt_out. Removing \
someone who did not insist costs one contact; keeping someone who did is worse.
"""


class ReplyVerdict(BaseModel):
    """Structured output contract for the classifier."""

    classification: Literal["interested", "rejection", "auto_reply", "opt_out"] = Field(
        description="Exactly one category."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


@dataclass
class IncomingMessage:
    """A message pulled from the inbox."""

    message_id: str | None = None
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)
    from_address: str = ""
    subject: str = ""
    body: str = ""
    is_bounce: bool = False
    raw_text: str = ""

    def candidate_message_ids(self) -> list[str]:
        """Message-IDs this could be a reply to, most reliable first.

        In-Reply-To and References are the correct answer for a human reply. A
        bounce usually carries the original only inside the attached report, so
        the raw text is scanned as a fallback.
        """
        ids: list[str] = []
        if self.in_reply_to:
            ids.append(self.in_reply_to)
        ids.extend(ref for ref in self.references if ref not in ids)
        for found in MESSAGE_ID_PATTERN.findall(self.raw_text or ""):
            if found not in ids and found != self.message_id:
                ids.append(found)
        return ids


@dataclass
class ReplyOutcome:
    """What was done about one incoming message."""

    message: IncomingMessage
    matched_message_id: str | None
    classification: ReplyClassification | None
    is_bounce: bool = False
    suppressed: bool = False
    used_model: bool = False


@dataclass
class ReplyReport:
    outcomes: list[ReplyOutcome] = field(default_factory=list)
    unmatched: int = 0

    @property
    def bounces(self) -> int:
        return sum(1 for o in self.outcomes if o.is_bounce)

    @property
    def suppressions(self) -> int:
        return sum(1 for o in self.outcomes if o.suppressed)

    def count(self, classification: ReplyClassification) -> int:
        return sum(1 for o in self.outcomes if o.classification is classification)


# ─────────────────────────── detection ───────────────────────────


def looks_like_bounce(message: IncomingMessage | Message) -> bool:
    """True for a delivery status notification.

    Deterministic and header-based. A bounce is a fact about delivery, not an
    opinion about content, so no model is involved.
    """
    if isinstance(message, IncomingMessage):
        sender = (message.from_address or "").lower()
        return message.is_bounce or any(marker in sender for marker in BOUNCE_SENDERS)

    sender = (message.get("From") or "").lower()
    content_type = (message.get_content_type() or "").lower()
    report_type = (message.get_param("report-type") or "").lower()

    if any(marker in sender for marker in BOUNCE_SENDERS):
        return True
    if content_type in BOUNCE_CONTENT_TYPES:
        return True
    return report_type == "delivery-status"


def looks_like_opt_out(text: str) -> bool:
    """True for an unmistakable removal request."""
    return bool(_OPT_OUT_RE.search(text or ""))


# ─────────────────────────── classification ───────────────────────────


class ReplyClassifier:
    """Classifies a human reply into one of four categories."""

    def __init__(self, client: Any, *, model: str, max_tokens: int = 512) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def classify(self, body: str, *, subject: str = "") -> ReplyClassification:
        """Classify one reply.

        The regex pre-check runs first and short-circuits an explicit opt-out
        without spending a token.
        """
        if looks_like_opt_out(body) or looks_like_opt_out(subject):
            return ReplyClassification.OPT_OUT

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=CLASSIFIER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Subject: {subject}\n\nReply:\n{body}",
                }
            ],
            output_format=ReplyVerdict,
        )

        verdict = getattr(response, "parsed_output", None)
        if verdict is None:
            # Unreadable answer: treat as a human rejection rather than guessing
            # optimistically. It stops follow-ups, which is the safe direction.
            log.warning("classifier returned no parsed output; defaulting to rejection")
            return ReplyClassification.REJECTION

        return ReplyClassification(verdict.classification)


# ─────────────────────────── IMAP ───────────────────────────


class InboxPoller:
    """Fetches recent messages over IMAP.

    The IMAP class is injected, so tests exercise parsing and matching without
    a connection.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        address: str,
        password: str,
        folder: str = "INBOX",
        imap_factory: Callable[..., Any] = imaplib.IMAP4_SSL,
    ) -> None:
        self.host = host
        self.port = port
        self.address = address
        self.password = password
        self.folder = folder
        self.imap_factory = imap_factory

    def fetch_replies(self, *, since_days: int = 7) -> list[IncomingMessage]:
        since = (utcnow() - timedelta(days=since_days)).strftime("%d-%b-%Y")
        messages: list[IncomingMessage] = []

        client = self.imap_factory(self.host, self.port)
        try:
            client.login(self.address, self.password)
            client.select(self.folder)
            status, data = client.search(None, f'(SINCE "{since}")')
            if status != "OK":
                log.warning("IMAP search failed with status %s", status)
                return []

            for uid in _uids(data):
                status, payload = client.fetch(uid, "(RFC822)")
                if status != "OK" or not payload:
                    continue
                raw = _first_bytes(payload)
                if raw is None:
                    continue
                messages.append(parse_message(raw))
        finally:
            try:
                client.logout()
            except Exception:  # pragma: no cover — best-effort cleanup
                log.debug("IMAP logout failed", exc_info=True)

        return messages


def _uids(data: Any) -> list[bytes]:
    if not data or not data[0]:
        return []
    first = data[0]
    if isinstance(first, bytes):
        return first.split()
    return []


def _first_bytes(payload: Any) -> bytes | None:
    for part in payload:
        if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], bytes):
            return part[1]
    return None


def parse_message(raw: bytes) -> IncomingMessage:
    """Parse a raw RFC822 message into the shape this module works with."""
    parsed = email.message_from_bytes(raw)
    references = (parsed.get("References") or "").split()

    return IncomingMessage(
        message_id=parsed.get("Message-ID"),
        in_reply_to=(parsed.get("In-Reply-To") or "").strip() or None,
        references=[r.strip() for r in references if r.strip()],
        from_address=parseaddr(parsed.get("From") or "")[1],
        subject=parsed.get("Subject") or "",
        body=_extract_body(parsed),
        is_bounce=looks_like_bounce(parsed),
        raw_text=raw.decode("utf-8", errors="replace"),
    )


def _extract_body(parsed: Message) -> str:
    """Best-effort plain-text body."""
    if parsed.is_multipart():
        for part in parsed.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    return payload.decode("utf-8", errors="replace").strip()
        return ""

    payload = parsed.get_payload(decode=True)
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace").strip()
    raw = parsed.get_payload()
    return raw.strip() if isinstance(raw, str) else ""


# ─────────────────────────── consequences ───────────────────────────


def process_replies(
    session: Session,
    messages: list[IncomingMessage],
    classifier: ReplyClassifier,
) -> ReplyReport:
    """Apply the consequences of every reply.

    Order matters:

    1. Match the message to something we sent. Unrelated inbox mail is ignored.
    2. A bounce is recorded as a bounce and never classified.
    3. An opt-out writes a suppression **immediately**, in the same
       transaction as the reply itself, so a crash cannot leave a person
       recorded as opted out but still mailable.
    4. Any human reply stops all future follow-ups by virtue of being stored.
    """
    report = ReplyReport()

    for message in messages:
        matched = _match(session, message)
        if matched is None:
            report.unmatched += 1
            log.debug("ignoring unmatched inbox message from %s", message.from_address)
            continue

        if looks_like_bounce(message):
            record_bounce(session, message_id=matched)
            session.commit()
            report.outcomes.append(
                ReplyOutcome(message, matched, classification=None, is_bounce=True)
            )
            continue

        used_model = not (
            looks_like_opt_out(message.body) or looks_like_opt_out(message.subject)
        )
        classification = classifier.classify(message.body, subject=message.subject)

        record_reply(
            session,
            message_id=matched,
            classification=classification,
            body=message.body,
        )

        suppressed = False
        if classification is ReplyClassification.OPT_OUT:
            target = message.from_address or ""
            if target:
                add_suppression(session, target, reason="opt_out")
                suppressed = True
                log.info("suppressed %s at their request", target)

        # One commit covers the reply and its suppression together.
        session.commit()

        report.outcomes.append(
            ReplyOutcome(
                message=message,
                matched_message_id=matched,
                classification=classification,
                suppressed=suppressed,
                used_model=used_model,
            )
        )

    return report


def _match(session: Session, message: IncomingMessage) -> str | None:
    """Find the message-id of ours that this incoming message answers."""
    for candidate in message.candidate_message_ids():
        found = session.scalar(
            select(Outreach.message_id).where(Outreach.message_id == candidate)
        )
        if found:
            return found
    return None


def suppress_domain_of(session: Session, address: str, reason: str = "manual") -> None:
    """Suppress every address at the same domain."""
    domain = email_domain(address)
    if domain:
        add_suppression(session, domain, reason=reason)
