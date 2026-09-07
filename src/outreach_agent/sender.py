"""Delivery.

One interface, three implementations: SMTP over a Gmail app password (the
default), a Gmail API adapter that is deliberately not implemented yet, and a
dry-run sender that delivers nothing.

The dry-run guarantee in this project is structural. ``build_sender`` returns a
DryRunSender unless the caller explicitly asks for live delivery, so a dry run
does not merely skip the send — it never holds an object capable of sending.
"""

from __future__ import annotations

import logging
import smtplib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Any

log = logging.getLogger(__name__)


class SendError(Exception):
    """Delivery failed. The runner records it and moves to the next contact."""


@dataclass
class OutboundEmail:
    """One message, ready to deliver."""

    to_email: str
    subject: str
    body: str
    to_name: str | None = None
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)

    @property
    def to_header(self) -> str:
        return formataddr((self.to_name, self.to_email)) if self.to_name else self.to_email


@dataclass(frozen=True)
class SendResult:
    """What happened. ``message_id`` is what threads a future reply back."""

    message_id: str
    delivered: bool
    detail: str = ""


class Sender(ABC):
    """The interface every delivery backend implements."""

    @abstractmethod
    def send(self, message: OutboundEmail) -> SendResult:
        """Deliver one message, or raise SendError."""


class SmtpSender(Sender):
    """Gmail over SMTP with an app password.

    A connection is opened per message rather than held for the length of a
    run: sends are minutes apart by design, and a socket left open across a
    three-hour run is a socket that will be closed by someone else.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        from_address: str,
        from_name: str,
        smtp_factory: Callable[..., Any] = smtplib.SMTP_SSL,
        timeout: float = 30.0,
    ) -> None:
        if not username or not password:
            raise SendError("SMTP sender needs both a username and an app password")

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.from_name = from_name
        self.smtp_factory = smtp_factory
        self.timeout = timeout

    def send(self, message: OutboundEmail) -> SendResult:
        email_message, message_id = self.build_message(message)
        try:
            with self.smtp_factory(self.host, self.port, timeout=self.timeout) as smtp:
                smtp.login(self.username, self.password)
                smtp.send_message(email_message)
        except smtplib.SMTPAuthenticationError as exc:
            raise SendError(
                "Gmail rejected the credentials. GMAIL_APP_PASSWORD must be a 16-character "
                f"app password, not your account password. ({exc})"
            ) from exc
        except (smtplib.SMTPException, OSError) as exc:
            raise SendError(f"SMTP delivery failed: {exc}") from exc

        return SendResult(message_id=message_id, delivered=True)

    def build_message(self, message: OutboundEmail) -> tuple[EmailMessage, str]:
        """Build the MIME message and its Message-ID.

        Separated from send() so the headers can be asserted on without any
        delivery machinery involved.
        """
        domain = self.from_address.partition("@")[2] or None
        message_id = make_msgid(domain=domain)

        email_message = EmailMessage()
        email_message["Message-ID"] = message_id
        email_message["From"] = formataddr((self.from_name, self.from_address))
        email_message["To"] = message.to_header
        email_message["Subject"] = message.subject

        # A mailto, and no List-Unsubscribe-Post header. The one-click header
        # asserts that the URL accepts an unauthenticated POST; there is no web
        # server in this project, so claiming it would be false and is
        # penalised by some receivers.
        email_message["List-Unsubscribe"] = (
            f"<mailto:{self.from_address}?subject=unsubscribe>"
        )

        if message.in_reply_to:
            email_message["In-Reply-To"] = message.in_reply_to
            references = message.references or [message.in_reply_to]
            email_message["References"] = " ".join(references)

        email_message.set_content(message.body)
        return email_message, message_id


class GmailApiSender(Sender):
    """Placeholder for OAuth-based Gmail API delivery.

    Behind the same interface on purpose: switching backends later is a config
    change, not a rewrite. It raises rather than silently doing nothing, so
    nobody discovers the gap by wondering why no mail arrived.
    """

    def __init__(self, **_: Any) -> None:
        pass

    def send(self, message: OutboundEmail) -> SendResult:
        raise NotImplementedError(
            "The Gmail API sender is not implemented yet. Use SENDER_BACKEND=smtp "
            "with a Gmail app password."
        )


class DryRunSender(Sender):
    """Delivers nothing, ever. Records what would have been sent.

    This is what a run without --live is given. There is no code path from here
    to a socket.
    """

    def __init__(self, *, from_address: str = "dry-run@localhost") -> None:
        self.from_address = from_address
        self.outbox: list[OutboundEmail] = []

    def send(self, message: OutboundEmail) -> SendResult:
        self.outbox.append(message)
        domain = self.from_address.partition("@")[2] or "localhost"
        message_id = make_msgid(domain=f"dryrun.{domain}")
        log.info("[dry-run] would send to %s: %s", message.to_email, message.subject)
        return SendResult(message_id=message_id, delivered=False, detail="dry run")


def build_sender(config: Any, *, live: bool) -> Sender:
    """Choose a delivery backend.

    The only place a live sender is constructed. A caller that does not ask for
    live delivery cannot accidentally receive one.
    """
    if not live:
        return DryRunSender(
            from_address=getattr(config, "gmail_address", "") or "dry-run@localhost"
        )

    backend = getattr(config, "sender_backend", "smtp")
    if backend == "gmail_api":
        return GmailApiSender()

    return SmtpSender(
        host=config.smtp_host,
        port=config.smtp_port,
        username=config.gmail_address,
        password=config.gmail_app_password,
        from_address=config.gmail_address,
        from_name=config.sender_name,
    )
