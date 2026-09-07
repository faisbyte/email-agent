"""Persistence: models, sessions, and every query the rest of the app makes.

Two invariants live here and are never bypassed by anything upstream:

1. ``contacts.email`` is UNIQUE, and every write and lookup normalises through
   ``normalize_email`` so casing cannot dodge the constraint.
2. ``is_suppressed`` matches the full address *and* the bare domain, and the
   runner calls it immediately before every send rather than once at selection
   time.

Dialect-agnostic by construction: nothing in this module branches on the
database backend. Swapping DATABASE_URL to Postgres later needs no code change.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

# ─────────────────────────── time handling ───────────────────────────
#
# Every timestamp written is timezone-aware UTC. SQLite hands them back naive,
# so anything that does arithmetic on a value read from the database passes it
# through as_utc() first. Skipping this is how a daily cap silently resets at
# the wrong hour.


def utcnow() -> datetime:
    """Current time, timezone-aware, UTC. The only clock this app reads."""
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to a naive datetime read back from the database."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_day_start(moment: datetime | None = None) -> datetime:
    """Midnight UTC of the day containing ``moment``."""
    now = moment or utcnow()
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def normalize_email(raw: str) -> str:
    """Lowercase and strip an address.

    Applied on every write and every lookup. Without it, ``Bob@Corp.com`` and
    ``bob@corp.com`` are two rows and the same human is emailed twice.
    """
    return (raw or "").strip().lower()


def email_domain(raw: str) -> str:
    """Bare domain of an address, lowercased. Empty string if malformed."""
    normalized = normalize_email(raw)
    _, _, domain = normalized.partition("@")
    return domain


# ─────────────────────────── enums ───────────────────────────


class OutreachStatus(enum.StrEnum):
    """Lifecycle of a single outbound message.

    DRY_RUN rows are rehearsals: they record what would have been sent and
    deliberately do not count as contact.
    """

    DRY_RUN = "dry_run"
    SENT = "sent"
    FAILED = "failed"
    BOUNCED = "bounced"


class ReplyClassification(enum.StrEnum):
    """What a human reply meant. A bounce is not here — it is a status."""

    INTERESTED = "interested"
    REJECTION = "rejection"
    AUTO_REPLY = "auto_reply"
    OPT_OUT = "opt_out"


class RunMode(enum.StrEnum):
    DRY_RUN = "dry_run"
    LIVE = "live"


#: Statuses meaning "this person has been contacted, do not contact again".
#: FAILED is included on purpose: an SMTP error can be raised *after* the
#: message was handed off, and a duplicate email is worse than a missed one.
#: Retrying a failure is a deliberate manual act, never automatic.
CONTACTED_STATUSES = frozenset(
    {OutreachStatus.SENT.value, OutreachStatus.FAILED.value, OutreachStatus.BOUNCED.value}
)

#: Replies that came from a person. Any of these stops all follow-ups.
#: AUTO_REPLY is excluded — an out-of-office is not a human saying no.
HUMAN_REPLY_CLASSIFICATIONS = frozenset(
    {
        ReplyClassification.INTERESTED.value,
        ReplyClassification.REJECTION.value,
        ReplyClassification.OPT_OUT.value,
    }
)


# ─────────────────────────── models ───────────────────────────


class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    domain: Mapped[str | None] = mapped_column(String(255), index=True)
    apollo_org_id: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    contacts: Mapped[list[Contact]] = relationship(back_populates="organization")

    def __repr__(self) -> str:
        return f"<Organization {self.id} {self.name!r} {self.domain!r}>"


class Contact(Base):
    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int | None] = mapped_column(ForeignKey("organizations.id"), index=True)
    apollo_person_id: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(255))

    # The invariant. Normalised on every write and lookup.
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)

    linkedin_url: Mapped[str | None] = mapped_column(String(512))
    location: Mapped[str | None] = mapped_column(String(255))
    revealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    organization: Mapped[Organization | None] = relationship(back_populates="contacts")
    outreach: Mapped[list[Outreach]] = relationship(
        back_populates="contact", order_by="Outreach.id"
    )

    @property
    def first_name(self) -> str:
        return (self.name or "").strip().split(" ")[0] if self.name else ""

    def __repr__(self) -> str:
        return f"<Contact {self.id} {self.email!r}>"


class Outreach(Base):
    """One outbound message. Append-only: a follow-up is a new row.

    Each sent message has its own ``message_id``, and reply threading matches on
    it, so one row per message is what makes threading work at all.
    ``follow_up_count`` is the ordinal within the thread: 0 for the initial
    send, 1 for the first follow-up.
    """

    __tablename__ = "outreach"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"), nullable=False, index=True)
    campaign: Mapped[str | None] = mapped_column(String(128), index=True)
    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    message_id: Mapped[str | None] = mapped_column(String(512), unique=True, index=True)
    follow_up_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    replied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reply_classification: Mapped[str | None] = mapped_column(String(16), index=True)
    reply_body: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    contact: Mapped[Contact] = relationship(back_populates="outreach")

    def __repr__(self) -> str:
        return f"<Outreach {self.id} contact={self.contact_id} {self.status}>"


class Suppression(Base):
    """A do-not-contact entry: either a full address or a bare domain."""

    __tablename__ = "suppressions"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_or_domain: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True
    )
    reason: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    def __repr__(self) -> str:
        return f"<Suppression {self.email_or_domain!r}>"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sent_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mode: Mapped[str | None] = mapped_column(String(16))
    campaign: Mapped[str | None] = mapped_column(String(128))

    def __repr__(self) -> str:
        return f"<Run {self.id} {self.mode} sent={self.sent_count} errors={self.error_count}>"


# ─────────────────────────── engine / session ───────────────────────────


def make_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create an engine for any SQLAlchemy URL.

    No dialect branching. SQLite gets one connection-arg tweak so a session can
    be used across the run; everything else uses driver defaults.
    """
    kwargs: dict[str, object] = {"echo": echo, "future": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    """Create the schema. Used by `outreach init-db` and by the test fixtures.

    This is not a migration tool and does not pretend to be one.
    """
    Base.metadata.create_all(engine)


# ─────────────────────────── writes ───────────────────────────


def upsert_organization(
    session: Session,
    *,
    name: str | None = None,
    domain: str | None = None,
    apollo_org_id: str | None = None,
) -> Organization | None:
    """Insert an organization, or return the existing one.

    Matches on apollo_org_id first, then domain. Returns None when there is
    nothing identifying to store.
    """
    domain = (domain or "").strip().lower() or None
    apollo_org_id = (apollo_org_id or "").strip() or None
    if not any([name, domain, apollo_org_id]):
        return None

    existing: Organization | None = None
    if apollo_org_id:
        existing = session.scalar(
            select(Organization).where(Organization.apollo_org_id == apollo_org_id)
        )
    if existing is None and domain:
        existing = session.scalar(select(Organization).where(Organization.domain == domain))

    if existing is not None:
        # Backfill anything we did not know before, never overwrite.
        existing.name = existing.name or name
        existing.domain = existing.domain or domain
        existing.apollo_org_id = existing.apollo_org_id or apollo_org_id
        return existing

    org = Organization(name=name, domain=domain, apollo_org_id=apollo_org_id)
    session.add(org)
    session.flush()
    return org


def upsert_contact(
    session: Session,
    *,
    email: str,
    name: str | None = None,
    title: str | None = None,
    org: Organization | None = None,
    apollo_person_id: str | None = None,
    linkedin_url: str | None = None,
    location: str | None = None,
    revealed_at: datetime | None = None,
) -> Contact:
    """Insert a contact, or return the existing row for that address.

    The insert runs inside a SAVEPOINT. Catching IntegrityError on a bare
    session poisons the enclosing transaction and every later write in the run
    fails; scoping it to a savepoint means the rollback undoes only the failed
    insert.
    """
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        raise ValueError(f"{email!r} is not a usable email address")

    existing = _find_contact(session, normalized, apollo_person_id)
    if existing is not None:
        return _backfill_contact(existing, name, title, org, apollo_person_id, linkedin_url,
                                 location, revealed_at)

    contact = Contact(
        email=normalized,
        name=name,
        title=title,
        org_id=org.id if org is not None else None,
        apollo_person_id=(apollo_person_id or None),
        linkedin_url=linkedin_url,
        location=location,
        revealed_at=revealed_at,
    )
    try:
        with session.begin_nested():
            session.add(contact)
            session.flush()
    except IntegrityError:
        # Someone else inserted the same person between our SELECT and our
        # INSERT. The savepoint rolled back, which already discarded the pending
        # object — expunging it here would raise. The outer transaction is
        # intact, so simply re-read the row that won. Either unique column
        # (email or apollo_person_id) may be the one that collided.
        existing = _find_contact(session, normalized, apollo_person_id)
        if existing is None:  # pragma: no cover — an unexpected constraint
            raise
        return _backfill_contact(existing, name, title, org, apollo_person_id, linkedin_url,
                                 location, revealed_at)
    return contact


def _find_contact(
    session: Session, normalized_email: str, apollo_person_id: str | None
) -> Contact | None:
    """Look a contact up by either of its unique columns."""
    found = session.scalar(select(Contact).where(Contact.email == normalized_email))
    if found is not None:
        return found
    if apollo_person_id:
        return session.scalar(
            select(Contact).where(Contact.apollo_person_id == apollo_person_id)
        )
    return None


def _backfill_contact(
    contact: Contact,
    name: str | None,
    title: str | None,
    org: Organization | None,
    apollo_person_id: str | None,
    linkedin_url: str | None,
    location: str | None,
    revealed_at: datetime | None,
) -> Contact:
    """Fill in fields we did not have before. Never overwrite known values."""
    contact.name = contact.name or name
    contact.title = contact.title or title
    contact.org_id = contact.org_id or (org.id if org is not None else None)
    contact.apollo_person_id = contact.apollo_person_id or (apollo_person_id or None)
    contact.linkedin_url = contact.linkedin_url or linkedin_url
    contact.location = contact.location or location
    contact.revealed_at = contact.revealed_at or revealed_at
    return contact


def add_suppression(
    session: Session, email_or_domain: str, reason: str | None = None
) -> Suppression:
    """Add a do-not-contact entry. Idempotent.

    Accepts either a full address or a bare domain. Suppressing a domain kills
    every address on it.
    """
    key = normalize_email(email_or_domain)
    if not key:
        raise ValueError("cannot suppress an empty value")

    existing = session.scalar(select(Suppression).where(Suppression.email_or_domain == key))
    if existing is not None:
        return existing

    entry = Suppression(email_or_domain=key, reason=reason)
    try:
        with session.begin_nested():
            session.add(entry)
            session.flush()
    except IntegrityError:
        # As in upsert_contact: the savepoint rollback discarded the pending row.
        found = session.scalar(select(Suppression).where(Suppression.email_or_domain == key))
        if found is None:  # pragma: no cover
            raise
        return found
    return entry


def record_send(
    session: Session,
    *,
    contact: Contact,
    campaign: str | None,
    subject: str,
    body: str,
    status: OutreachStatus,
    message_id: str | None = None,
    error: str | None = None,
    follow_up_count: int = 0,
) -> Outreach:
    """Write one outbound message to history.

    ``sent_at`` is set only for real deliveries; a dry-run row is dated by
    ``created_at`` so it can be inspected without polluting the daily count.
    """
    row = Outreach(
        contact_id=contact.id,
        campaign=campaign,
        subject=subject,
        body=body,
        status=status.value,
        message_id=message_id,
        error=error,
        follow_up_count=follow_up_count,
        sent_at=utcnow() if status is OutreachStatus.SENT else None,
    )
    session.add(row)
    session.flush()
    return row


def record_reply(
    session: Session,
    *,
    message_id: str,
    classification: ReplyClassification,
    body: str | None = None,
) -> Outreach | None:
    """Attach a classified reply to the message it answers.

    Returns None when the In-Reply-To header does not match anything we sent,
    which is normal for unrelated mail in the inbox.
    """
    row = session.scalar(select(Outreach).where(Outreach.message_id == message_id))
    if row is None:
        return None
    row.replied_at = utcnow()
    row.reply_classification = classification.value
    row.reply_body = body
    session.flush()
    return row


def record_bounce(session: Session, *, message_id: str) -> Outreach | None:
    """Mark a sent message as bounced.

    Called by inbox.py's deterministic bounce detection, which runs before any
    model call. This is what makes BOUNCED a status the code actually writes.
    """
    row = session.scalar(select(Outreach).where(Outreach.message_id == message_id))
    if row is None:
        return None
    row.status = OutreachStatus.BOUNCED.value
    session.flush()
    return row


def start_run(session: Session, *, mode: RunMode, campaign: str | None = None) -> Run:
    run = Run(mode=mode.value, campaign=campaign, started_at=utcnow())
    session.add(run)
    session.flush()
    return run


def finish_run(session: Session, run: Run, *, sent_count: int, error_count: int) -> Run:
    run.finished_at = utcnow()
    run.sent_count = sent_count
    run.error_count = error_count
    session.flush()
    return run


# ─────────────────────────── reads ───────────────────────────


def is_suppressed(session: Session, email: str) -> bool:
    """True if this address, or its whole domain, is suppressed.

    Called immediately before every send. Not once at selection time — a
    suppression added mid-run must take effect on the very next send.
    """
    normalized = normalize_email(email)
    if not normalized:
        return True  # An address we cannot read is one we must not mail.

    domain = email_domain(normalized)
    keys = [normalized] + ([domain] if domain else [])
    found = session.scalar(
        select(func.count()).select_from(Suppression).where(Suppression.email_or_domain.in_(keys))
    )
    return bool(found)


def already_contacted(session: Session, email: str) -> bool:
    """True if this address has ever been sent a message.

    Global across campaigns by design: one address, one email, ever. Dry-run
    rows do not count; sent, failed and bounced do.
    """
    normalized = normalize_email(email)
    if not normalized:
        return False

    count = session.scalar(
        select(func.count())
        .select_from(Outreach)
        .join(Contact, Contact.id == Outreach.contact_id)
        .where(Contact.email == normalized, Outreach.status.in_(CONTACTED_STATUSES))
    )
    return bool(count)


def has_human_reply(session: Session, contact_id: int) -> bool:
    """True if a person has replied to any message in this contact's history.

    One of these stops every future follow-up. An auto-reply does not count.
    """
    count = session.scalar(
        select(func.count())
        .select_from(Outreach)
        .where(
            Outreach.contact_id == contact_id,
            Outreach.reply_classification.in_(HUMAN_REPLY_CLASSIFICATIONS),
        )
    )
    return bool(count)


def sent_today(session: Session, *, now: datetime | None = None) -> int:
    """Messages actually sent since midnight UTC.

    Counted from the database, not from a loop variable, so the daily cap
    survives a crash, a restart, and three terminals open at once.
    """
    since = utc_day_start(now)
    count = session.scalar(
        select(func.count())
        .select_from(Outreach)
        .where(Outreach.status == OutreachStatus.SENT.value, Outreach.sent_at >= since)
    )
    return int(count or 0)


def candidates(session: Session, *, limit: int) -> list[Contact]:
    """Contacts that have never been mailed and are not suppressed.

    Takes no campaign argument on purpose. Deduplication is global — one
    address, one email, ever — so campaign-scoped selection would contradict
    the guarantee. Which campaign composed a send is recorded on the outreach
    row for reporting, and is never a filter here.

    The runner re-checks suppression and prior contact immediately before each
    send anyway; this query is an optimisation, never the safety net.
    """
    if limit <= 0:
        return []

    contacted = (
        select(Outreach.contact_id)
        .where(Outreach.status.in_(CONTACTED_STATUSES))
        .scalar_subquery()
    )
    suppressed_keys = select(Suppression.email_or_domain).scalar_subquery()

    stmt = (
        select(Contact)
        .where(
            Contact.email.is_not(None),
            Contact.email != "",
            Contact.id.not_in(contacted),
            Contact.email.not_in(suppressed_keys),
        )
        .order_by(Contact.id)
        .limit(limit)
    )
    rows = list(session.scalars(stmt))
    # Domain-level suppression is not expressible portably in the subquery
    # above, so it is applied here. is_suppressed remains the authority.
    return [c for c in rows if not is_suppressed(session, c.email)]


def follow_up_candidates(
    session: Session,
    *,
    limit: int,
    max_follow_ups: int,
    days_between: int,
    now: datetime | None = None,
) -> list[Outreach]:
    """Latest sent message for each contact that is due a follow-up.

    The inverse of candidates(): this selects on the *presence* of a sent row,
    not its absence. Every follow-up target is by definition already contacted,
    so applying already_contacted() here would skip every candidate.

    Requires: a sent message at least ``days_between`` days old, fewer than
    ``max_follow_ups`` follow-ups so far, no human reply, no suppression.
    """
    if limit <= 0 or max_follow_ups <= 0:
        return []

    cutoff = (now or utcnow()) - timedelta(days=days_between)

    # Contacts with any human reply are out entirely.
    replied = (
        select(Outreach.contact_id)
        .where(Outreach.reply_classification.in_(HUMAN_REPLY_CLASSIFICATIONS))
        .scalar_subquery()
    )
    # Bounced addresses are dead; do not keep writing to them.
    bounced = (
        select(Outreach.contact_id)
        .where(Outreach.status == OutreachStatus.BOUNCED.value)
        .scalar_subquery()
    )

    stmt = (
        select(Outreach)
        .join(Contact, Contact.id == Outreach.contact_id)
        .where(
            Outreach.status == OutreachStatus.SENT.value,
            Outreach.contact_id.not_in(replied),
            Outreach.contact_id.not_in(bounced),
        )
        .order_by(Outreach.contact_id, Outreach.follow_up_count.desc(), Outreach.id.desc())
    )

    latest_per_contact: dict[int, Outreach] = {}
    for row in session.scalars(stmt):
        latest_per_contact.setdefault(row.contact_id, row)

    due: list[Outreach] = []
    for row in latest_per_contact.values():
        if row.follow_up_count >= max_follow_ups:
            continue
        sent_at = as_utc(row.sent_at)
        if sent_at is None or sent_at > cutoff:
            continue
        if is_suppressed(session, row.contact.email):
            continue
        due.append(row)
        if len(due) >= limit:
            break
    return due


def counts_by_status(session: Session) -> dict[str, int]:
    """Outreach row counts per status, for `outreach stats`."""
    rows = session.execute(
        select(Outreach.status, func.count()).group_by(Outreach.status)
    ).all()
    return {status: int(count) for status, count in rows}
