"""The orchestration loop.

Ordinary, deterministic Python. It decides who gets written to, enforces the
caps, and records what happened. Claude is called by the composer for the words
and by the classifier for the replies; neither decides anything here.

Three properties this module is built around:

* **Dry-run is structural.** A live sender is constructed only by
  ``build_sender(..., live=True)``. Nothing in this loop branches into sending;
  it sends through whatever sender it was handed, and in a dry run that object
  cannot deliver.
* **The cap is counted from the database**, at the top of every iteration, so
  it survives a crash, a restart, and two terminals running at once.
* **Suppression and prior contact are re-checked immediately before each
  send**, not once when candidates were selected. A suppression added during a
  run takes effect on the very next message.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .apollo_client import ApolloClient, ApolloOutOfCredits
from .campaigns import Campaign
from .composer import Composer, has_footer
from .config import Config
from .sender import OutboundEmail, Sender, SendError
from .store import (
    Contact,
    Outreach,
    OutreachStatus,
    RunMode,
    already_contacted,
    candidates,
    finish_run,
    follow_up_candidates,
    has_human_reply,
    is_suppressed,
    normalize_email,
    record_send,
    sent_today,
    start_run,
    upsert_contact,
    upsert_organization,
    utcnow,
)

log = logging.getLogger(__name__)

#: The ceiling configuration cannot raise. A DAILY_CAP above this is clamped,
#: loudly, every run. It exists so a typo in .env cannot turn a careful tool
#: into a spam cannon.
HARD_DAILY_CAP = 50

#: Consecutive send failures that abort a run. A wrong app password should cost
#: five attempts, not the whole list.
MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class RunResult:
    """What a run did. Returned whether it completed or stopped early."""

    run_id: int | None = None
    mode: str = RunMode.DRY_RUN.value
    sent: int = 0
    skipped: int = 0
    failed: int = 0
    cap: int = 0
    stopped_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    skips: dict[str, int] = field(default_factory=dict)

    def note_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skips[reason] = self.skips.get(reason, 0) + 1

    @property
    def summary(self) -> str:
        parts = [f"sent={self.sent}", f"skipped={self.skipped}", f"failed={self.failed}"]
        if self.stopped_reason:
            parts.append(f"stopped={self.stopped_reason}")
        return " ".join(parts)


def effective_daily_cap(configured: int) -> int:
    """The cap actually applied.

    Configuration can lower the ceiling but never raise it.
    """
    if configured > HARD_DAILY_CAP:
        log.warning(
            "DAILY_CAP=%d exceeds the hard ceiling of %d and has been clamped. "
            "The ceiling lives in code and configuration cannot raise it.",
            configured,
            HARD_DAILY_CAP,
        )
        return HARD_DAILY_CAP
    return max(0, configured)


# ─────────────────────────── discovery ───────────────────────────


def discover(
    config: Config,
    campaign: Campaign,
    session: Session,
    apollo: ApolloClient,
    *,
    limit: int = 25,
    per_page: int = 25,
) -> dict[str, int]:
    """Search Apollo and store the people found. Sends nothing.

    Enrichment spends a credit, so suppression and prior contact are checked
    *before* paying to reveal an address. Paying to learn the email of someone
    already emailed is the most avoidable waste in the whole system.
    """
    stats = {"seen": 0, "enriched": 0, "stored": 0, "skipped": 0, "credits": 0}
    page = 1

    while stats["stored"] < limit:
        try:
            people = apollo.search_people(campaign.apollo, page=page, per_page=per_page)
        except ApolloOutOfCredits as exc:
            log.warning("stopping discovery: %s", exc)
            stats["stopped_out_of_credits"] = 1
            break

        if not people:
            break

        for person in people:
            if stats["stored"] >= limit:
                break
            stats["seen"] += 1

            known_email = normalize_email(person.email or "")
            if known_email and _should_skip_email(session, known_email):
                stats["skipped"] += 1
                continue

            if not person.has_usable_email:
                try:
                    enriched = apollo.enrich_person(person.apollo_id)
                except ApolloOutOfCredits as exc:
                    log.warning("stopping discovery: %s", exc)
                    stats["stopped_out_of_credits"] = 1
                    stats["credits"] = apollo.credits_used
                    return stats
                stats["enriched"] += 1
                if enriched is None or not enriched.has_usable_email:
                    stats["skipped"] += 1
                    continue
                person = enriched

            email = normalize_email(person.email or "")
            if not email or _should_skip_email(session, email):
                stats["skipped"] += 1
                continue

            org = upsert_organization(
                session,
                name=person.organization_name,
                domain=person.organization_domain,
                apollo_org_id=person.apollo_org_id,
            )
            upsert_contact(
                session,
                email=email,
                name=person.name,
                title=person.title,
                org=org,
                apollo_person_id=person.apollo_id,
                linkedin_url=person.linkedin_url,
                location=person.location,
                revealed_at=utcnow(),
            )
            session.commit()
            stats["stored"] += 1

        page += 1

    stats["credits"] = apollo.credits_used
    return stats


def _should_skip_email(session: Session, email: str) -> bool:
    return is_suppressed(session, email) or already_contacted(session, email)


# ─────────────────────────── the send loop ───────────────────────────


def run(
    config: Config,
    campaign: Campaign,
    session: Session,
    composer: Composer,
    sender: Sender,
    cv_text: str,
    *,
    live: bool = False,
    limit: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> RunResult:
    """Send the initial email to contacts who have never been written to.

    ``live`` here is a label for the record, not a switch: whether mail can
    actually leave is decided by which sender the caller built.
    """
    rng = rng or random.Random()
    mode = RunMode.LIVE if live else RunMode.DRY_RUN
    result = RunResult(mode=mode.value, cap=effective_daily_cap(config.daily_cap))

    run_row = start_run(session, mode=mode, campaign=campaign.slug)
    session.commit()
    result.run_id = run_row.id

    consecutive_failures = 0

    try:
        pool = candidates(session, limit=(limit or result.cap) * 3 or 1)
        log.info("run %s: %d candidate(s), cap %d", run_row.id, len(pool), result.cap)

        for contact in pool:
            # Recomputed every iteration, from the database. Two processes
            # running at once still cannot exceed the daily cap between them.
            already_sent = sent_today(session)
            if already_sent >= result.cap:
                result.stopped_reason = f"daily cap of {result.cap} reached"
                log.info("stopping: %s", result.stopped_reason)
                break

            if limit is not None and result.sent >= limit:
                result.stopped_reason = f"run limit of {limit} reached"
                break

            # Cheap pre-filter, so a skipped contact costs neither a pause nor
            # a composition call.
            if not _clear_to_send(session, contact, result):
                continue

            try:
                email = composer.compose(contact, campaign, cv_text)
            except Exception as exc:  # noqa: BLE001 — one bad contact must not end the run
                # Deliberately no outreach row. A `failed` row means "this may
                # have reached SMTP", which makes already_contacted() true for
                # this address forever. A contact we never managed to write an
                # email for has not been contacted, and must stay eligible for
                # the next run. The run's error_count still records the fault.
                result.failed += 1
                result.errors.append(f"compose failed for {contact.email}: {exc}")
                log.warning("compose failed for %s: %s", contact.email, exc)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    result.stopped_reason = (
                        f"{consecutive_failures} consecutive failures"
                    )
                    break
                continue

            # Last gate before delivery. The composer guarantees this; checking
            # it here means a future change to the composer cannot quietly send
            # an email with no way out of the list.
            if not has_footer(email.body, campaign.removal_line):
                # As above: refused before delivery, so no row is written and
                # the contact remains eligible.
                result.failed += 1
                result.errors.append(f"missing removal line for {contact.email}")
                log.error("refusing to send to %s: removal line missing", contact.email)
                continue

            # Pause before the send rather than after it. That yields exactly
            # one gap between consecutive sends and, crucially, none after the
            # last one — a run must not end by sitting idle for three minutes.
            if result.sent > 0:
                _pause(sleep, rng, config, live=live)

            # The authoritative check, with nothing between it and delivery.
            # Minutes may have passed inside the pause and the composition
            # call, and the reply poller may have written a suppression during
            # them. A check made before that window is not a check.
            if not _final_gate(session, contact, result):
                continue

            outbound = OutboundEmail(
                to_email=contact.email,
                to_name=contact.name,
                subject=email.subject,
                body=email.body,
            )

            sent_ok = _deliver(
                session,
                sender,
                outbound,
                contact=contact,
                campaign=campaign,
                composed=email,
                live=live,
                result=result,
                follow_up_count=0,
            )

            if sent_ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    result.stopped_reason = f"{consecutive_failures} consecutive failures"
                    log.error("aborting run: %s", result.stopped_reason)
                    break
    finally:
        # Always runs. Sends already made are real and must be recorded even if
        # the loop died halfway.
        finish_run(session, run_row, sent_count=result.sent, error_count=result.failed)
        session.commit()

    return result


def run_follow_ups(
    config: Config,
    campaign: Campaign,
    session: Session,
    composer: Composer,
    sender: Sender,
    cv_text: str,
    *,
    live: bool = False,
    limit: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> RunResult:
    """Follow up with contacts who were written to and did not reply.

    This pass selects on the *presence* of a sent message. It deliberately does
    not apply ``already_contacted`` — every follow-up target is by definition
    already contacted, so inheriting that check would skip all of them.

    Everything else still applies: the daily cap, the delay, suppression, and
    the human-reply check.
    """
    rng = rng or random.Random()
    mode = RunMode.LIVE if live else RunMode.DRY_RUN
    result = RunResult(mode=mode.value, cap=effective_daily_cap(config.daily_cap))

    run_row = start_run(session, mode=mode, campaign=f"{campaign.slug}:follow_up")
    session.commit()
    result.run_id = run_row.id

    consecutive_failures = 0

    try:
        due = follow_up_candidates(
            session,
            limit=limit or result.cap,
            max_follow_ups=campaign.follow_up.max_follow_ups,
            days_between=campaign.follow_up.days_between,
        )
        log.info("follow-up run %s: %d due, cap %d", run_row.id, len(due), result.cap)

        for previous in due:
            if sent_today(session) >= result.cap:
                result.stopped_reason = f"daily cap of {result.cap} reached"
                break
            if limit is not None and result.sent >= limit:
                result.stopped_reason = f"run limit of {limit} reached"
                break

            contact = previous.contact

            # Re-checked immediately before sending, exactly as in run().
            if is_suppressed(session, contact.email):
                result.note_skip("suppressed")
                continue
            if has_human_reply(session, contact.id):
                result.note_skip("replied")
                continue

            try:
                email = composer.compose_follow_up(
                    contact,
                    campaign,
                    cv_text,
                    previous_subject=previous.subject,
                    previous_body=previous.body,
                )
            except Exception as exc:  # noqa: BLE001
                result.failed += 1
                result.errors.append(f"compose failed for {contact.email}: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    result.stopped_reason = f"{consecutive_failures} consecutive failures"
                    break
                continue

            if not has_footer(email.body, campaign.removal_line):
                result.failed += 1
                result.errors.append(f"missing removal line for {contact.email}")
                continue

            if result.sent > 0:
                _pause(sleep, rng, config, live=live)

            if not _final_gate(session, contact, result, check_contacted=False):
                continue

            outbound = OutboundEmail(
                to_email=contact.email,
                to_name=contact.name,
                subject=email.subject,
                body=email.body,
                in_reply_to=previous.message_id,
                references=_thread_references(session, contact.id),
            )

            sent_ok = _deliver(
                session,
                sender,
                outbound,
                contact=contact,
                campaign=campaign,
                composed=email,
                live=live,
                result=result,
                follow_up_count=previous.follow_up_count + 1,
            )

            if sent_ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    result.stopped_reason = f"{consecutive_failures} consecutive failures"
                    break
    finally:
        finish_run(session, run_row, sent_count=result.sent, error_count=result.failed)
        session.commit()

    return result


# ─────────────────────────── shared steps ───────────────────────────


def _clear_to_send(session: Session, contact: Contact, result: RunResult) -> bool:
    """The checks that run immediately before every single send.

    Not once at selection time. A suppression written by the reply poller while
    this run is sleeping between sends must take effect on the next message.
    """
    if is_suppressed(session, contact.email):
        log.info("skipping %s: suppressed", contact.email)
        result.note_skip("suppressed")
        return False

    if already_contacted(session, contact.email):
        log.info("skipping %s: already contacted", contact.email)
        result.note_skip("already_contacted")
        return False

    if has_human_reply(session, contact.id):
        log.info("skipping %s: already replied", contact.email)
        result.note_skip("replied")
        return False

    return True


def _final_gate(
    session: Session,
    contact: Contact,
    result: RunResult,
    *,
    check_contacted: bool = True,
) -> bool:
    """The last check, run immediately before handing a message to the sender.

    ``_clear_to_send`` runs earlier as a pre-filter, but a pause of up to three
    minutes and a composition call sit between it and delivery. This re-reads
    the suppression list so a removal request that arrived during that window
    still takes effect on this very message.

    ``check_contacted`` is False for follow-ups: the previous send is exactly
    what makes a contact eligible there.
    """
    if is_suppressed(session, contact.email):
        log.info("skipping %s at the last moment: suppressed during the pause", contact.email)
        result.note_skip("suppressed")
        return False

    if check_contacted and already_contacted(session, contact.email):
        log.info("skipping %s at the last moment: contacted by another run", contact.email)
        result.note_skip("already_contacted")
        return False

    if has_human_reply(session, contact.id):
        log.info("skipping %s at the last moment: they replied during the pause", contact.email)
        result.note_skip("replied")
        return False

    return True


def _deliver(
    session: Session,
    sender: Sender,
    outbound: OutboundEmail,
    *,
    contact: Contact,
    campaign: Campaign,
    composed: Any,
    live: bool,
    result: RunResult,
    follow_up_count: int,
) -> bool:
    """Hand one message to the sender and record the outcome. Never raises."""
    try:
        send_result = sender.send(outbound)
    except SendError as exc:
        log.warning("send failed for %s: %s", contact.email, exc)
        result.failed += 1
        result.errors.append(f"send failed for {contact.email}: {exc}")
        record_send(
            session,
            contact=contact,
            campaign=campaign.slug,
            subject=composed.subject,
            body=composed.body,
            status=OutreachStatus.FAILED,
            error=str(exc),
            follow_up_count=follow_up_count,
        )
        session.commit()
        return False
    except Exception as exc:  # noqa: BLE001 — an unexpected backend fault is still one contact
        log.exception("unexpected send failure for %s", contact.email)
        result.failed += 1
        result.errors.append(f"send failed for {contact.email}: {exc}")
        record_send(
            session,
            contact=contact,
            campaign=campaign.slug,
            subject=composed.subject,
            body=composed.body,
            status=OutreachStatus.FAILED,
            error=str(exc),
            follow_up_count=follow_up_count,
        )
        session.commit()
        return False

    record_send(
        session,
        contact=contact,
        campaign=campaign.slug,
        subject=composed.subject,
        body=composed.body,
        status=OutreachStatus.SENT if live else OutreachStatus.DRY_RUN,
        message_id=send_result.message_id,
        follow_up_count=follow_up_count,
    )
    session.commit()

    result.sent += 1
    log.info(
        "%s %s: %s",
        "sent to" if live else "[dry-run] would send to",
        contact.email,
        composed.subject,
    )
    return True


def _pause(
    sleep: Callable[[float], None],
    rng: random.Random,
    config: Config,
    *,
    live: bool,
) -> None:
    """Wait a random interval before the next send.

    A dry run does not wait: there is nothing to pace, and making a rehearsal
    take three hours would mean nobody ever rehearses.
    """
    delay = rng.uniform(config.min_delay_seconds, config.max_delay_seconds)
    if not live:
        log.debug("[dry-run] would pause %.0fs", delay)
        return
    log.info("pausing %.0fs before the next send", delay)
    sleep(delay)


def _thread_references(session: Session, contact_id: int) -> list[str]:
    """Every message-id we have sent this contact, oldest first."""
    rows = session.scalars(
        select(Outreach.message_id)
        .where(Outreach.contact_id == contact_id, Outreach.message_id.is_not(None))
        .order_by(Outreach.id)
    )
    return [r for r in rows if r]
