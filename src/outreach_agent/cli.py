"""Command line entry point.

Every command loads and validates configuration before it does anything else,
so a missing key stops the run at the start rather than partway through a list.

``run`` is a dry run unless ``--live`` is passed, and ``--live`` additionally
asks for typed confirmation unless ``--yes`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from sqlalchemy.exc import OperationalError

from . import __version__
from .apollo_client import ApolloClient, ApolloError
from .campaigns import CampaignError, load_campaign
from .composer import Composer
from .config import Config, ConfigError, load_config, load_cv
from .inbox import InboxPoller, ReplyClassifier, process_replies
from .runner import HARD_DAILY_CAP, discover, effective_daily_cap, run, run_follow_ups
from .sender import build_sender
from .store import (
    add_suppression,
    counts_by_status,
    create_all,
    make_engine,
    make_session_factory,
    sent_today,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="outreach",
        description="Deterministic outreach automation. Dry-run by default.",
    )
    parser.add_argument("--version", action="version", version=f"outreach-agent {__version__}")
    parser.add_argument(
        "--campaign",
        help="Path to a campaign TOML file. Defaults to the CAMPAIGN environment variable.",
    )
    parser.add_argument("--log-level", help="Override LOG_LEVEL for this invocation.")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Create the database schema.")

    search = sub.add_parser("search", help="Find contacts in Apollo. Sends nothing.")
    search.add_argument("--limit", type=int, default=25, help="How many contacts to store.")

    send = sub.add_parser("run", help="Send the first email to new contacts.")
    send.add_argument(
        "--live",
        action="store_true",
        help="Actually deliver mail. Without this flag nothing is sent.",
    )
    send.add_argument("--limit", type=int, help="Stop after this many sends.")
    send.add_argument("--yes", action="store_true", help="Skip the live-run confirmation.")

    follow = sub.add_parser("follow-up", help="Send follow-ups to contacts who did not reply.")
    follow.add_argument("--live", action="store_true", help="Actually deliver mail.")
    follow.add_argument("--limit", type=int, help="Stop after this many sends.")
    follow.add_argument("--yes", action="store_true", help="Skip the live-run confirmation.")

    poll = sub.add_parser("poll-replies", help="Read replies, classify them, apply opt-outs.")
    poll.add_argument("--since", type=int, default=7, help="How many days back to read.")

    suppress = sub.add_parser("suppress", help="Never contact an address or domain again.")
    suppress.add_argument("target", help="An email address or a bare domain.")
    suppress.add_argument("--reason", default="manual")

    sub.add_parser("stats", help="Show what has been sent.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    live = getattr(args, "live", False)

    try:
        config = load_config(require_send=live)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG

    logging.basicConfig(
        level=(args.log_level or config.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    engine = make_engine(config.database_url)
    session_factory = make_session_factory(engine)

    try:
        if args.command == "init-db":
            create_all(engine)
            print(f"Schema created at {config.database_url}")
            return EXIT_OK

        if args.command == "stats":
            with session_factory() as session:
                return _stats(session, config)

        if args.command == "suppress":
            with session_factory() as session:
                add_suppression(session, args.target, reason=args.reason)
                session.commit()
            print(f"Suppressed {args.target}. They will never be contacted.")
            return EXIT_OK

        campaign = load_campaign(args.campaign or config.campaign_path)

        if args.command == "search":
            return _search(config, campaign, session_factory, args)

        if args.command in {"run", "follow-up"}:
            return _send(
                config,
                campaign,
                session_factory,
                args,
                follow_up=args.command == "follow-up",
            )

        if args.command == "poll-replies":
            return _poll(config, session_factory, args)

    except (ConfigError, CampaignError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG
    except ApolloError as exc:
        print(f"Apollo error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except OperationalError as exc:
        if "no such table" in str(exc):
            print(
                "The database has no schema yet. Run `outreach init-db` first.",
                file=sys.stderr,
            )
            return EXIT_CONFIG
        raise
    except KeyboardInterrupt:
        print("\nInterrupted. Everything sent so far has been recorded.", file=sys.stderr)
        return EXIT_FAILED

    parser.error(f"unknown command {args.command}")
    return EXIT_FAILED


# ─────────────────────────── commands ───────────────────────────


def _stats(session, config: Config) -> int:
    counts = counts_by_status(session)
    cap = effective_daily_cap(config.daily_cap)
    print("Outreach by status:")
    for status, count in sorted(counts.items()):
        print(f"  {status:<10} {count}")
    if not counts:
        print("  (nothing yet)")
    print(f"\nSent today: {sent_today(session)} of {cap} (hard ceiling {HARD_DAILY_CAP})")
    return EXIT_OK


def _search(config: Config, campaign, session_factory, args) -> int:
    apollo = ApolloClient(
        config.apollo_api_key, enrichment_budget=config.apollo_enrichment_budget
    )
    with session_factory() as session:
        stats = discover(config, campaign, session, apollo, limit=args.limit)

    print(
        f"Searched {stats['seen']} people, stored {stats['stored']}, "
        f"skipped {stats['skipped']}, spent {stats['credits']} credit(s)."
    )
    if stats.get("stopped_out_of_credits"):
        print("Stopped early: Apollo credits or the per-run budget ran out.")
    return EXIT_OK


def _send(config: Config, campaign, session_factory, args, *, follow_up: bool) -> int:
    live = args.live
    cv_text = load_cv(config.cv_path)

    if live and not args.yes and not _confirm(config, campaign, follow_up=follow_up):
        print("Cancelled. Nothing was sent.")
        return EXIT_OK

    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    composer = Composer(client, model=config.anthropic_model)

    # The only place a live sender is constructed. Without --live this returns
    # a DryRunSender, so the loop below is not capable of delivering mail.
    sender = build_sender(config, live=live)

    with session_factory() as session:
        do = run_follow_ups if follow_up else run
        result = do(config, campaign, session, composer, sender, cv_text, live=live,
                    limit=args.limit)

    label = "Sent" if live else "Would send (dry run)"
    print(f"\n{label}: {result.sent}")
    print(f"Skipped: {result.skipped}  {_format_skips(result.skips)}")
    print(f"Failed: {result.failed}")
    if result.stopped_reason:
        print(f"Stopped: {result.stopped_reason}")
    for error in result.errors[:10]:
        print(f"  ! {error}")
    if not live:
        print("\nThis was a dry run. Nothing was delivered. Add --live to send.")

    return EXIT_FAILED if result.failed and not result.sent else EXIT_OK


def _poll(config: Config, session_factory, args) -> int:
    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    classifier = ReplyClassifier(client, model=config.anthropic_model)
    poller = InboxPoller(
        host=config.imap_host,
        port=config.imap_port,
        address=config.gmail_address,
        password=config.gmail_app_password,
        folder=config.imap_folder,
    )

    messages = poller.fetch_replies(since_days=args.since)
    with session_factory() as session:
        report = process_replies(session, messages, classifier)

    print(f"Read {len(messages)} message(s), matched {len(report.outcomes)}.")
    print(f"  bounces:      {report.bounces}")
    print(f"  suppressions: {report.suppressions}")
    print(f"  unmatched:    {report.unmatched}")
    return EXIT_OK


def _confirm(config: Config, campaign, *, follow_up: bool) -> bool:
    cap = effective_daily_cap(config.daily_cap)
    kind = "follow-up emails" if follow_up else "first-contact emails"
    print("About to send real email.")
    print(f"  campaign: {campaign.name}")
    print(f"  from:     {config.sender_name} <{config.gmail_address}>")
    print(f"  sending:  {kind}, up to {cap} today")
    answer = input("Type 'yes' to continue: ").strip().lower()
    return answer == "yes"


def _format_skips(skips: dict[str, int]) -> str:
    if not skips:
        return ""
    return "(" + ", ".join(f"{reason}: {count}" for reason, count in sorted(skips.items())) + ")"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
