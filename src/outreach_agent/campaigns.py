"""Campaign templates.

A campaign is content, not configuration: who you are, what you want, the tone
to strike, who to look for in Apollo, and how to follow up. It lives in a TOML
file so a non-programmer can edit it, and it is read with the standard library.

config.py answers "can this run at all". This module answers "what is this run
trying to say".
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CampaignError(Exception):
    """Raised when a campaign file is missing, malformed, or incomplete."""

    def __init__(self, problems: list[str], path: Path | None = None) -> None:
        self.problems = problems
        self.path = path
        where = f" in {path}" if path else ""
        joined = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Campaign is not usable{where}:\n{joined}")


@dataclass(frozen=True)
class ApolloFilters:
    """Search filters handed to Apollo's people search.

    Every field is optional on its own, but a campaign with no filters at all
    would ask Apollo for the entire world, so at least one is required.
    """

    person_titles: list[str] = field(default_factory=list)
    person_seniorities: list[str] = field(default_factory=list)
    person_locations: list[str] = field(default_factory=list)
    organization_locations: list[str] = field(default_factory=list)
    organization_num_employees_ranges: list[str] = field(default_factory=list)
    q_organization_domains: list[str] = field(default_factory=list)
    q_keywords: str = ""

    def is_empty(self) -> bool:
        return not any(
            [
                self.person_titles,
                self.person_seniorities,
                self.person_locations,
                self.organization_locations,
                self.organization_num_employees_ranges,
                self.q_organization_domains,
                self.q_keywords,
            ]
        )


@dataclass(frozen=True)
class FollowUpPolicy:
    max_follow_ups: int = 1
    days_between: int = 5
    goal: str = ""


@dataclass(frozen=True)
class Campaign:
    """A validated campaign template."""

    slug: str
    name: str
    sender_persona: str
    goal: str
    tone: list[str]
    subject_guidance: str
    max_words: int
    removal_line: str
    apollo: ApolloFilters
    follow_up: FollowUpPolicy
    source_path: Path | None = None


_REQUIRED_TEXT_FIELDS = ("name", "sender_persona", "goal", "subject_guidance", "removal_line")


def load_campaign(path: Path | str) -> Campaign:
    """Read and validate a campaign TOML file.

    Raises:
        CampaignError: listing every problem found in the file.
    """
    p = Path(path)
    try:
        raw = p.read_bytes()
    except FileNotFoundError as exc:
        raise CampaignError([f"file not found: {p}"], p) from exc
    except OSError as exc:
        raise CampaignError([f"could not be read: {exc}"], p) from exc

    try:
        data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise CampaignError([f"is not valid TOML: {exc}"], p) from exc

    return _build(data, p)


def _build(data: dict[str, Any], path: Path | None) -> Campaign:
    problems: list[str] = []

    def text(key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key!r} is required and must be a non-empty string")
            return ""
        return value.strip()

    values = {key: text(key) for key in _REQUIRED_TEXT_FIELDS}

    tone = data.get("tone", [])
    if not isinstance(tone, list) or not all(isinstance(t, str) for t in tone):
        problems.append("'tone' must be a list of strings")
        tone = []

    max_words = data.get("max_words", 150)
    if not isinstance(max_words, int) or isinstance(max_words, bool) or max_words <= 0:
        problems.append(f"'max_words' must be a positive whole number, got {max_words!r}")
        max_words = 150

    removal_line = values["removal_line"]
    if removal_line and ("http://" in removal_line or "https://" in removal_line):
        # There is no web server in this project, so a link in the removal line
        # would point at nothing. Removal is a reply, and it must say so.
        problems.append(
            "'removal_line' must not contain a URL — this project serves no unsubscribe "
            "page. Ask the recipient to reply instead."
        )

    apollo = _build_apollo(data.get("apollo", {}), problems)
    follow_up = _build_follow_up(data.get("follow_up", {}), problems)

    if problems:
        raise CampaignError(problems, path)

    slug = path.stem if path is not None else values["name"].lower().replace(" ", "_")

    return Campaign(
        slug=slug,
        name=values["name"],
        sender_persona=values["sender_persona"],
        goal=values["goal"],
        tone=[t.strip() for t in tone if t.strip()],
        subject_guidance=values["subject_guidance"],
        max_words=max_words,
        removal_line=removal_line,
        apollo=apollo,
        follow_up=follow_up,
        source_path=path,
    )


def _build_apollo(section: Any, problems: list[str]) -> ApolloFilters:
    if not isinstance(section, dict):
        problems.append("'[apollo]' must be a table")
        return ApolloFilters()

    def str_list(key: str) -> list[str]:
        value = section.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            problems.append(f"'apollo.{key}' must be a list of strings")
            return []
        return [v.strip() for v in value if v.strip()]

    keywords = section.get("q_keywords", "")
    if not isinstance(keywords, str):
        problems.append("'apollo.q_keywords' must be a string")
        keywords = ""

    filters = ApolloFilters(
        person_titles=str_list("person_titles"),
        person_seniorities=str_list("person_seniorities"),
        person_locations=str_list("person_locations"),
        organization_locations=str_list("organization_locations"),
        organization_num_employees_ranges=str_list("organization_num_employees_ranges"),
        q_organization_domains=str_list("q_organization_domains"),
        q_keywords=keywords.strip(),
    )
    if filters.is_empty():
        problems.append(
            "'[apollo]' has no filters. An unfiltered search asks for everyone; "
            "set at least person_titles or q_keywords."
        )
    return filters


def _build_follow_up(section: Any, problems: list[str]) -> FollowUpPolicy:
    if not isinstance(section, dict):
        problems.append("'[follow_up]' must be a table")
        return FollowUpPolicy()

    def positive_int(key: str, default: int) -> int:
        value = section.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            problems.append(f"'follow_up.{key}' must be a non-negative whole number")
            return default
        return value

    goal = section.get("goal", "")
    if not isinstance(goal, str):
        problems.append("'follow_up.goal' must be a string")
        goal = ""

    days = positive_int("days_between", 5)
    if days == 0:
        problems.append("'follow_up.days_between' must be at least 1")
        days = 5

    return FollowUpPolicy(
        max_follow_ups=positive_int("max_follow_ups", 1),
        days_between=days,
        goal=goal.strip(),
    )
