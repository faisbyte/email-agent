"""Apollo.io client: people search and email enrichment.

Two things this module takes seriously.

**Credits.** Search is free; enrichment is not. Every enrichment is counted
against a per-run budget the caller sets, and the client refuses to exceed it
rather than discovering the overspend on the invoice. Running out on Apollo's
side raises a typed error that the runner can catch and finish cleanly.

**Backoff.** 429 and 5xx retry with exponential backoff and full jitter,
honouring Retry-After when it is present. Everything else fails immediately —
retrying a 400 just spends time being wrong.

The HTTP session and the sleep function are both injected, which is how the
tests exercise every path without opening a socket.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.apollo.io"
SEARCH_PATH = "/api/v1/mixed_people/search"
MATCH_PATH = "/api/v1/people/match"

#: Statuses worth trying again. Anything else is a decision, not a hiccup.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Apollo marks unusable addresses with these. Sending to them wastes a send
#: and hurts deliverability.
UNUSABLE_EMAIL_STATUSES = frozenset({"unavailable", "bounced", "invalid"})

#: Placeholder Apollo returns instead of an address that has not been revealed.
LOCKED_EMAIL_MARKERS = ("email_not_unlocked", "domain.com")


# ─────────────────────────── errors ───────────────────────────


class ApolloError(Exception):
    """Base for everything this client raises."""


class ApolloAuthError(ApolloError):
    """401/403 — the API key is missing, wrong, or lacks permission."""


class ApolloRateLimited(ApolloError):
    """429 that survived every retry."""


class ApolloOutOfCredits(ApolloError):
    """No credits left to reveal an address.

    The runner catches this and finishes the run cleanly: the sends already
    made are real and must be recorded, and dying here would strand them.
    """


class ApolloBudgetExhausted(ApolloOutOfCredits):
    """The per-run enrichment budget is spent.

    Subclasses ApolloOutOfCredits deliberately: a caller that handles "no more
    credits" wants the same behaviour for "no more budget".
    """


class ApolloNotFound(ApolloError):
    """404."""


class ApolloServerError(ApolloError):
    """5xx that survived every retry."""


class ApolloBadResponse(ApolloError):
    """A 2xx whose body was not the JSON we expected."""


# ─────────────────────────── data ───────────────────────────


@dataclass
class ApolloPerson:
    """A person as Apollo describes them.

    Kept as a whole object rather than reduced to an email: the enrichment call
    spends a credit and returns verified-email status, LinkedIn, location and
    seniority along with the address. Throwing that away means paying twice for
    data already in hand.
    """

    apollo_id: str
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    title: str | None = None
    email: str | None = None
    email_status: str | None = None
    linkedin_url: str | None = None
    location: str | None = None
    seniority: str | None = None
    organization_name: str | None = None
    organization_domain: str | None = None
    apollo_org_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def has_usable_email(self) -> bool:
        """True only for a real, revealed, not-known-bad address."""
        if not self.email or "@" not in self.email:
            return False
        lowered = self.email.lower()
        if any(marker in lowered for marker in LOCKED_EMAIL_MARKERS):
            return False
        status = (self.email_status or "").lower()
        return status not in UNUSABLE_EMAIL_STATUSES

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> ApolloPerson:
        """Build from an Apollo person object, defensively.

        Apollo's payloads vary between endpoints and change over time, so every
        field is optional and nothing here assumes a key exists.
        """
        org = data.get("organization") or {}
        location = ", ".join(
            part
            for part in (data.get("city"), data.get("state"), data.get("country"))
            if isinstance(part, str) and part.strip()
        )
        return cls(
            apollo_id=str(data.get("id") or ""),
            name=data.get("name"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            title=data.get("title"),
            email=data.get("email"),
            email_status=data.get("email_status"),
            linkedin_url=data.get("linkedin_url"),
            location=location or None,
            seniority=data.get("seniority"),
            organization_name=org.get("name") or data.get("organization_name"),
            organization_domain=org.get("primary_domain") or org.get("website_url"),
            apollo_org_id=str(org.get("id")) if org.get("id") else None,
            raw=data,
        )


# ─────────────────────────── client ───────────────────────────


class ApolloClient:
    """Talks to Apollo. Injectable transport and clock, so it is fully testable."""

    def __init__(
        self,
        api_key: str,
        *,
        http: requests.Session | Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        max_backoff: float = 60.0,
        timeout: float = 30.0,
        enrichment_budget: int = 50,
    ) -> None:
        if not api_key:
            raise ApolloAuthError("Apollo API key is empty")

        self.api_key = api_key
        self.http = http if http is not None else requests.Session()
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.timeout = timeout

        self.enrichment_budget = enrichment_budget
        self.credits_used = 0

    # ── public API ────────────────────────────────────────────

    def search_people(
        self,
        filters: Any,
        *,
        page: int = 1,
        per_page: int = 25,
    ) -> list[ApolloPerson]:
        """Search for people. Costs no credits.

        ``filters`` is a campaigns.ApolloFilters, or anything with the same
        attribute names.
        """
        payload = self._search_payload(filters, page=page, per_page=per_page)
        data = self._request("POST", SEARCH_PATH, payload)

        people = data.get("people")
        if people is None:
            people = data.get("contacts", [])
        if not isinstance(people, list):
            raise ApolloBadResponse(
                f"search returned {type(people).__name__}, expected a list of people"
            )
        return [ApolloPerson.from_payload(p) for p in people if isinstance(p, dict)]

    def enrich_person(
        self,
        person_id: str,
        *,
        reveal_personal_emails: bool = False,
    ) -> ApolloPerson | None:
        """Reveal a person's details. **This spends a credit.**

        Returns the whole person, not just the address: the credit buys the
        email status, LinkedIn URL, location and seniority too. The caller
        pulls the email off the object.

        Returns None when Apollo has no match — a miss, not an error.

        Raises:
            ApolloBudgetExhausted: the per-run budget is spent (checked before
                the request, so no credit is burned discovering it).
            ApolloOutOfCredits: Apollo has no credits left.
        """
        if self.credits_remaining <= 0:
            raise ApolloBudgetExhausted(
                f"per-run enrichment budget of {self.enrichment_budget} is spent "
                f"({self.credits_used} used). Raise APOLLO_ENRICHMENT_BUDGET to continue."
            )

        payload: dict[str, Any] = {"id": person_id}
        if reveal_personal_emails:
            payload["reveal_personal_emails"] = True

        # Counted before the response is read: a request that reaches Apollo may
        # have spent the credit even if we fail to parse the reply. Undercounting
        # spend is the more expensive mistake.
        self.credits_used += 1

        data = self._request("POST", MATCH_PATH, payload)
        person = data.get("person")
        if not isinstance(person, dict):
            return None
        return ApolloPerson.from_payload(person)

    @property
    def credits_remaining(self) -> int:
        return max(0, self.enrichment_budget - self.credits_used)

    # ── request plumbing ──────────────────────────────────────

    def _search_payload(self, filters: Any, *, page: int, per_page: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page": max(1, page),
            "per_page": max(1, min(per_page, 100)),
        }
        for key in (
            "person_titles",
            "person_seniorities",
            "person_locations",
            "organization_locations",
            "organization_num_employees_ranges",
            "q_organization_domains",
        ):
            value = getattr(filters, key, None)
            if value:
                payload[key] = list(value)

        keywords = getattr(filters, "q_keywords", None)
        if keywords:
            payload["q_keywords"] = keywords
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "accept": "application/json",
            "x-api-key": self.api_key,
        }

    def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.request(
                    method,
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                # A connection problem is a hiccup, not a decision: retry it.
                last_error = exc
                if attempt >= self.max_retries:
                    raise ApolloServerError(
                        f"{method} {path} failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                self._wait(attempt, None)
                continue

            status = getattr(response, "status_code", None)

            if status in RETRYABLE_STATUSES:
                if attempt >= self.max_retries:
                    if status == 429:
                        raise ApolloRateLimited(
                            f"rate limited by Apollo after {attempt + 1} attempts"
                        )
                    raise ApolloServerError(
                        f"Apollo returned {status} after {attempt + 1} attempts"
                    )
                self._wait(attempt, self._retry_after(response))
                continue

            return self._handle_final_response(response, status, path)

        raise ApolloServerError(  # pragma: no cover — loop always returns or raises
            f"{method} {path} exhausted retries: {last_error}"
        )

    def _handle_final_response(
        self, response: Any, status: int | None, path: str
    ) -> dict[str, Any]:
        if status == 401 or status == 403:
            raise ApolloAuthError(
                f"Apollo rejected the API key ({status}). Check APOLLO_API_KEY."
            )
        if status == 402:
            raise ApolloOutOfCredits("Apollo reports no credits remaining")
        if status == 404:
            raise ApolloNotFound(f"Apollo has no such resource: {path}")

        body = self._json(response, path)

        # Some plans report exhaustion in the body of an otherwise-fine response.
        if status in (200, 422) and _mentions_credit_exhaustion(body):
            raise ApolloOutOfCredits(f"Apollo reports credit exhaustion: {_error_text(body)}")

        if status is None or not (200 <= status < 300):
            raise ApolloError(f"Apollo returned {status} for {path}: {_error_text(body)}")

        return body

    def _json(self, response: Any, path: str) -> dict[str, Any]:
        try:
            body = response.json()
        except (ValueError, AttributeError) as exc:
            raise ApolloBadResponse(f"Apollo returned non-JSON for {path}") from exc
        if not isinstance(body, dict):
            raise ApolloBadResponse(
                f"Apollo returned {type(body).__name__} for {path}, expected an object"
            )
        return body

    def _retry_after(self, response: Any) -> float | None:
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None

    def _wait(self, attempt: int, retry_after: float | None) -> None:
        """Exponential backoff with full jitter, or Retry-After when given."""
        if retry_after is not None:
            delay = min(retry_after, self.max_backoff)
        else:
            ceiling = min(self.backoff_base * (2**attempt), self.max_backoff)
            delay = self.rng.uniform(0, ceiling)
        log.debug("apollo backoff: attempt %d, sleeping %.2fs", attempt + 1, delay)
        self.sleep(delay)


def _error_text(body: dict[str, Any]) -> str:
    for key in ("error", "message", "error_message", "errors"):
        value = body.get(key)
        if value:
            return str(value)
    return str(body)[:200]


def _mentions_credit_exhaustion(body: dict[str, Any]) -> bool:
    text = _error_text(body).lower()
    if "credit" not in text:
        return False
    return any(word in text for word in ("insufficient", "exhaust", "no ", "out of", "limit"))
