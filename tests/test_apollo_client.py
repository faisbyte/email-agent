"""Tests for the Apollo client.

Backoff, typed errors, and credit accounting — the three things that decide
whether a run degrades gracefully or burns money.
"""

from __future__ import annotations

import random

import pytest
from tests.fakes import FakeHttp, FakeResponse, RecordingSleep, connection_error, person_payload

from outreach_agent.apollo_client import (
    ApolloAuthError,
    ApolloBadResponse,
    ApolloBudgetExhausted,
    ApolloClient,
    ApolloError,
    ApolloNotFound,
    ApolloOutOfCredits,
    ApolloPerson,
    ApolloRateLimited,
    ApolloServerError,
)
from outreach_agent.campaigns import ApolloFilters


def make_client(responses=None, *, budget: int = 50, max_retries: int = 3):
    http = FakeHttp(responses)
    sleep = RecordingSleep()
    client = ApolloClient(
        "test-key",
        http=http,
        sleep=sleep,
        rng=random.Random(1234),
        max_retries=max_retries,
        backoff_base=1.0,
        enrichment_budget=budget,
    )
    return client, http, sleep


# ─────────────────────────── construction ───────────────────────────


def test_empty_api_key_is_refused_immediately():
    with pytest.raises(ApolloAuthError):
        ApolloClient("")


def test_api_key_is_sent_as_a_header():
    client, http, _ = make_client([FakeResponse(200, {"people": []})])
    client.search_people(ApolloFilters(person_titles=["Recruiter"]))

    assert http.calls[0]["headers"]["x-api-key"] == "test-key"


# ─────────────────────────── search ───────────────────────────


def test_search_builds_a_payload_from_campaign_filters():
    client, http, _ = make_client([FakeResponse(200, {"people": []})])
    filters = ApolloFilters(
        person_titles=["Technical Recruiter"],
        person_seniorities=["manager"],
        person_locations=["Australia"],
        q_keywords="hiring",
    )
    client.search_people(filters, page=2, per_page=10)

    payload = http.calls[0]["json"]
    assert payload["person_titles"] == ["Technical Recruiter"]
    assert payload["person_seniorities"] == ["manager"]
    assert payload["q_keywords"] == "hiring"
    assert payload["page"] == 2
    assert payload["per_page"] == 10


def test_search_omits_empty_filters():
    client, http, _ = make_client([FakeResponse(200, {"people": []})])
    client.search_people(ApolloFilters(person_titles=["Recruiter"]))

    payload = http.calls[0]["json"]
    assert "person_locations" not in payload
    assert "q_keywords" not in payload


def test_search_caps_per_page():
    client, http, _ = make_client([FakeResponse(200, {"people": []})])
    client.search_people(ApolloFilters(person_titles=["X"]), per_page=5000)
    assert http.calls[0]["json"]["per_page"] == 100


def test_search_parses_people():
    client, _, _ = make_client([FakeResponse(200, {"people": [person_payload("p1")]})])
    people = client.search_people(ApolloFilters(person_titles=["X"]))

    assert len(people) == 1
    person = people[0]
    assert person.apollo_id == "p1"
    assert person.title == "Technical Recruiter"
    assert person.organization_domain == "corp.com"
    assert person.location == "Sydney, NSW, Australia"


def test_search_costs_no_credits():
    client, _, _ = make_client([FakeResponse(200, {"people": [person_payload()]})])
    client.search_people(ApolloFilters(person_titles=["X"]))
    assert client.credits_used == 0


def test_search_rejects_a_body_that_is_not_a_list_of_people():
    client, _, _ = make_client([FakeResponse(200, {"people": "nope"})])
    with pytest.raises(ApolloBadResponse):
        client.search_people(ApolloFilters(person_titles=["X"]))


# ─────────────────────────── usable email detection ───────────────────────────


@pytest.mark.parametrize(
    ("email", "status", "usable"),
    [
        ("bob@corp.com", "verified", True),
        ("bob@corp.com", None, True),
        ("email_not_unlocked@domain.com", None, False),
        ("bob@corp.com", "bounced", False),
        ("bob@corp.com", "unavailable", False),
        (None, "verified", False),
        ("nonsense", "verified", False),
    ],
)
def test_has_usable_email(email, status, usable):
    person = ApolloPerson(apollo_id="p1", email=email, email_status=status)
    assert person.has_usable_email is usable


# ─────────────────────────── enrichment and credits ───────────────────────────


def test_enrich_returns_the_whole_person_not_just_an_email():
    """The credit buys more than an address; the call site gets all of it."""
    client, _, _ = make_client([FakeResponse(200, {"person": person_payload("p9")})])
    person = client.enrich_person("p9")

    assert isinstance(person, ApolloPerson)
    assert person.email == "bob@corp.com"
    assert person.email_status == "verified"
    assert person.linkedin_url.endswith("p9")
    assert person.seniority == "manager"
    assert person.apollo_org_id == "o1"


def test_enrich_counts_a_credit():
    client, _, _ = make_client([FakeResponse(200, {"person": person_payload()})])
    assert client.credits_used == 0
    client.enrich_person("p1")
    assert client.credits_used == 1
    assert client.credits_remaining == 49


def test_enrich_returns_none_when_apollo_has_no_match():
    client, _, _ = make_client([FakeResponse(200, {"person": None})])
    assert client.enrich_person("nobody") is None


def test_budget_exhaustion_raises_before_spending_anything():
    client, http, _ = make_client(
        [FakeResponse(200, {"person": person_payload()})] * 3, budget=2
    )
    client.enrich_person("p1")
    client.enrich_person("p2")

    with pytest.raises(ApolloBudgetExhausted):
        client.enrich_person("p3")

    assert http.call_count == 2, "the refused enrichment must not reach Apollo"
    assert client.credits_used == 2


def test_budget_exhaustion_is_catchable_as_out_of_credits():
    """A caller handling 'no credits' wants the same behaviour for 'no budget'."""
    client, _, _ = make_client([], budget=0)
    with pytest.raises(ApolloOutOfCredits):
        client.enrich_person("p1")


def test_402_raises_out_of_credits():
    client, _, _ = make_client([FakeResponse(402, {"error": "no credits"})])
    with pytest.raises(ApolloOutOfCredits):
        client.enrich_person("p1")


def test_credit_exhaustion_reported_in_a_200_body():
    """Some plans report exhaustion in the body of an otherwise-fine response."""
    client, _, _ = make_client([FakeResponse(200, {"error": "Insufficient credits remaining"})])
    with pytest.raises(ApolloOutOfCredits):
        client.enrich_person("p1")


def test_an_unrelated_200_error_body_is_not_mistaken_for_credit_exhaustion():
    client, _, _ = make_client([FakeResponse(200, {"person": person_payload()})])
    assert client.enrich_person("p1") is not None


# ─────────────────────────── typed errors ───────────────────────────


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ApolloAuthError),
        (403, ApolloAuthError),
        (404, ApolloNotFound),
        (402, ApolloOutOfCredits),
    ],
)
def test_status_codes_map_to_typed_errors(status, expected):
    client, _, _ = make_client([FakeResponse(status, {"error": "nope"})])
    with pytest.raises(expected):
        client.search_people(ApolloFilters(person_titles=["X"]))


def test_non_json_body_raises_bad_response():
    client, _, _ = make_client([FakeResponse(200, invalid_json=True)])
    with pytest.raises(ApolloBadResponse):
        client.search_people(ApolloFilters(person_titles=["X"]))


# ─────────────────────────── retries and backoff ───────────────────────────


def test_429_is_retried_then_succeeds():
    client, http, sleep = make_client(
        [
            FakeResponse(429, {}),
            FakeResponse(429, {}),
            FakeResponse(200, {"people": [person_payload()]}),
        ]
    )
    people = client.search_people(ApolloFilters(person_titles=["X"]))

    assert len(people) == 1
    assert http.call_count == 3
    assert sleep.count == 2


def test_429_that_never_clears_raises_rate_limited():
    client, http, sleep = make_client([FakeResponse(429, {})] * 10, max_retries=3)
    with pytest.raises(ApolloRateLimited):
        client.search_people(ApolloFilters(person_titles=["X"]))

    assert http.call_count == 4, "initial attempt plus three retries"
    assert sleep.count == 3


def test_retry_after_header_is_honoured_exactly():
    client, _, sleep = make_client(
        [
            FakeResponse(429, {}, headers={"Retry-After": "7"}),
            FakeResponse(200, {"people": []}),
        ]
    )
    client.search_people(ApolloFilters(person_titles=["X"]))

    assert sleep.calls == [7.0]


def test_a_nonsense_retry_after_falls_back_to_jittered_backoff():
    client, _, sleep = make_client(
        [
            FakeResponse(429, {}, headers={"Retry-After": "soon"}),
            FakeResponse(200, {"people": []}),
        ]
    )
    client.search_people(ApolloFilters(person_titles=["X"]))

    assert sleep.count == 1
    assert 0 <= sleep.calls[0] <= 1.0


def test_backoff_grows_exponentially_within_the_jitter_ceiling():
    """Full jitter: each delay is drawn from [0, base * 2**attempt]."""
    client, _, sleep = make_client([FakeResponse(503, {})] * 10, max_retries=4)
    with pytest.raises(ApolloServerError):
        client.search_people(ApolloFilters(person_titles=["X"]))

    assert sleep.count == 4
    for attempt, delay in enumerate(sleep.calls):
        assert 0 <= delay <= 1.0 * (2**attempt)


def test_backoff_is_capped():
    client, _, sleep = make_client([FakeResponse(503, {})] * 30, max_retries=20)
    client.max_backoff = 5.0
    with pytest.raises(ApolloServerError):
        client.search_people(ApolloFilters(person_titles=["X"]))

    assert max(sleep.calls) <= 5.0


def test_5xx_is_retried():
    client, http, _ = make_client(
        [FakeResponse(500, {}), FakeResponse(200, {"people": []})]
    )
    client.search_people(ApolloFilters(person_titles=["X"]))
    assert http.call_count == 2


def test_a_connection_error_is_retried_then_gives_up():
    client, http, sleep = make_client([connection_error()] * 10, max_retries=2)
    with pytest.raises(ApolloServerError):
        client.search_people(ApolloFilters(person_titles=["X"]))

    assert http.call_count == 3
    assert sleep.count == 2


def test_a_connection_error_that_clears_succeeds():
    client, http, _ = make_client(
        [connection_error(), FakeResponse(200, {"people": [person_payload()]})]
    )
    assert len(client.search_people(ApolloFilters(person_titles=["X"]))) == 1


def test_a_400_is_not_retried():
    """Retrying a decision just spends time being wrong."""
    client, http, sleep = make_client([FakeResponse(400, {"error": "bad filter"})] * 5)
    with pytest.raises(ApolloError):
        client.search_people(ApolloFilters(person_titles=["X"]))

    assert http.call_count == 1
    assert sleep.count == 0


def test_out_of_credits_is_never_retried():
    client, http, _ = make_client([FakeResponse(402, {})] * 5)
    with pytest.raises(ApolloOutOfCredits):
        client.enrich_person("p1")
    assert http.call_count == 1
