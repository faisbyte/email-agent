"""Test doubles for every external service.

Nothing here touches a network. The socket kill-switch in conftest.py enforces
that; these fakes are what make the code testable once it is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests

# ─────────────────────────── Apollo ───────────────────────────


@dataclass
class FakeResponse:
    """Stands in for a requests.Response."""

    status_code: int = 200
    payload: Any = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    invalid_json: bool = False

    def json(self) -> Any:
        if self.invalid_json:
            raise ValueError("not json")
        return self.payload


class FakeHttp:
    """Scripted HTTP transport.

    Give it a list of FakeResponse objects (or exceptions to raise) and it
    returns them in order, recording every call. That is what makes retry and
    backoff behaviour testable without waiting or connecting.
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            return FakeResponse(200, {"people": []})
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def call_count(self) -> int:
        return len(self.calls)


class RecordingSleep:
    """Captures sleep durations instead of spending them."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)

    @property
    def count(self) -> int:
        return len(self.calls)


def person_payload(
    person_id: str = "p1",
    *,
    email: str | None = "bob@corp.com",
    email_status: str | None = "verified",
    name: str = "Bob Roberts",
    title: str = "Technical Recruiter",
    org_id: str = "o1",
    org_name: str = "Corp",
    org_domain: str = "corp.com",
) -> dict[str, Any]:
    """An Apollo person object, shaped like the real thing."""
    return {
        "id": person_id,
        "name": name,
        "first_name": name.split(" ")[0],
        "last_name": name.split(" ")[-1],
        "title": title,
        "email": email,
        "email_status": email_status,
        "linkedin_url": f"https://linkedin.com/in/{person_id}",
        "city": "Sydney",
        "state": "NSW",
        "country": "Australia",
        "seniority": "manager",
        "organization": {"id": org_id, "name": org_name, "primary_domain": org_domain},
    }


def connection_error(message: str = "boom") -> requests.RequestException:
    return requests.ConnectionError(message)


# ─────────────────────────── Anthropic ───────────────────────────


@dataclass
class FakeParsedMessage:
    """Stands in for anthropic's ParsedMessage."""

    parsed_output: Any
    stop_reason: str = "end_turn"


class FakeMessages:
    def __init__(self, owner: FakeAnthropic) -> None:
        self._owner = owner

    def parse(self, **kwargs: Any) -> FakeParsedMessage:
        self._owner.calls.append(kwargs)
        if self._owner.error is not None:
            raise self._owner.error
        if not self._owner.outputs:
            raise AssertionError("FakeAnthropic ran out of scripted outputs")
        nxt = self._owner.outputs.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return FakeParsedMessage(parsed_output=nxt)


class FakeAnthropic:
    """Scripted Anthropic client.

    Returns the parsed_output objects it was given, in order, and records every
    request so tests can assert on what was actually sent to the model.
    """

    def __init__(self, outputs: list[Any] | None = None, error: Exception | None = None) -> None:
        self.outputs = list(outputs or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.messages = FakeMessages(self)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_system(self) -> str:
        system = self.calls[-1].get("system", "")
        if isinstance(system, list):
            return "\n".join(block.get("text", "") for block in system)
        return system

    def last_user_text(self) -> str:
        messages = self.calls[-1].get("messages", [])
        parts = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(b.get("text", "") for b in content if isinstance(b, dict))
        return "\n".join(parts)


# ─────────────────────────── SMTP ───────────────────────────


class FakeSMTP:
    """Stands in for smtplib.SMTP_SSL, as a context manager."""

    instances: list[FakeSMTP] = []

    def __init__(
        self,
        host: str = "",
        port: int = 0,
        *,
        fail_on_login: Exception | None = None,
        fail_on_send: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        self.host = host
        self.port = port
        self.logged_in_as: str | None = None
        self.sent_messages: list[Any] = []
        self.fail_on_login = fail_on_login
        self.fail_on_send = fail_on_send
        self.quit_called = False
        FakeSMTP.instances.append(self)

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.quit_called = True

    def login(self, user: str, password: str) -> None:
        if self.fail_on_login:
            raise self.fail_on_login
        self.logged_in_as = user
        self.password = password

    def send_message(self, message: Any) -> dict[str, Any]:
        if self.fail_on_send:
            raise self.fail_on_send
        self.sent_messages.append(message)
        return {}


class FakeSMTPFactory:
    """Builds FakeSMTP instances and remembers them."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.built: list[FakeSMTP] = []

    def __call__(self, host: str, port: int, **kwargs: Any) -> FakeSMTP:
        smtp = FakeSMTP(host, port, **self.kwargs)
        self.built.append(smtp)
        return smtp

    @property
    def all_sent(self) -> list[Any]:
        return [m for s in self.built for m in s.sent_messages]


class ExplodingSender:
    """A Sender that fails the test if anything tries to send through it.

    Used to prove that a dry run cannot deliver mail: the live sender is not
    merely bypassed, it is never even asked.
    """

    def __init__(self) -> None:
        self.attempts = 0

    def send(self, message: Any) -> Any:
        self.attempts += 1
        raise AssertionError(
            "A live send was attempted. Dry-run must never reach a real sender."
        )
