# Outreach Agent — Specification

> **Note on this file.** The original SPEC.md was not included in the `baseline
> repo` commit and is no longer on disk (it was never tracked, and it is not
> gitignored). What follows is sections 5, 6 and 7 rewritten to match the
> composer work, under their original numbering, so they slot back into the full
> document if you still have it. Sections 1–4 and 8–10 are not reproduced here.

---

## 5. Modules

### `composer.py`

`Composer(client, model, sender_name, personal_website="", max_tokens=1500)`
with `compose(contact, campaign, cv_text) -> ComposedEmail(subject, body)` and
`compose_follow_up(..., previous_subject, previous_body)`.

Uses the official `anthropic` SDK, `client.messages.parse(...)` with a Pydantic
`EmailDraft(subject, body)` as `output_format`. Verified against
`anthropic==1.4.0`: `parse` is on the stable `client.messages` namespace, needs
no beta header, and `claude-sonnet-5` is on the documented structured-outputs
support list.

#### The email is five sentences

A fixed content budget, stated in the system prompt:

1. Who the sender is.
2. What the sender is doing at the moment.
3–4. One specific, true reason for writing to **this** person or organisation.
5. The ask: whether there are opportunities worth talking about.

No "I hope this finds you well." No paragraph praising the company. If a
sentence is not doing one of those four jobs, it does not ship.

#### The model writes only the middle

`EmailDraft.body` is the five sentences and nothing else. The greeting, the
sign-off, the website line and the removal line are assembled by
`render_message()` in Python:

```
Hi {first_name},

{body}

Thanks,
{sender_name}
{personal_website}        <- omitted entirely when unset
{campaign.removal_line}
```

This split is the point of the module. It turns the sentence count, the link and
the way out of the list into mechanical guarantees rather than instructions
handed to a language model in hope. **Only `body` is ever counted against the
five-sentence cap** — the scaffolding is free.

The system prompt states explicitly that the greeting, sign-off, name, links and
opt-out line are added automatically and that writing them produces duplicates.
A draft that includes them anyway is rejected and retried.

`first_name(full_name)` takes the first whitespace-separated token after
stripping a leading title (Dr, Mr, Ms, Mrs, Miss, Prof), and falls back to the
whole string when there is only one token. `"Dr. Sarah Chen"` and `"Sarah"` both
greet Sarah. An unusable name yields `"Hi there,"`.

#### Enforcing the cap

`count_sentences(text)` splits on `[.!?]+(?:\s|$)`. The system prompt forbids the
three things that would fool it — abbreviations with full stops (`e.g.`, `Inc.`),
ellipses, and decimal numbers — which is what keeps a crude splitter honest for
text this short. No `nltk`, no `spacy`; counting to five does not need a parser.

`validate_draft(draft, campaign) -> list[str]` returns every problem in language
the model can act on. It rejects: an empty body, more than
`campaign.max_sentences` sentences, an empty subject, a line break in the
subject, markdown (bold, backticks, links, headings, bullets, block quotes),
emoji, and any greeting or sign-off the model added anyway.

If there are problems, the API call is retried **exactly once** with a corrective
user turn stating the actual count and the limit. The retry conversation is
`[user, assistant(the rejected draft), user(correction)]`, so the model sees what
it produced. If the retry still fails validation, `CompositionError` is raised.

**Nothing is truncated.** The ask is the last sentence, so cutting the body short
produces an email that stops before its own point.

#### The CV

`CV_PATH`, plain text, read once at startup by `config.py` — not per contact.
The whole text goes in the system prompt, which is the stable, cache-friendly
half; per-contact facts go in the user turn.

The hard rule, stated plainly in the prompt: every factual claim about the sender
must be traceable to the CV text. No inferred seniority, no invented years of
experience, no claimed familiarity with the company, no "I have long admired".

#### A composition failure is not a send failure

When `compose()` raises, the runner logs it, increments the run's `error_count`,
and moves to the next contact **without writing an outreach row**.

This is load-bearing. A `failed` row means "we may have handed this to SMTP",
which makes `already_contacted()` return True for that address forever. A contact
we never managed to write an email for has not been contacted, and must stay
eligible for the next run. A *send* failure still writes its row, because SMTP
was genuinely reached.

---

## 6. Configuration surface

| Variable | Required | Default | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | — | composing + classifying |
| `APOLLO_API_KEY` | yes | — | search + enrichment |
| `SENDER_NAME` | **yes** | — | signs every email; Python writes the sign-off, so there is no mode, dry run included, in which it is optional |
| `GMAIL_APP_PASSWORD` | for `--live` | — | Google app password, not the account password |
| `GMAIL_ADDRESS` | for `--live` | — | the From address |
| `PERSONAL_WEBSITE` | no | unset | full URL including scheme; validated as http/https at startup. Rendered on its own line above the removal line; unset, the line disappears with no blank gap |
| `CV_PATH` | no | `data/cv.txt` | plain text, capped at 20,000 characters |
| `DATABASE_URL` | no | `sqlite:///outreach.db` | Postgres/Supabase later, zero code change |
| `ANTHROPIC_MODEL` | no | `claude-sonnet-5` | validated against the structured-outputs support list at startup |
| `DAILY_CAP` | no | `25` | clamped to `HARD_DAILY_CAP = 50` |
| `MIN_DELAY_SECONDS` / `MAX_DELAY_SECONDS` | no | `30` / `180` | |
| `APOLLO_ENRICHMENT_BUDGET` | no | `50` | credit ceiling per run |
| `SMTP_HOST` / `SMTP_PORT` | no | `smtp.gmail.com` / `465` | |
| `IMAP_HOST` / `IMAP_PORT` / `IMAP_FOLDER` | no | `imap.gmail.com` / `993` / `INBOX` | |
| `SENDER_BACKEND` | no | `smtp` | `gmail_api` raises today |
| `LOG_LEVEL` | no | `INFO` | |

`SENDER_NAME` moved from live-only to always-required: every composed email is
signed, so a dry run without a name would rehearse an email that cannot exist.

`load_config(require_send=..., require_cv=...)`. `require_cv` is set by the
commands that compose (`run`, `follow-up`); it reads the CV and folds any problem
into the same aggregated `ConfigError` as everything else, so one pass fixes the
`.env`. A missing CV names the path. An oversized CV is refused rather than
silently sending a novel with every request. `init-db` and `stats` do not demand
a CV.

### Campaign TOML

`max_words` stays as soft guidance passed to the model. `max_sentences = 5` is
the hard gate enforced in Python after the model answers — the one that actually
decides whether an email ships. Both are set in `job_search.toml` and
`partnerships.toml`.

---

## 7. Testing

`pytest`, in-memory SQLite, every external service injected as a fake. Two
autouse fixtures enforce what would otherwise rely on discipline:
`socket.socket.connect` raises, and `time.sleep` raises for anything over a
second.

### `test_composer.py`

- a five-sentence body passes through unchanged, with one API call
- a six-sentence body triggers exactly one retry
- the correction states the actual count and the limit
- the retry conversation shows the model its own rejected draft
- a six-sentence retry raises `CompositionError`, after exactly two calls
- the cap comes from `campaign.max_sentences`, not a constant
- the removal line is present in every rendered message
- the website line appears when `PERSONAL_WEBSITE` is set
- with it unset, the message has no blank line where it would have been
- neither the removal line nor the greeting nor the sign-off counts toward the
  five — the whole message exceeds five sentences while the body does not
- `first_name` handles `"Sarah Chen"`, `"Sarah"`, `"Dr. Sarah Chen"`, and empty
- the CV reaches the system prompt and **not** the user turn
- the system prompt is byte-identical between contacts (cache breakpoint)
- the model is told not to write the scaffolding, and that claims must trace to
  the CV
- greetings, sign-offs, markdown, emoji and subject line breaks are each rejected

### `test_config.py`

- a missing CV file raises `ConfigError` at `load_config(require_cv=True)`,
  before any API client exists, naming the path
- the CV is only required for commands that compose
- an oversized CV is rejected; one at exactly the limit is accepted
- CV problems are reported alongside every other problem, not one at a time
- `PERSONAL_WEBSITE` accepts full http/https URLs and rejects bare hostnames,
  `www.` forms and non-web schemes
- `SENDER_NAME` is required even for a dry run

### `test_runner.py`

- a `CompositionError` writes **no** outreach row, increments `error_count`, and
  leaves the contact eligible on the next run
- by contrast, a send failure does write a row and marks the address contacted
