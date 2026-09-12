# Outreach Agent

Pull contacts from Apollo.io, write each one a personalised email with Claude,
send it from your Gmail, and record everything so the same person is never
contacted twice.

Deterministic Python drives the loop. Claude is called for exactly two things:
composing an email body and classifying a reply. Nothing else is delegated to a
model, so the behaviour you read in the code is the behaviour you get.

The first shipped campaign is job-search outreach to talent acquisition people.
That is a configuration file, not the architecture — a second campaign ships
alongside it to keep the general path honest.

> **Dry-run is the default.** `outreach run` composes and records but sends
> nothing. Delivery requires an explicit `--live` flag and a typed confirmation.

---

## What it does

1. **`search`** — queries Apollo for people matching your campaign's filters and
   stores them. Suppression and prior contact are checked *before* paying a
   credit to reveal an address.
2. **`run`** — for each new contact: checks the suppression list, checks whether
   they have been written to before, asks Claude for a subject and five
   sentences, assembles the rest of the email in Python, sends, records. Waits a
   random 30–180 seconds between sends.
3. **`poll-replies`** — reads the inbox over IMAP, detects bounces from headers,
   catches explicit opt-outs with a regex, and asks Claude to classify the rest
   as interested / rejection / auto-reply / opt-out. An opt-out is suppressed
   immediately.
4. **`follow-up`** — one nudge to people who never replied, subject to the same
   caps and checks. Any human reply stops all follow-ups permanently.

---

## Setup

You need Python 3.11 or newer.

```bash
git clone <your-fork-url> email-agent
cd email-agent

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Then fill in three keys:

```bash
cp .env.example .env
```

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `APOLLO_API_KEY` | Apollo → Settings → Integrations → API |
| `GMAIL_APP_PASSWORD` | Enable 2-Step Verification, then [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) — a 16-character app password, **not** your account password |

Set `SENDER_NAME` in `.env` — it signs every email, so it is required even for a
dry run. `PERSONAL_WEBSITE` is optional; when set it appears on its own line
under your name.

Add your CV and create the database:

```bash
cp data/cv.example.txt data/cv.txt
$EDITOR data/cv.txt

outreach init-db
```

**About the CV.** Export it to **plain text** — no PDF, no DOCX. Every factual
claim Claude makes about you must be traceable to this file, so anything not in
it will not appear in an email. Keep it under 20,000 characters; the whole file
is sent with every composition, and the app refuses to start above that.

`.env`, `data/cv.txt` and `*.db` are all gitignored. `data/cv.example.txt` is the
stub to copy, and it is the only CV file in the repo.

---

## Use

```bash
# 1. Find people. Costs Apollo credits. Sends nothing.
outreach search --limit 25

# 2. Rehearse. Composes real emails, delivers none.
outreach run

# 3. Read what it wrote, then send for real.
outreach run --live

# Later:
outreach poll-replies          # classify replies, apply opt-outs
outreach follow-up --live      # nudge people who never replied
outreach stats                 # what has gone out, and today's usage
outreach suppress someone@corp.com --reason "asked by email"
```

`outreach suppress` accepts a bare domain too — `outreach suppress corp.com`
blocks every address there.

---

## What the emails look like

Five sentences, and a fixed budget for them:

1. Who you are.
2. What you are doing at the moment.
3–4. One specific, true reason for writing to *this* person or organisation.
5. The ask.

No "I hope this finds you well." No paragraph of praise for the company. If a
sentence is not doing one of those four jobs, it does not ship.

**The model writes only the middle.** It returns a subject and those five
sentences; Python assembles everything around them:

```
Hi Sarah,

I am a backend engineer in Sydney. I am wrapping up a payments service handling
about 40,000 transactions a day. Acme posted two platform engineering roles last
month. The one on the billing team lines up with what I have been building. Are
there openings worth a conversation?

Thanks,
Faisal Naveed
https://faisal.dev
If you'd rather not hear from me, reply with 'no thanks' and I won't write again.
```

That split is deliberate. The greeting, the sign-off, the website line and the
removal line become mechanical guarantees instead of instructions we hand to a
model and hope it follows — and only the five sentences are ever counted against
the cap. The website sits outside the five on purpose: spending one of five
sentences on "you can see my work at…" is a bad trade, and a bare URL under the
sign-off gets clicked just as often.

If the model returns six sentences, it gets **one** corrective retry stating the
actual count. If the retry also fails, composition raises and that contact is
skipped. Nothing is truncated — the ask is the last sentence, so cutting the body
short produces an email that stops before its own point.

`max_sentences` is per-campaign; `max_words` is soft guidance passed to the
model.

## Safety properties

These are enforced in code and covered by tests, not left to discipline.

**Dry-run is structural.** A sender capable of delivering mail is constructed
only by `build_sender(..., live=True)`. Without `--live` the loop is handed a
`DryRunSender`, so it does not skip the send — it never holds an object that
could send.

**The daily cap has a ceiling configuration cannot raise.** `HARD_DAILY_CAP` in
`runner.py` is 50. A `DAILY_CAP` above it is clamped, loudly, on every run. The
count comes from the database at the top of every iteration, so it survives a
crash, a restart, and two terminals running at once.

**Nobody is written to twice.** `contacts.email` is UNIQUE, addresses are
normalised on every read and write, and `already_contacted()` is global across
campaigns. A *failed* send counts as contact: an SMTP error can be raised after
the message was handed off, and a duplicate is worse than a miss.

**Suppression is checked immediately before delivery.** Not once when candidates
were selected. Up to three minutes of pause and one composition call sit between
selection and sending, and a removal request that lands in that window still
takes effect on that very message.

**Every email offers a way out.** The removal line is appended by Python after
the model returns, and the runner refuses to send a body that lacks it. A
guarantee that depends on a model following an instruction is not a guarantee.

**A human reply ends the sequence.** Interested, rejection and opt-out all stop
follow-ups. An out-of-office does not — it is not a person saying no.

**A composition failure is not a send failure.** If Claude cannot produce a
usable email, no outreach row is written at all. A `failed` row means "this may
have reached SMTP", which marks the address as contacted forever; a contact we
never managed to write an email for stays eligible for the next run.

---

## Configuration

Everything lives in `.env`; see `.env.example` for the annotated list.

The two worth knowing:

- **`DATABASE_URL`** defaults to `sqlite:///outreach.db`. Point it at
  `postgresql+psycopg://...` and nothing else changes — there is one data layer,
  and no code branches on the backend.
- **`ANTHROPIC_MODEL`** defaults to `claude-sonnet-5`. Both LLM calls depend on
  structured outputs, so the model is validated against a support list at
  startup; an unsupported one fails before the first contact, not at the
  seventeenth.

### Campaigns

A campaign is a TOML file: who you are, what you want, the tone, the Apollo
filters, and the follow-up policy. Copy `campaigns/job_search.toml`, change it,
and point `CAMPAIGN` at it. No code changes.

```bash
outreach run --campaign campaigns/partnerships.toml
```

---

## Development

```bash
pytest            # 325 tests, no network, no clock
ruff check src tests
```

The suite runs against in-memory SQLite with every external service injected as
a fake. Two guarantees are enforced by autouse fixtures rather than by
remembering:

- **`socket.socket.connect` raises.** If any test reaches for the network, it
  fails loudly instead of quietly making a real call.
- **`time.sleep` raises for anything over a second.** A test that forgot to
  inject a fake clock would otherwise pass while taking hours.

### Layout

| Module | Responsibility |
|---|---|
| `config.py` | Environment loading and validation. Reports every problem at once, at startup. |
| `campaigns.py` | Campaign templates — the content, as distinct from the configuration. |
| `store.py` | SQLAlchemy models, sessions, and every query. Dialect-agnostic. |
| `apollo_client.py` | People search and enrichment. Credit-aware, exponential backoff, typed errors. |
| `composer.py` | The Anthropic call that writes the email. |
| `sender.py` | `Sender` interface, `SmtpSender`, `GmailApiSender` (stub), `DryRunSender`. |
| `inbox.py` | IMAP polling, bounce detection, reply classification. |
| `runner.py` | The loop: caps, jitter, per-send checks, dry-run. |
| `cli.py` | Entry point. |

### Not built yet

- **Migrations.** `init-db` calls `create_all()`. There is no Alembic setup, so
  a schema change today means recreating the database.
- **The Gmail API sender.** `GmailApiSender` exists behind the interface and
  raises `NotImplementedError`. SMTP with an app password is the working path.

---

## A word on using this

Cold outreach is easy to do badly. The caps, the delays, the removal line and
the suppression list are here because sending less, to fewer, better-chosen
people is both kinder and more effective. Raising the ceiling is a code change,
deliberately.

Check your jurisdiction's rules on unsolicited email before you send anything.

## License

MIT. See [LICENSE](LICENSE).
