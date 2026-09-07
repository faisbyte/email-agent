"""Tests for campaign loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from outreach_agent.campaigns import CampaignError, load_campaign

REPO_CAMPAIGNS = Path(__file__).resolve().parent.parent / "campaigns"

VALID = """
name = "Test"
sender_persona = "An engineer."
goal = "Start a conversation."
tone = ["Direct."]
subject_guidance = "Short."
max_words = 120
removal_line = "Reply 'no thanks' and I won't write again."

[apollo]
person_titles = ["Recruiter"]

[follow_up]
max_follow_ups = 1
days_between = 5
goal = "One nudge."
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "campaign.toml"
    path.write_text(text)
    return path


# ─────────────────────────── the shipped campaigns ───────────────────────────


@pytest.mark.parametrize("name", ["job_search.toml", "partnerships.toml"])
def test_the_shipped_campaigns_load(name):
    """A broken example in the repo is a broken first experience."""
    campaign = load_campaign(REPO_CAMPAIGNS / name)
    assert campaign.name
    assert campaign.removal_line
    assert campaign.apollo.person_titles


def test_the_two_shipped_campaigns_are_genuinely_different():
    """The job search is one configuration, not the architecture."""
    job = load_campaign(REPO_CAMPAIGNS / "job_search.toml")
    partnerships = load_campaign(REPO_CAMPAIGNS / "partnerships.toml")

    assert job.goal != partnerships.goal
    assert job.apollo.person_titles != partnerships.apollo.person_titles
    assert job.follow_up.max_follow_ups != partnerships.follow_up.max_follow_ups


# ─────────────────────────── loading ───────────────────────────


def test_a_valid_campaign_parses(tmp_path):
    campaign = load_campaign(write(tmp_path, VALID))

    assert campaign.name == "Test"
    assert campaign.max_words == 120
    assert campaign.slug == "campaign"
    assert campaign.follow_up.days_between == 5


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(CampaignError, match="file not found"):
        load_campaign(tmp_path / "absent.toml")


def test_invalid_toml_is_reported(tmp_path):
    with pytest.raises(CampaignError, match="not valid TOML"):
        load_campaign(write(tmp_path, "name = = ="))


def test_every_missing_field_is_reported_at_once(tmp_path):
    with pytest.raises(CampaignError) as exc:
        load_campaign(write(tmp_path, 'name = "Only a name"'))

    problems = " ".join(exc.value.problems)
    for field in ("sender_persona", "goal", "subject_guidance", "removal_line"):
        assert field in problems


def test_a_removal_line_containing_a_url_is_rejected(tmp_path):
    """There is no web server here, so a link would point at nothing."""
    text = VALID.replace(
        "removal_line = \"Reply 'no thanks' and I won't write again.\"",
        'removal_line = "Unsubscribe at https://example.com/unsub"',
    )
    with pytest.raises(CampaignError, match="must not contain a URL"):
        load_campaign(write(tmp_path, text))


def test_an_unfiltered_apollo_search_is_rejected(tmp_path):
    """An empty filter set asks Apollo for everyone."""
    text = VALID.replace('person_titles = ["Recruiter"]', "")
    with pytest.raises(CampaignError, match="no filters"):
        load_campaign(write(tmp_path, text))


def test_a_negative_word_limit_is_rejected(tmp_path):
    text = VALID.replace("max_words = 120", "max_words = -5")
    with pytest.raises(CampaignError, match="positive whole number"):
        load_campaign(write(tmp_path, text))


def test_a_zero_day_follow_up_gap_is_rejected(tmp_path):
    text = VALID.replace("days_between = 5", "days_between = 0")
    with pytest.raises(CampaignError, match="at least 1"):
        load_campaign(write(tmp_path, text))


def test_tone_must_be_a_list_of_strings(tmp_path):
    text = VALID.replace('tone = ["Direct."]', 'tone = "Direct."')
    with pytest.raises(CampaignError, match="list of strings"):
        load_campaign(write(tmp_path, text))


def test_follow_up_defaults_apply_when_the_section_is_absent(tmp_path):
    text = VALID.split("[follow_up]")[0]
    campaign = load_campaign(write(tmp_path, text))

    assert campaign.follow_up.max_follow_ups == 1
    assert campaign.follow_up.days_between == 5
