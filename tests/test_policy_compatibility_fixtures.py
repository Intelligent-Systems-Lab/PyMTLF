import json
from pathlib import Path


def test_policy_oracle_has_phase_one_families_and_provenance():
    path = Path(__file__).parent / "fixtures" / "policy_compatibility" / "policy_oracle_v1.json"
    oracle = json.loads(path.read_text(encoding="utf-8"))

    assert oracle["schema_version"] == "1.0"
    assert oracle["baseline"]["repository"] == "NWDAF"
    assert len(oracle["baseline"]["commit"]) == 40
    fixtures = oracle["fixtures"]
    assert len(fixtures) == 14
    assert len({fixture["fixture_id"] for fixture in fixtures}) == len(fixtures)
    for fixture in fixtures:
        assert fixture["classification"] in {"historical_invariant", "approved_change"}
        assert fixture["logical_clock"].endswith("Z")
        assert fixture["provenance"]["commit"] == oracle["baseline"]["commit"]
        assert fixture["provenance"]["test_name"].startswith("Test")
        assert isinstance(fixture["policy_config"], dict)
        assert fixture["ordered_reports"]
        assert fixture["expected_steps"]
        assert isinstance(fixture["retrain_trigger_expected"], bool)
        assert fixture["dedup_stale_behavior"]
