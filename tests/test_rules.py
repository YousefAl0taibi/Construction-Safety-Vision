from safety_vision.rules import RuleConfig, SafetyRuleEngine
from safety_vision.types import Detection


def detection(label: str, box: tuple[int, int, int, int], track_id=None) -> Detection:
    class_id = {"person": 0, "helmet": 1, "safety_vest": 2, "forklift": 5}.get(label, 0)
    return Detection(*box, 0.9, class_id, label, track_id)


def test_missing_helmet_requires_temporal_confirmation():
    engine = SafetyRuleEngine(RuleConfig(window_size=3, minimum_positive_frames=2, cooldown_seconds=0))
    person = detection("person", (0, 0, 100, 200), track_id=7)
    vest = detection("safety_vest", (20, 70, 80, 150))
    assert engine.evaluate([person, vest], 0) == []
    assert engine.evaluate([person, vest], 1) == []
    events = engine.evaluate([person, vest], 2)
    assert [event.event_type for event in events] == ["MISSING_HELMET"]


def test_ppe_inside_person_suppresses_violation():
    engine = SafetyRuleEngine(RuleConfig(window_size=1, minimum_positive_frames=1, cooldown_seconds=0))
    person = detection("person", (0, 0, 100, 200), track_id=3)
    helmet = detection("helmet", (30, 10, 70, 45))
    vest = detection("safety_vest", (20, 70, 80, 150))
    assert engine.evaluate([person, helmet, vest], 0) == []


def test_danger_zone_requires_machine_presence():
    engine = SafetyRuleEngine(RuleConfig(window_size=1, minimum_positive_frames=1, cooldown_seconds=0))
    person = detection("person", (20, 20, 80, 100), track_id=1)
    helmet = detection("helmet", (35, 25, 65, 45))
    vest = detection("safety_vest", (30, 55, 70, 90))
    zone = [[(0, 0), (200, 0), (200, 200), (0, 200)]]
    assert engine.evaluate([person, helmet, vest], 0, danger_zones=zone) == []
    forklift = detection("forklift", (120, 20, 190, 100))
    events = engine.evaluate([person, helmet, vest, forklift], 1, danger_zones=zone)
    assert events[0].event_type == "DANGER_ZONE_INTRUSION"

