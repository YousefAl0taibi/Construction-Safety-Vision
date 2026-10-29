from __future__ import annotations

from pathlib import Path

import yaml

from .detector import OnnxYoloDetector, load_labels
from .rules import RuleConfig, SafetyRuleEngine


def build_components(config_path: str | Path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    detector = OnnxYoloDetector(
        model_path=config["model_path"],
        labels=load_labels(config["labels_path"]),
        input_size=config.get("input_size", 640),
        confidence_threshold=config.get("confidence_threshold", 0.35),
        iou_threshold=config.get("iou_threshold", 0.45),
        provider=config.get("provider", "auto"),
    )
    rules = SafetyRuleEngine(
        RuleConfig(
            window_size=config.get("violation_window", 20),
            minimum_positive_frames=config.get("violation_min_frames", 15),
            cooldown_seconds=config.get("event_cooldown_seconds", 10),
        )
    )
    return detector, rules

