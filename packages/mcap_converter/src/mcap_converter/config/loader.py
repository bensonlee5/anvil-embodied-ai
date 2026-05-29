"""Configuration loader for the unified YAML format.

Schema (joint and EE share the same top-level keys; ``data_space`` switches
encoding only)::

    data_space: "joint" | "ee"
    observation_topics:
      <arm_id>: <topic>
      ...
    action_topics:
      <arm_id>:
        topic: <topic>
        joint_order: [...]
      ...                              # empty in EE mode
    joint_names:                       # joint mode only
      separator: "_"
      source:  { follower: observation }
      arms:    { l: left, r: right }
    observation_feature_mapping:       # new configs use others: []
      state: position
      others: []
    camera_topics: [...]
    camera_topic_mapping: { <topic>: <name>, ... }
    image_resolution: [W, H]

Legacy formats (singular ``robot_state_topic`` field, topic-keyed
``action_topics``, ``robot_state_topics`` plural, ``motor_feature_mapping``,
``action_from_observation*``) are no longer accepted.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .schema import (
    ActionTopicSpec,
    DataConfig,
    FeatureMapping,
    JointNamePattern,
)


class ConfigLoader:
    """Load configuration from YAML / dict into a :class:`DataConfig`."""

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def from_yaml(config_path: str) -> DataConfig:
        return ConfigLoader.from_dict(ConfigLoader.load_yaml(config_path))

    @staticmethod
    def get_default() -> DataConfig:
        return DataConfig()

    # ------------------------------------------------------------------
    # New unified format parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_joint_name_pattern(pattern_dict: Optional[Dict]) -> JointNamePattern:
        if not pattern_dict:
            return JointNamePattern()
        defaults = JointNamePattern()
        return JointNamePattern(
            source=pattern_dict.get("source", defaults.source),
            arms=pattern_dict.get("arms", defaults.arms),
            separator=pattern_dict.get("separator", defaults.separator),
        )

    @staticmethod
    def _parse_observation_topics(value: Any) -> Dict[str, str]:
        if not value:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                "observation_topics must be a mapping of arm_id -> topic, "
                f"got {type(value).__name__}"
            )
        out: Dict[str, str] = {}
        for arm_id, topic in value.items():
            if not isinstance(topic, str) or not topic:
                raise ValueError(
                    f"observation_topics[{arm_id!r}] must be a non-empty topic string"
                )
            out[str(arm_id)] = topic
        return out

    @staticmethod
    def _parse_action_topics(value: Any) -> Dict[str, ActionTopicSpec]:
        """Parse ``action_topics: { arm_id: { topic, joint_order } }``.

        Empty / missing → ``{}`` (EE mode, or joint "act-from-obs" opt-in).
        """
        if not value:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                "action_topics must be a mapping of arm_id -> { topic, joint_order }, "
                f"got {type(value).__name__}"
            )
        out: Dict[str, ActionTopicSpec] = {}
        for arm_id, spec in value.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"action_topics[{arm_id!r}] must be a mapping with keys "
                    f"'topic' and 'joint_order'; got {type(spec).__name__}"
                )
            topic = spec.get("topic", "")
            if not topic or not isinstance(topic, str):
                raise ValueError(
                    f"action_topics[{arm_id!r}].topic must be a non-empty string"
                )
            joint_order = list(spec.get("joint_order") or [])
            out[str(arm_id)] = ActionTopicSpec(topic=topic, joint_order=joint_order)
        return out

    @staticmethod
    def _parse_feature_mapping(
        mapping_dict: Optional[Dict], default: FeatureMapping
    ) -> FeatureMapping:
        if not mapping_dict:
            return default
        return FeatureMapping(
            state=mapping_dict.get("state", default.state),
            others=list(mapping_dict.get("others", default.others)),
        )

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> DataConfig:
        defaults = DataConfig()

        data_space = str(config_dict.get("data_space", defaults.data_space))
        if data_space not in ("joint", "ee"):
            raise ValueError(
                f"data_space must be 'joint' or 'ee'; got {data_space!r}"
            )

        observation_topics = ConfigLoader._parse_observation_topics(
            config_dict.get("observation_topics")
        )
        action_topics = ConfigLoader._parse_action_topics(
            config_dict.get("action_topics")
        )

        joint_name_pattern = ConfigLoader._parse_joint_name_pattern(
            config_dict.get("joint_names") or config_dict.get("joint_name_pattern")
        )

        observation_feature_mapping = ConfigLoader._parse_feature_mapping(
            config_dict.get("observation_feature_mapping"),
            defaults.observation_feature_mapping,
        )
        action_feature_mapping = ConfigLoader._parse_feature_mapping(
            config_dict.get("action_feature_mapping"),
            defaults.action_feature_mapping,
        )

        return DataConfig(
            data_space=data_space,
            observation_topics=observation_topics,
            action_topics=action_topics,
            joint_name_pattern=joint_name_pattern,
            observation_feature_mapping=observation_feature_mapping,
            action_feature_mapping=action_feature_mapping,
            camera_topics=list(config_dict.get("camera_topics") or defaults.camera_topics),
            camera_topic_mapping=dict(
                config_dict.get("camera_topic_mapping") or defaults.camera_topic_mapping
            ),
            image_resolution=list(
                config_dict.get("image_resolution") or defaults.image_resolution
            ),
        )
