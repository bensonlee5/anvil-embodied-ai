"""Configuration schema for MCAP to LeRobot conversion.

The unified config (introduced with EE-space support) has a single shape that
works for joint and EE modes:

    data_space:         "joint" | "ee"
    observation_topics: { arm_id -> topic }
    action_topics:      { arm_id -> ActionTopicSpec }   # empty in EE mode
    joint_names:        JointNamePattern (joint mode only — splits /joint_states by arm)
    camera_topics:      [...]
    camera_topic_mapping: { topic -> dataset_camera_name }
    image_resolution:   [W, H]

The legacy joint-extraction code reads ``robot_state_topic`` (single string) and
``action_command_topics`` (``{topic -> ActionTopicConfig}``). Both are exposed
as ``@property`` derivations over the new fields so the joint extractor stays
byte-identical apart from the attribute name.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class JointNamePattern:
    """Parsing rules for joint names inside a shared /joint_states topic.

    Joint names follow ``{source}{separator}{arm}{separator}{joint_id}``::

        "follower_l_joint1" -> observation, left arm, joint1

    Only ``source`` and ``arms`` mappings are required. The ``source`` map
    classifies a name prefix as either ``observation`` or ``action`` (the
    latter only relevant in leader-follower mode, which is not used by the
    new unified configs). ``arms`` maps the per-arm identifier letter to its
    canonical name.
    """

    source: Dict[str, str] = field(
        default_factory=lambda: {
            "leader": "action",
            "follower": "observation",
        }
    )
    arms: Dict[str, str] = field(
        default_factory=lambda: {
            "r": "right",
            "l": "left",
        }
    )
    separator: str = "_"

    @property
    def role_prefix(self) -> Dict[str, str]:
        """Alias kept for joint-extractor code that reads `role_prefix`."""
        return self.source

    @property
    def robot_prefix(self) -> Dict[str, str]:
        """Alias kept for joint-extractor code that reads `robot_prefix`."""
        return self.arms


@dataclass
class ActionTopicConfig:
    """Internal/legacy view of an action command topic, keyed by topic name.

    Returned by :pyattr:`DataConfig.action_command_topics` so the joint
    extractor methods can keep reading ``.arm`` / ``.joint_order``.
    """

    arm: str = ""
    joint_order: List[str] = field(default_factory=list)


@dataclass
class ActionTopicSpec:
    """User-facing per-arm action source.

    The new YAML format is ``action_topics: { arm_id: ActionTopicSpec }``. In
    EE mode this whole map is empty; in joint mode each entry provides the
    Float64MultiArray command topic and the joint ordering that maps the
    flat data array to canonical joint slots.
    """

    topic: str = ""
    joint_order: List[str] = field(default_factory=list)


@dataclass
class FeatureMapping:
    """Selects which JointState fields to extract for a given role.

    New unified configs set ``others: []`` for both observation and action;
    velocity/effort are dropped going forward.
    """

    state: str = "position"
    others: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Unified converter config.

    ``data_space`` is the only switch between joint and EE conversion paths.
    Arm scope is determined entirely by the keys of ``observation_topics`` —
    there is no separate ``arms`` block. Insertion order of
    ``observation_topics`` defines the per-arm concatenation order in the
    output ``observation.state`` / ``action`` features.
    """

    data_space: str = "joint"
    observation_topics: Dict[str, str] = field(default_factory=dict)
    action_topics: Dict[str, ActionTopicSpec] = field(default_factory=dict)

    joint_name_pattern: JointNamePattern = field(default_factory=JointNamePattern)

    observation_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(state="position", others=[])
    )
    action_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(state="position", others=[])
    )

    camera_topics: List[str] = field(default_factory=list)
    camera_topic_mapping: Dict[str, str] = field(default_factory=dict)
    image_resolution: List[int] = field(default_factory=lambda: [640, 480])

    # Kept defaulted-empty so the legacy DataExtractor batch class doesn't
    # blow up at import time. Never populated by the new loader.
    robot_state_topics: List[str] = field(default_factory=list)
    motor_feature_mapping: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties (keep joint-extraction code byte-identical)
    # ------------------------------------------------------------------

    @property
    def is_ee(self) -> bool:
        return self.data_space == "ee"

    @property
    def arms(self) -> List[str]:
        """Arms in concatenation order, taken from observation_topics insertion order."""
        return list(self.observation_topics.keys())

    @property
    def robot_state_topic(self) -> str:
        """Single distinct value of observation_topics.

        Joint mode always shares ``/joint_states`` across arms, so this
        collapses to one topic. Raises if multiple distinct values were
        listed (which would be nonsensical for the joint path). Returns the
        empty string when observation_topics is empty.
        """
        topics = set(self.observation_topics.values())
        if not topics:
            return ""
        if len(topics) > 1:
            raise ValueError(
                "observation_topics has multiple distinct values "
                f"{sorted(topics)}; robot_state_topic is undefined."
            )
        return next(iter(topics))

    @property
    def action_command_topics(self) -> Dict[str, ActionTopicConfig]:
        """``{topic -> ActionTopicConfig(arm, joint_order)}`` for joint-path code.

        Inverts the new per-arm ``action_topics`` map into the topic-keyed
        shape the joint extractor methods consume. Empty in EE mode and in
        the joint "act-from-obs" opt-in (empty ``action_topics``).
        """
        out: Dict[str, ActionTopicConfig] = {}
        for arm_id, spec in self.action_topics.items():
            if not spec.topic:
                continue
            out[spec.topic] = ActionTopicConfig(arm=arm_id, joint_order=list(spec.joint_order))
        return out


# Default configuration (empty maps — convert.py always supplies a real config).
DEFAULT_DATA_CONFIG = DataConfig()
