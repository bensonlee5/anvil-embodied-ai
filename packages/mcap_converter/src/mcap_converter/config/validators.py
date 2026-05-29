"""Configuration validation for the unified mcap_converter config."""

from typing import List

from .schema import DataConfig, FeatureMapping, JointNamePattern


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""


def validate_joint_name_pattern(pattern: JointNamePattern) -> List[str]:
    """Joint mode: source/arms/separator must be coherent."""
    errors: List[str] = []
    if not pattern.source:
        errors.append("joint_names.source cannot be empty")
    else:
        for prefix, role in pattern.source.items():
            if role not in ("observation", "action"):
                errors.append(
                    f"joint_names.source: prefix {prefix!r} maps to {role!r}; "
                    "must be 'observation' or 'action'"
                )
        if "observation" not in set(pattern.source.values()):
            errors.append("joint_names.source must include an 'observation' mapping")
    if not pattern.separator:
        errors.append("joint_names.separator cannot be empty")
    return errors


def validate_feature_mapping(mapping: FeatureMapping, name: str) -> List[str]:
    errors: List[str] = []
    valid_fields = {"position", "velocity", "effort"}
    if not mapping.state:
        errors.append(f"{name}.state cannot be empty")
    elif mapping.state not in valid_fields:
        errors.append(f"{name}.state {mapping.state!r} is not a valid JointState field")
    for f in mapping.others:
        if f not in valid_fields:
            errors.append(f"{name}.others contains invalid field {f!r}")
    return errors


def validate_config(config: DataConfig) -> None:
    """Raises :class:`ConfigurationError` when the unified config is malformed."""
    errors: List[str] = []

    # data_space
    if config.data_space not in ("joint", "ee"):
        errors.append(f"data_space must be 'joint' or 'ee'; got {config.data_space!r}")

    # observation_topics — required, defines arm scope
    if not config.observation_topics:
        errors.append(
            "observation_topics cannot be empty; list one topic per arm "
            "(e.g. { left: /joint_states, right: /joint_states })."
        )

    if config.data_space == "joint":
        # All arms must share a single /joint_states topic — robot_state_topic property
        # collapses to it. Reading the property validates uniqueness.
        try:
            _ = config.robot_state_topic
        except ValueError as exc:
            errors.append(str(exc))

        errors.extend(validate_joint_name_pattern(config.joint_name_pattern))

        for arm_id, spec in config.action_topics.items():
            if not spec.topic:
                errors.append(f"action_topics[{arm_id!r}].topic cannot be empty")
            if not spec.joint_order:
                errors.append(
                    f"action_topics[{arm_id!r}].joint_order cannot be empty in joint mode; "
                    "specify the ordered joint list matching Float64MultiArray.data."
                )
            if arm_id not in config.observation_topics:
                errors.append(
                    f"action_topics[{arm_id!r}] references an arm not present in observation_topics"
                )

    if config.data_space == "ee":
        if config.action_topics:
            errors.append(
                "action_topics must be empty in ee mode; the EE action is derived "
                "from the same /ee_pose_<arm> topics listed in observation_topics."
            )

    # observation_feature_mapping
    errors.extend(validate_feature_mapping(
        config.observation_feature_mapping, "observation_feature_mapping"
    ))
    errors.extend(validate_feature_mapping(
        config.action_feature_mapping, "action_feature_mapping"
    ))

    # cameras
    if not config.camera_topics:
        errors.append("camera_topics cannot be empty")
    if not config.camera_topic_mapping:
        errors.append("camera_topic_mapping cannot be empty")
    else:
        for t in config.camera_topics:
            if t not in config.camera_topic_mapping:
                errors.append(f"camera topic {t!r} missing from camera_topic_mapping")

    # image_resolution
    if not config.image_resolution or len(config.image_resolution) != 2:
        errors.append("image_resolution must be [width, height]")
    elif any(dim <= 0 for dim in config.image_resolution):
        errors.append("image_resolution dimensions must be positive")

    if errors:
        raise ConfigurationError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )


def validate_topics_exist(config: DataConfig, available_topics: List[str]) -> None:
    """Cross-check that observation/action/camera topics are present in the MCAP."""
    missing: List[str] = []
    seen_obs = set()

    for arm_id, topic in config.observation_topics.items():
        if topic in seen_obs:
            continue  # Shared /joint_states across arms — only check once
        seen_obs.add(topic)
        if topic not in available_topics:
            missing.append(f"observation_topics[{arm_id!r}]: {topic}")

    for arm_id, spec in config.action_topics.items():
        if spec.topic and spec.topic not in available_topics:
            missing.append(f"action_topics[{arm_id!r}].topic: {spec.topic}")

    for topic in config.camera_topics:
        if topic not in available_topics:
            missing.append(f"camera_topic: {topic}")

    if missing:
        raise ConfigurationError(
            "Topics not found in MCAP file:\n  - "
            + "\n  - ".join(missing)
            + f"\n\nAvailable topics: {available_topics}"
        )
