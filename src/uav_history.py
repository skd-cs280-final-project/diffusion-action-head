"""Causal UAV observation history, shared by offline training and online policy.

World vectors use AirSim's NED convention and quaternions are ``[x, y, z, w]``.
All positions and velocities are rotated into the *current yaw* frame, whose
vertical axis stays aligned with gravity. No labels, future poses, finite
difference velocities, or assumed frame durations enter this representation.
"""

from collections.abc import Mapping
import math
from numbers import Real

import torch


STATE_SCHEMA_VERSION = 1
STATE_FEATURE_NAMES = (
    "relative_x", "relative_y", "relative_z",
    "roll_sin", "roll_cos", "pitch_sin", "pitch_cos",
    "relative_yaw_sin", "relative_yaw_cos",
    "velocity_x", "velocity_y", "velocity_z", "age_seconds",
)
STATE_VALUE_DIM = len(STATE_FEATURE_NAMES)
STATE_DIM = 2 * STATE_VALUE_DIM
HISTORY_SCHEMA = {
    "version": STATE_SCHEMA_VERSION,
    "state_dim": STATE_DIM,
    "features": list(STATE_FEATURE_NAMES),
    "layout": "13 values followed by 13 per-feature validity bits",
    "position_scale_m": 20.0,
    "velocity_scale_m_s": 5.0,
    "age_scale_s": 10.0,
    "coordinate_frame": "current position and yaw; world NED gravity axis",
    "quaternion_order": "xyzw",
    "timestamp_unit": "AirSim nanoseconds",
    "padding": "left zero padding; bool token mask; current observation last",
    "clip": [-1.0, 1.0],
    "missing": "zero value and zero feature-validity bit; no inferred velocity",
}


def _state(record):
    if not isinstance(record, Mapping):
        raise ValueError("Every history observation must be a mapping.")
    if "sensors" in record:
        sensors = record["sensors"]
        value = sensors.get("state") if isinstance(sensors, Mapping) else None
    else:
        value = record
    if not isinstance(value, Mapping):
        raise ValueError("History observation is missing sensors.state.")
    return value


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _timestamp(state):
    # The plain timestamp field is explicitly AirSim nanoseconds, never seconds
    # guessed from magnitude. Preserve Python ints before subtracting epoch time.
    value = state.get("timestamp")
    return value if _finite(value) and value > 0 else None


def observation_identity(record):
    """Identify a repeated observation without treating hovering as a copy.

    Shared with the offline log loader so both paths select the same K unique
    observations even when their raw streams contain duplicate timestamps.
    """
    state = _state(record)
    timestamp = _timestamp(state)
    if timestamp is not None:
        return ("timestamp_ns", timestamp)
    for source in (record, state):
        for name in ("frame_id", "frame_index", "frame", "img_name"):
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                return (name, value)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return (name, value)
    # Repeated object references are copies; matching poses without matching IDs
    # can be genuine stationary observations and must survive.
    return ("object_identity", id(state))


def _pose(state):
    vectors = []
    for name, size in (("position", 3), ("orientation", 4)):
        value = state.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != size or not all(map(_finite, value)):
            raise ValueError(f"History state requires finite {name} with {size} coordinates.")
        vectors.append([float(component) for component in value])
    position, quaternion = vectors
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-8 or not math.isfinite(norm):
        raise ValueError("History orientation must be a nonzero finite xyzw quaternion.")
    x, y, z, w = (component / norm for component in quaternion)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return position, roll, pitch, yaw


def encode_state_history(records, history_length):
    """Return ``float32[K, 26]`` values/validity and a ``bool[K]`` token mask.

    ``records`` is an oldest-to-current sequence of observation dictionaries
    (``sensors.state``) or state dictionaries. Duplicate timestamps/frame IDs
    are removed before selecting the newest K records. A missing velocity or
    timestamp is masked; missing or malformed position/orientation fails.
    ``K=0`` returns empty tensors, allowing old checkpoints to disable history.
    """
    if type(history_length) is not int or history_length < 0:
        raise ValueError("history_length must be a nonnegative integer.")
    encoded = torch.zeros((history_length, STATE_DIM), dtype=torch.float32)
    mask = torch.zeros(history_length, dtype=torch.bool)
    if history_length == 0:
        return encoded, mask
    records = list(records)
    if not records:
        raise ValueError("State history requires at least the current observation.")

    # Traverse backwards so the final/current observation is retained even if
    # the simulator appended several references to the same terminal state.
    selected, seen = [], set()
    for record in reversed(records):
        state = _state(record)
        key = observation_identity(record)
        if key not in seen:
            selected.append(state)
            seen.add(key)
        if len(selected) == history_length:
            break
    selected.reverse()
    poses = [_pose(state) for state in selected]
    current_position, _, _, current_yaw = poses[-1]
    current_time = _timestamp(selected[-1])
    c, s = math.cos(current_yaw), math.sin(current_yaw)
    previous_timestamp = None
    for offset, (state, pose) in enumerate(zip(selected, poses), history_length - len(selected)):
        position, roll, pitch, yaw = pose
        dx, dy, dz = [position[axis] - current_position[axis] for axis in range(3)]
        values = [
            (c * dx + s * dy) / 20.0, (-s * dx + c * dy) / 20.0, dz / 20.0,
            math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch),
            math.sin(yaw - current_yaw), math.cos(yaw - current_yaw),
            0.0, 0.0, 0.0, 0.0,
        ]
        valid = [1.0] * 9 + [0.0] * 4
        velocity = state.get("linear_velocity")
        if isinstance(velocity, (list, tuple)) and len(velocity) == 3:
            if _finite(velocity[0]) and _finite(velocity[1]):
                vx, vy = float(velocity[0]), float(velocity[1])
                values[9:11] = [(c * vx + s * vy) / 5.0, (-s * vx + c * vy) / 5.0]
                valid[9:11] = [1.0, 1.0]
            if _finite(velocity[2]):
                values[11], valid[11] = float(velocity[2]) / 5.0, 1.0
        timestamp = _timestamp(state)
        if timestamp is not None:
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError("History timestamps must be ordered from oldest to current.")
            previous_timestamp = timestamp
        if timestamp is not None and current_time is not None:
            if timestamp > current_time:
                raise ValueError("History contains an observation later than the current timestamp.")
            values[12], valid[12] = (current_time - timestamp) * 1e-9 / 10.0, 1.0
        encoded[offset] = torch.tensor(
            [max(-1.0, min(1.0, value)) for value in values] + valid, dtype=torch.float32,
        )
        mask[offset] = True
    return encoded, mask
