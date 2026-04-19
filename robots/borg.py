"""Borg robot: bimanual 6-DOF arms with grippers, driven over ZMQ.

A companion ``leborg_bridge`` ROS 2 node (see ``bridge/leborg_bridge.py``) runs
in the user's ROS 2 environment and forwards ``/joint_states``, two
``std_msgs/Bool`` gripper-contact topics, and three camera streams over
ZMQ to this client. Actions flow back the other way and are republished
on ``/pid_controller/reference/state``.

Splitting ROS 2 and LeRobot into two processes is required because the
two stacks pin incompatible Python versions; they cannot coexist in a
single interpreter.

Wire protocol (msgpack bytes, single-frame) — see ``bridge/leborg_bridge.py``
for the authoritative definition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property

import cv2
import msgpack
import numpy as np
import zmq

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

# ── Joint names (must match the order used during data collection) ───────────

LEFT_ARM_JOINTS = [
    "l_arm_pivot_1_joint",
    "l_arm_pivot_2_joint",
    "l_arm_pivot_3_joint",
    "l_arm_pivot_4_joint",
    "l_arm_pivot_5_joint",
    "l_arm_pivot_6_joint",
]
RIGHT_ARM_JOINTS = [
    "r_arm_pivot_1_joint",
    "r_arm_pivot_2_joint",
    "r_arm_pivot_3_joint",
    "r_arm_pivot_4_joint",
    "r_arm_pivot_5_joint",
    "r_arm_pivot_6_joint",
]

# Observation state names — MUST match the "names" list in
# info.json → features → observation.state so that build_dataset_frame
# can look them up by name in the dict returned by get_observation().
OBS_STATE_NAMES = [
    "action.l_arm_pivot_1_joint",
    "action.l_arm_pivot_2_joint",
    "action.l_arm_pivot_3_joint",
    "action.l_arm_pivot_4_joint",
    "action.l_arm_pivot_5_joint",
    "action.l_arm_pivot_6_joint",
    "action.l_gripper_position",
    "state.l_gripper_contact",
    "action.r_arm_pivot_1_joint",
    "action.r_arm_pivot_2_joint",
    "action.r_arm_pivot_3_joint",
    "action.r_arm_pivot_4_joint",
    "action.r_arm_pivot_5_joint",
    "action.r_arm_pivot_6_joint",
    "action.r_gripper_position",
    "state.r_gripper_contact",
]

# Action names — must match info.json → features → action → names
ACTION_NAMES = [
    "action.l_arm_pivot_1_joint",
    "action.l_arm_pivot_2_joint",
    "action.l_arm_pivot_3_joint",
    "action.l_arm_pivot_4_joint",
    "action.l_arm_pivot_5_joint",
    "action.l_arm_pivot_6_joint",
    "action.l_gripper_position",
    "action.r_arm_pivot_1_joint",
    "action.r_arm_pivot_2_joint",
    "action.r_arm_pivot_3_joint",
    "action.r_arm_pivot_4_joint",
    "action.r_arm_pivot_5_joint",
    "action.r_arm_pivot_6_joint",
    "action.r_gripper_position",
]

# Camera config: key → (topic — informational only, height, width)
CAMERAS = {
    "cam_head": ("/oak/rgb/image_raw", 720, 1280),
    "cam_left_wrist": ("/left_gripper_camera/image_raw", 720, 1280),
    "cam_right_wrist": ("/right_gripper_camera/image_raw", 720, 1280),
}


# ── Config ───────────────────────────────────────────────────────────────────


@RobotConfig.register_subclass("borg")
@dataclass
class BorgConfig(RobotConfig):
    """Configuration for the Borg bimanual robot."""

    # ZMQ bridge connection
    bridge_host: str = "localhost"
    obs_port: int = 5555
    cmd_port: int = 5556

    # Per-call poll timeout while streaming observations (milliseconds).
    recv_timeout_ms: int = 50

    # How long to wait for the first observation at connect() time (seconds).
    connection_timeout: float = 10.0

    # Image resolution (height, width) per camera — required for
    # observation_features; must match what the bridge ships.
    camera_resolution: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {k: (v[1], v[2]) for k, v in CAMERAS.items()}
    )

    # TODO: Adjust these joint name lists if your URDF differs from the defaults
    left_arm_joints: list[str] = field(default_factory=lambda: list(LEFT_ARM_JOINTS))
    right_arm_joints: list[str] = field(default_factory=lambda: list(RIGHT_ARM_JOINTS))

    # Names of the gripper position fields inside the JointState message.
    # On the real robot these are published as-is; in Isaac Lab the URDF
    # uses "finger_joint" for both grippers, so sim_gripper_remap maps
    # the duplicated URDF name to the per-side canonical names below.
    left_gripper_joint: str = "l_gripper_position"
    right_gripper_joint: str = "r_gripper_position"

    # Isaac Lab publishes "finger_joint" for both grippers.  Because the
    # bridge ships list-based JointState (preserving order) we can
    # disambiguate by finding which arm's last joint precedes each
    # occurrence.  Set to "" to disable remapping (real robot).
    sim_gripper_joint_name: str = "finger_joint"

    # Names of the gripper contact sensor fields (observation-only). The
    # bridge injects these into joint_state under these same names so the
    # ros_to_obs mapping below works without modification.
    left_gripper_contact_joint: str = "l_gripper_contact"
    right_gripper_contact_joint: str = "r_gripper_contact"


# ── Robot ────────────────────────────────────────────────────────────────────


class Borg(Robot):
    """Borg bimanual robot driven through the ZMQ bridge."""

    config_class = BorgConfig
    name = "borg"

    def __init__(self, config: BorgConfig):
        super().__init__(config)
        self.config = config

        # ZMQ state (populated in connect)
        self._zmq_ctx: zmq.Context | None = None
        self._obs_socket: zmq.Socket | None = None
        self._cmd_socket: zmq.Socket | None = None

        # Cached last observation so get_observation() returns the most
        # recent frame even when no new message has arrived in the last
        # recv_timeout_ms — mirrors the LeKiwi client pattern.
        self._last_joint_state: dict[str, float] | None = None
        self._last_images: dict[str, np.ndarray] = {}

        self._connected = False

    # ── Feature definitions ──────────────────────────────────────────────

    @cached_property
    def observation_features(self) -> dict:
        features: dict = {}
        for name in OBS_STATE_NAMES:
            features[name] = float
        for cam_key in CAMERAS:
            h, w = self.config.camera_resolution[cam_key]
            features[cam_key] = (h, w, 3)
        return features

    @cached_property
    def action_features(self) -> dict:
        return {name: float for name in ACTION_NAMES}

    # ── Connection ───────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        cfg = self.config
        self._zmq_ctx = zmq.Context()

        # PULL observations from the bridge. Set CONFLATE *before* connect
        # so it actually takes effect.
        self._obs_socket = self._zmq_ctx.socket(zmq.PULL)
        self._obs_socket.setsockopt(zmq.CONFLATE, 1)
        self._obs_socket.connect(f"tcp://{cfg.bridge_host}:{cfg.obs_port}")

        # PUSH actions to the bridge.
        self._cmd_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self._cmd_socket.connect(f"tcp://{cfg.bridge_host}:{cfg.cmd_port}")

        self._wait_for_data()
        self._connected = True

        if calibrate and not self.is_calibrated:
            self.calibrate()

    def _wait_for_data(self) -> None:
        deadline = time.time() + self.config.connection_timeout
        while time.time() < deadline:
            self._recv_latest()
            if self._last_joint_state is not None and all(
                k in self._last_images for k in CAMERAS
            ):
                return

        missing = []
        if self._last_joint_state is None:
            missing.append("joint_state")
        for k in CAMERAS:
            if k not in self._last_images:
                missing.append(k)
        raise TimeoutError(
            f"Timed out ({self.config.connection_timeout}s) waiting for data on: "
            f"{missing}. Is leborg_bridge running at "
            f"tcp://{self.config.bridge_host}:{self.config.obs_port}?"
        )

    # ── Calibration (no-op for Borg) ─────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ── ZMQ receive / decode ─────────────────────────────────────────────

    def _recv_latest(self) -> None:
        """Poll the obs socket; if a message is available, drain to the
        latest and decode it into ``self._last_joint_state`` /
        ``self._last_images``. No-op if no new message arrives within
        ``recv_timeout_ms``.
        """
        assert self._obs_socket is not None
        poller = zmq.Poller()
        poller.register(self._obs_socket, zmq.POLLIN)
        try:
            socks = dict(poller.poll(self.config.recv_timeout_ms))
        except zmq.ZMQError:
            return
        if self._obs_socket not in socks:
            return

        # Drain to the most recent message (belt-and-suspenders with CONFLATE=1).
        raw = None
        while True:
            try:
                raw = self._obs_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if raw is None:
            return

        try:
            payload = msgpack.unpackb(raw, raw=False)
        except Exception:
            # Keep the previous cache; drop this frame.
            return

        js = payload.get("joint_state")
        if not isinstance(js, dict) or "name" not in js or "position" not in js:
            return

        names = list(js["name"])
        positions = list(js["position"])

        # Isaac Lab publishes the same URDF name ("finger_joint") for
        # both left and right grippers.  Disambiguate by occurrence
        # order: the first is the left gripper, the second is the right.
        sim_name = self.config.sim_gripper_joint_name
        if sim_name:
            occurrence = 0
            for i, n in enumerate(names):
                if n == sim_name:
                    if occurrence == 0:
                        names[i] = self.config.left_gripper_joint
                    elif occurrence == 1:
                        names[i] = self.config.right_gripper_joint
                    occurrence += 1

        joint_state = {
            str(name): float(pos) for name, pos in zip(names, positions)
        }

        images: dict[str, np.ndarray] = {}
        decode_failed = False
        for cam_key, jpg_bytes in (payload.get("images") or {}).items():
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                decode_failed = True
                break
            # The bridge ships BGR; policies trained on the dataset expect
            # RGB (the old code used CvBridge(desired_encoding="rgb8")).
            images[cam_key] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Treat a partial observation as a dropped frame — keep the last
        # full cache rather than clobbering it with a partial update that
        # would trip KeyError in get_observation().
        if decode_failed:
            return

        self._last_joint_state = joint_state
        self._last_images = images

    # ── Observation ──────────────────────────────────────────────────────

    def get_observation(self) -> dict:
        if not self._connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        self._recv_latest()
        if self._last_joint_state is None:
            raise RuntimeError("No observation received from the bridge yet.")

        joint_state = self._last_joint_state
        images = self._last_images

        # Map ROS joint names → dataset observation.state names.
        # The dataset names use "action." prefix for controllable joints
        # and "state." prefix for sensor-only values (gripper contacts).
        cfg = self.config
        ros_to_obs = {
            **{j: f"action.{j}" for j in cfg.left_arm_joints},
            cfg.left_gripper_joint: f"action.{cfg.left_gripper_joint}",
            cfg.left_gripper_contact_joint: f"state.{cfg.left_gripper_contact_joint}",
            **{j: f"action.{j}" for j in cfg.right_arm_joints},
            cfg.right_gripper_joint: f"action.{cfg.right_gripper_joint}",
            cfg.right_gripper_contact_joint: f"state.{cfg.right_gripper_contact_joint}",
        }
        missing_joint_names = [
            ros_name for ros_name in ros_to_obs if ros_name not in joint_state
        ]
        if missing_joint_names:
            raise KeyError(
                "Missing joint names in latest JointState message: "
                f"{missing_joint_names}. Available names: {sorted(joint_state)}"
            )

        obs: dict = {}
        for ros_name, obs_name in ros_to_obs.items():
            obs[obs_name] = joint_state[ros_name]

        for cam_key in CAMERAS:
            if cam_key not in images:
                raise KeyError(f"Missing camera frame for {cam_key}")
            obs[cam_key] = images[cam_key]

        return obs

    # ── Actions ──────────────────────────────────────────────────────────

    def send_action(self, action: dict) -> dict:
        if not self._connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        cfg = self.config

        # Build the joint names and positions in the order the controller expects.
        # TODO: Adjust this ordering if your PID controller expects a different layout.
        joint_names = (
            cfg.left_arm_joints
            + [cfg.left_gripper_joint]
            + cfg.right_arm_joints
            + [cfg.right_gripper_joint]
        )
        action_keys = [f"action.{name}" for name in joint_names]
        missing_action_keys = [key for key in action_keys if key not in action]
        if missing_action_keys:
            raise KeyError(
                "Missing action keys in policy output: "
                f"{missing_action_keys}. Available keys: {sorted(action)}"
            )

        positions = [float(action[f"action.{name}"]) for name in joint_names]
        payload = {"name": joint_names, "position": positions}

        assert self._cmd_socket is not None
        self._cmd_socket.send(msgpack.packb(payload, use_bin_type=True))
        return action

    # ── Disconnect ───────────────────────────────────────────────────────

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        try:
            if self._obs_socket is not None:
                self._obs_socket.close(linger=0)
            if self._cmd_socket is not None:
                self._cmd_socket.close(linger=0)
            if self._zmq_ctx is not None:
                self._zmq_ctx.term()
        finally:
            self._obs_socket = None
            self._cmd_socket = None
            self._zmq_ctx = None
