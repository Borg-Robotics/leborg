#!/usr/bin/env python
"""Standalone ROS 2 ↔ ZMQ bridge for the Borg robot.

Runs in the user's ROS 2 environment (no LeRobot dependency). Forwards
the joint state, gripper-contact bools, and three camera streams to a
LeRobot client over ZMQ, and publishes the actions received back over
ZMQ to /pid_controller/reference/state.

This node deliberately does not depend on cv_bridge: sensor_msgs/Image
is converted manually via numpy. Only rgb8 and bgr8 encodings are
accepted — anything else is rejected loudly.

Wire protocol (msgpack bytes, single-frame):

    Observation (bridge → client, PUSH tcp://*:5555):
        {"stamp": float,
         "joint_state": {"name": [str, ...], "position": [float, ...]},
         "images": {cam_key: bytes, ...},            # JPEG-encoded
         "image_stamps": {cam_key: float, ...}}

    Action (client → bridge, PULL tcp://*:5556):
        {"name": [str, ...], "position": [float, ...]}

Both sockets use ZMQ_CONFLATE=1 to get latest-wins semantics, mirroring
the pattern in lerobot/src/lerobot/robots/lekiwi/lekiwi_host.py. Clients
may connect/reconnect freely.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import cv2
import msgpack
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to a BGR uint8 numpy array.

    Supports only rgb8 and bgr8 encodings. Raises ValueError otherwise.
    """
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(
            f"Unsupported image encoding {msg.encoding!r}; "
            "only 'rgb8' and 'bgr8' are supported."
        )
    expected_step = msg.width * 3
    if msg.step == expected_step:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
    else:
        # Non-contiguous rows (rare): strip row padding.
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = buf[:, :expected_step].reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img  # BGR


class BorgBridge(Node):
    """ROS 2 node that bridges Borg sensors/actuators to/from ZMQ."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("leborg_bridge")
        self._args = args

        # ── Latest data (protected by _lock) ──────────────────────────────
        self._lock = threading.Lock()
        self._latest_joint_state: dict | None = None  # {"name":[...], "position":[...]}
        self._latest_left_contact: float | None = None
        self._latest_right_contact: float | None = None
        self._latest_images: dict[str, np.ndarray] = {}
        self._latest_image_stamps: dict[str, float] = {}

        # Camera key → topic (keys are the LeRobot-side observation keys)
        self._camera_topics: dict[str, str] = {
            "cam_head": args.cam_head_topic,
            "cam_left_wrist": args.cam_left_wrist_topic,
            "cam_right_wrist": args.cam_right_wrist_topic,
        }

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(
            JointState, args.joint_state_topic, self._joint_state_cb, 10
        )
        self.create_subscription(
            Bool, args.left_contact_topic, self._left_contact_cb, 10
        )
        self.create_subscription(
            Bool, args.right_contact_topic, self._right_contact_cb, 10
        )
        for cam_key, topic in self._camera_topics.items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, key=cam_key: self._image_cb(key, msg),
                10,
            )

        # ── Publisher ────────────────────────────────────────────────────
        self._action_pub = self.create_publisher(JointState, args.action_topic, 10)

        # ── ZMQ (CONFLATE must be set before bind) ───────────────────────
        self._ctx = zmq.Context()
        self._obs_socket = self._ctx.socket(zmq.PUSH)
        self._obs_socket.setsockopt(zmq.CONFLATE, 1)
        self._obs_socket.bind(f"tcp://*:{args.obs_port}")

        self._cmd_socket = self._ctx.socket(zmq.PULL)
        self._cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self._cmd_socket.bind(f"tcp://*:{args.cmd_port}")

        # Observability: track whether we've ever published, and warn
        # periodically about which ROS sources haven't produced yet.
        self._publish_count: int = 0
        self._startup_time = time.time()
        self._last_diag_time = 0.0

        # ── Timers (run on the default single-threaded executor) ─────────
        period = 1.0 / args.pub_rate_hz
        self.create_timer(period, self._publish_observation)
        self.create_timer(period, self._poll_and_publish_action)
        self.create_timer(2.0, self._diagnostics)

        self.get_logger().info(
            f"leborg_bridge up: obs PUSH tcp://*:{args.obs_port}, "
            f"cmd PULL tcp://*:{args.cmd_port}, rate={args.pub_rate_hz} Hz"
        )
        self.get_logger().info(
            "Subscribed: "
            f"joint_state={args.joint_state_topic}, "
            f"left_contact={args.left_contact_topic}, "
            f"right_contact={args.right_contact_topic}, "
            f"cam_head={args.cam_head_topic}, "
            f"cam_left_wrist={args.cam_left_wrist_topic}, "
            f"cam_right_wrist={args.cam_right_wrist_topic}"
        )

    # ── Subscription callbacks ───────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joint_state = {
                "name": list(msg.name),
                "position": [float(p) for p in msg.position],
            }

    def _left_contact_cb(self, msg: Bool) -> None:
        with self._lock:
            self._latest_left_contact = 1.0 if msg.data else 0.0

    def _right_contact_cb(self, msg: Bool) -> None:
        with self._lock:
            self._latest_right_contact = 1.0 if msg.data else 0.0

    def _image_cb(self, cam_key: str, msg: Image) -> None:
        try:
            img = image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(f"[{cam_key}] {e}")
            return
        with self._lock:
            self._latest_images[cam_key] = img
            self._latest_image_stamps[cam_key] = time.time()

    # ── Timer callbacks ──────────────────────────────────────────────────

    def _missing_sources(self) -> list[str]:
        """Return the list of ROS topics that haven't produced a message yet."""
        missing: list[str] = []
        with self._lock:
            if self._latest_joint_state is None:
                missing.append(self._args.joint_state_topic)
            if self._latest_left_contact is None:
                missing.append(self._args.left_contact_topic)
            if self._latest_right_contact is None:
                missing.append(self._args.right_contact_topic)
            for cam_key, topic in self._camera_topics.items():
                if cam_key not in self._latest_images:
                    missing.append(topic)
        return missing

    def _diagnostics(self) -> None:
        """Periodically report bridge status until the first observation
        is published, then fall silent. Helps diagnose the common case
        where the bridge binds successfully but never has data to send.
        """
        if self._publish_count > 0:
            return
        missing = self._missing_sources()
        if missing:
            self.get_logger().warn(
                f"No observation published yet ({time.time() - self._startup_time:.1f}s "
                f"since startup). Waiting on: {missing}"
            )
        else:
            self.get_logger().info(
                "All sources ready; first observation should ship on the next tick."
            )

    def _publish_observation(self) -> None:
        with self._lock:
            js = self._latest_joint_state
            left = self._latest_left_contact
            right = self._latest_right_contact
            images = dict(self._latest_images)
            image_stamps = dict(self._latest_image_stamps)

        # Wait until every source has produced at least one message.
        if js is None or left is None or right is None:
            return
        if any(k not in images for k in self._camera_topics):
            return

        # Merge gripper contacts into joint_state under synthetic names so
        # the LeRobot-side ros_to_obs mapping works without modification.
        names = list(js["name"]) + [
            self._args.left_contact_joint_name,
            self._args.right_contact_joint_name,
        ]
        positions = list(js["position"]) + [left, right]

        encoded: dict[str, bytes] = {}
        for cam_key, img in images.items():
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._args.jpeg_quality]
            )
            if not ok:
                self.get_logger().warn(f"[{cam_key}] JPEG encode failed; dropping frame")
                return
            encoded[cam_key] = buf.tobytes()

        payload = {
            "stamp": time.time(),
            "joint_state": {"name": names, "position": positions},
            "images": encoded,
            "image_stamps": image_stamps,
        }
        try:
            self._obs_socket.send(msgpack.packb(payload, use_bin_type=True), zmq.NOBLOCK)
        except zmq.Again:
            # No client attached yet (CONFLATE drops when HWM reached).
            return

        if self._publish_count == 0:
            self.get_logger().info(
                f"First observation published ({len(names)} joints, "
                f"{len(encoded)} cameras, {sum(len(b) for b in encoded.values()) // 1024} KiB)"
            )
        self._publish_count += 1

    def _poll_and_publish_action(self) -> None:
        try:
            raw = self._cmd_socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        try:
            cmd = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self.get_logger().error(f"Action decode failed: {e}")
            return
        if not isinstance(cmd, dict) or "name" not in cmd or "position" not in cmd:
            self.get_logger().error(f"Invalid action payload: {cmd!r}")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(cmd["name"])
        msg.position = [float(p) for p in cmd["position"]]
        # velocity/effort intentionally empty — the PID controller ignores them.
        self._action_pub.publish(msg)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        try:
            self._obs_socket.close(linger=0)
            self._cmd_socket.close(linger=0)
            self._ctx.term()
        except Exception as e:  # pragma: no cover - cleanup best-effort
            self.get_logger().warn(f"ZMQ shutdown raised: {e}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--obs-port", type=int, default=5555,
                   help="ZMQ PUSH port for observations (default: 5555)")
    p.add_argument("--cmd-port", type=int, default=5556,
                   help="ZMQ PULL port for action commands (default: 5556)")
    p.add_argument("--pub-rate-hz", type=float, default=20.0,
                   help="Observation publish / action poll rate in Hz (default: 20)")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="cv2.imencode JPEG quality (default: 90)")

    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--action-topic", default="/pid_controller/reference/state")
    p.add_argument("--left-contact-topic", default="/l_arm_gripper/holding")
    p.add_argument("--right-contact-topic", default="/r_arm_gripper/holding")
    p.add_argument("--cam-head-topic", default="/oak/rgb/image_raw")
    p.add_argument("--cam-left-wrist-topic", default="/left_gripper_camera/image_raw")
    p.add_argument("--cam-right-wrist-topic", default="/right_gripper_camera/image_raw")

    # Synthetic joint names used to merge the Bool contact topics into the
    # joint_state dict sent to the client. These must match
    # BorgConfig.left_gripper_contact_joint / right_gripper_contact_joint.
    p.add_argument("--left-contact-joint-name", default="l_gripper_contact")
    p.add_argument("--right-contact-joint-name", default="r_gripper_contact")

    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)

    rclpy.init()
    node = BorgBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt; shutting down.")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
