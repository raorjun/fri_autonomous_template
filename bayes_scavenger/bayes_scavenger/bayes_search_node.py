import csv
import json
import math
import random
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from bayes_scavenger.bayes_engine import BayesianSearchEngine
from bayes_scavenger.search_config import load_search_config
from bayes_scavenger.search_policy import choose_next_location


def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw_rad / 2.0)
    q.w = math.cos(yaw_rad / 2.0)
    return q


class BayesSearchNode(Node):
    def __init__(self) -> None:
        super().__init__("bayes_search_node")

        self.declare_parameter("config_path", "")

        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path parameter is required")

        config = load_search_config(config_path)
        search_cfg = config["search"]
        detector_cfg = config["detector"]
        yolo_cfg = config["yolo"]
        likelihoods = config["likelihoods"]

        self.declare_parameter("goal_topic", search_cfg["goal_topic"])
        self.declare_parameter("navigation_mode", search_cfg["navigation_mode"])
        self.declare_parameter("strategy", search_cfg["strategy"])
        self.declare_parameter("arrival_radius_m", float(search_cfg["arrival_radius_m"]))
        self.declare_parameter("scan_duration_sec", float(search_cfg["scan_duration_sec"]))
        self.declare_parameter("revisit_penalty", float(search_cfg["revisit_penalty"]))
        self.declare_parameter(
            "observation_confidence_threshold",
            float(search_cfg["observation_confidence_threshold"]),
        )
        self.declare_parameter("auto_advance_sec", float(search_cfg["auto_advance_sec"]))

        self.target_label = str(search_cfg["target_label"])
        self.detector_labels = {
            str(search_cfg["target_label"]),
            str(detector_cfg["target_mode"]),
            str(yolo_cfg["target_label"]),
        }
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.navigation_mode = str(self.get_parameter("navigation_mode").value).lower()
        self.search_strategy = str(self.get_parameter("strategy").value).lower()
        self.arrival_radius_m = float(self.get_parameter("arrival_radius_m").value)
        self.scan_duration_sec = float(self.get_parameter("scan_duration_sec").value)
        self.revisit_penalty = float(self.get_parameter("revisit_penalty").value)
        self.conf_threshold = float(self.get_parameter("observation_confidence_threshold").value)
        self.auto_advance_sec = float(self.get_parameter("auto_advance_sec").value)
        self.marker_topic = str(search_cfg["marker_topic"])
        self.sequence_order = tuple(search_cfg["sequence_order"])
        self.rng = random.Random(int(search_cfg["random_seed"]))

        if self.navigation_mode not in {"action", "topic"}:
            raise ValueError("navigation_mode must be either 'action' or 'topic'")
        if self.search_strategy not in {"bayes", "random", "sequential"}:
            raise ValueError("strategy must be bayes, random, or sequential")

        self.waypoints = config["zones"]
        self.engine = BayesianSearchEngine(
            config["priors"],
            likelihoods["positive_detection"],
            likelihoods["negative_detection"],
            false_positive_rate=likelihoods["false_positive_rate"],
            revisit_penalty=self.revisit_penalty,
        )

        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)
        self.debug_goal_pub = self.create_publisher(PoseStamped, "/bayes/current_goal_pose", 10)
        self.belief_pub = self.create_publisher(String, "/bayes/beliefs", 10)
        self.status_pub = self.create_publisher(String, "/bayes/status", 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.obs_sub = self.create_subscription(String, "/detector/observation", self.observation_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.pose_callback, 10
        )

        self.nav_client: Optional[ActionClient] = None
        if self.navigation_mode == "action":
            self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.current_pose: Optional[Tuple[float, float]] = None
        self.current_goal_name: Optional[str] = None
        self.current_goal_pose: Optional[PoseStamped] = None
        self.goal_sent_time: Optional[float] = None
        self.goal_arrival_time: Optional[float] = None
        self.goal_request_future = None
        self.goal_result_future = None
        self.active_goal_handle = None
        self.last_action_retry_sec = 0.0
        self.state = "idle"
        self.found = False
        self.started = False

        self.history_file = None
        self.history_writer = None
        self._open_history_log(str(search_cfg["log_history_path"]))

        self.timer = self.create_timer(0.5, self.control_loop)
        self.publish_status(
            f"Bayes search node ready in {self.navigation_mode} navigation mode using {self.search_strategy} search."
        )
        self.publish_beliefs("initial")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _open_history_log(self, path: str) -> None:
        if not path:
            return
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_file = output_path.open("w", newline="", encoding="utf-8")
        self.history_writer = csv.writer(self.history_file)
        self.history_writer.writerow(
            [
                "timestamp_sec",
                "event",
                "state",
                "strategy",
                "navigation_mode",
                "current_goal",
                "zone",
                "belief",
                "visited_count",
                "found",
            ]
        )

    def destroy_node(self):
        if self.history_file is not None and not self.history_file.closed:
            self.history_file.close()
        return super().destroy_node()

    def pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.current_pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )

        if self.navigation_mode == "topic" and self.state == "traveling":
            distance = self.distance_to_current_goal()
            if distance is not None and distance <= self.arrival_radius_m:
                self.mark_arrived()

    def publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _write_history_rows(self, event: str) -> None:
        if self.history_writer is None or self.history_file is None:
            return
        timestamp_sec = self._now_sec()
        for zone in self.waypoints:
            self.history_writer.writerow(
                [
                    f"{timestamp_sec:.6f}",
                    event,
                    self.state,
                    self.search_strategy,
                    self.navigation_mode,
                    self.current_goal_name or "",
                    zone,
                    f"{self.engine.beliefs[zone]:.6f}",
                    self.engine.visited_counts[zone],
                    self.found,
                ]
            )
        self.history_file.flush()

    def publish_zone_markers(self) -> None:
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for index, zone_name in enumerate(self.waypoints):
            zone = self.waypoints[zone_name]
            belief = float(self.engine.beliefs[zone_name])
            scale = 0.25 + 0.75 * belief

            sphere = Marker()
            sphere.header.frame_id = zone.get("frame_id", "map")
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "bayes_zone"
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(zone["x"])
            sphere.pose.position.y = float(zone["y"])
            sphere.pose.position.z = 0.15
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = scale
            sphere.scale.y = scale
            sphere.scale.z = 0.15
            sphere.color.a = 0.8
            sphere.color.r = 1.0
            sphere.color.g = max(0.1, 1.0 - belief)
            sphere.color.b = 0.2
            marker_array.markers.append(sphere)

            label = Marker()
            label.header.frame_id = zone.get("frame_id", "map")
            label.header.stamp = self.get_clock().now().to_msg()
            label.ns = "bayes_zone_label"
            label.id = 1000 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(zone["x"])
            label.pose.position.y = float(zone["y"])
            label.pose.position.z = 0.55
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.a = 1.0
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            prefix = "* " if zone_name == self.current_goal_name else ""
            label.text = f"{prefix}{zone_name}\nP={belief:.2f}"
            marker_array.markers.append(label)

        self.marker_pub.publish(marker_array)

    def publish_beliefs(self, event: str) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "event": event,
                "state": self.state,
                "strategy": self.search_strategy,
                "navigation_mode": self.navigation_mode,
                "beliefs": self.engine.beliefs,
                "current_goal": self.current_goal_name,
                "visited_counts": self.engine.visited_counts,
                "target_label": self.target_label,
                "found": self.found,
            }
        )
        self.belief_pub.publish(msg)
        self.publish_zone_markers()
        self._write_history_rows(event)

    def make_goal(self, zone_name: str) -> PoseStamped:
        zone = self.waypoints[zone_name]
        yaw_rad = math.radians(float(zone.get("yaw_deg", 0.0)))
        msg = PoseStamped()
        msg.header.frame_id = zone.get("frame_id", "map")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(zone["x"])
        msg.pose.position.y = float(zone["y"])
        msg.pose.position.z = float(zone.get("z", 0.0))
        msg.pose.orientation = yaw_to_quaternion(yaw_rad)
        return msg

    def choose_next_zone(self) -> str:
        return choose_next_location(
            self.search_strategy,
            self.engine,
            rng=self.rng,
            sequence_order=self.sequence_order,
            current_pose=self.current_pose,
            waypoints=self.waypoints,
        )

    def distance_to_current_goal(self) -> Optional[float]:
        if self.current_pose is None or self.current_goal_pose is None:
            return None
        dx = float(self.current_goal_pose.pose.position.x) - self.current_pose[0]
        dy = float(self.current_goal_pose.pose.position.y) - self.current_pose[1]
        return math.hypot(dx, dy)

    def _dispatch_action_goal(self) -> bool:
        if self.nav_client is None or self.current_goal_pose is None or self.current_goal_name is None:
            return False
        if not self.nav_client.wait_for_server(timeout_sec=0.1):
            self.publish_status("Waiting for Nav2 action server on navigate_to_pose.")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.current_goal_pose
        self.goal_request_future = self.nav_client.send_goal_async(goal_msg)
        self.goal_request_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future) -> None:
        self.goal_request_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self.publish_status(f"Nav2 goal request failed: {exc}")
            self.state = "idle"
            self.current_goal_name = None
            self.current_goal_pose = None
            return

        if goal_handle is None or not goal_handle.accepted:
            self.publish_status(f"Goal to {self.current_goal_name} was rejected.")
            self.state = "idle"
            self.current_goal_name = None
            self.current_goal_pose = None
            return

        self.active_goal_handle = goal_handle
        self.publish_status(f"Nav2 accepted goal for {self.current_goal_name}.")
        self.goal_result_future = goal_handle.get_result_async()
        self.goal_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future) -> None:
        self.goal_result_future = None
        self.active_goal_handle = None

        try:
            wrapped_result = future.result()
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self.publish_status(f"Nav2 result retrieval failed: {exc}")
            self.state = "idle"
            self.current_goal_name = None
            self.current_goal_pose = None
            return

        status = wrapped_result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.mark_arrived()
            return

        self.publish_status(f"Navigation to {self.current_goal_name} ended with status {status}.")
        self.state = "idle"
        self.current_goal_name = None
        self.current_goal_pose = None
        self.publish_beliefs("navigation_failed")

    def send_goal(self, zone_name: str) -> None:
        self.current_goal_name = zone_name
        self.current_goal_pose = self.make_goal(zone_name)
        self.goal_sent_time = self._now_sec()
        self.goal_arrival_time = None
        self.goal_request_future = None
        self.goal_result_future = None
        self.active_goal_handle = None
        self.started = True
        self.state = "traveling"

        self.debug_goal_pub.publish(self.current_goal_pose)
        if self.navigation_mode == "topic":
            self.goal_pub.publish(self.current_goal_pose)
        else:
            self.last_action_retry_sec = self.goal_sent_time
            self._dispatch_action_goal()

        self.publish_status(f"Sending robot to zone {zone_name}.")
        self.publish_beliefs("goal_sent")

    def mark_arrived(self) -> None:
        if self.current_goal_name is None or self.state == "scanning":
            return
        self.goal_arrival_time = self._now_sec()
        self.state = "scanning"
        self.publish_status(f"Arrived at {self.current_goal_name}, scanning now.")
        self.publish_beliefs("arrived")

    def update_beliefs(self, observation_positive: bool) -> None:
        if self.current_goal_name is None:
            return

        self.engine.update(self.current_goal_name, observation_positive)
        event_name = "positive_update" if observation_positive else "negative_update"
        self.publish_beliefs(event_name)

    def complete_negative_scan(self) -> None:
        if self.current_goal_name is None:
            return
        scanned_zone = self.current_goal_name
        self.update_beliefs(False)
        self.publish_status(f"No target confirmed in {scanned_zone}. Selecting next zone.")
        self.current_goal_name = None
        self.current_goal_pose = None
        self.state = "idle"
        next_zone = self.choose_next_zone()
        self.send_goal(next_zone)

    def observation_callback(self, msg: String) -> None:
        if self.found or self.state != "scanning" or self.current_goal_name is None:
            return

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received invalid detector JSON")
            return

        detected = bool(data.get("detected", False))
        confidence = float(data.get("confidence", 0.0))
        label = str(data.get("label", ""))
        label_matches = label in self.detector_labels

        if detected and confidence >= self.conf_threshold and label_matches:
            self.update_beliefs(True)
            self.found = True
            self.state = "found"
            self.publish_status(
                f"Target found in {self.current_goal_name} with confidence {confidence:.2f}."
            )
            self.publish_beliefs("found")

    def control_loop(self) -> None:
        if self.found or self.state == "found":
            return

        now_sec = self._now_sec()

        if not self.started and self.state == "idle":
            self.send_goal(self.choose_next_zone())
            return

        if self.state == "traveling":
            if self.navigation_mode == "topic":
                distance = self.distance_to_current_goal()
                if distance is not None and distance <= self.arrival_radius_m:
                    self.mark_arrived()
                    return
                if self.goal_sent_time is not None and (now_sec - self.goal_sent_time) >= self.auto_advance_sec:
                    self.mark_arrived()
                    return

            if self.navigation_mode == "action":
                no_active_goal = (
                    self.goal_request_future is None
                    and self.goal_result_future is None
                    and self.active_goal_handle is None
                )
                if no_active_goal and self.current_goal_name is not None and self.current_goal_pose is not None:
                    if (now_sec - self.last_action_retry_sec) >= 1.0:
                        self.last_action_retry_sec = now_sec
                        self._dispatch_action_goal()
            return

        if self.state == "scanning" and self.goal_arrival_time is not None:
            if (now_sec - self.goal_arrival_time) >= self.scan_duration_sec:
                self.complete_negative_scan()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BayesSearchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
