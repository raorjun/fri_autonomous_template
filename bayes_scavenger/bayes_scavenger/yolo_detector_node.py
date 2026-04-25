from __future__ import annotations

import json
from typing import Dict, Tuple

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from bayes_scavenger.search_config import load_search_config

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency for robot deployment
    YOLO = None


class YoloDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detector_node")
        self.declare_parameter("config_path", "")
        config_path = str(self.get_parameter("config_path").value)
        yolo_cfg = {}

        if config_path:
            config = load_search_config(config_path)
            yolo_cfg = config["yolo"]

        self.declare_parameter("camera_topic", yolo_cfg.get("camera_topic", "/image_raw"))
        self.declare_parameter("model_path", yolo_cfg.get("model_path", "yolov8n.pt"))
        self.declare_parameter("target_label", yolo_cfg.get("target_label", "person"))
        self.declare_parameter("confidence_threshold", float(yolo_cfg.get("confidence_threshold", 0.45)))
        self.declare_parameter("image_size", int(yolo_cfg.get("image_size", 640)))
        self.declare_parameter("device", yolo_cfg.get("device", "cpu"))

        self.bridge = CvBridge()
        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.target_label = str(self.get_parameter("target_label").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.device = str(self.get_parameter("device").value)

        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Install it on the robot with `pip install ultralytics`."
            )

        self.model = YOLO(self.model_path)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.debug_pub = self.create_publisher(Image, "/detector/debug_image", 10)
        self.obs_pub = self.create_publisher(String, "/detector/observation", 10)
        self.get_logger().info(
            f"Listening on {self.camera_topic} with YOLO model {self.model_path} for {self.target_label}"
        )

    @staticmethod
    def _make_observation(detected: bool, confidence: float, bbox: Tuple[int, int, int, int], label: str) -> Dict:
        return {
            "label": label,
            "detected": detected,
            "confidence": float(confidence),
            "bbox": list(map(int, bbox)),
        }

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        annotated = frame.copy()

        detected = False
        best_confidence = 0.0
        best_bbox = (0, 0, 0, 0)

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if results:
            result = results[0]
            names = result.names
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    class_name = str(names[class_id])
                    confidence = float(box.conf[0].item())
                    if class_name != self.target_label or confidence < best_confidence:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    best_confidence = confidence
                    best_bbox = (x1, y1, x2 - x1, y2 - y1)
                    detected = True

        if detected:
            x, y, w, h = best_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{self.target_label}: {best_confidence:.2f}",
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        obs = self._make_observation(detected, best_confidence, best_bbox, self.target_label)
        obs_msg = String()
        obs_msg.data = json.dumps(obs)
        self.obs_pub.publish(obs_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
