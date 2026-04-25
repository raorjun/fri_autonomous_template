import json
from typing import Dict, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from bayes_scavenger.search_config import load_search_config
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ObjectDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('object_detector_node')
        self.declare_parameter('config_path', '')
        config_path = str(self.get_parameter('config_path').value)
        detector_cfg = {}

        if config_path:
            config = load_search_config(config_path)
            detector_cfg = config['detector']

        self.declare_parameter('camera_topic', detector_cfg.get('camera_topic', '/image_raw'))
        self.declare_parameter('target_mode', detector_cfg.get('target_mode', 'red'))
        self.declare_parameter('min_area_px', int(detector_cfg.get('min_area_px', 1800)))
        self.declare_parameter('blur_kernel', int(detector_cfg.get('blur_kernel', 5)))
        self.declare_parameter('red_1_lower', detector_cfg.get('red_1_lower', [0, 120, 70]))
        self.declare_parameter('red_1_upper', detector_cfg.get('red_1_upper', [10, 255, 255]))
        self.declare_parameter('red_2_lower', detector_cfg.get('red_2_lower', [170, 120, 70]))
        self.declare_parameter('red_2_upper', detector_cfg.get('red_2_upper', [180, 255, 255]))
        self.declare_parameter('blue_lower', detector_cfg.get('blue_lower', [100, 120, 50]))
        self.declare_parameter('blue_upper', detector_cfg.get('blue_upper', [140, 255, 255]))
        self.declare_parameter('green_lower', detector_cfg.get('green_lower', [40, 60, 50]))
        self.declare_parameter('green_upper', detector_cfg.get('green_upper', [85, 255, 255]))

        self.bridge = CvBridge()
        self.camera_topic = self.get_parameter('camera_topic').value
        self.target_mode = self.get_parameter('target_mode').value
        self.min_area_px = int(self.get_parameter('min_area_px').value)
        self.blur_kernel = int(self.get_parameter('blur_kernel').value)
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, qos_profile_sensor_data
        )
        self.debug_pub = self.create_publisher(Image, '/detector/debug_image', 10)
        self.obs_pub = self.create_publisher(String, '/detector/observation', 10)
        self.get_logger().info(f'Listening on {self.camera_topic} using {self.target_mode} detector')

    def _get_bounds(self, key: str) -> np.ndarray:
        return np.array(self.get_parameter(key).value, dtype=np.uint8)

    def _build_mask(self, hsv: np.ndarray) -> np.ndarray:
        if self.target_mode == 'red':
            lower1 = self._get_bounds('red_1_lower')
            upper1 = self._get_bounds('red_1_upper')
            lower2 = self._get_bounds('red_2_lower')
            upper2 = self._get_bounds('red_2_upper')
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif self.target_mode == 'blue':
            mask = cv2.inRange(hsv, self._get_bounds('blue_lower'), self._get_bounds('blue_upper'))
        elif self.target_mode == 'green':
            mask = cv2.inRange(hsv, self._get_bounds('green_lower'), self._get_bounds('green_upper'))
        else:
            raise ValueError(f'Unsupported target_mode: {self.target_mode}')

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _make_observation(self, detected: bool, confidence: float, bbox: Tuple[int, int, int, int]) -> Dict:
        return {
            'label': self.target_mode,
            'detected': detected,
            'confidence': float(confidence),
            'bbox': list(map(int, bbox)),
        }

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.blur_kernel > 1:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0.0
        best_bbox = (0, 0, 0, 0)
        detected = False
        annotated = frame.copy()
        frame_area = float(frame.shape[0] * frame.shape[1]) if frame.size else 1.0

        if contours:
            best_contour = max(contours, key=cv2.contourArea)
            best_area = float(cv2.contourArea(best_contour))
            if best_area >= self.min_area_px:
                x, y, w, h = cv2.boundingRect(best_contour)
                best_bbox = (x, y, w, h)
                detected = True
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    f'{self.target_mode}: {best_area:.0f}px',
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

        confidence = min(best_area / max(frame_area * 0.12, 1.0), 1.0) if detected else 0.0
        obs = self._make_observation(detected, confidence, best_bbox)
        obs_msg = String()
        obs_msg.data = json.dumps(obs)
        self.obs_pub.publish(obs_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
