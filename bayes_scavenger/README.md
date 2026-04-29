# bayes_scavenger

A ROS 2 package that searches for a lost object (e.g. a backpack) across known locations using Bayesian inference — the same way a person mentally retraces their steps.

Eou lost your keys. You start with a gut feeling about where they might be (prior probabilities), check each spot, and update your confidence based on what you find or don't find. That's exactly what this does, formalized with Bayes' theorem.

## How it works

The robot visits three locations in Anna Hiss Gymnasium, scanning each one:

| Zone | Location |
|------|----------|
| `elevator_lab` | Elevator by Dr. Hart's lab |
| `tv_desk_room` | TV room |
| `last_elevator` | Elevator by the medical robotics lab |

Each zone starts with a roughly equal prior probability (~0.33). After scanning a zone:
- **Object found** → beliefs collapse, search ends
- **Object not found** → that zone's probability drops, others rise, and the robot moves to the highest-probability remaining zone

## Pub / Sub

| Topic | Direction | Type | Purpose |
|-------|-----------|------|---------|
| `/goal_pose` | Publish | `PoseStamped` | Sends navigation goals to the robot |
| `/bayes/beliefs` | Publish | `String` (JSON) | Current probability of each zone |
| `/bayes/status` | Publish | `String` | Human-readable status updates |
| `/bayes/zones` | Publish | `MarkerArray` | RViz visualization of beliefs |
| `/detector/observation` | Subscribe | `String` (JSON) | Incoming detections from the camera |
| `/amcl_pose` | Subscribe | `PoseWithCovarianceStamped` | Robot's current position |

## Running

```bash
ros2 launch bayes_scavenger bayes_scavenger.launch.py
```

Config is in `config/search_config.yaml`. Detection works with either a color detector or YOLO (`detector_type:=yolo`).
