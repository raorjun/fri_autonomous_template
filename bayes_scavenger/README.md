# Bayes Scavenger Hunt (ROS 2, Python)

This package is set up for an "optimal Bayesian search" project:
- discrete room or zone beliefs like `zone_1`, `zone_2`, `zone_3`
- Bayesian belief updates after a failed or successful search
- three strategies for comparison: `bayes`, `random`, `sequential`
- real robot navigation through Nav2 `navigate_to_pose` or a topic fallback
- RViz-friendly zone markers and belief/status topics
- a color-detector fallback plus an optional YOLO detector node
- simulator and evaluation scripts for your write-up

## Core idea
The robot keeps a probability distribution over a small set of zones. After each scan:
- a failed search pushes probability away from the current zone
- a strong positive detection pushes belief toward the current zone
- the next destination is chosen by the active strategy

## Topics
### Subscribed
- `/image_raw` (`sensor_msgs/msg/Image`): camera stream
- `/detector/observation` (`std_msgs/msg/String`): detector output as JSON
- `/amcl_pose` (`geometry_msgs/msg/PoseWithCovarianceStamped`): current pose for RViz/map tests

### Published
- `/bayes/beliefs` (`std_msgs/msg/String`): JSON belief/state report
- `/bayes/status` (`std_msgs/msg/String`): human-readable status
- `/bayes/current_goal_pose` (`geometry_msgs/msg/PoseStamped`): current target zone pose
- `/bayes/zones` (`visualization_msgs/msg/MarkerArray`): zone belief markers for RViz
- `/goal_pose` (`geometry_msgs/msg/PoseStamped`): only used when `navigation_mode:=topic`
- `/detector/debug_image` (`sensor_msgs/msg/Image`): annotated camera image

## Configure the project
Edit `config/search_config.yaml`:
1. Replace the example `zones` with your real map coordinates from RViz.
2. Set `priors` so likely rooms start higher.
3. Tune `likelihoods` to match how reliable your detector is in each zone.
4. Choose `navigation_mode: action` if you are using Nav2 on the robot.
5. Set `strategy` to `bayes`, `random`, or `sequential`.
6. Set `log_history_path` if you want CSV belief traces from a live robot run.

## Build
From the root of the ROS 2 workspace that contains this repo:
```bash
cd ~/your_ros2_ws
colcon build --packages-select bayes_scavenger
source install/setup.bash
```

## Run on the robot or in RViz
Default launch path for the project right now is YOLO:
```bash
ros2 launch bayes_scavenger bayes_scavenger.launch.py
```

Color detector fallback:
```bash
ros2 launch bayes_scavenger bayes_scavenger.launch.py detector_type:=color navigation_mode:=action strategy:=bayes
```

YOLO detector:
```bash
ros2 launch bayes_scavenger bayes_scavenger.launch.py detector_type:=yolo navigation_mode:=action strategy:=bayes
```

If another detector already publishes `/detector/observation`, skip the built-in detector:
```bash
ros2 launch bayes_scavenger bayes_scavenger.launch.py detector_type:=external navigation_mode:=action strategy:=bayes
```

`RViz` is the correct spelling, and the helpful live views here are:
- map plus AMCL pose
- `/bayes/current_goal_pose`
- `/bayes/zones`

## Detector options
The included color detector is useful for quick bench testing. The YOLO node expects `ultralytics` to be installed on the robot:
```bash
pip install ultralytics
```

Both detectors publish the same observation JSON format, so you can swap them without changing the Bayesian node.

## Simulator without ROS
Single run:
```bash
python3 -m bayes_scavenger.simulation --config config/search_config.yaml --strategy bayes --target zone_2 --seed 7 --history-csv logs/bayes_history.csv
```

Compare strategies across many trials:
```bash
python3 -m bayes_scavenger.evaluation --config config/search_config.yaml --trials 50 --summary-csv logs/summary.csv --history-csv logs/convergence.csv --history-strategy bayes
```

After installing the package, the same commands are available as:
```bash
simulate_bayes_search --config config/search_config.yaml --strategy bayes
evaluate_bayes_search --config config/search_config.yaml --trials 50
```

## Write-up metrics this supports
- reduced search time: compare `bayes` vs `random` vs `sequential`
- distance traveled: printed and recorded by the simulator/evaluator
- success rate: reported by `evaluation.py`
- probability convergence: use the belief CSV to graph `P(object in zone)` over time

## Good next customizations
- replace example zone coordinates with real map coordinates
- point the camera topic to the robot camera
- set the true YOLO target label you care about
- retune `scan_duration_sec` and confidence thresholds after a few robot runs
