from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('bayes_scavenger')
    default_config = PathJoinSubstitution([package_share, 'config', 'search_config.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('config_path', default_value=default_config),
        DeclareLaunchArgument('detector_type', default_value='yolo'),
        DeclareLaunchArgument('navigation_mode', default_value='action'),
        DeclareLaunchArgument('strategy', default_value='bayes'),
        Node(
            package='bayes_scavenger',
            executable='object_detector_node',
            name='object_detector_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('detector_type'), "' == 'color'"])
            ),
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
            }],
        ),
        Node(
            package='bayes_scavenger',
            executable='yolo_detector_node',
            name='yolo_detector_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('detector_type'), "' == 'yolo'"])
            ),
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
            }],
        ),
        Node(
            package='bayes_scavenger',
            executable='bayes_search_node',
            name='bayes_search_node',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
                'navigation_mode': LaunchConfiguration('navigation_mode'),
                'strategy': LaunchConfiguration('strategy'),
            }],
        ),
    ])
