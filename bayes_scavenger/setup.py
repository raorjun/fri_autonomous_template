from setuptools import find_packages, setup
from glob import glob

package_name = 'bayes_scavenger'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Swathi Karthikeyan',
    maintainer_email='arjun@utexas.edu',
    description='Bayesian scavenger hunt search package for a ROS 2 robot.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'object_detector_node = bayes_scavenger.object_detector_node:main',
            'bayes_search_node = bayes_scavenger.bayes_search_node:main',
            'yolo_detector_node = bayes_scavenger.yolo_detector_node:main',
        ],
    },
)
