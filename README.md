# Setup #

The folder should be added to the home directory, as the ros2-sign-recognition folder will be the project folder. The git ignore automatically skips all build files, only the source is synchronised.

If everything done as sad, then the commands written later work as intended (for building and starting the project)


# ros2-sign-recognition #

This is a ROS2 project for the Cognitive Robotics Laboratory course of the Mechatronics Engineering MSc of the Budapest University of Technology and Economics created by József Ferenczi, Mór Sas, Máté Horváth and Barnabás Szabó.

The task is to simulate a robot with autonomous driving with additional capabilities of sign or pedestrian recognition.

## How to use this project repository

1. Change directory to $USER's home folder and clone the repo
```bash
cd;
git clone https://github.com/barnus877/ros2-sign-recognition;
```
2. To run the project paste the following code into the terminal
```bash
cd ~/ros2-sign-recognition && colcon build && source install/setup.bash && ros2 launch sign_recognition_bringup world_teleopt.launch.py 
```
