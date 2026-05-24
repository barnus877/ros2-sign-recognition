# ros2-sign-recognition #

This is a ROS2 project for the Cognitive Robotics Laboratory course of the Mechatronics Engineering MSc of the Budapest University of Technology and Economics created by József Ferenczi, Mór Sas, Máté Horváth and Barnabás Szabó.

The task is to simulate a robot with autonomous driving with additional capabilities of sign or pedestrian recognition.

cd;
mkdir -p projekt;
cd projekt;
git clone https://github.com/barnus877/ros2-sign-recognition;

cd ~/projekt && colcon build && source install/setup.bash && ros2 launch sign_recognition_bringup world_teleopt.launch.py 

ros2 run sign_recognition_py helloka_py
