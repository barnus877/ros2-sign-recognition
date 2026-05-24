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

## How to use Gazebo simulation

### Add objects to the world

1. Run the project.
2. In Gazebo GUI click the 3 dots (`6`) on the top right corner.
   ![alt text](https://raw.githubusercontent.com/MOGI-ROS/Week-1-8-Cognitive-robotics/main/assets/gazebo-1.png)
3. Add models through the `Resource Spawner` within the `plug-in browser`.
4. Select folder: `~/ros2-sign-recognition/install/sign_recognition_bringup/share/sign_recognition_bringup/gazebo_models/`
   ![alt text](https://raw.githubusercontent.com/MOGI-ROS/Week-1-8-Cognitive-robotics/main/assets/gazebo-2.png)

### How to save world.sdf file

1. Replace all `file:///home/$USER/ros2-sign-recognition/install/sign_recognition_bringup/share/sign_recognition_bringup/gazebo_models/`
   ```bash
   <include>
     <uri>file:///home/$USER/ros2-sign-recognition/install/sign_recognition_bringup/share/sign_recognition_bringup/gazebo_models/palya_mogi</uri>
     <name>palya</name>
     <pose>-1.2731360914953953 0.024185007175546058 0 0 0 0</pose>
   </include>
   ```
   with relative path `model://`
   ```bash
   <include>
     <uri>model://palya_mogi</uri>
     <name>palya</name>
     <pose>-1.2731360914953953 0.024185007175546058 0 0 0 0</pose>
   </include>
   ```

2. Delete `turtlebot3_burger`
   ```bash
   <include>
          <uri>file:///home/$USER/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models/turtlebot3_burger</uri>
       <name>burger</name>
       <pose>4.1577145665028317 2.4008829921053554 0.010008771952750758 6.9807543737599483e-07 -0.012459121963041841 1.   3190277154170922</pose>
   </include>
   ```

### Update launch file

In `src/sign_recognition_bringup/launch/world_teleopt.launch.py` update `world_sign.sdf` with the newly created `.sdf` file.
```bash
world_arg = DeclareLaunchArgument(
    'world', default_value='world_sign.sdf',
    description='Name of the Gazebo world file to load'
)
```
