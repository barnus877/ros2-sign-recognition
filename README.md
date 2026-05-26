# Setup #

The folder should be added to the home directory, as the ros2-sign-recognition folder will be the project folder. The git ignore automatically skips all build files, only the source is synchronised.

If everything done as sad, then the commands written later work as intended (for building and starting the project)


# ros2-sign-recognition #

This is a ROS2 project for the Cognitive Robotics Laboratory course of the Mechatronics Engineering MSc of the Budapest University of Technology and Economics created by József Ferenczi, Mór Sas, Máté Horváth, Barnabás Szabó and Dániel Sándor.

The task is to simulate a robot with autonomous driving with additional capabilities of sign or pedestrian recognition.

## How to use this project repository

1. Change directory to `$USER`'s home folder and clone the repo
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

## How to start the line following algorithm

1. Run the project according to the description above: [How to use this project repository](#how-to-use-this-project-repository)

2. Paste the following code into a new terminal
   ```bash
   cd ~/ros2-sign-recognition && source install/setup.bash && ros2 run sign_recognition_py line_follower
   ```

## Save training images

Run the `save_training_images` node that can save training images by pressing the `s` key, but before that, make sure that `self.save_path` is set to your own directory by changing `$USER` in the node:

```python
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(
            CompressedImage,
            'image_raw/compressed',  # Replace with your topic name
            self.image_callback,
            1  # Queue size of 1
        )

        self.save_path = "/home/$USER/ros2-sign-recognition/src/sign_recognition_py/saved_images/"
```

If the path is set up correctly we can build and source the project:
```bash
cd ~/ros2-sign-recognition && colcon build && source install/setup.bash
```

and we can run the node:
```bash
ros2 run sign_recognition_py save_training_images
```

# Neural network

To label the saved images we just simply have to copy the images to the suitable folder under the `training_images` folder. We distinguish 6 labels:
- Limit 5
- Limit 40
- Limit no
- Lived place
- No sign
- Stop

## Train the neural network

There is a simple python training script in the package, called `train_network.py`.
First, navigate to the right folder then run the script:

```bash
cd ~/ros2-sign-recognition/src/sign_recognition_py/sign_recognition_py/
python train_network.py
```




