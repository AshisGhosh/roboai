# RoboAI: Playground + Framework for applying LLM/VLMs to Robots in Sim

### Update Videos:

* **May 27 2024** - [VIDEO](https://www.youtube.com/watch?v=ycvPWq4JfEI) - Robot learning task relevant information and factoring that in the plan -- integrated with [OmniGibson](https://behavior.stanford.edu/omnigibson/) from Stanford/NVIDIA
* **May 8 2024** - [VIDEO](https://www.youtube.com/watch?v=sg3PTz5q6kc) - Robot going from plain text to grasping attempt -- integrated with ROS2, MoveIt2, a grasping model and Isaac Sim. 

## Simulation Frameworks

### MuJoCo & Robosuite

[Mujoco](https://mujoco.org/) is Google DeepMind's physics simulation. 

[Robosuite](https://robosuite.ai/) is a modular framework built on top of MuJoCo.

In the `/robosim` folder you'll find a Robosuite/MuJoCo sim environment:
* Focused on Panda arm grasping objects in pick and place environment
* Camera views to focus on objects
* Markers to indicate robot goal and grasp targets
* Simple API to control the robot


### Isaac Sim

[Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) is NVIDIA's robot simulation powered by GPUs. 

Isaac Sim offers advanced tooling as well as close to real rendering. This was adopted to better test vision models. 

Isaac Sim does not support external async frameworks as well - the development towards it in this project is still in progress and may need some re-architecting.

The simulation
* Focuses on the Panda arm on a table with objects to grasp
* Cameras for different views
* Initial work on Markers - rendering/material support is still WIP


## Models & LLM Framework

The high-level goal is to be able to command a robot to complete a long-horizon task with natural language. 

An example would be to "clear the messy table". 

### LLMs

LLMs are used in planning layer. Once the scene is understood an LLM (either iteratively or with CoT/ToT) to generate a robot affordable plan. 

Currently focused on free models hosted on [openrouter.ai](https://openrouter.ai).

### VLMs

VLMs are an extremely fast changing space. Current work is focused on:
* [moondream2](https://huggingface.co/vikhyatk/moondream2)
* [VILA-2.7b](https://huggingface.co/Efficient-Large-Model/VILA-2.7b) -- inference running on a Jetson Orin Nano (not in this repo) using [NanoLLM](https://dusty-nv.github.io/NanoLLM/index.html)