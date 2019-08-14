# Reacher - reinforcement learning DDPG implementation
============

Reacher is a reinforcement learning algorithm based on [DDPG paper](https://arxiv.org/pdf/1509.02971) with some modifications to improve performance of the model. In this environment, a double-jointed arm can move to target locations. 

---

## Features
- Buffer size tuning
- Batch size tuning
- Gamma factor tuning
- TAU tuning for soft updates

---


## Screenshot

![Reacher - solving environment](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

## Loss

![Reacher - loss](https://github.com/andreiliphd/reacher-ddpg-ppo/blob/master/output_40_2.png)



---

## Setup
1. Clone this repo: 
```
git clone https://github.com/andreiliphd/reacher-ddpg.git
```

2. Create and activate a new environment with Python 3.6.
```
conda create --name reacher python=3.6
conda activate reacher
```

3. Create an IPython kernel for the `reacher` environment.
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Before running code in a notebook, change the kernel to match the `reacher` environment by using the drop-down Kernel menu.
![Change Kernel](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)

---


## Installation

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the directory of GitHub repository files.


## Usage

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Follow the instructions in `Report.ipynb` to get started with training your agents!  

The environment considered solved if each agent mean is above 30.0


---

## License
You can check out the full license in the LICENSE file.

This project is licensed under the terms of the **MIT** license.
