# MuJoCo Dual-Arm Cooperative Manipulation

This repository provides a physics-based simulation of a dual-arm robotic system performing cooperative manipulation tasks. It is implemented using the **MuJoCo** physics engine and Python.

This project serves as an unofficial partial reproduction of the cooperative manipulation control architecture presented in the academic paper: 
*"Mobile Preassembly Systems with Cooperative Dual-Arm Manipulation - A Concept for Industrial Applications in the Near Future"*.

![Simulation Demo](data/grasping_process.gif) 

## 📄 Reference Paper
* **Title**: Mobile Preassembly Systems with Cooperative Dual-Arm Manipulation - A Concept for Industrial Applications in the Near Future 
* **Authors**: Jonas Wittmann , Johannes Rainer , Daniel Rixen , Mathias Laile , Johannes Fottner
* **Conference**: ISR Europe 2023 

## ✨ Key Features

This implementation bypasses standard kinematic position control, utilizing **Joint Torque Commands** to achieve realistic physics and dynamic interaction with rigid objects.

* **Quintic Spline Trajectory Planning**: Generates smooth object-level trajectories and derives the corresponding end-effector reference positions, velocities, and accelerations based on grasp kinematics.
* **Master-Slave Hybrid Control**: 
  * **Master Arm**: Implements a stiff Cartesian impedance controller to accurately track the desired object trajectory.
  * **Slave Arm**: Employs a hybrid-parallel position/force controller to support motion tracking while actively limiting internal object squeezing forces.
* **Null-Space Damping**: Integrates a null-space damping term to restrict the redundant degrees of freedom (elbow motions) of the manipulators, preventing self-collisions during the transport phase.
* **Smooth Grasping State Machine**: Features a meticulously tuned multi-phase task sequence (Hover -> Descend -> Soft Grasp -> Lift -> Transport -> Release) to ensure stable contacts and prevent physics engine instability during rigid body grasping.
* **Automated Data Pipeline**: Automatically records high-quality 30FPS MP4 simulation videos using `mujoco.Renderer` and OpenCV, alongside exporting end-effector XYZ trajectory data (`.txt`/`.csv`) for all task phases.

## ⚙️ Dependencies

To run this simulation, you need to set up a Python environment with the following dependencies.

Prerequisites:

Python 3.8 or higher

Ubuntu 22.04 (Recommended) or other standard OS

Install required packages via pip:

    pip install mujoco numpy opencv-python

## 🚀 Usage

1. Clone the repository:

        git clone git@github.com:WangSuhan/Dual-arm-cooperative-manipulation-simulation-in-Mujoco.git

2. Run the simulation:
Navigate to the scripts directory and execute the main python file:

        cd scripts
        python run_demo.py
  
## 📁 Repository Structure

```text
├── models/
│   └── dual_panda_torque.xml    # MuJoCo XML model with symmetric dual-arm setup
│   └── franka_emika_panda    
├── scripts/
│   ├── run_demo.py                  # Main simulation loop, state machine, and data recording
├── data/                        # Auto-generated output directory
│   ├── grasping_process.mp4     # Simulation video recording
│   └── traj_l_xyz_*.txt         # Exported XYZ trajectory data for each phase
├── src/                        # class
│   ├── dynamics_model.py     
│   ├── force_estimator.py     
│   ├── impedance_controller.py     
│   ├── robot_interface.py     
│   └── trajectory_planner
├── tests                         
│   ├── visualize_scene.py      # to test the scene setting     
└── README.md


