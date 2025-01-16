# Autonomous Racetrack Simulation with Reinforcement Learning

This repository contains the implementation of a custom racetrack simulation environment and reinforcement learning training scripts. The project is focused on benchmarking multiple RL algorithms, customizing the environment, and evaluating agent performance in diverse scenarios.

## Overview

The main objectives of this project are:
- Implement a custom racetrack environment.
- Benchmark reinforcement learning algorithms (SAC, PPO, A2C, TD3).
- Evaluate the adaptability of agents across diverse racetrack scenarios.
- Leverage GPU acceleration and parallel environments for efficient training.

Our custom environment is heavily based on [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv), an open-source project for training autonomous driving agents. While HighwayEnv provides a solid foundation, we introduced significant modifications, including custom rewards, dynamic scenario generation, and enhanced metrics to better suit racetrack-style simulations.

## How to run

If your machine has:
- GPU
- cuda/cudatoolkit>=11.8

simply import the python environment ms_env.yaml into conda.

Otherwise, some libraries like torch+cuda will be incompatible.

## Training

The training is performed using `train_model.py`, which leverages [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithms. Key features of the script include:
- Support for GPU and CPU training.
- Configurable parallel environments (up to 20 environments).
- Logging of training progress using TensorBoard.

## Visualization and Debugging

The `view_model.py` script is a key tool for evaluating trained agents. It renders episodes and provides insights into:
- Agent behavior and actions.
- Rewards received during episodes.
- Debugging environment settings to refine rewards and penalties.

This tool has proven essential for identifying issues like unsafe lane changes or off-track behavior and tweaking reward configurations accordingly.

## Benchmarking

### Diverse Scenarios
The custom environment supports a variety of scenarios activated during training via the `different_scenarios` configuration. These include:
- Varying racetrack sizes.
- Randomized agent speeds and starting positions.
- Different numbers of adversary vehicles.

Benchmarking across diverse scenarios ensures a robust evaluation of each algorithm's adaptability and performance.

### Selected Algorithms
SAC and PPO were chosen for benchmarking based on their superior performance during initial evaluations. Both algorithms were trained for 5 million timesteps:
- SAC: Trained on GPU with 15 parallel environments (11 hours).
- PPO: Trained on CPU with 20 parallel environments (8 hours).

## Learned SAC Policy

  <p align="center">
    <img src="models/sac.gif" alt="Learned Policy" height="400">
    <br> Learned Policy
  </p>

TODO:
- Higher proximity penalty
- Higher distance to front car threshold

## References

1. HighwayEnv: [Farama Foundation GitHub Repository](https://github.com/Farama-Foundation/HighwayEnv)
2. Stable-Baselines3: [Documentation](https://stable-baselines3.readthedocs.io/)
3. TensorBoard: [TensorFlow Visualization Tool](https://www.tensorflow.org/tensorboard)
