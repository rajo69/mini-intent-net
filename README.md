# Mini-Intent-Net

A small-scale simulation of an Intent-Driven Network for optical systems, inspired by the PhD project "Autonomous Intent-Driven Optical Networks."

This project demonstrates the core concept of translating a high-level human "intent" into an optimized network configuration using Deep Reinforcement Learning (DRL).

## Components

- **`src/translator.py`**: A basic module to parse a natural language command into a machine-readable goal.
- **`src/environment.py`**: A simulated optical network environment built with `networkx`. The DRL agent will operate within this environment.
- **`src/agent.py`**: The DRL agent responsible for learning optimal paths based on the translated intent.
- **`main.py`**: The main script to run a simulation.
