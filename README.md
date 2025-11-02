# Hi there 👋 I am Andreas Jaramillo

## 🚀 AI/ML Systems Engineer & Aspiring Aerospace Software Developer

Highly skilled **Computer Scientist** specializing in low-level systems and **Machine Learning infrastructure**. Currently pursuing a Bachelor of Science degree in Computer Science at **Johns Hopkins University**.

Proficient in **C** and **Python**, with experience developing **backend ML systems** (as an Ex-Googler) and leading **Computer Vision research**. Committed to leveraging deep learning expertise and systems proficiency to contribute to **Aerospace ML applications** and robust space-based systems.

### Skills:

* Languages: C, C++, Python

* Deep Learning Frameworks: PyTorch, TensorFlow

---

# 🤖 Machine Learning & Systems Projects

The following projects showcase my ability to integrate low-level systems thinking with advanced machine learning and search infrastructure, focusing on efficiency, robustness, and information retrieval.

## [1. Rank-Optimizing Bidirectional Search Engine ](https://github.com/andyjaramillo/Bidirectional-With-Ranking)

A novel enhancement to traditional informed search mechanisms, shifting the primary optimization goal from finding the minimum-cost path to optimizing the **rank** of the discovered solution path. This framework significantly improves search efficiency, particularly in large state spaces.

### 💡 Overview

Traditional search algorithms (like A\*) focus exclusively on reaching a specific goal state with the shortest path. This project introduces a paradigm where the search is optimized to rank the discovered solution, allowing for the discovery of more *relevant* or *efficient* paths that may not be the absolute minimum cost but provide higher utility based on a defined ranking metric.

### 🚀 Key Features

* **Rank Optimization:** Custom heuristic design to optimize solution path rank instead of just path cost.

* **Bidirectional Search Integration:** Implements a **bidirectional search** strategy, simultaneously searching forward from the initial state and backward from the goal state. This technique drastically reduces the explored state space, improving overall efficiency and exploration depth.

* **Visualization Support:** Includes integration with `pygame` for visual representation of the search process and path discovery.

## [2. Robust PyTorch Staged Execution Framework](https://github.com/andyjaramillo/pytorch_stage_layer_recovery)

A modular PyTorch wrapper designed to enhance the reliability and crash-resilience of deep learning model training pipelines, particularly useful for long-running or volatile training environments.

### 🛡️ Crash Resilience and Modular Execution

This implementation introduces `Stage` and `StagedModule` classes, effectively creating checkpointable execution stages within a standard PyTorch `nn.Module`.

* The **`Stage`** class wraps individual layers, providing a standardized point for exception handling and logging (as demonstrated by the `try...except` block).

* The **`StagedModule`** implements a `run` loop with a customizable `restartMethod`, allowing the training process to gracefully handle simulated or real crashes by resuming or re-initializing the current epoch, minimizing data loss and maximizing training time utilization. This pattern is crucial for reliable ML infrastructure.

### 💻 Core Implementation (PyTorch)

The code demonstrates how to wrap standard PyTorch layers (`nn.Flatten`, `nn.Linear`, etc.) within the custom `Stage` module, and how the overall execution flow is managed by the `StagedModule` for error recovery.

<details>
<summary>View Code Implementation</summary>

```python
import torch
import torch.nn as nn
import random
from typing import Callable

class Stage(nn.Module):
    """Wraps a PyTorch module to provide a consistent stage boundary for error handling."""
    def __init__(self, wrapped_module: nn.Module):
        super().__init__()
        self.wrapped_module = wrapped_module
        # Ensure the wrapped module is properly registered in the computational graph
        self.add_module('wrapped_module', wrapped_module) 
        
    def forward(self, *args, **kwargs):
        try:
            return self.wrapped_module(*args, **kwargs)
        except Exception as e:
            # Custom logging/reporting logic before re-raising or handling a stage crash
            print(f"Exception caught in stage: {self.wrapped_module}")
            # Re-running the module for demonstration, actual implementation would involve checkpointing/logging
            return self.wrapped_module(*args, **kwargs)


class StagedModule:
    """Manages the overall training lifecycle, implementing restart logic."""
    stages = [] # Placeholder, could be used for tracking
    def __init__(self, module: nn.Module, restartMethod: Callable, epochs: int):
        # Initializing the inner module and tracking state
        module.__init__() 
        self.module = module
        self.restartMethod = restartMethod # Callable to reset/load state for restart
        self.epochs = epochs
        self.t = 0 # Current epoch counter
        
    def run(self):
        """Main training loop with crash recovery logic."""
        try:
            while self.t < self.epochs:
                print(f"Epoch {self.t+1}\n-------------------------------")
                # Call the external method (e.g., a function to load data and run one epoch)
                self.restartMethod() 
                self.t += 1
                
        except Exception as e:
            # Handle the simulated crash and attempt to resume/restart
            print(f"Catastrophic failure at epoch {self.t}. Attempting to run again...")
            self.run() # Recursive call to restart the training loop

class DummyLayer(nn.Module):
    """Simulates a layer that occasionally raises an exception."""
    def __init__(self):
        super().__init__()

    def forward(self, x=None):
        random_int = random.randint(1, 500)
        if random_int % 499 == 0:
            raise Exception("Simulated Crash")
        return x # Pass through if no crash

class NeuralNetwork(nn.Module):
    """A sample network using the Stage wrapper."""
    def __init__(self):
        super().__init__()
        self.flatten = Stage(nn.Flatten())
        self.linear_relu_stack = nn.Sequential(
            Stage(nn.Linear(28*28, 512)),
            Stage(nn.ReLU()),
            Stage(nn.Linear(512, 512)),
            Stage(nn.ReLU()),
            Stage(nn.Linear(512, 10)),
        )
        self.dummy = Stage(DummyLayer())

    def forward(self, x):
        x = self.flatten(x)
        # The crash simulation layer is executed here
        x = self.dummy(x) 
        logits = self.linear_relu_stack(x)
        return logits

```
