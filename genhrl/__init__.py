"""
GenHRL: Generative Hierarchical Reinforcement Learning Framework

A comprehensive framework for:
- Generating hierarchical RL tasks from natural language
- Training hierarchical policies with IsaacLab integration
- Managing skill libraries and task orchestration
"""

__version__ = "0.1.0"
__author__ = "GenHRL Development Team"

# Core imports
from .generation import TaskGenerator, SkillLibrary
from .training import TrainingOrchestrator

# Check for IsaacLab at import time
import importlib.util
import warnings

def check_isaaclab():
    """Check if IsaacLab is available."""
    try:
        # Try to import IsaacLab
        spec = importlib.util.find_spec("isaaclab")
        if spec is None:
            warnings.warn(
                "IsaacLab not found. Please install IsaacLab to use training features. "
                "See https://isaac-sim.github.io/IsaacLab/ for installation instructions.",
                ImportWarning
            )
            return False
        return True
    except Exception as e:
        warnings.warn(f"Error checking for IsaacLab: {e}", ImportWarning)
        return False

# Check IsaacLab availability
ISAACLAB_AVAILABLE = check_isaaclab()

__all__ = [
    "TaskGenerator",
    "SkillLibrary", 
    "TrainingOrchestrator",
    "ISAACLAB_AVAILABLE",
]