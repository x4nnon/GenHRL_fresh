#!/usr/bin/env python3
"""
Example usage script for GenHRL testing framework.

This shows how to use the various testing tools:
1. Scientific testing for task generation
2. Skill execution testing 
3. Visualization of skill hierarchies
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd: str, description: str):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def main():
    """Main function demonstrating testing tools."""
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "genhrl" / "testing").exists():
        print("❌ Please run this script from the GenHRL_v2 root directory")
        sys.exit(1)
    
    print("🧪 GenHRL Testing Framework Examples")
    print("====================================")
    
    # Example 1: Test skill execution for a specific task
    print("\n1️⃣ Example: Test skill execution")
    print("This will test all skills in a task to see which ones work vs fail")
    print("Note: Requires an existing task. Replace 'Create_Steps' with your task name.")
    
    cmd1 = "python -m genhrl.testing.test_skill_execution Create_Steps --steps 250"
    print(f"Command to run: {cmd1}")
    print("(Not running automatically - replace task name with your actual task)")
    
    # Example 2: Scientific testing
    print("\n2️⃣ Example: Scientific testing framework")
    print("This generates multiple tasks with different seeds for evaluation")
    print("Requires GENHRL_API_KEY environment variable to be set")
    
    if 'GENHRL_API_KEY' in os.environ:
        cmd2 = "python -m genhrl.testing.scientific_testing"
        print(f"Command to run: {cmd2}")
        print("(Not running automatically to avoid API usage)")
    else:
        print("Set GENHRL_API_KEY environment variable first:")
        print("export GENHRL_API_KEY=your_key_here")
    
    # Example 3: Visualization
    print("\n3️⃣ Example: Visualize skill hierarchies")
    print("This creates visual diagrams of generated skill hierarchies")
    
    cmd3 = "python genhrl/testing/visualize_skill_hierarchies.py --format text"
    print(f"Command to run: {cmd3}")
    print("(This will show hierarchies in text format)")
    
    # Example 4: Analysis
    print("\n4️⃣ Example: Analyze test results")
    print("This analyzes results from scientific testing")
    
    cmd4 = "python -m genhrl.testing.analyze_results genhrl/testing/results/*.json"
    print(f"Command to run: {cmd4}")
    print("(Requires existing result files)")
    
    print("\n📝 Summary of available testing tools:")
    print("- test_skill_execution.py: Test individual skill execution")
    print("- scientific_testing.py: Generate tasks with multiple seeds")
    print("- visualize_skill_hierarchies.py: Create hierarchy visualizations")
    print("- analyze_results.py: Analyze scientific testing results")
    
    print("\n💡 Quick start for skill testing:")
    print("1. Make sure you have a generated task (e.g., from 'genhrl generate')")
    print("2. Activate conda environment: conda activate env_isaaclab")
    print("3. Run: python -m genhrl.testing.test_skill_execution YOUR_TASK_NAME")
    
    print("\n📚 See README.md in genhrl/testing/ for detailed documentation")

if __name__ == "__main__":
    main()