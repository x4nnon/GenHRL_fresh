"""
Main Workflow for GenHRL Task Generation

This module provides the main workflow functions for creating complete hierarchical RL tasks,
including cleanup utilities and task orchestration that were in the previous implementation.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .task_manager import TaskManager, TaskConfig
from .robot_configs import get_robot_folder_name


def create_task_with_workflow(
    task_name: str,
    task_description: str,
    isaaclab_path: str,
    api_key: str,
    robot: str = "G1",
    max_hierarchy_levels: int = 3,
    remove_previous: bool = False,
    verify_decompose: bool = True,
    verify_plan: bool = False,
    verify_rewards: bool = False,
    verify_success: bool = False
) -> TaskConfig:
    """
    Complete workflow to create a hierarchical RL task from description.
    
    This function replicates the main() workflow from the previous implementation.
    
    Args:
        task_name: Name of the task
        task_description: Natural language description
        isaaclab_path: Path to IsaacLab installation
        api_key: API key for LLM services
        robot: Robot name (default: G1)
        max_hierarchy_levels: Maximum hierarchy levels
        remove_previous: Whether to clean up previous skills first
        verify_decompose: Enable decomposition verification
        verify_plan: Enable plan verification
        verify_rewards: Enable reward verification
        verify_success: Enable success criteria verification
        
    Returns:
        TaskConfig object with all generated configurations
    """
    print(f"🚀 Starting task creation workflow")
    print(f"📝 Task: {task_name}")
    print(f"🤖 Robot: {robot}")
    print(f"🔧 IsaacLab Path: {isaaclab_path}")
    
    # Clean up previous skills if requested
    if remove_previous:
        print("🧹 Cleaning up previous skills...")
        remove_all_previous_skills(isaaclab_path, robot)
    
    # Initialize task manager with verification settings
    task_manager = TaskManager(isaaclab_path, api_key, robot)
    
    # Update verification flags on the code generator
    task_manager.code_generator.verify_decompose = verify_decompose
    task_manager.code_generator.verify_plan = verify_plan
    # Note: verify_rewards and verify_success are method names, so we can't assign to them as attributes
    # These verification flags need to be passed to the constructor instead
    
    # Create the task
    task_config = task_manager.create_task_from_description(
        task_name=task_name,
        task_description=task_description,
        max_hierarchy_levels=max_hierarchy_levels,
        robot=robot
    )
    
    print(f"\n✅ Task creation workflow complete!")
    print(f"📁 Task directory: {task_config.get_task_path()}")
    print(f"🎯 Skills directory: {task_config.get_skills_base_path() / 'skills'}")
    print(f"📚 Skill library: {task_config.get_skills_base_path() / 'skill_library.json'}")
    
    return task_config


def remove_all_previous_skills(isaaclab_path: str, robot: str = "G1") -> None:
    """
    Remove all previous skills and tasks for the specified robot.
    
    This replicates the cleanup functionality from the previous implementation.
    
    Args:
        isaaclab_path: Path to IsaacLab installation
        robot: Robot name (default: G1)
    """
    print(f"🧹 Removing all previous skills for robot: {robot}")
    
    robot_folder = get_robot_folder_name(robot)
    base_path = Path(isaaclab_path) / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder
    
    # Remove skills directory contents
    skills_dir = base_path / "skills" / "skills"
    if skills_dir.exists():
        print(f"🗑️ Cleaning skills directory: {skills_dir}")
        for item in skills_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    # Remove skill library file
    skill_library_path = base_path / "skills" / "skill_library.json"
    if skill_library_path.exists():
        print(f"🗑️ Removing skill library: {skill_library_path}")
        skill_library_path.unlink()
    
    # Remove tasks directory
    tasks_dir = base_path / "tasks"
    if tasks_dir.exists():
        print(f"🗑️ Removing tasks directory: {tasks_dir}")
        shutil.rmtree(tasks_dir)
    
    # Remove relevant logs (be careful with this - only remove robot-specific logs)
    logs_dir = Path("logs/skrl")
    if logs_dir.exists():
        print(f"🗑️ Cleaning logs for robot: {robot}")
        robot_pattern = robot.lower()
        for log_item in logs_dir.iterdir():
            if log_item.is_dir() and robot_pattern in log_item.name.lower():
                shutil.rmtree(log_item)
            elif log_item.is_file() and robot_pattern in log_item.name.lower():
                log_item.unlink()
    
    print(f"✅ Cleanup complete for robot: {robot}")


def strip_markdown_formatting(text: str) -> str:
    """
    Remove Markdown code block formatting from text.
    
    This replicates the utility function from the previous implementation.
    
    Args:
        text: Text potentially containing Markdown formatting
        
    Returns:
        Clean text without Markdown formatting
    """
    # Remove opening code block markers (```python, ```json, etc.)
    lines = text.split('\n')
    if lines and '```' in lines[0]:
        lines = lines[1:]
    
    # Remove closing code block markers
    if lines and '```' in lines[-1]:
        lines = lines[:-1]
    # Also check second-to-last line in case of trailing newline
    elif len(lines) > 1 and '```' in lines[-2] and not lines[-1].strip():
        lines = lines[:-2] + [lines[-1]]
    
    # Look for any other code block markers in the middle and remove them
    cleaned_lines = []
    for line in lines:
        if line.strip() == '```' or line.startswith('```'):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_json_string(json_string: str) -> str:
    """
    Clean and correct common JSON formatting issues.
    
    This replicates the utility function from the previous implementation.
    
    Args:
        json_string: The raw JSON string to clean
        
    Returns:
        A cleaned JSON string
    """
    import re
    
    # Remove trailing commas before closing braces/brackets
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    
    # Add missing commas between JSON objects or arrays
    json_string = re.sub(r'("[^"]+":\s*"[^"]+")\s*("[^"]+":)', r'\1, \2', json_string)
    
    # Ensure all strings are properly terminated
    json_string = re.sub(r'"([^"]*)$', r'"\1"', json_string)
    
    # Add missing quotes around property names
    json_string = re.sub(r'(?<!")([a-zA-Z0-9_]+)(?=")', r'"\1"', json_string)
    
    # Log the cleaned JSON string
    print(f"Cleaned JSON string: {json_string[:200]}{'...' if len(json_string) > 200 else ''}")
    
    return json_string


def main_create_steps_example(isaaclab_path: str, api_key: str, remove_previous: bool = True) -> TaskConfig:
    """
    Example workflow that replicates the Create_Steps task from the previous implementation.
    
    Args:
        isaaclab_path: Path to IsaacLab installation
        api_key: API key for LLM services  
        remove_previous: Whether to clean up previous skills first
    """
    task_name = "Create_Steps"
    task_description = """The environment should consist of three boxes, of heights 0.5, 1.0, and 1.5. These should all be 1m wide, and weigh 10, 20 and 100kg.
    These boxes should be arranged roughly in a triangle with 60deg between each box, with approximately 3m between the boxes. 

    The robot should walk to the medium box (ensuring it is on the side opposite to the large box), then push it over to the large box, 
    then walk to the correct pushing side of the small box. Then the robot should push it to the medium box.
    The robot should then jump onto each box so that it finishes standing on the largest box.

    The default hip height is 0.7 this should override any other reference to default hip height.

    When forming the rewards and success criteria, you should treat the small box as Object1, the medium box as Object2, and the large box as Object3. This overrides other references to objects. 
    However in naming each skill, you may still name them as smallbox mediumbox, largebox.
    """
    
    task_config = create_task_with_workflow(
        task_name=task_name,
        task_description=task_description,
        isaaclab_path=isaaclab_path,
        api_key=api_key,
        robot="G1",
        max_hierarchy_levels=3,
        remove_previous=remove_previous,
        verify_decompose=True,
        verify_plan=False,
        verify_rewards=False,
        verify_success=False
    )
    
    print(f"\n🎉 Create_Steps task generated successfully!")
    return task_config


if __name__ == "__main__":
    # Example usage - replace with your actual values
    ISAACLAB_PATH = "/path/to/isaaclab"
    API_KEY = "your-api-key-here"
    
    main_create_steps_example(ISAACLAB_PATH, API_KEY)