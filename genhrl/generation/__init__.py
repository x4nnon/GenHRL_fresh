"""
Generation Module for GenHRL

Contains task generation, skill library management, and code generation capabilities.
"""

from .code_generator import HierarchicalCodeGenerator as TaskGenerator
from .skill_library import SkillLibrary
from .task_manager import TaskManager
from .robot_configs import get_robot_config, get_available_robots, get_robot_folder_name
from .main_workflow import (
    create_task_with_workflow, 
    remove_all_previous_skills, 
    main_create_steps_example,
    strip_markdown_formatting,
    clean_json_string
)

__all__ = [
    "TaskGenerator",
    "SkillLibrary", 
    "TaskManager",
    "get_robot_config",
    "get_available_robots", 
    "get_robot_folder_name",
    "create_task_with_workflow",
    "remove_all_previous_skills",
    "main_create_steps_example",
    "strip_markdown_formatting",
    "clean_json_string",
]