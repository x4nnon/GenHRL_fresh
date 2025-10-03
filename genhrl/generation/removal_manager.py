"""
Removal Manager for GenHRL Tasks and Skills

This module provides comprehensive removal functionality for tasks and skills,
ensuring all related files, directories, and logs are properly cleaned up.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import glob

from .robot_configs import get_robot_folder_name
from .skill_library import SkillLibrary


class RemovalManager:
    """
    Manages comprehensive removal of tasks and skills with proper cleanup.
    
    This class handles:
    - Complete task removal (all skills, config files, logs)
    - Individual skill removal (from library, filesystem, logs)
    - Dependency checking and cleanup
    - Log directory cleanup
    """
    
    def __init__(self, isaaclab_path: str, robot: str = "G1"):
        """
        Initialize the removal manager.
        
        Args:
            isaaclab_path: Path to IsaacLab installation
            robot: Robot name (default: G1)
        """
        self.isaaclab_path = Path(isaaclab_path)
        self.robot = robot
        self.robot_folder = get_robot_folder_name(robot)
        
        # Create base paths
        self.base_path = self.isaaclab_path / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / self.robot_folder
        self.tasks_base_path = self.base_path / "tasks"
        self.skills_base_path = self.base_path / "skills"
        self.logs_base_path = Path("logs/skrl")
    
    def remove_task(self, task_name: str, confirm: bool = True) -> bool:
        """
        Remove a complete task including all its skills and related files.
        
        Args:
            task_name: Name of the task to remove
            confirm: Whether to require confirmation before deletion
            
        Returns:
            True if task was removed successfully, False otherwise
        """
        if confirm:
            print(f"⚠️  WARNING: This will permanently delete task '{task_name}' and ALL its skills!")
            print(f"   - Task directory: {self.tasks_base_path / task_name}")
            print(f"   - Skills directory: {self.skills_base_path / task_name}")
            print(f"   - All training logs and checkpoints")
            response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
            if response != 'yes':
                print("Operation cancelled.")
                return False
        
        print(f"🗑️  Removing task: {task_name}")
        
        # Get all skills for this task before removal
        task_skills = self._get_task_skills(task_name)
        
        # Remove task directory
        task_path = self.tasks_base_path / task_name
        if task_path.exists():
            print(f"   📁 Removing task directory: {task_path}")
            shutil.rmtree(task_path)
        
        # Remove skills directory
        task_skills_path = self.skills_base_path / task_name
        if task_skills_path.exists():
            print(f"   🎯 Removing skills directory: {task_skills_path}")
            shutil.rmtree(task_skills_path)
        
        # Clean up logs for all skills in this task
        if task_skills:
            print(f"   🧹 Cleaning logs for {len(task_skills)} skills...")
            for skill_name in task_skills:
                self._remove_skill_logs(skill_name)
        
        print(f"✅ Successfully removed task: {task_name}")
        return True
    
    def remove_skill(self, task_name: str, skill_name: str, confirm: bool = True) -> bool:
        """
        Remove an individual skill from a task.
        
        Args:
            task_name: Name of the task containing the skill
            skill_name: Name of the skill to remove
            confirm: Whether to require confirmation before deletion
            
        Returns:
            True if skill was removed successfully, False otherwise
        """
        if confirm:
            print(f"⚠️  WARNING: This will permanently delete skill '{skill_name}' from task '{task_name}'!")
            response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
            if response != 'yes':
                print("Operation cancelled.")
                return False
        
        print(f"🗑️  Removing skill: {skill_name} from task: {task_name}")
        
        # Load skill library
        skill_library_path = self.skills_base_path / task_name / "skill_library.json"
        if not skill_library_path.exists():
            print(f"❌ Skill library not found: {skill_library_path}")
            return False
        
        skill_library = SkillLibrary(str(skill_library_path))
        
        # Check if skill exists in library
        if skill_name not in skill_library.skills["skills"]:
            print(f"❌ Skill '{skill_name}' not found in task '{task_name}'")
            return False
        
        # Check for dependencies
        dependencies = self._check_skill_dependencies(skill_name, skill_library)
        if dependencies:
            print(f"⚠️  Warning: Skill '{skill_name}' is used by other skills: {dependencies}")
            if confirm:
                response = input("Remove anyway? This may break dependent skills. (yes/no): ").lower().strip()
                if response != 'yes':
                    print("Operation cancelled.")
                    return False
        
        # Remove skill from library
        print(f"   📚 Removing skill from library...")
        skill_library.remove_skill(skill_name)
        
        # Remove skill directory
        skill_path = self.skills_base_path / task_name / "skills" / skill_name
        if skill_path.exists():
            print(f"   📁 Removing skill directory: {skill_path}")
            shutil.rmtree(skill_path)
        
        # Remove skill logs
        print(f"   🧹 Cleaning skill logs...")
        self._remove_skill_logs(skill_name)
        
        # Update hierarchy if this was the root skill
        self._update_hierarchy_after_skill_removal(task_name, skill_name, skill_library)
        
        print(f"✅ Successfully removed skill: {skill_name}")
        return True
    
    def remove_skill_logs(self, skill_name: str) -> bool:
        """
        Remove all logs and training data for a specific skill.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            True if logs were removed successfully
        """
        print(f"🧹 Removing logs for skill: {skill_name}")
        return self._remove_skill_logs(skill_name)
    
    def _get_task_skills(self, task_name: str) -> List[str]:
        """Get all skill names for a task."""
        skill_library_path = self.skills_base_path / task_name / "skill_library.json"
        if not skill_library_path.exists():
            return []
        
        try:
            skill_library = SkillLibrary(str(skill_library_path))
            return skill_library.get_full_skill_sequence(task_name)
        except Exception as e:
            print(f"Warning: Could not load skill library: {e}")
            return []
    
    def _check_skill_dependencies(self, skill_name: str, skill_library: SkillLibrary) -> List[str]:
        """Check which other skills depend on this skill."""
        skill_info = skill_library.get_skill_info(skill_name)
        if not skill_info:
            return []
        
        return skill_info.get("used_by", [])
    
    def _remove_skill_logs(self, skill_name: str) -> bool:
        """Remove all logs and training data for a skill."""
        removed_any = False
        
        if not self.logs_base_path.exists():
            return removed_any
        
        # Try different log directory patterns
        log_patterns = [
            skill_name.lower(),  # Primitive skills use lowercase
            skill_name,          # Composite skills use original case
            f"*{skill_name.lower()}*",  # Fuzzy match with lowercase
            f"*{skill_name}*"    # Fuzzy match with original case
        ]
        
        for pattern in log_patterns:
            # Direct directory match
            log_dir = self.logs_base_path / pattern
            if '*' not in pattern and log_dir.exists():
                print(f"      🗂️  Removing log directory: {log_dir}")
                shutil.rmtree(log_dir)
                removed_any = True
            
            # Glob pattern match
            elif '*' in pattern:
                matching_dirs = list(self.logs_base_path.glob(pattern))
                for log_dir in matching_dirs:
                    if log_dir.is_dir():
                        print(f"      🗂️  Removing log directory: {log_dir}")
                        shutil.rmtree(log_dir)
                        removed_any = True
        
        # Also clean up any experiment directories that might contain this skill
        for log_dir in self.logs_base_path.iterdir():
            if log_dir.is_dir():
                # Check if any subdirectories contain this skill name
                for subdir in log_dir.iterdir():
                    if subdir.is_dir() and skill_name.lower() in subdir.name.lower():
                        print(f"      🗂️  Removing experiment directory: {subdir}")
                        shutil.rmtree(subdir)
                        removed_any = True
        
        return removed_any
    
    def _update_hierarchy_after_skill_removal(self, task_name: str, skill_name: str, skill_library: SkillLibrary) -> None:
        """Update the task hierarchy after removing a skill."""
        if task_name in skill_library.skills["hierarchies"]:
            hierarchy = skill_library.skills["hierarchies"][task_name]
            
            # If the removed skill was the root skill, we need to update the hierarchy
            if hierarchy["structure"]["name"] == skill_name:
                # Check if there are children that can become the new root
                children = hierarchy["structure"].get("children", [])
                if len(children) == 1:
                    # Single child becomes new root
                    hierarchy["structure"] = children[0]
                    skill_library._save_library()
                    print(f"   🔄 Updated hierarchy: {children[0]['name']} is now the root skill")
                elif len(children) > 1:
                    print(f"   ⚠️  Warning: Multiple children found. Hierarchy may need manual adjustment.")
                else:
                    # No children - remove the entire hierarchy
                    skill_library.remove_hierarchy(task_name)
                    print(f"   🗑️  Removed empty hierarchy for task: {task_name}")
    
    def list_removable_tasks(self) -> List[str]:
        """List all tasks that can be removed."""
        if not self.tasks_base_path.exists():
            return []
        
        return [d.name for d in self.tasks_base_path.iterdir() if d.is_dir()]
    
    def list_removable_skills(self, task_name: str) -> List[str]:
        """List all skills in a task that can be removed."""
        task_skills_path = self.skills_base_path / task_name / "skills"
        if not task_skills_path.exists():
            return []
        
        return [d.name for d in task_skills_path.iterdir() if d.is_dir()]
    
    def get_removal_impact(self, task_name: str, skill_name: Optional[str] = None) -> Dict:
        """
        Get information about what will be removed.
        
        Args:
            task_name: Name of the task
            skill_name: Name of the skill (if removing individual skill)
            
        Returns:
            Dictionary with removal impact information
        """
        impact = {
            "type": "skill" if skill_name else "task",
            "task_name": task_name,
            "skill_name": skill_name,
            "directories": [],
            "log_directories": [],
            "affected_skills": [],
            "file_count": 0
        }
        
        if skill_name:
            # Individual skill removal impact
            skill_library_path = self.skills_base_path / task_name / "skill_library.json"
            if skill_library_path.exists():
                skill_library = SkillLibrary(str(skill_library_path))
                impact["affected_skills"] = self._check_skill_dependencies(skill_name, skill_library)
            
            skill_path = self.skills_base_path / task_name / "skills" / skill_name
            if skill_path.exists():
                impact["directories"].append(str(skill_path))
                impact["file_count"] = sum(1 for _ in skill_path.rglob('*') if _.is_file())
            
            # Check for logs
            log_patterns = [skill_name.lower(), skill_name, f"*{skill_name.lower()}*", f"*{skill_name}*"]
            for pattern in log_patterns:
                if '*' not in pattern:
                    log_dir = self.logs_base_path / pattern
                    if log_dir.exists():
                        impact["log_directories"].append(str(log_dir))
                else:
                    matching_dirs = list(self.logs_base_path.glob(pattern))
                    impact["log_directories"].extend([str(d) for d in matching_dirs if d.is_dir()])
        else:
            # Complete task removal impact
            task_path = self.tasks_base_path / task_name
            task_skills_path = self.skills_base_path / task_name
            
            if task_path.exists():
                impact["directories"].append(str(task_path))
                impact["file_count"] += sum(1 for _ in task_path.rglob('*') if _.is_file())
            
            if task_skills_path.exists():
                impact["directories"].append(str(task_skills_path))
                impact["file_count"] += sum(1 for _ in task_skills_path.rglob('*') if _.is_file())
            
            # Get all skills that will be affected
            impact["affected_skills"] = self._get_task_skills(task_name)
            
            # Check for logs for all skills
            for skill in impact["affected_skills"]:
                log_patterns = [skill.lower(), skill, f"*{skill.lower()}*", f"*{skill}*"]
                for pattern in log_patterns:
                    if '*' not in pattern:
                        log_dir = self.logs_base_path / pattern
                        if log_dir.exists() and str(log_dir) not in impact["log_directories"]:
                            impact["log_directories"].append(str(log_dir))
                    else:
                        matching_dirs = list(self.logs_base_path.glob(pattern))
                        for d in matching_dirs:
                            if d.is_dir() and str(d) not in impact["log_directories"]:
                                impact["log_directories"].append(str(d))
        
        return impact
    
    def dry_run_removal(self, task_name: str, skill_name: Optional[str] = None) -> None:
        """
        Show what would be removed without actually removing anything.
        
        Args:
            task_name: Name of the task
            skill_name: Name of the skill (if removing individual skill)
        """
        impact = self.get_removal_impact(task_name, skill_name)
        
        print(f"\n🔍 DRY RUN: {impact['type'].title()} Removal Impact")
        print(f"   Target: {task_name}" + (f" > {skill_name}" if skill_name else ""))
        
        if impact["directories"]:
            print(f"\n📁 Directories to be removed ({len(impact['directories'])}):")
            for dir_path in impact["directories"]:
                print(f"   - {dir_path}")
        
        if impact["log_directories"]:
            print(f"\n🗂️  Log directories to be removed ({len(impact['log_directories'])}):")
            for log_dir in impact["log_directories"]:
                print(f"   - {log_dir}")
        
        if impact["affected_skills"]:
            print(f"\n🎯 Skills affected ({len(impact['affected_skills'])}):")
            for skill in impact["affected_skills"]:
                print(f"   - {skill}")
        
        print(f"\n📊 Total files to be removed: {impact['file_count']}")
        
        if skill_name and impact["affected_skills"]:
            print(f"\n⚠️  WARNING: Skill '{skill_name}' is used by {len(impact['affected_skills'])} other skills!")
            print("   Removing it may break dependent skills.")