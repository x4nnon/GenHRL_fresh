"""
Skill Library for the iGen framework.

This module manages the hierarchical skill library, storing and organizing
skills and their relationships for reuse across different tasks.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class SkillLibrary:
    """
    Manages the hierarchical skill library for iGen.
    
    This class handles:
    - Storage and retrieval of skills (primitive and composite)
    - Skill hierarchy management
    - Skill sequencing and dependencies
    - Library persistence to JSON
    """
    
    def __init__(self, library_path: str):
        """
        Initialize the skill library.
        
        Args:
            library_path: Path to the skill library JSON file
        """
        self.library_path = Path(library_path)
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills = self._load_library()
    
    def _load_library(self) -> Dict:
        """Load the skill library from JSON file or create new one if it doesn't exist."""
        if self.library_path.exists():
            print(f"Loading existing skill library from {self.library_path}")
            try:
                with open(self.library_path, 'r') as f:
                    data = json.load(f)
                    
                # Basic validation for expected keys
                if "skills" not in data or "hierarchies" not in data:
                    print(f"Warning: Skill library file {self.library_path} is missing expected keys. Initializing new structure.")
                    return {"skills": {}, "hierarchies": {}}
                return data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {self.library_path}. Initializing new library.")
                return {"skills": {}, "hierarchies": {}}
        
        print(f"No existing skill library found at {self.library_path}. Creating new library.")
        return {
            "skills": {},      # All individual skills indexed by name (primitives and composites)
            "hierarchies": {}  # Top-level task hierarchies stored by task name
        }
    
    def add_skill(self, name: str, task_path: str, description: str,
                  object_config_paths: List[str],
                  is_primitive: bool = False,
                  parent_skill: Optional[str] = None,
                  sub_skills: Optional[List[str]] = None) -> None:
        """
        Add a new skill (primitive or composite) to the library or update existing one.
        
        Args:
            name: Skill name
            task_path: Path to the skill's task directory
            description: Skill description
            object_config_paths: List of object configuration file paths
            is_primitive: Whether this is a primitive skill
            parent_skill: Name of parent skill (if any)
            sub_skills: List of sub-skill names (for composite skills)
        """
        if sub_skills is None:
            sub_skills = []
        
        if name in self.skills["skills"]:
            print(f"Skill '{name}' already exists in library. Updating...")
            skill_entry = self.skills["skills"][name]
            
            # Update paths if new ones are provided
            if object_config_paths:
                for path in object_config_paths:
                    if path not in skill_entry["object_config_paths"]:
                        skill_entry["object_config_paths"].append(path)
            
            # Update description if different
            if description != skill_entry["description"]:
                skill_entry["description"] = description
            
            # Update primitive status
            skill_entry["is_primitive"] = is_primitive
            
            # Update sub_skills if provided
            if sub_skills:
                skill_entry["sub_skills"] = sub_skills
            
            # Update parent references (skills that use this skill)
            if parent_skill and parent_skill not in skill_entry["used_by"]:
                skill_entry["used_by"].append(parent_skill)
        else:
            print(f"Adding new skill '{name}' to library...")
            self.skills["skills"][name] = {
                "task_path": task_path,
                "policy_path": f"trained_policies/{name}.pt",
                "description": description,
                "object_config_paths": object_config_paths,
                "is_primitive": is_primitive,
                "sub_skills": sub_skills,  # Skills this skill directly uses
                "used_by": [parent_skill] if parent_skill else []  # Skills that use this skill
            }
    
    def add_hierarchy(self, task_name: str, skills_hierarchy, task_description: str) -> None:
        """
        Add a complete skill hierarchy to the library.
        
        Args:
            task_name: Name of the task
            skills_hierarchy: Hierarchical skill structure (Dict for level 3+, List for level 2, Dict for level 1)
            task_description: Description of the task
        """
        print(f"\nAdding hierarchy for task '{task_name}' to library...")
        
        # Handle different hierarchy formats based on structure
        if isinstance(skills_hierarchy, list):
            # Level 2: List of skills - create a virtual root node
            print("Processing level 2 hierarchy (list of skills)")
            virtual_root = {
                "name": task_name,
                "description": task_description,
                "children": skills_hierarchy
            }
            processed_hierarchy = virtual_root
        elif isinstance(skills_hierarchy, dict):
            if "children" not in skills_hierarchy or not skills_hierarchy.get("children"):
                # Level 1: Single task without children
                print("Processing level 1 hierarchy (single task)")
                processed_hierarchy = skills_hierarchy
            else:
                # Level 3+: Standard dictionary format with children
                print("Processing level 3+ hierarchy (nested structure)")
                processed_hierarchy = skills_hierarchy
        else:
            raise ValueError(f"Invalid skills_hierarchy format: {type(skills_hierarchy)}")
        
        # Store the processed hierarchy structure
        self.skills["hierarchies"][task_name] = {
            "created_at": datetime.datetime.now().isoformat(),
            "description": task_description,
            "structure": processed_hierarchy  # Store the processed hierarchy
        }
        
        # Recursively process the hierarchy to add/update individual skills
        self._process_skill_node(processed_hierarchy, parent_skill=None, task_name=task_name)
        
        self._save_library()
    
    def _process_skill_node(self, skill_node: Dict, parent_skill: Optional[str], task_name: str) -> None:
        """
        Recursively process a node in the skill hierarchy.
        
        Args:
            skill_node: Node in the skill hierarchy
            parent_skill: Name of parent skill (if any)
            task_name: Name of the task this hierarchy belongs to
        """
        name = skill_node["name"]
        description = skill_node["description"]
        children = skill_node.get("children", [])
        
        # Object config path relative to task's skill directory - now in skills directory
        object_config_paths = [f"../object_config.json"]
        
        is_primitive = not bool(children)  # A skill is primitive if it has no children
        sub_skill_names = [child["name"] for child in children]
        
        # Add/update this skill in the main skills dictionary (paths are relative to task's skill directory)
        self.add_skill(
            name=name,
            task_path=f"skills/{name}",  # This is relative to the task's skills directory
            description=description,
            object_config_paths=object_config_paths,
            is_primitive=is_primitive,
            parent_skill=parent_skill,
            sub_skills=sub_skill_names
        )
        
        # Recursively process children
        for child_node in children:
            self._process_skill_node(child_node, parent_skill=name, task_name=task_name)
    
    def extract_primitive_skills(self, skills_hierarchy: Dict) -> List[Dict]:
        """
        Extract all primitive skills (leaves) from the hierarchy into a flat list.
        
        Args:
            skills_hierarchy: Hierarchical skill structure
            
        Returns:
            List of primitive skill dictionaries
        """
        primitive_skills = []
        
        def find_leaves(node: Dict) -> None:
            is_primitive = not bool(node.get("children", []))
            if is_primitive:
                skill_info = {
                    "name": node["name"],
                    "description": node["description"],
                    "object_config_paths": node.get("object_config_paths", []),
                    "is_primitive": True
                }
                # Avoid adding duplicates if a primitive is used multiple times
                if node["name"] not in [p["name"] for p in primitive_skills]:
                    primitive_skills.append(skill_info)
            else:
                for child in node.get("children", []):
                    find_leaves(child)
        
        find_leaves(skills_hierarchy)
        return primitive_skills
    
    def get_primitive_skill_sequence(self, hierarchy_name: str) -> List[str]:
        """
        Get the ordered sequence of primitive skills for a given hierarchy.
        
        Args:
            hierarchy_name: Name of the hierarchy
            
        Returns:
            Ordered list of primitive skill names
        """
        if hierarchy_name not in self.skills["hierarchies"]:
            print(f"Error: Hierarchy '{hierarchy_name}' not found.")
            return []
        
        hierarchy = self.skills["hierarchies"][hierarchy_name]["structure"]
        primitive_sequence = []
        
        def traverse_for_primitives(node: Dict) -> None:
            is_primitive = not bool(node.get("children", []))
            if is_primitive:
                primitive_sequence.append(node["name"])
            else:
                # Assume children should be executed in order
                for child in node.get("children", []):
                    traverse_for_primitives(child)
        
        traverse_for_primitives(hierarchy)
        return primitive_sequence
    
    def get_full_skill_sequence(self, hierarchy_name: str) -> List[str]:
        """
        Get the ordered sequence of ALL skills (primitive and composite) 
        for a given hierarchy using depth-first traversal.
        
        Args:
            hierarchy_name: Name of the hierarchy
            
        Returns:
            Ordered list of all skill names
        """
        if hierarchy_name not in self.skills["hierarchies"]:
            print(f"Error: Hierarchy '{hierarchy_name}' not found.")
            return []
        
        hierarchy = self.skills["hierarchies"][hierarchy_name]["structure"]
        full_sequence = []
        
        def traverse_depth_first(node: Dict) -> None:
            # Add the current node first (parent before children in DFT)
            full_sequence.append(node["name"])
            
            # Then traverse children
            for child in node.get("children", []):
                traverse_depth_first(child)
        
        traverse_depth_first(hierarchy)
        return full_sequence
    
    def get_skill_info(self, skill_name: str) -> Optional[Dict]:
        """
        Get information about a specific skill.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Skill information dictionary or None if not found
        """
        return self.skills["skills"].get(skill_name)
    
    def get_hierarchy_info(self, hierarchy_name: str) -> Optional[Dict]:
        """
        Get information about a specific hierarchy.
        
        Args:
            hierarchy_name: Name of the hierarchy
            
        Returns:
            Hierarchy information dictionary or None if not found
        """
        return self.skills["hierarchies"].get(hierarchy_name)
    
    def list_skills(self) -> List[str]:
        """Get a list of all skill names."""
        return list(self.skills["skills"].keys())
    
    def list_hierarchies(self) -> List[str]:
        """Get a list of all hierarchy names."""
        return list(self.skills["hierarchies"].keys())
    
    def list_primitive_skills(self) -> List[str]:
        """Get a list of all primitive skill names."""
        return [name for name, info in self.skills["skills"].items() 
                if info.get("is_primitive", False)]
    
    def list_composite_skills(self) -> List[str]:
        """Get a list of all composite skill names."""
        return [name for name, info in self.skills["skills"].items() 
                if not info.get("is_primitive", False)]
    
    def remove_skill(self, skill_name: str) -> bool:
        """
        Remove a skill from the library.
        
        Args:
            skill_name: Name of the skill to remove
            
        Returns:
            True if skill was removed, False if not found
        """
        if skill_name in self.skills["skills"]:
            del self.skills["skills"][skill_name]
            self._save_library()
            print(f"Removed skill '{skill_name}' from library")
            return True
        return False
    
    def remove_hierarchy(self, hierarchy_name: str) -> bool:
        """
        Remove a hierarchy from the library.
        
        Args:
            hierarchy_name: Name of the hierarchy to remove
            
        Returns:
            True if hierarchy was removed, False if not found
        """
        if hierarchy_name in self.skills["hierarchies"]:
            del self.skills["hierarchies"][hierarchy_name]
            self._save_library()
            print(f"Removed hierarchy '{hierarchy_name}' from library")
            return True
        return False
    
    def _save_library(self) -> None:
        """Save the skill library to JSON file."""
        print(f"Saving updated skill library to {self.library_path}")
        with open(self.library_path, 'w') as f:
            json.dump(self.skills, f, indent=4)
    
    def export_library(self, export_path: str) -> None:
        """
        Export the library to a different file.
        
        Args:
            export_path: Path to export the library to
        """
        export_path_obj = Path(export_path)
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(self.skills, f, indent=4)
        print(f"Exported skill library to {export_path_obj}")
    
    def import_library(self, import_path: str, merge: bool = True) -> None:
        """
        Import skills from another library file.
        
        Args:
            import_path: Path to import the library from
            merge: Whether to merge with existing library (True) or replace it (False)
        """
        import_path_obj = Path(import_path)
        if not import_path_obj.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        with open(import_path, 'r') as f:
            imported_data = json.load(f)
        
        if merge:
            # Merge skills and hierarchies
            self.skills["skills"].update(imported_data.get("skills", {}))
            self.skills["hierarchies"].update(imported_data.get("hierarchies", {}))
        else:
            # Replace entire library
            self.skills = imported_data
        
        self._save_library()
        print(f"Imported skill library from {import_path}")
    
    def get_library_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the library.
        
        Returns:
            Dictionary with library statistics
        """
        total_skills = len(self.skills["skills"])
        primitive_skills = len(self.list_primitive_skills())
        composite_skills = len(self.list_composite_skills())
        total_hierarchies = len(self.skills["hierarchies"])
        
        return {
            "total_skills": total_skills,
            "primitive_skills": primitive_skills,
            "composite_skills": composite_skills,
            "total_hierarchies": total_hierarchies,
            "skill_names": self.list_skills(),
            "hierarchy_names": self.list_hierarchies()
        }