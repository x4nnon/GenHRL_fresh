#!/usr/bin/env python3
"""
Example usage of the SkillHierarchyVisualizer class.

This script demonstrates how to use the visualizer programmatically
rather than through the command line interface.
"""

from visualize_skill_hierarchies import SkillHierarchyVisualizer
from pathlib import Path


def main():
    """Example usage of the SkillHierarchyVisualizer."""
    
    # Path to the tasks directory
    tasks_path = Path(__file__).parent.parent.parent / "IsaacLab" / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "G1_generated" / "tasks"
    
    # Create visualizer
    visualizer = SkillHierarchyVisualizer(tasks_path)
    
    print("🔍 Example: Finding task directories...")
    task_dirs = visualizer.find_task_directories()
    print(f"Found {len(task_dirs)} tasks: {[d.name for d in task_dirs[:3]]}{'...' if len(task_dirs) > 3 else ''}")
    
    if task_dirs:
        print("\n🌳 Example: Visualizing first task...")
        visualizer.visualize_task(task_dirs[0], max_width=80)
        
        print("\n📊 Example: Task comparison...")
        print("You could extend this to compare hierarchies between seeds:")
        
        # Find tasks with same base name but different seeds
        task_groups = {}
        for task_dir in task_dirs:
            # Extract base name (everything before the last underscore)
            parts = task_dir.name.split('_')
            if len(parts) >= 2 and parts[-1].startswith('seed'):
                base_name = '_'.join(parts[:-1])
                if base_name not in task_groups:
                    task_groups[base_name] = []
                task_groups[base_name].append(task_dir)
        
        for base_name, variants in task_groups.items():
            if len(variants) > 1:
                print(f"  - {base_name}: {len(variants)} variants ({', '.join([v.name.split('_')[-1] for v in variants])})")
                break
    
    print("\n✅ Example complete!")
    print("\nTo run the full visualization:")
    print("  python genhrl/testing/visualize_skill_hierarchies.py")


if __name__ == "__main__":
    main()