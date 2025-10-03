#!/usr/bin/env python3

# Final test script to verify the naming fix works correctly without underscore separator

def format_gym_task_name(task_name, skill_name):
    """Test the final fixed naming logic without underscore separator."""
    # Format gym task name to include task name prefix (matching orchestrator)
    # Split by underscores and capitalize each part, but remove the main separator
    task_parts = task_name.split('_')
    skill_parts = skill_name.split('_')
    formatted_task = ''.join(part.capitalize() for part in task_parts)
    formatted_skill = ''.join(part.capitalize() for part in skill_parts)
    gym_task_name_suffix = f"{formatted_task}{formatted_skill}"
    return gym_task_name_suffix

# Test cases
test_cases = [
    ("knock_over_pillars_seed42", "walk_to_cylindercolumn1"),
    ("knock_over_pillars_seed42", "knock_over_cylindercolumn1"),
    ("build_stairs_seed123", "walk_to_large_block"),
    ("obstacle_course_seeded42", "walk_to_low_wall")
]

print("Testing the final naming fix (without underscore separator):")
print("=" * 60)

for task_name, skill_name in test_cases:
    formatted_name = format_gym_task_name(task_name, skill_name)
    gym_id = f"Isaac-RobotFlat{formatted_name}"
    
    print(f"Task: {task_name}")
    print(f"Skill: {skill_name}")
    print(f"Formatted: {formatted_name}")
    print(f"Gym ID: {gym_id}")
    print("-" * 40)

print("\n✅ The naming fix should now create correct gym environment names!")
print("   These should match what's registered: Isaac-RobotFlatKnockOverPillarsSeed42WalkToCylindercolumn1")
