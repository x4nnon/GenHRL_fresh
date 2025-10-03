#!/usr/bin/env python3
"""
Patch script to update the orchestrator with the new step logic
"""

import re

def apply_patch():
    with open('genhrl/training/orchestrator.py', 'r') as f:
        content = f.read()
    
    # Update the training plan display
    old_pattern = r'print\(f"\\nSimple training plan for \{self\.task_name\}:"\)\s+print\(f"Each skill will train for exactly \{steps_per_skill\} steps"\)'
    new_pattern = '''print(f"\\nSimple training plan for {self.task_name}:")
        
        # Show step requirements
        if min_primitive_steps or min_composite_steps:
            print("📊 Step requirements:")
            if min_primitive_steps:
                print(f"   🏃 Primitive skills: {min_primitive_steps} steps")
            else:
                print(f"   🏃 Primitive skills: {steps_per_skill} steps (default)")
            if min_composite_steps:
                print(f"   🧩 Composite skills: {min_composite_steps} steps")
            else:
                print(f"   🧩 Composite skills: {steps_per_skill} steps (default)")
        else:
            print(f"📊 Each skill will train for exactly {steps_per_skill} steps")'''
    
    content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)
    
    # Update the training loop
    old_loop = r'# Train each skill in sequence\s+for i, skill_name in enumerate\(self\.training_order\):\s+print\(f"\\n\[\{i\+1\}/\{len\(self\.training_order\)\}\] Training \{skill_name\}\.\.\."\)\s+success = self\.train_skill_simple\(\s+skill_name,\s+steps_per_skill,'
    new_loop = '''# Train each skill in sequence
        for i, skill_name in enumerate(self.training_order):
            print(f"\\n[{i+1}/{len(self.training_order)}] Training {skill_name}...")
            
            # Determine steps for this skill
            is_primitive = self.is_skill_primitive(skill_name)
            if is_primitive and min_primitive_steps is not None:
                skill_steps = min_primitive_steps
                print(f"   🏃 Using primitive steps: {skill_steps}")
            elif not is_primitive and min_composite_steps is not None:
                skill_steps = min_composite_steps
                print(f"   🧩 Using composite steps: {skill_steps}")
            else:
                skill_steps = steps_per_skill
                print(f"   📊 Using default steps: {skill_steps}")
            
            success = self.train_skill_simple(
                skill_name, 
                skill_steps,'''
    
    content = re.sub(old_loop, new_loop, content, flags=re.MULTILINE | re.DOTALL)
    
    with open('genhrl/training/orchestrator.py', 'w') as f:
        f.write(content)
    
    print("Orchestrator patch applied successfully!")

if __name__ == "__main__":
    apply_patch()
