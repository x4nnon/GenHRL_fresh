#!/usr/bin/env python3
"""
Skill Execution Testing Script for GenHRL

This script tests each skill within a task by running it for 250 steps to identify
which skills run successfully vs which ones encounter errors.

Usage:
    python -m genhrl.testing.test_skill_execution TASK_NAME [options]
    
The script will:
1. Get all skills for the specified task
2. Run each skill for exactly 250 steps
3. Track success/failure for each skill
4. Report summary of working vs broken skills

Note: This script must be run with the env_isaaclab conda environment activated.
"""

import argparse
import os
import sys
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add the parent directory to path to import genhrl modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genhrl.training import TrainingOrchestrator


@dataclass
class SkillTestResult:
    """Result of testing a single skill."""
    skill_name: str
    is_primitive: bool
    success: bool
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    process_output_tail: Optional[str] = None
    execution_time: float = 0.0


class SkillExecutionTester:
    """Tester for individual skill execution within a task."""
    
    def __init__(self, isaaclab_path: str, task_name: str, robot: str = "G1"):
        """
        Initialize the skill execution tester.
        
        Args:
            isaaclab_path: Path to IsaacLab installation
            task_name: Name of the task to test
            robot: Robot name (default: G1)
        """
        self.isaaclab_path = isaaclab_path
        self.task_name = task_name
        self.robot = robot
        
        # Verify IsaacLab path exists
        if not Path(isaaclab_path).exists():
            raise ValueError(f"IsaacLab path does not exist: {isaaclab_path}")
        
        # Initialize the training orchestrator
        try:
            self.orchestrator = TrainingOrchestrator(isaaclab_path, task_name, robot)
        except FileNotFoundError as e:
            raise ValueError(f"Task '{task_name}' not found for robot '{robot}': {e}")
        
        self.test_results: List[SkillTestResult] = []
    
    def check_conda_environment(self) -> bool:
        """Check if we're running in the env_isaaclab conda environment."""
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        if conda_env != 'env_isaaclab':
            print(f"❌ Warning: Not running in env_isaaclab environment (current: {conda_env})")
            print("   Please activate the environment with: conda activate env_isaaclab")
            return False
        print(f"✅ Running in conda environment: {conda_env}")
        return True
    
    def get_skills_to_test(self) -> List[str]:
        """Get the list of all skills for the task."""
        try:
            skills = self.orchestrator.get_training_order()
            if not skills:
                raise ValueError(f"No skills found for task '{self.task_name}'")
            return skills
        except Exception as e:
            raise ValueError(f"Failed to get skills for task '{self.task_name}': {e}")
    
    def test_single_skill(self, skill_name: str, steps: int = 250) -> SkillTestResult:
        """
        Test a single skill by running it for the specified number of steps.
        
        Args:
            skill_name: Name of the skill to test
            steps: Number of steps to run (default: 250)
            
        Returns:
            SkillTestResult with success/failure information
        """
        print(f"\n{'='*60}")
        print(f"🧪 Testing skill: {skill_name}")
        print(f"{'='*60}")
        
        is_primitive = self.orchestrator.is_skill_primitive(skill_name)
        skill_type = "Primitive" if is_primitive else "Composite"
        print(f"Type: {skill_type}")
        if not is_primitive:
            print("🎲 Using random policies for L0 sub-skills (testing mode)")
        print(f"Steps: {steps}")
        
        # Show the exact command that will be run for debugging
        command = self.orchestrator.build_simple_training_command(
            skill_name=skill_name,
            steps=steps,
            record_video=False,
            num_envs=512,
            seed=42,
            use_random_policies=not is_primitive
        )
        print(f"🔧 Full Command: {command}")
        print(f"🔧 IsaacLab part: {command.split('&&')[-1].strip()}")  # Show just the isaaclab.sh part
        
        start_time = time.time()
        
        try:
            # Run the command ourselves so we can capture output and traceback
            completed = subprocess.run(
                command,
                shell=True,
                executable='/bin/bash',
                cwd=str(self.isaaclab_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )

            execution_time = time.time() - start_time

            if completed.returncode == 0:
                print(f"✅ {skill_name} completed successfully in {execution_time:.1f}s")
                return SkillTestResult(
                    skill_name=skill_name,
                    is_primitive=is_primitive,
                    success=True,
                    execution_time=execution_time
                )

            # Extract traceback (if any) from captured output; include tail for context
            output_text = completed.stdout or ""
            traceback_start = output_text.rfind("Traceback (most recent call last):")
            extracted_tb = output_text[traceback_start:] if traceback_start != -1 else None
            tail_len = 8000  # keep a reasonable chunk for debugging
            tail_text = output_text[-tail_len:] if output_text else None

            print(f"❌ {skill_name} failed during execution (exit code {completed.returncode})")
            return SkillTestResult(
                skill_name=skill_name,
                is_primitive=is_primitive,
                success=False,
                error_message=f"Training process returned failure (exit code {completed.returncode})",
                error_traceback=extracted_tb,
                process_output_tail=tail_text,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            print(f"❌ {skill_name} failed with exception: {error_msg}")
            return SkillTestResult(
                skill_name=skill_name,
                is_primitive=is_primitive,
                success=False,
                error_message=error_msg,
                error_traceback=traceback.format_exc(),
                process_output_tail=None,
                execution_time=execution_time
            )
    
    def test_all_skills(self, steps: int = 250) -> List[SkillTestResult]:
        """
        Test all skills in the task.
        
        Args:
            steps: Number of steps to run each skill (default: 250)
            
        Returns:
            List of SkillTestResult objects
        """
        # Check conda environment
        if not self.check_conda_environment():
            print("Consider running: conda activate env_isaaclab")
            print("Continuing anyway...\n")
        
        # Get list of skills
        try:
            skills = self.get_skills_to_test()
        except ValueError as e:
            print(f"❌ Error: {e}")
            return []
        
        print(f"\n🎯 Testing task: {self.task_name} (Robot: {self.robot})")
        print(f"📋 Found {len(skills)} skills to test")
        print(f"⚡ Each skill will run for {steps} steps")
        
        # List all skills that will be tested
        print(f"\nSkills to test:")
        for i, skill in enumerate(skills, 1):
            skill_type = "[P]" if self.orchestrator.is_skill_primitive(skill) else "[C]"
            print(f"  {i:2d}. {skill} {skill_type}")
        
        print(f"\n🚀 Starting skill testing...")
        
        # Test each skill
        self.test_results = []
        for i, skill in enumerate(skills, 1):
            print(f"\n[{i}/{len(skills)}]", end=" ")
            result = self.test_single_skill(skill, steps)
            self.test_results.append(result)
            
            # Brief pause between tests
            time.sleep(1)
        
        return self.test_results
    
    def print_summary(self) -> None:
        """Print a summary of the test results."""
        if not self.test_results:
            print("No test results to summarize.")
            return
        
        successful_skills = [r for r in self.test_results if r.success]
        failed_skills = [r for r in self.test_results if not r.success]
        
        # Count by type
        successful_primitives = [r for r in successful_skills if r.is_primitive]
        successful_composites = [r for r in successful_skills if not r.is_primitive]
        failed_primitives = [r for r in failed_skills if r.is_primitive]
        failed_composites = [r for r in failed_skills if not r.is_primitive]
        
        total_time = sum(r.execution_time for r in self.test_results)
        
        print(f"\n{'='*80}")
        print(f"📊 SKILL EXECUTION TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Task: {self.task_name} (Robot: {self.robot})")
        print(f"Total skills tested: {len(self.test_results)}")
        print(f"Total execution time: {total_time:.1f} seconds")
        print()
        
        # Success summary
        print(f"✅ SUCCESSFUL SKILLS: {len(successful_skills)}/{len(self.test_results)}")
        if successful_primitives:
            print(f"   Primitives ({len(successful_primitives)}):")
            for result in successful_primitives:
                print(f"     • {result.skill_name} ({result.execution_time:.1f}s)")
        if successful_composites:
            print(f"   Composites ({len(successful_composites)}):")
            for result in successful_composites:
                print(f"     • {result.skill_name} ({result.execution_time:.1f}s)")
        
        # Failure summary
        if failed_skills:
            print(f"\n❌ FAILED SKILLS: {len(failed_skills)}/{len(self.test_results)}")
            if failed_primitives:
                print(f"   Primitives ({len(failed_primitives)}):")
                for result in failed_primitives:
                    error_preview = result.error_message[:50] + "..." if result.error_message and len(result.error_message) > 50 else result.error_message
                    print(f"     • {result.skill_name} - {error_preview}")
            if failed_composites:
                print(f"   Composites ({len(failed_composites)}):")
                for result in failed_composites:
                    error_preview = result.error_message[:50] + "..." if result.error_message and len(result.error_message) > 50 else result.error_message
                    print(f"     • {result.skill_name} - {error_preview}")
        else:
            print(f"\n🎉 ALL SKILLS EXECUTED SUCCESSFULLY!")
        
        # Success rate
        success_rate = len(successful_skills) / len(self.test_results) * 100
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        if failed_skills:
            print(f"\n🔍 Skills that need attention:")
            for result in failed_skills:
                print(f"   • {result.skill_name}")

            # Print detailed tracebacks to aid debugging
            print(f"\n🧯 Detailed error tracebacks:")
            for result in failed_skills:
                print(f"\n--- {result.skill_name} ---")
                if result.error_traceback:
                    print(result.error_traceback)
                else:
                    print("No Python exception traceback was captured.")
                    if result.error_message:
                        print(f"Message: {result.error_message}")
                    if result.process_output_tail:
                        print("\nOutput tail (last lines):\n")
                        print(result.process_output_tail)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test execution of all skills within a GenHRL task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all skills in a task with 250 steps each
  python -m genhrl.testing.test_skill_execution Create_Steps
  
  # Test with custom number of steps
  python -m genhrl.testing.test_skill_execution Create_Steps --steps 500
  
  # Test with different robot
  python -m genhrl.testing.test_skill_execution Create_Steps --robot H1
  
  # Use custom IsaacLab path
  python -m genhrl.testing.test_skill_execution Create_Steps --isaaclab-path /custom/path
  
Note: Make sure to activate the conda environment first:
  conda activate env_isaaclab
        """
    )
    
    parser.add_argument(
        "task_name", 
        help="Name of the task to test"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=250,
        help="Number of steps to run each skill (default: 250)"
    )
    parser.add_argument(
        "--robot", 
        default="G1",
        help="Robot type (default: G1). Available: G1, H1, Anymal_B, Anymal_C, Anymal_D, A1, Go1, Go2, Spot, Digit, Franka, UR10"
    )
    parser.add_argument(
        "--isaaclab-path", 
        default="./IsaacLab",
        help="Path to IsaacLab installation (default: ./IsaacLab)"
    )
    
    args = parser.parse_args()
    
    # Resolve IsaacLab path
    isaaclab_path = Path(args.isaaclab_path).resolve()
    
    try:
        # Create tester
        tester = SkillExecutionTester(
            isaaclab_path=str(isaaclab_path),
            task_name=args.task_name,
            robot=args.robot
        )
        
        # Run tests
        results = tester.test_all_skills(steps=args.steps)
        
        # Print summary
        tester.print_summary()
        
        # Exit with error code if any skills failed
        failed_count = sum(1 for r in results if not r.success)
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} skill(s) failed execution testing.")
            sys.exit(1)
        else:
            print(f"\n🎉 All skills passed execution testing!")
            sys.exit(0)
            
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Testing interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()