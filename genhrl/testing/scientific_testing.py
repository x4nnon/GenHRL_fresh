#!/usr/bin/env python3
"""
Scientific Testing Framework for GenHRL

This script generates multiple tasks across different seeds for scientific evaluation.
It creates 5 different tasks, each generated with 3 different seeds to test generation consistency and robustness.

The script:
1. Cleans only the specific task directories that will be regenerated (preserves other existing tasks)
2. Generates tasks with different seeds for reproducibility testing
3. Automatically registers all generated skills at the end for training

Usage:
    python -m genhrl.testing.scientific_testing --api-key YOUR_KEY [options]
"""

import argparse
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add the parent directory to path to import genhrl modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genhrl.generation import TaskManager


@dataclass
class TaskDefinition:
    """Definition of a task for testing."""
    name: str
    description: str
    max_hierarchy_levels: int = 3
    expected_skills: Optional[List[str]] = None  # Optional: expected skills for validation


@dataclass
class TestResult:
    """Result of a single test run."""
    task_name: str
    seed: int
    generation_success: bool
    generation_time: float
    generated_skills: List[str]
    error_message: Optional[str] = None


class ScientificTester:
    """Main testing class for scientific evaluation of GenHRL."""
    
    def __init__(self, isaaclab_path: str, api_key: str, robot: str = "G1", provider: str = "google"):
        self.isaaclab_path = isaaclab_path
        self.api_key = api_key
        self.robot = robot
        self.provider = provider
        self.results: List[TestResult] = []
        
        # Create results directory
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup skills base path for cleaning
        self._setup_skills_path()
        
    def define_test_tasks(self) -> List[TaskDefinition]:
        """Define the test tasks. Modify descriptions here for your experiments."""
        
        # TODO: Fill in your task descriptions here
        tasks = [
            TaskDefinition(
                name="obstacle_course",
                description=""" Environment set-up description: The environment should have five objects in it. A wide low wall, 5m in the y-axis and 0.5m in the z and 0.3m in x axis. A large sphere 1m radius. A wide high wall, 5m in the y axis 1m in the z axis and 0.3m in the x axis. A small sphere 0.2m radius. and a block cube of 0.5m cubed. These objects should be positioned along the x axis in a line, with 3m separating each. The order should be: 1) Low wall, 2) large sphere, 3) high wall, 4) small sphere and 5) block. The robot will start at 0,0,0 and the first object (low wall) should be 3m away in the x axis. Such that the robot meets each object sequentially like an obstacle course.  
        
        Robot task description: The robot should first walk to the low wall and then jump over it, then the robot should push the large sphere into the high wall. The sphere hitting the high wall should push it over. The robot should then walk to a small sphere and kick it away from the wall. Finally the robot should walk to a 0.5x0.5x0.5 block and jump on top of it. """, 
                max_hierarchy_levels=3
            ),
        #     TaskDefinition(
        #         name="build_stairs", 
        #         description="""Environment set-up description: The environment should have three objects in it. A small block measuring x=1m y=1m and z=0.3m. A medium block measuring x=1m y=1m and z=0.6m. A large block measuring x=1m y=1m and z=0.9m. These blocks should be positioned in a triangle each 4m from the origin (0,0,0) which is also where the robot will be.
        
        # Robot task description: The robot should  should walk to each of the blocks in turn and push them into a sensible place near the other blocks such that it forms something resembling stairs.
        #  Once the robot has built the stairs then it should jump onto each in turn so that it climbs the stairs.""", 
        #         max_hierarchy_levels=3
        #     ),
            TaskDefinition(
                name="move_three_objects",
                description=""" Environment set-up description: The environment should have four objects. three of these objects should be 0.5m cubed blocks. These blocks should be arranged in a small triangle of 1m on each side and this centre of this triangle should be 2m from the robot (the robot will be at 0,0,0). The fourth object should be a platform x=2m y=2m and z=0.001. This platform should be positioned 4m from the triangle of cubes.
        
        Robot task description: The robot should move each of the cubes from their start location onto the platform. The robot will need to move each objet individually. The robot should just push the object and not pick them up.""", 
                max_hierarchy_levels=2
            ),
            TaskDefinition(
                name="doorway_and_goal",
                description=""" Environment set-up description: The environment should consist of three objects. Two of these objects should be very heavy (1000kg) cubes with a z of 1.5m x of 0.5 and a y of 5m. These form the 'walls' and they should be positioned such that there is a doorway gap of 0.5m. I.e the y centre of both objects should be 5.5m apart. which would leave a 0.5m gap. The third object should be a small block of 0.3m cubed. This should be positioned 2m past the doorway in the y axis and 3m down in the y axis.

        Robot task description: The robot should walk through the doorway and then walk to the small block.
                """,
                max_hierarchy_levels=3
            ),
            TaskDefinition(
                name="knock_over_pillars",
                description=""" Environment set-up description: The environment should consist of five objects. Each object is identical and is a cylinder column, with z dimension of 2m and a radius of 0.3m. These five objects should be placed at equal distance from the robot approximately 4m away. 

        Robot task description: The robot should walk to each of the objects in turn and knock them over such that they are on the floor. It should treat each object as a separate skill. """,
                max_hierarchy_levels=1
            )
        ]
        
        # Validate that all descriptions are filled
        empty_descriptions = [task.name for task in tasks if not task.description.strip()]
        if empty_descriptions:
            print(f"❌ Error: Please fill in descriptions for tasks: {empty_descriptions}")
            print("Edit the task descriptions in the define_test_tasks() method.")
            sys.exit(1)
            
        return tasks
    
    def _setup_skills_path(self) -> None:
        """Setup the skills base path for cleaning and registration."""
        try:
            # Import robot configuration helpers
            sys.path.append(str(Path(__file__).parent.parent / "generation"))
            from robot_configs import get_robot_folder_name
            robot_folder = get_robot_folder_name(self.robot)
        except ImportError:
            # Fallback for backward compatibility
            robot_folder = f"{self.robot}_generated"
        
        self.robot_folder = robot_folder
        self.skills_base_path = Path(self.isaaclab_path) / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "skills"
    
    def clean_specific_skills(self, task_names_to_clean: List[str]) -> None:
        """Clean only the specific task directories that will be regenerated."""
        if not self.skills_base_path.exists():
            print(f"📁 Skills folder does not exist yet: {self.skills_base_path}")
            print("   Will be created during task generation")
            return
            
        print(f"🧹 Cleaning specific tasks from skills folder: {self.skills_base_path}")
        
        # Get list of existing task directories
        existing_tasks = [d.name for d in self.skills_base_path.iterdir() if d.is_dir() and not d.name.startswith('__')]
        
        # Find tasks to remove (those that match our task names)
        tasks_to_remove = [task for task in existing_tasks if task in task_names_to_clean]
        tasks_to_keep = [task for task in existing_tasks if task not in task_names_to_clean]
        
        if tasks_to_remove:
            print(f"   🗑️  Removing {len(tasks_to_remove)} task directories: {tasks_to_remove}")
            for task_name in tasks_to_remove:
                task_dir = self.skills_base_path / task_name
                if task_dir.exists() and task_dir.is_dir():
                    shutil.rmtree(task_dir)
                    print(f"      Removed: {task_name}")
        else:
            print("   ℹ️  No existing tasks found that match the tasks to be generated")
            
        if tasks_to_keep:
            print(f"   ✅ Preserving {len(tasks_to_keep)} existing task directories: {tasks_to_keep}")
        
        # Also clean any skill_library.json that might contain references to removed tasks
        skill_library_path = self.skills_base_path / "skill_library.json"
        if skill_library_path.exists() and tasks_to_remove:
            print(f"   🔄 Updating skill_library.json to remove references to cleaned tasks")
            try:
                with open(skill_library_path, 'r') as f:
                    skill_library = json.load(f)
                
                # Remove hierarchies for cleaned tasks
                if "hierarchies" in skill_library:
                    removed_hierarchies = []
                    for task_name in tasks_to_remove:
                        if task_name in skill_library["hierarchies"]:
                            del skill_library["hierarchies"][task_name]
                            removed_hierarchies.append(task_name)
                    
                    if removed_hierarchies:
                        # Save updated skill library
                        with open(skill_library_path, 'w') as f:
                            json.dump(skill_library, f, indent=2)
                        print(f"      Removed {len(removed_hierarchies)} task hierarchies from skill_library.json")
                        
            except Exception as e:
                print(f"   ⚠️  Warning: Could not update skill_library.json: {e}")
        
        print("   ✅ Specific task cleanup completed")
    
    def register_all_skills(self) -> None:
        """Register all generated skills using the register_all_skills script."""
        print(f"\n🔧 Registering all generated skills...")
        
        try:
            # Run the register_all_skills script
            script_path = Path(__file__).parent.parent / "scripts" / "register_all_skills.py"
            
            cmd = [
                sys.executable, "-m", "genhrl.scripts.register_all_skills",
                "--isaaclab-path", self.isaaclab_path,
                "--robot", self.robot
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            
            if result.returncode == 0:
                print("✅ Skills registration completed successfully!")
                # Print the output from the registration script
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        print(f"   {line}")
            else:
                print("❌ Skills registration failed!")
                if result.stderr:
                    print("Error output:")
                    for line in result.stderr.strip().split('\n'):
                        print(f"   {line}")
                if result.stdout:
                    print("Standard output:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"   {line}")
                        
        except Exception as e:
            print(f"❌ Error running skills registration: {e}")
    
    def generate_task_with_seed(self, task_def: TaskDefinition, seed: int) -> TestResult:
        """Generate a single task with a specific seed."""
        
        # Create task name with seed suffix
        task_name_with_seed = f"{task_def.name}_seed{seed}"
        
        print(f"\n🧪 Testing: {task_name_with_seed}")
        print(f"📝 Description: {task_def.description}")
        print(f"🌱 Seed: {seed}")
        
        result = TestResult(
            task_name=task_name_with_seed,
            seed=seed,
            generation_success=False,
            generation_time=0.0,
            generated_skills=[]
        )
        
        try:
            # Task Generation
            print(f"🚀 Starting task generation...")
            start_time = time.time()
            
            task_manager = TaskManager(self.isaaclab_path, self.api_key, robot=self.robot)
            task_config = task_manager.create_task_from_description(
                task_name_with_seed, 
                task_def.description, 
                max_hierarchy_levels=task_def.max_hierarchy_levels,
                robot=self.robot
            )
            
            result.generation_time = time.time() - start_time
            result.generation_success = True
            
            # Extract generated skills from task config
            try:
                # TODO: Extract skills directly from task_config without TrainingOrchestrator
                # For now, we'll extract from the filesystem or task config
                skills_path = task_config.get_skills_base_path() / 'skills'
                if skills_path.exists():
                    skill_dirs = [d.name for d in skills_path.iterdir() if d.is_dir()]
                    result.generated_skills = skill_dirs
                    print(f"✅ Generated {len(skill_dirs)} skills: {skill_dirs}")
                else:
                    result.generated_skills = []
                    print(f"⚠️  Skills directory not found: {skills_path}")
            except Exception as e:
                print(f"⚠️  Could not extract skill list: {e}")
                result.generated_skills = []
            
            print(f"✅ Task generation completed in {result.generation_time:.2f}s")
            
        except Exception as e:
            result.generation_time = time.time() - start_time
            result.error_message = str(e)
            print(f"❌ Task generation failed: {e}")
            return result
        
        return result
    

    
    def run_full_experiment(self, seeds: List[int] = [42, 123, 456]) -> None:
        """Run the full scientific experiment."""
        
        print("🧬 Starting GenHRL Scientific Testing Framework")
        print("=" * 60)
        print(f"🤖 Robot: {self.robot}")
        print(f"🌱 Seeds: {seeds}")
        print(f"🎯 Focus: Task Generation Only")
        print("=" * 60)
        
        tasks = self.define_test_tasks()
        
        # Generate list of all task names that will be created
        task_names_to_generate = []
        for task_def in tasks:
            for seed in seeds:
                task_name_with_seed = f"{task_def.name}_seed{seed}"
                task_names_to_generate.append(task_name_with_seed)
        
        print(f"📋 Will generate {len(task_names_to_generate)} tasks: {task_names_to_generate}")
        
        # Clean only the specific tasks that will be regenerated
        self.clean_specific_skills(task_names_to_generate)
        
        total_experiments = len(tasks) * len(seeds)
        experiment_count = 0
        
        for task_def in tasks:
            print(f"\n📋 Task: {task_def.name}")
            print(f"📝 Description: {task_def.description}")
            print(f"🏗️  Max hierarchy levels: {task_def.max_hierarchy_levels}")
            
            for seed in seeds:
                experiment_count += 1
                print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
                
                # Generate task
                result = self.generate_task_with_seed(task_def, seed)
                
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
        
        # Final summary
        self.print_summary()
        
        # Register all generated skills
        self.register_all_skills()
        
    def save_results(self) -> None:
        """Save results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"scientific_test_results_{timestamp}.json"
        
        # Convert results to dict for JSON serialization
        results_data = {
            "metadata": {
                "robot": self.robot,
                "provider": self.provider,
                "isaaclab_path": self.isaaclab_path,
                "timestamp": timestamp,
                "total_experiments": len(self.results),
                "focus": "generation_only"
            },
            "results": [
                {
                    "task_name": r.task_name,
                    "seed": r.seed,
                    "generation_success": r.generation_success,
                    "generation_time": r.generation_time,
                    "generated_skills": r.generated_skills,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"💾 Results saved to: {results_file}")
    
    def print_summary(self) -> None:
        """Print a summary of all test results."""
        print("\n" + "=" * 80)
        print("📊 SCIENTIFIC TESTING SUMMARY")
        print("=" * 80)
        
        if not self.results:
            print("No results to summarize.")
            return
        
        # Overall statistics
        total_experiments = len(self.results)
        generation_successes = sum(1 for r in self.results if r.generation_success)
        
        print(f"Total experiments: {total_experiments}")
        print(f"Generation success rate: {generation_successes}/{total_experiments} ({100*generation_successes/total_experiments:.1f}%)")
        
        # Average times
        gen_times = [r.generation_time for r in self.results if r.generation_success]
        
        if gen_times:
            print(f"Average generation time: {sum(gen_times)/len(gen_times):.2f}s")
        
        # Per-task breakdown
        print("\n📋 Per-task breakdown:")
        task_names = set(r.task_name.rsplit('_seed', 1)[0] for r in self.results)
        
        for task_name in sorted(task_names):
            task_results = [r for r in self.results if r.task_name.startswith(task_name)]
            gen_success = sum(1 for r in task_results if r.generation_success)
            
            print(f"  {task_name}: {gen_success}/{len(task_results)} generation success")
            
            # Show skill consistency across seeds
            skills_per_seed = {}
            for r in task_results:
                if r.generation_success:
                    skills_per_seed[r.seed] = set(r.generated_skills)
            
            if len(skills_per_seed) > 1:
                # Check consistency
                all_skills = set()
                for skills in skills_per_seed.values():
                    all_skills.update(skills)
                
                common_skills = set.intersection(*skills_per_seed.values()) if skills_per_seed else set()
                print(f"    Skill consistency: {len(common_skills)}/{len(all_skills)} skills common across seeds")
        
        # Errors
        errors = [r for r in self.results if r.error_message]
        if errors:
            print(f"\n❌ Errors encountered in {len(errors)} experiments:")
            for r in errors:
                print(f"  {r.task_name}: {r.error_message}")


def main():
    """Main entry point for scientific testing."""
    parser = argparse.ArgumentParser(
        description="GenHRL Scientific Testing Framework - Generates tasks, cleans specific task directories, and registers all skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Clean only the specific task directories that will be regenerated (preserves other existing tasks)
2. Generate multiple tasks with different seeds for reproducibility testing  
3. Automatically register all generated skills at the end for training

Examples:
  # Set API key as environment variable
  export GENHRL_API_KEY=your_key_here
  python -m genhrl.testing.scientific_testing
  
  # Or pass API key directly
  python -m genhrl.testing.scientific_testing --api-key YOUR_KEY
  
  # Use custom seeds
  python -m genhrl.testing.scientific_testing --seeds 42,123,456,789,999
  
  # Custom robot and provider
  python -m genhrl.testing.scientific_testing --robot H1 --provider anthropic
        """
    )
    
    # Arguments
    parser.add_argument("--api-key", help="API key for LLM services (or set GENHRL_API_KEY env var)")
    parser.add_argument("--isaaclab-path", default="./IsaacLab", help="Path to IsaacLab installation")
    parser.add_argument("--robot", default="G1", help="Robot type to use")
    parser.add_argument("--provider", choices=["google", "anthropic"], default="google", help="LLM provider")
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated list of seeds")

    
    args = parser.parse_args()
    
    # Get API key with fallback to environment variable
    api_key = args.api_key or os.getenv("GENHRL_API_KEY")
    if not api_key:
        print("Error: API key required. Use --api-key or set GENHRL_API_KEY environment variable")
        sys.exit(1)
    
    # Parse seeds
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: Seeds must be comma-separated integers")
        sys.exit(1)
    
    # Verify IsaacLab path
    isaaclab_path = Path(args.isaaclab_path).resolve()
    if not isaaclab_path.exists():
        print(f"Error: IsaacLab path does not exist: {isaaclab_path}")
        sys.exit(1)
    
    # Create tester
    tester = ScientificTester(
        isaaclab_path=str(isaaclab_path),
        api_key=api_key,
        robot=args.robot,
        provider=args.provider
    )
    
    # Run experiments
    tester.run_full_experiment(seeds=seeds)
    
    print(f"\n🎉 Scientific testing complete!")
    print(f"📁 Results saved in: {tester.results_dir}")
    print(f"🔧 All generated skills have been registered for training!")
    print(f"✅ Existing tasks in skills folder were preserved (only regenerated tasks were cleaned)")


if __name__ == "__main__":
    main()