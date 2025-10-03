import os
import sys
import json
import time
import uuid
import subprocess
import traceback
import select
import fcntl
import signal
import atexit
import psutil
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import shutil
from datetime import datetime
from threading import Thread
import queue

app = Flask(__name__)
CORS(app)

# Store active SSE connections and their message queues
progress_connections = {}
progress_queues = {}
# Store active subprocesses for cancellation
active_processes = {}
# Store current progress status for polling
progress_status = {}

# Store active training sessions
active_training_sessions = {}
training_progress_queues = {}
training_progress_connections = {}
training_progress_status = {}

# Configuration
ISAACLAB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../IsaacLab'))
GENHRL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Track sessions where the user requested an early finish (advance to next skill)
early_finish_sessions: set[str] = set()

# === CLEANUP FUNCTIONS ===

def cleanup_isaac_sim_processes():
    """Clean up all Isaac Sim related processes more reliably."""
    print("[DEBUG] Reliably cleaning up Isaac Sim processes...")
    
    killed_count = 0
    # Use GENHRL_PATH to be specific to this project's Isaac Lab instance
    genhrl_path_str = str(GENHRL_PATH)

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            cwd = proc.info['cwd'] if proc.info['cwd'] else ''

            # Target processes running isaaclab.sh from our project directory
            # This is a much safer anchor than broad keywords.
            is_our_isaac_process = ('isaaclab.sh' in cmdline and genhrl_path_str in cwd)

            if is_our_isaac_process:
                print(f"[DEBUG] Found target Isaac Sim process {proc.info['pid']}: {cmdline[:100]} in {cwd}")
                try:
                    parent = psutil.Process(proc.info['pid'])
                    # Kill the entire process group started by the shell script
                    pgid = os.getpgid(parent.pid)
                    print(f"[DEBUG] Killing process group {pgid} for PID {parent.pid}")
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(1) # Give it a moment to die gracefully
                except (psutil.NoSuchProcess, ProcessLookupError):
                    # Process might have died in the meantime
                    pass
                except Exception as e:
                    print(f"[ERROR] Failed to kill process group for PID {proc.info['pid']}: {e}")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print(f"[DEBUG] Killed {killed_count} Isaac Sim process groups.")

def cleanup_training_processes():
    """Clean up all active training processes"""
    print("[DEBUG] Cleaning up active training processes...")
    
    # Clean up tracked active processes
    for session_id, process in list(active_processes.items()):
        try:
            if process and process.poll() is None:  # Process is still running
                print(f"[DEBUG] Terminating training process for session {session_id}")
                try:
                    # Kill process group
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    # Fallback to individual process kill
                    process.terminate()
                    time.sleep(1)
                    if process.poll() is None:
                        process.kill()
        except Exception as e:
            print(f"[DEBUG] Error cleaning up process for session {session_id}: {e}")
    
    # Clear the active processes dict
    active_processes.clear()

def cleanup_all():
    """Comprehensive cleanup of all processes and resources"""
    print("[DEBUG] Performing comprehensive cleanup...")
    
    try:
        # Clean up training processes first
        cleanup_training_processes()
        
        # Then clean up any remaining Isaac Sim processes
        cleanup_isaac_sim_processes()
        
        # Clear all connection tracking
        progress_connections.clear()
        progress_queues.clear()
        training_progress_connections.clear()
        training_progress_queues.clear()
        active_training_sessions.clear()
        
        print("[DEBUG] Cleanup completed")
    except Exception as e:
        print(f"[ERROR] Error during cleanup: {e}")

def cleanup_memory_between_skills():
    """Comprehensive memory cleanup to prevent corruption between skill transitions."""
    print("[DEBUG] Performing memory cleanup between skills...")
    
    try:
        # Force garbage collection
        import gc
        collected = gc.collect()
        print(f"[DEBUG] Garbage collection: {collected} objects collected")
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[DEBUG] CUDA cache cleared")
        except ImportError:
            pass
        
        # Clear any remaining Isaac Sim processes more aggressively
        cleanup_isaac_sim_processes()
        
        # Wait a moment for processes to fully terminate
        time.sleep(2)
        
        # Additional cleanup for any remaining GPU processes
        try:
            import subprocess
            # Kill any remaining nvidia-smi processes that might be hanging
            subprocess.run(['pkill', '-f', 'nvidia-smi'], capture_output=True)
        except Exception:
            pass
        
        print("[DEBUG] Memory cleanup completed")
        
    except Exception as e:
        print(f"[ERROR] Error during memory cleanup: {e}")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print(f"[DEBUG] Received signal {sig}, cleaning up...")
    cleanup_all()
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup_all)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_task_path(robot, task_name):
    """Get the path to a task directory"""
    robot_folder = f"{robot}_generated"
    return os.path.join(ISAACLAB_PATH, 'source/isaaclab_tasks/isaaclab_tasks/manager_based', 
                       robot_folder, 'tasks', task_name)

def get_skills_path(robot, task_name):
    """Get the path to a task's skills directory"""
    robot_folder = f"{robot}_generated"
    return os.path.join(ISAACLAB_PATH, 'source/isaaclab_tasks/isaaclab_tasks/manager_based', 
                       robot_folder, 'skills', task_name)

def path_exists(path):
    """Check if a path exists"""
    return os.path.exists(path)

def read_json(file_path):
    """Read JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def read_text(file_path):
    """Read text file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except:
        return None

def get_skill_details(skill_path, skill_name):
    """Get detailed information about a skill"""
    skill_data = {
        'name': skill_name,
        'path': skill_path
    }
    
    # Read reward configuration
    rewards_path = os.path.join(skill_path, 'TaskRewardsCfg.py')
    if path_exists(rewards_path):
        skill_data['rewards'] = read_text(rewards_path)
    
    # Read success criteria
    success_path = os.path.join(skill_path, 'SuccessTerminationCfg.py')
    if path_exists(success_path):
        skill_data['success'] = read_text(success_path)
    
    # Read skill config
    config_path = os.path.join(skill_path, f"{skill_name.lower()}_cfg.py")
    if path_exists(config_path):
        skill_data['config'] = read_text(config_path)
    
    # Check for training data/checkpoints
    checkpoints_path = os.path.join(skill_path, 'checkpoints')
    skill_data['hasCheckpoints'] = path_exists(checkpoints_path)
    
    # Check for success states
    success_states_path = os.path.join(skill_path, 'success_states')
    skill_data['hasSuccessStates'] = path_exists(success_states_path)
    if skill_data['hasSuccessStates']:
        try:
            success_files = os.listdir(success_states_path)
            skill_data['successStatesCount'] = len([f for f in success_files if f.endswith('.pt')])
        except:
            skill_data['successStatesCount'] = 0
    
    return skill_data

# API Routes

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks for a robot"""
    try:
        robot = request.args.get('robot', 'G1')
        robot_folder = f"{robot}_generated"
        tasks_path = os.path.join(ISAACLAB_PATH, 'source/isaaclab_tasks/isaaclab_tasks/manager_based', 
                                 robot_folder, 'tasks')
        
        if not path_exists(tasks_path):
            return jsonify([])
        
        task_dirs = os.listdir(tasks_path)
        tasks = []
        
        for task_dir in task_dirs:
            task_path = os.path.join(tasks_path, task_dir)
            description_file = os.path.join(task_path, 'description.txt')
            
            if path_exists(description_file):
                description = read_text(description_file)
                skills_hierarchy = read_json(os.path.join(task_path, 'skills_hierarchy.json'))
                object_config = read_json(os.path.join(task_path, 'object_config.json'))
                
                # Get creation time
                try:
                    created_at = datetime.fromtimestamp(os.path.getctime(task_path)).isoformat()
                except:
                    created_at = datetime.now().isoformat()
                
                tasks.append({
                    'name': task_dir,
                    'description': description,
                    'skillsHierarchy': skills_hierarchy,
                    'objectConfig': object_config,
                    'robot': robot,
                    'createdAt': created_at
                })
        
        return jsonify(tasks)
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        return jsonify({'error': 'Failed to fetch tasks'}), 500

@app.route('/api/tasks/<task_name>', methods=['GET'])
def get_task_details(task_name):
    """Get detailed information about a specific task"""
    try:
        robot = request.args.get('robot', 'G1')
        
        task_path = get_task_path(robot, task_name)
        skills_base_path = get_skills_path(robot, task_name)
        
        if not path_exists(task_path):
            return jsonify({'error': 'Task not found'}), 404
        
        description = read_text(os.path.join(task_path, 'description.txt'))
        skills_hierarchy = read_json(os.path.join(task_path, 'skills_hierarchy.json'))
        object_config = read_json(os.path.join(skills_base_path, 'object_config.json'))
        skill_library = read_json(os.path.join(skills_base_path, 'skill_library.json'))
        
        # Get individual skills
        skills_path = os.path.join(skills_base_path, 'skills')
        skills = []
        
        if path_exists(skills_path):
            skill_dirs = os.listdir(skills_path)
            for skill_dir in skill_dirs:
                skill_path = os.path.join(skills_path, skill_dir)
                if os.path.isdir(skill_path):
                    skill_data = get_skill_details(skill_path, skill_dir)
                    skills.append(skill_data)
        
        return jsonify({
            'name': task_name,
            'description': description,
            'skillsHierarchy': skills_hierarchy,
            'objectConfig': object_config,
            'skillLibrary': skill_library,
            'skills': skills,
            'robot': robot,
            'paths': {
                'taskPath': task_path,
                'skillsBasePath': skills_base_path,
                'skillsPath': skills_path
            }
        })
    except Exception as e:
        print(f"Error fetching task details: {e}")
        return jsonify({'error': 'Failed to fetch task details'}), 500

@app.route('/api/tasks', methods=['POST'])
def create_task():
    """Create a new task"""
    try:
        data = request.get_json()
        task_name = data.get('taskName')
        task_description = data.get('taskDescription')
        robot = data.get('robot', 'G1')
        max_hierarchy_levels = data.get('maxHierarchyLevels', 3)
        api_key = data.get('apiKey')
        model = data.get('model', 'gemini-2.5-pro')
        backup_model = data.get('backupModel', 'gemini-2.5-flash')
        
        # Determine provider from model name (extended heuristic)
        provider = 'google' if any(k in model.lower() for k in ['gemini', 'flash']) else 'anthropic'
        
        if not all([task_name, task_description, api_key]):
            return jsonify({'error': 'Missing required fields: taskName, taskDescription, apiKey'}), 400
        
        # Generate session ID for progress tracking
        session_id = str(uuid.uuid4())
        
        # Create progress queue immediately
        progress_queues[session_id] = queue.Queue()
        print(f"[DEBUG] Created progress queue for session {session_id}")
        
        # Send initial progress message immediately (estimate 7 minimum steps, will be updated dynamically)
        send_progress(session_id, 'starting', 0, 7, 'Starting task creation...', 'Initializing task generation process')
        
        # Start task creation in a separate thread
        thread = Thread(target=create_task_async, args=(
            session_id, task_name, task_description, robot, max_hierarchy_levels, api_key, model, backup_model, provider
        ))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'sessionId': session_id,
            'message': 'Task creation started. Use the sessionId to track progress.'
        })
        
    except Exception as e:
        print(f"Error creating task: {e}")
        return jsonify({'error': 'Failed to create task'}), 500

def create_task_async(session_id, task_name, task_description, robot, max_hierarchy_levels, api_key, model, backup_model, provider):
    """Create task asynchronously with progress tracking"""
    try:
        print(f"Starting task creation for session {session_id}")
        send_progress(session_id, 'starting', 0, 7, 'Starting Python process...', 'Launching task generation subprocess')
        
        # Create Python script for task generation
        python_script = f'''
import sys
import os
import json
import time
import traceback

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Add the GenHRL path
try:
    sys.path.append('{GENHRL_PATH}')
    print("STDERR: Python path added successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"STDERR: Error adding python path: {{e}}", file=sys.stderr, flush=True)

# Progress tracking
SESSION_ID = "{session_id}"

def send_progress(stage, current_stage, total_stages, message, details=""):
    try:
        progress_data = {{
            "type": "progress",
            "sessionId": SESSION_ID,
            "stage": stage,
            "currentStage": current_stage,
            "totalStages": total_stages,
            "message": message,
            "details": details
        }}
        print(f"PROGRESS:{{json.dumps(progress_data)}}", flush=True)
        print(f"STDERR: Progress - Stage {{current_stage}}/{{total_stages}}: {{message}}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"STDERR: Progress error: {{e}}", file=sys.stderr, flush=True)

def send_completion(success, message, task_name=None, error=None):
    try:
        completion_data = {{
            "type": "complete",
            "sessionId": SESSION_ID,
            "success": success,
            "message": message,
            "taskName": task_name,
            "error": error
        }}
        print(f"COMPLETE:{{json.dumps(completion_data)}}", flush=True)
        print(f"STDERR: Completion - Success: {{success}}, Message: {{message}}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"STDERR: Completion error: {{e}}", file=sys.stderr, flush=True)

try:
    print("STDERR: ========== TASK GENERATION STARTED ==========", file=sys.stderr, flush=True)
    send_progress("initializing", 1, 7, "Initializing task generation...", "Setting up Python environment")
    time.sleep(1)
    
    # Test basic imports first
    send_progress("importing", 1, 7, "Testing imports...", "Checking if GenHRL modules are accessible")
    print("STDERR: Testing GenHRL imports...", file=sys.stderr, flush=True)
    
    try:
        from genhrl.generation.task_manager import TaskManager
        print("STDERR: TaskManager import successful", file=sys.stderr, flush=True)
        send_progress("initializing", 2, 7, "Imports successful", "GenHRL modules loaded correctly")
        time.sleep(0.5)
    except Exception as import_error:
        error_msg = f"Import failed: {{str(import_error)}}"
        print(f"STDERR: {{error_msg}}", file=sys.stderr, flush=True)
        print(f"STDERR: Traceback: {{traceback.format_exc()}}", file=sys.stderr, flush=True)
        send_completion(False, error_msg, None, str(import_error))
        sys.exit(1)
    
    # Import other required modules
    import json
    import os
    from pathlib import Path
    from genhrl.generation.robot_configs import get_robot_folder_name
    
    # Initialize TaskManager
    send_progress("setup", 2, 7, "Setting up task manager...", "Validating inputs and creating TaskManager instance")
    print("STDERR: Creating TaskManager instance...", file=sys.stderr, flush=True)
    
    try:
        task_manager = TaskManager("{ISAACLAB_PATH}", "{api_key}", "{robot}", provider="{provider}", model="{model}", backup_model="{backup_model}")
        os.environ['ROBOT_NAME'] = "{robot}"
        print("STDERR: TaskManager created successfully", file=sys.stderr, flush=True)
        send_progress("setup", 3, 7, "Task manager ready", "Environment configured successfully")
        time.sleep(0.5)
    except Exception as tm_error:
        error_msg = f"TaskManager creation failed: {{str(tm_error)}}"
        print(f"STDERR: {{error_msg}}", file=sys.stderr, flush=True)
        print(f"STDERR: Traceback: {{traceback.format_exc()}}", file=sys.stderr, flush=True)
        send_completion(False, error_msg, None, str(tm_error))
        sys.exit(1)
    
    # Step 3: Generate object configuration
    send_progress("objects", 3, 7, "Generating scene objects...", "Creating object configurations for the environment")
    print("STDERR: Generating objects configuration...", file=sys.stderr, flush=True)
    
    try:
        objects_config = task_manager.code_generator.generate_objects_config("""{task_description}""")
        print("STDERR: Objects configuration generated", file=sys.stderr, flush=True)
        send_progress("objects", 4, 7, "Objects generated", "Scene object configuration completed")
        time.sleep(0.5)
    except Exception as obj_error:
        error_msg = f"Object generation failed: {{str(obj_error)}}"
        print(f"STDERR: {{error_msg}}", file=sys.stderr, flush=True)
        send_completion(False, error_msg, None, str(obj_error))
        sys.exit(1)
    
    # Step 4: Decompose task into hierarchical skills
    send_progress("decompose", 4, 7, "Decomposing task hierarchy...", "Breaking down the task into hierarchical skills using LLM")
    print("STDERR: Starting task decomposition...", file=sys.stderr, flush=True)
    
    try:
        robot_folder = get_robot_folder_name("{robot}")
        skills_base_path = Path("{ISAACLAB_PATH}") / "source/isaaclab_tasks/isaaclab_tasks/manager_based" / robot_folder / "skills" / "{task_name}"
        object_config_path = str(skills_base_path / "object_config.json")
        
        print(f"STDERR: Skills will be created at: {{skills_base_path}}", file=sys.stderr, flush=True)
        
        skills_hierarchy = task_manager.code_generator.decompose_task(
            "{task_name}", """{task_description}""", object_config_path, objects_config, max_hierarchy_levels={max_hierarchy_levels}
        )
        
        print("STDERR: Task decomposition completed", file=sys.stderr, flush=True)
        send_progress("decompose", 5, 7, "Hierarchy created", "Task successfully decomposed into skill hierarchy")
        time.sleep(0.5)
    except Exception as decomp_error:
        error_msg = f"Task decomposition failed: {{str(decomp_error)}}"
        print(f"STDERR: {{error_msg}}", file=sys.stderr, flush=True)
        print(f"STDERR: Traceback: {{traceback.format_exc()}}", file=sys.stderr, flush=True)
        send_completion(False, error_msg, None, str(decomp_error))
        sys.exit(1)
    
    # Step 5: Create task configuration and setup directories
    send_progress("config", 5, 7, "Creating task configuration...", "Setting up task configuration and directory structure")
    print("STDERR: Creating task configuration...", file=sys.stderr, flush=True)
    
    try:
        from genhrl.generation.task_manager import TaskConfig
        task_config = TaskConfig(
            name="{task_name}",
            description="""{task_description}""",
            isaaclab_path="{ISAACLAB_PATH}",
            robot="{robot}",
            objects_config=json.loads(objects_config),
            skills_hierarchy=skills_hierarchy
        )
        
        # Create task directory and basic files
        task_path = task_config.get_task_path()
        task_path.mkdir(parents=True, exist_ok=True)
        print(f"STDERR: Created task directory: {{task_path}}", file=sys.stderr, flush=True)
        
        # Write task description
        with open(task_path / "description.txt", 'w') as f:
            f.write(task_config.description)
        print("STDERR: Wrote description.txt", file=sys.stderr, flush=True)
        
        # Write skills hierarchy
        with open(task_path / "skills_hierarchy.json", 'w') as f:
            json.dump(task_config.skills_hierarchy, f, indent=2)
        print("STDERR: Wrote skills_hierarchy.json", file=sys.stderr, flush=True)
        
        # Create task-specific skill directories
        task_skills_base = task_config.get_skills_base_path()
        task_skills_base.mkdir(parents=True, exist_ok=True)
        print(f"STDERR: Created skills directory: {{task_skills_base}}", file=sys.stderr, flush=True)
        
        # Write object configuration to both locations
        with open(task_skills_base / "object_config.json", 'w') as f:
            json.dump(task_config.objects_config, f, indent=2)
        with open(task_path / "object_config.json", 'w') as f:
            json.dump(task_config.objects_config, f, indent=2)
        print("STDERR: Wrote object_config.json to both locations", file=sys.stderr, flush=True)
        
        # Initialize skill library with task-specific JSON file
        from genhrl.generation.skill_library import SkillLibrary
        skill_library_path = task_config.get_skill_library_path()
        skill_library = SkillLibrary(str(skill_library_path))
        skill_library.add_hierarchy(
            task_config.name, 
            task_config.skills_hierarchy, 
            task_config.description
        )
        
        print("STDERR: Task configuration created", file=sys.stderr, flush=True)
        send_progress("config", 6, 7, "Configuration ready", "Task configuration object created successfully")
        time.sleep(0.5)
    except Exception as config_error:
        error_msg = f"Task configuration failed: {{str(config_error)}}"
        print(f"STDERR: {{error_msg}}", file=sys.stderr, flush=True)
        send_completion(False, error_msg, None, str(config_error))
        sys.exit(1)

    # Step 6: Generate individual skill files (dynamic progress based on skill count)
    try:
        skills_dir = task_config.get_task_skills_path()
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all skills in execution order
        full_skill_sequence = skill_library.get_full_skill_sequence("{task_name}")
        total_skills = len(full_skill_sequence)
        
        # Calculate dynamic total steps: 6 base steps + (3 steps per skill) + 1 finalize step
        dynamic_total_steps = 6 + (total_skills * 3) + 1
        
        print(f"STDERR: Found {{total_skills}} skills to generate: {{full_skill_sequence}}", file=sys.stderr, flush=True)
        print(f"STDERR: Dynamic total steps calculated: {{dynamic_total_steps}}", file=sys.stderr, flush=True)
        
        # Create skill descriptions mapping
        all_skill_descriptions = []
        for skill_name in full_skill_sequence:
            if skill_name in skill_library.skills["skills"]:
                description = skill_library.skills["skills"][skill_name].get("description", "N/A")
                all_skill_descriptions.append(description)
            else:
                all_skill_descriptions.append("N/A")
        
        # Generate files for each skill with detailed progress
        for skill_index, skill_name in enumerate(full_skill_sequence):
            skill_info = skill_library.skills["skills"].get(skill_name)
            if not skill_info:
                continue
            
            skill_path = skills_dir / skill_name
            if skill_path.exists():
                print(f"STDERR: Skill directory already exists: {{skill_name}}", file=sys.stderr, flush=True)
                continue
            
            # Calculate current skill's base step (each skill gets 3 dedicated steps)
            skill_base_step = 6 + (skill_index * 3)  # Steps start after the 6 base steps
            
            # Create skill directory structure
            skill_path.mkdir(parents=True, exist_ok=True)
            (skill_path / "success_states").mkdir(exist_ok=True)
            print(f"STDERR: Created skill directory: {{skill_path}}", file=sys.stderr, flush=True)
            
            # Step A: Generate reward functions (most time-consuming)
            send_progress("skill_rewards", skill_base_step + 1, dynamic_total_steps, 
                         f"Generating rewards for skill: {{skill_name}}", 
                         f"Creating reward functions for skill {{skill_index + 1}}/{{total_skills}}: {{skill_info.get('description', 'N/A')[:60]}}")
            print(f"STDERR: Generating rewards for skill {{skill_index + 1}}/{{total_skills}}: {{skill_name}}", file=sys.stderr, flush=True)
            
            rewards_code = task_manager.code_generator.generate_task_rewards(
                task_config.description,
                json.dumps(task_config.objects_config),
                skill_name,
                skill_info["description"],
                full_skill_sequence,
                all_skill_descriptions
            )
            
            with open(skill_path / "TaskRewardsCfg.py", 'w') as f:
                f.write(task_manager._strip_markdown(rewards_code))
            print(f"STDERR: Rewards generated for {{skill_name}}", file=sys.stderr, flush=True)
            
            # Step B: Generate success criteria
            send_progress("skill_success", skill_base_step + 2, dynamic_total_steps,
                         f"Generating success criteria for: {{skill_name}}",
                         f"Creating success conditions for skill {{skill_index + 1}}/{{total_skills}}: {{skill_info.get('description', 'N/A')[:60]}}")
            print(f"STDERR: Generating success criteria for: {{skill_name}}", file=sys.stderr, flush=True)
            
            success_code = task_manager.code_generator.generate_success_criteria(
                task_config.description,
                json.dumps(task_config.objects_config),
                skill_name,
                skill_info["description"],
                rewards_code,
                full_skill_sequence,
                all_skill_descriptions
            )
            
            # Create base success file if needed
            base_success_path = skill_path / "base_success.py"
            if not base_success_path.exists():
                try:
                    from genhrl.generation.prompts.success_pre_code import get_success_pre_code
                    with open(base_success_path, 'w') as f:
                        f.write(get_success_pre_code())
                except ImportError:
                    print("STDERR: Could not import success pre-code", file=sys.stderr, flush=True)
            
            # Write success termination configuration
            success_content = f\"\"\"
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

{{task_manager._strip_markdown(success_code)}}
\"\"\"
            
            with open(skill_path / "SuccessTerminationCfg.py", 'w') as f:
                f.write(success_content)
            print(f"STDERR: Success criteria generated for {{skill_name}}", file=sys.stderr, flush=True)
            
            # Step C: Setup configurations and templates
            send_progress("skill_config", skill_base_step + 3, dynamic_total_steps,
                         f"Setting up configurations for: {{skill_name}}",
                         f"Creating agent configs and skill templates for {{skill_index + 1}}/{{total_skills}}: {{skill_info.get('description', 'N/A')[:60]}}")
            print(f"STDERR: Setting up configurations for: {{skill_name}}", file=sys.stderr, flush=True)
            
            # Copy agents configuration templates
            agents_dst = skill_path / "agents"
            if not agents_dst.exists():
                task_manager._copy_agents_templates(agents_dst, skill_name, skill_info, task_config)
            
            # Copy skill config template
            import shutil
            template_path = Path("genhrl/generation/skill_config_template.py")
            config_path = skill_path / f"{{skill_name.lower()}}_cfg.py"
            if template_path.exists() and not config_path.exists():
                shutil.copy2(template_path, config_path)
                print(f"STDERR: Created skill config from template: {{config_path}}", file=sys.stderr, flush=True)
            
            # Ensure all necessary __init__.py files exist
            task_manager._ensure_init_files(skill_path)
            print(f"STDERR: Skill {{skill_name}} completed ({{skill_index + 1}}/{{total_skills}})", file=sys.stderr, flush=True)
        
        # Finalize
        send_progress("finalize", dynamic_total_steps, dynamic_total_steps, "Finalizing task creation...", "Adding gym registrations and final setup")
        task_manager._add_gym_registrations(task_config, skill_library)
        
        print("STDERR: ========== TASK GENERATION COMPLETED ==========", file=sys.stderr, flush=True)
        send_completion(True, "Task created successfully!", "{task_name}")
        print(f"SUCCESS: Task created at {{task_config.get_task_path()}}", flush=True)
        
    except Exception as skill_error:
        error_msg = f"Skill generation failed: {{str(skill_error)}}"
        print(f"STDERR: {{error_msg}}", file=sys.stderr, flush=True)
        print(f"STDERR: Traceback: {{traceback.format_exc()}}", file=sys.stderr, flush=True)
        send_completion(False, error_msg, None, str(skill_error))
        sys.exit(1)

except Exception as e:
    error_msg = f"Unexpected error: {{str(e)}}"
    print(f"STDERR: {{error_msg}}", file=sys.stderr, flush=True)
    print(f"STDERR: Traceback: {{traceback.format_exc()}}", file=sys.stderr, flush=True)
    send_completion(False, "Task creation failed due to unexpected error", None, error_msg)
    sys.exit(1)
'''
        
        # Start the Python process
        process = subprocess.Popen(
            ['python3', '-u', '-c', python_script],
            cwd=GENHRL_PATH,
            env={**os.environ, 'PYTHONPATH': GENHRL_PATH, 'PYTHONUNBUFFERED': '1'},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Store process for potential cancellation
        active_processes[session_id] = process
        
        stdout_lines = []
        stderr_lines = []
        completed_received = False
        
        print(f"[DEBUG] Starting real-time process monitoring for session {session_id}")
        
        # Process stdout and stderr in real-time using a non-blocking approach
        
        # Set stdout and stderr to non-blocking
        def set_nonblocking(fd):
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        set_nonblocking(process.stdout.fileno())
        set_nonblocking(process.stderr.fileno())
        
        # Process output in real-time
        while process.poll() is None:
            # Check if process was cancelled
            if session_id not in active_processes:
                print(f"[DEBUG] Process cancelled for session {session_id}")
                process.terminate()
                send_completion(session_id, False, 'Task creation cancelled', None, 'Process was cancelled by user')
                return
            
            # Use select to check for available data
            ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
            
            for stream in ready:
                try:
                    if stream == process.stdout:
                        line = stream.readline()
                        if line:
                            line = line.strip()
                            stdout_lines.append(line)
                            print(f'[DEBUG] Python stdout line: {line}')
                            
                            if line.startswith('PROGRESS:'):
                                try:
                                    progress_data = json.loads(line[9:])
                                    print(f'[DEBUG] Parsed progress data: {progress_data}')
                                    print(f'[DEBUG] Session {session_id} queue exists: {session_id in progress_queues}')
                                    send_progress(session_id, progress_data['stage'], progress_data['currentStage'], 
                                                progress_data['totalStages'], progress_data['message'], progress_data['details'])
                                    print(f'[DEBUG] Progress sent for session {session_id}')
                                except Exception as e:
                                    print(f'[ERROR] Error parsing progress data: {e}, Line: {line}')
                            elif line.startswith('COMPLETE:'):
                                try:
                                    complete_data = json.loads(line[9:])
                                    print(f'[DEBUG] Parsed completion data: {complete_data}')
                                    send_completion(session_id, complete_data['success'], complete_data['message'], 
                                                  complete_data['taskName'], complete_data['error'])
                                    completed_received = True
                                    print(f'[DEBUG] Completion sent for session {session_id}')
                                except Exception as e:
                                    print(f'[ERROR] Error parsing completion data: {e}, Line: {line}')
                            elif line.startswith('SUCCESS:'):
                                print('[DEBUG] Task creation success detected')
                            else:
                                # Forward generic stdout line to UI
                                send_generation_terminal_output(session_id, line)
                    
                    elif stream == process.stderr:
                        line = stream.readline()
                        if line:
                            line = line.strip()
                            stderr_lines.append(line)
                            print(f'[DEBUG] Python stderr line: {line}')
                            # Forward stderr line to UI as well
                            send_generation_terminal_output(session_id, line)
                    
                except Exception as e:
                    print(f'[ERROR] Error reading from stream: {e}')
            
            # Small delay to prevent busy waiting
            time.sleep(0.01)
        
        # Process any remaining output
        try:
            remaining_stdout = process.stdout.read()
            remaining_stderr = process.stderr.read()
            if remaining_stdout:
                for line in remaining_stdout.split('\n'):
                    if line.strip():
                        stdout_lines.append(line.strip())
                        print(f'[DEBUG] Final stdout line: {line.strip()}')
            if remaining_stderr:
                for line in remaining_stderr.split('\n'):
                    if line.strip():
                        stderr_lines.append(line.strip())
                        print(f'[DEBUG] Final stderr line: {line.strip()}')
        except:
            pass
        
        print(f'[DEBUG] Python process closed with code {process.returncode} for session {session_id}')
        
        # Clean up process reference
        if session_id in active_processes:
            del active_processes[session_id]
        
        # Ensure completion is sent if we haven't received one
        if not completed_received:
            stdout_text = '\n'.join(stdout_lines)
            stderr_text = '\n'.join(stderr_lines)
            
            if process.returncode == 0 and ('SUCCESS:' in stdout_text or 'Task created successfully' in stdout_text):
                send_completion(session_id, True, 'Task created successfully', task_name)
            else:
                error_message = stderr_text or f'Process exited with code {process.returncode}'
                send_completion(session_id, False, 'Task creation failed', None, error_message)
                
    except Exception as e:
        print(f'Error in task creation: {e}')
        send_completion(session_id, False, 'Failed to start Python process', None, str(e))
        # Clean up process reference
        if session_id in active_processes:
            del active_processes[session_id]

@app.route('/api/tasks/cancel/<session_id>', methods=['POST'])
def cancel_task(session_id):
    """Cancel an active task creation"""
    try:
        if session_id in active_processes:
            process = active_processes[session_id]
            process.terminate()
            
            # Wait a bit for graceful termination
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                process.kill()
                process.wait()
            
            # Clean up
            del active_processes[session_id]
            
            # Send cancellation notification
            send_completion(session_id, False, 'Task creation cancelled', None, 'Process was cancelled by user')
            
            return jsonify({'success': True, 'message': 'Task creation cancelled'})
        else:
            return jsonify({'success': False, 'message': 'No active process found for this session'}), 404
            
    except Exception as e:
        print(f"Error cancelling task: {e}")
        return jsonify({'error': 'Failed to cancel task'}), 500

@app.route('/api/robots', methods=['GET'])
def get_robots():
    """Get list of supported robots"""
    # This could be expanded to read from robot configs
    robots = ['G1', 'H1', 'Franka', 'UR10', 'Anymal_B', 'Anymal_C', 'Anymal_D', 'A1', 'Go1', 'Go2', 'Spot', 'Digit']
    return jsonify(robots)


@app.route('/api/tasks/<task_name>', methods=['DELETE'])
def delete_task(task_name):
    """Delete a task using comprehensive removal manager"""
    try:
        robot = request.args.get('robot', 'G1')
        skill_name = request.args.get('skill')  # Optional: remove specific skill only
        
        # Import removal manager
        sys.path.insert(0, GENHRL_PATH)
        from genhrl.generation.removal_manager import RemovalManager
        
        # Create removal manager
        removal_manager = RemovalManager(ISAACLAB_PATH, robot)
        
        # Perform removal without confirmation (UI handles confirmation)
        if skill_name:
            # Remove specific skill
            success = removal_manager.remove_skill(task_name, skill_name, confirm=False)
            message = f"Skill '{skill_name}' removed successfully" if success else "Failed to remove skill"
        else:
            # Remove entire task
            success = removal_manager.remove_task(task_name, confirm=False)
            message = "Task deleted successfully" if success else "Failed to delete task"
        
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        print(f"Error deleting task/skill: {e}")
        return jsonify({'error': 'Failed to delete task/skill'}), 500

@app.route('/api/tasks/<task_name>/removal-impact', methods=['GET'])
def get_removal_impact(task_name):
    """Get information about what will be removed"""
    try:
        robot = request.args.get('robot', 'G1')
        skill_name = request.args.get('skill')  # Optional: impact for specific skill only
        
        # Import removal manager
        sys.path.insert(0, GENHRL_PATH)
        from genhrl.generation.removal_manager import RemovalManager
        
        # Create removal manager
        removal_manager = RemovalManager(ISAACLAB_PATH, robot)
        
        # Get removal impact
        impact = removal_manager.get_removal_impact(task_name, skill_name)
        
        return jsonify(impact)
    except Exception as e:
        print(f"Error getting removal impact: {e}")
        return jsonify({'error': 'Failed to get removal impact'}), 500

@app.route('/api/progress/<session_id>', methods=['GET'])
def progress_stream(session_id):
    """Server-Sent Events endpoint for progress tracking"""
    print(f"[DEBUG] SSE connection requested for session {session_id}")
    
    def event_stream():
        # Use existing queue if it exists, otherwise create one
        if session_id not in progress_queues:
            progress_queues[session_id] = queue.Queue()
            print(f"[DEBUG] Created new queue for session {session_id}")
        else:
            print(f"[DEBUG] Using existing queue for session {session_id}, size: {progress_queues[session_id].qsize()}")
        
        # Store connection
        progress_connections[session_id] = True
        print(f"[DEBUG] SSE connection established for session {session_id}")
        
        # Send initial connection confirmation
        yield f"data: {json.dumps({'type': 'connected', 'sessionId': session_id})}\n\n"
        
        # Keep streaming messages
        try:
            while session_id in progress_connections:
                try:
                    # Wait for a message with timeout
                    data = progress_queues[session_id].get(timeout=30)
                    print(f"[DEBUG] Retrieved message from queue for {session_id}: {data}")
                    yield f"data: {json.dumps(data)}\n\n"
                    print(f"[DEBUG] Sent SSE message to client for {session_id}")
                    
                    # Check if this is a completion message
                    if data.get('type') == 'complete':
                        print(f"[DEBUG] Completion message sent, ending SSE for {session_id}")
                        break
                        
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    print(f"[DEBUG] Sending heartbeat for {session_id}")
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    
        except GeneratorExit:
            # Client disconnected
            print(f"[DEBUG] Client disconnected from SSE for {session_id}")
            pass
        finally:
            # Clean up
            print(f"[DEBUG] Cleaning up SSE connection for {session_id}")
            if session_id in progress_connections:
                del progress_connections[session_id]
            if session_id in progress_queues:
                del progress_queues[session_id]
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'})

@app.route('/api/progress-status/<session_id>', methods=['GET'])
def get_progress_status(session_id):
    """Get current progress status for polling"""
    if session_id in progress_status:
        return jsonify(progress_status[session_id])
    else:
        return jsonify({
            'type': 'unknown',
            'currentStage': 0,
            'totalStages': 7,  # Minimum: 6 base steps + 1 finalize step
            'message': 'No progress data available',
            'details': '',
            'timestamp': datetime.now().isoformat()
        })

def send_progress(session_id, stage, current_stage, total_stages, message, details=''):
    """Send progress update to connected clients"""
    print(f"[DEBUG] send_progress called for session {session_id}")
    print(f"[DEBUG] Available queues: {list(progress_queues.keys())}")
    print(f"[DEBUG] Progress data: stage={stage}, current={current_stage}, total={total_stages}, message={message}")
    
    # Store in progress_status for polling
    progress_status[session_id] = {
        'type': 'progress',
        'stage': stage,
        'currentStage': current_stage,
        'totalStages': total_stages,
        'message': message,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }
    print(f"[DEBUG] Stored progress status for session {session_id}: {progress_status[session_id]}")
    
    if session_id in progress_queues:
        data = {
            'type': 'progress',
            'stage': stage,
            'currentStage': current_stage,
            'totalStages': total_stages,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        try:
            progress_queues[session_id].put(data, timeout=1)
            print(f"[DEBUG] Progress update successfully queued for {session_id}")
            print(f"[DEBUG] Queue size after put: {progress_queues[session_id].qsize()}")
        except queue.Full:
            print(f"[ERROR] Progress queue full for session {session_id}")
    else:
        print(f"[ERROR] No progress queue found for session {session_id} when sending progress")

def send_completion(session_id, success, message, task_name=None, error=None):
    """Send completion notification to connected clients"""
    # Store in progress_status for polling
    progress_status[session_id] = {
        'type': 'complete',
        'success': success,
        'message': message,
        'taskName': task_name,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    print(f"[DEBUG] Stored completion status for session {session_id}: {progress_status[session_id]}")
    
    if session_id in progress_queues:
        data = {
            'type': 'complete',
            'success': success,
            'message': message,
            'taskName': task_name,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        try:
            progress_queues[session_id].put(data, timeout=1)
            print(f"Completion queued for {session_id}: {data}")
        except queue.Full:
            print(f"Progress queue full for session {session_id}")
        
        # Mark for cleanup (connection will be removed by the stream generator)
        if session_id in progress_connections:
            del progress_connections[session_id]
    else:
        print(f"No progress queue found for session {session_id} when sending completion")
    
    # Clean up progress status after completion
    if session_id in progress_status and progress_status[session_id]['type'] == 'complete':
        # Keep it for a bit so the frontend can poll it, will be cleaned up later
        pass

def send_training_terminal_output(session_id, line):
    """Send real-time terminal output to the UI"""
    # Ensure a queue exists so we never drop the message due to timing issues
    if session_id not in training_progress_queues:
        training_progress_queues[session_id] = queue.Queue()
        print(f"[DEBUG] Created training progress queue on-the-fly for session {session_id}")

    data = {
        'type': 'terminal',
        'line': line,
        'timestamp': datetime.now().isoformat()
    }
    try:
        training_progress_queues[session_id].put(data, timeout=0.1)
    except queue.Full:
        # If the queue is full we silently drop to avoid blocking the training loop
        pass  # Optionally log here if persistent drops are expected

def send_training_progress(session_id, stage, message, details):
    """Send training progress update to connected clients"""
    print(f"[DEBUG] send_training_progress called for session {session_id}: {stage} - {message}")
    
    # Store in training progress status for polling
    training_progress_status[session_id] = {
        'type': 'progress',
        'stage': stage,
        'message': message,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }
    
    if session_id in training_progress_queues:
        data = {
            'type': 'progress',
            'stage': stage,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        try:
            training_progress_queues[session_id].put(data, timeout=1)
            print(f"[DEBUG] Training progress update queued for {session_id}")
        except queue.Full:
            print(f"[ERROR] Training progress queue full for session {session_id}")
    else:
        print(f"[ERROR] No training progress queue found for session {session_id}")

def send_training_completion(session_id, success, message, error=None):
    """Send training completion notification to connected clients"""
    print(f"[DEBUG] send_training_completion called for session {session_id}: success={success}, message={message}")
    
    # Store in training progress status
    training_progress_status[session_id] = {
        'type': 'complete',
        'success': success,
        'message': message,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    
    if session_id in training_progress_queues:
        data = {
            'type': 'complete',
            'success': success,
            'message': message,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        try:
            training_progress_queues[session_id].put(data, timeout=1)
            print(f"Completion queued for training session {session_id}")
        except queue.Full:
            print(f"Training progress queue full for session {session_id}")
        
        # Mark for cleanup (connection will be removed by the stream generator)
        if session_id in training_progress_connections:
            del training_progress_connections[session_id]
    else:
        print(f"No training progress queue found for session {session_id} when sending completion")

@app.route('/api/tasks/<task_name>/skills/<skill_name>/code', methods=['PUT'])
def update_skill_code(task_name, skill_name):
    """Update the reward and success code for a specific skill"""
    try:
        robot = request.args.get('robot', 'G1')
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get the skill path
        skills_base_path = get_skills_path(robot, task_name)
        skill_path = os.path.join(skills_base_path, 'skills', skill_name)
        
        if not path_exists(skill_path):
            return jsonify({'error': f'Skill not found: {skill_name}'}), 404
        
        # Validate file types
        file_updates = {}
        
        if 'rewards' in data:
            file_updates['TaskRewardsCfg.py'] = data['rewards']
        
        if 'success' in data:
            file_updates['SuccessTerminationCfg.py'] = data['success']
        
        if 'config' in data:
            file_updates[f'{skill_name.lower()}_cfg.py'] = data['config']
        
        if not file_updates:
            return jsonify({'error': 'No valid code updates provided'}), 400
        
        # Backup existing files before writing (optional safety measure)
        backups = {}
        for filename in file_updates.keys():
            file_path = os.path.join(skill_path, filename)
            if path_exists(file_path):
                try:
                    backups[filename] = read_text(file_path)
                except:
                    pass
        
        # Write updated files
        written_files = []
        try:
            for filename, content in file_updates.items():
                file_path = os.path.join(skill_path, filename)
                
                # Validate Python syntax before writing
                try:
                    compile(content, filename, 'exec')
                except SyntaxError as e:
                    return jsonify({
                        'error': f'Syntax error in {filename}',
                        'details': f'Line {e.lineno}: {e.msg}',
                        'filename': filename
                    }), 400
                
                # Write the file
                with open(file_path, 'w') as f:
                    f.write(content)
                written_files.append(filename)
                print(f"Updated {filename} for skill {skill_name}")
        
        except Exception as e:
            # Try to restore from backups if something went wrong
            for filename in written_files:
                if filename in backups:
                    try:
                        file_path = os.path.join(skill_path, filename)
                        with open(file_path, 'w') as f:
                            f.write(backups[filename])
                    except:
                        pass
            
            return jsonify({
                'error': 'Failed to update files',
                'details': str(e)
            }), 500
        
        return jsonify({
            'success': True,
            'message': f'Successfully updated {len(written_files)} file(s)',
            'files': written_files
        })
        
    except Exception as e:
        print(f"Error updating skill code: {e}")
        return jsonify({'error': 'Failed to update skill code'}), 500

@app.route('/api/tasks/<task_name>/skills/<skill_name>', methods=['GET'])
def get_skill_code(task_name, skill_name):
    """Get the complete code for a specific skill"""
    try:
        robot = request.args.get('robot', 'G1')
        
        # Get the skill path
        skills_base_path = get_skills_path(robot, task_name)
        skill_path = os.path.join(skills_base_path, 'skills', skill_name)
        
        if not path_exists(skill_path):
            return jsonify({'error': f'Skill not found: {skill_name}'}), 404
        
        # Get skill details with complete code
        skill_data = get_skill_details(skill_path, skill_name)
        
        return jsonify(skill_data)
        
    except Exception as e:
        print(f"Error fetching skill code: {e}")
        return jsonify({'error': 'Failed to fetch skill code'}), 500

@app.route('/api/tasks/<task_name>/object_config', methods=['GET'])
def get_object_config(task_name):
    """Get the object configuration for a task"""
    try:
        robot = request.args.get('robot', 'G1')
        skills_base_path = get_skills_path(robot, task_name)
        object_config_path = os.path.join(skills_base_path, 'object_config.json')
        
        if not path_exists(object_config_path):
            return jsonify({'error': 'Object config not found'}), 404
            
        object_config = read_json(object_config_path)
        return jsonify(object_config)
        
    except Exception as e:
        print(f"Error getting object config: {e}")
        return jsonify({'error': 'Failed to get object config'}), 500

# Training API Endpoints

@app.route('/api/tasks/<task_name>/training/status', methods=['GET'])
def get_training_status(task_name):
    """Get current training status for a task"""
    try:
        robot = request.args.get('robot', 'G1')
        
        # Check if there's an active training session
        session_key = f"{robot}_{task_name}"
        if session_key in active_training_sessions:
            session_info = active_training_sessions[session_key]
            return jsonify({
                'isTraining': True,
                'sessionId': session_info['session_id'],
                'startTime': session_info['start_time'],
                'config': session_info['config'],
                'currentSkill': session_info.get('current_skill', ''),
                'progress': training_progress_status.get(session_info['session_id'], {})
            })
        
        # If no active session, get static training status
        skills_base_path = get_skills_path(robot, task_name)
        skills_path = os.path.join(skills_base_path, 'skills')
        
        if not path_exists(skills_path):
            return jsonify({'error': 'Task not found or no skills available'}), 404
        
        # Read skill library to get training order
        skill_library_path = os.path.join(skills_base_path, 'skill_library.json')
        if not path_exists(skill_library_path):
            return jsonify({'error': 'Skill library not found'}), 404
        
        skill_library = read_json(skill_library_path)
        
        # Get skill training status
        skill_status = {}
        if path_exists(skills_path):
            skill_dirs = os.listdir(skills_path)
            for skill_dir in skill_dirs:
                skill_path = os.path.join(skills_path, skill_dir)
                if os.path.isdir(skill_path):
                    # Check for policy
                    policy_path = os.path.join(skill_path, 'policy', 'agent.pt')
                    has_policy = path_exists(policy_path)
                    
                    # Check for success states
                    success_states_path = os.path.join(skill_path, 'success_states')
                    success_count = 0
                    if path_exists(success_states_path):
                        try:
                            success_files = os.listdir(success_states_path)
                            success_count = len([f for f in success_files if f.endswith('.pt')])
                        except:
                            success_count = 0
                    
                    # Determine if skill is primitive
                    is_primitive = False
                    if skill_library and 'skills' in skill_library and skill_dir in skill_library['skills']:
                        is_primitive = skill_library['skills'][skill_dir].get('is_primitive', False)
                    
                    skill_status[skill_dir] = {
                        'hasPolicy': has_policy,
                        'successStatesCount': success_count,
                        'isPrimitive': is_primitive,
                        'status': 'completed' if has_policy and success_count > 50 else 'pending'
                    }
        
        return jsonify({
            'isTraining': False,
            'skillStatus': skill_status,
            'taskName': task_name,
            'robot': robot
        })
        
    except Exception as e:
        print(f"Error getting training status: {e}")
        return jsonify({'error': 'Failed to get training status'}), 500

@app.route('/api/tasks/<task_name>/training/start', methods=['POST'])
def start_training(task_name):
    """Start training for a task"""
    try:
        # First, perform a cleanup to ensure no old processes are running
        print("[INFO] Cleaning up Isaac Sim processes before starting new training...")
        cleanup_isaac_sim_processes()

        robot = request.args.get('robot', 'G1')
        data = request.get_json() or {}
        
        # Check if already training
        session_key = f"{robot}_{task_name}"
        if session_key in active_training_sessions:
            return jsonify({'error': 'Training already in progress for this task'}), 400
        
        # Validate task exists
        task_path = get_task_path(robot, task_name)
        skills_base_path = get_skills_path(robot, task_name)
        
        if not path_exists(task_path) or not path_exists(skills_base_path):
            return jsonify({'error': 'Task not found'}), 404
        
        # Parse training configuration
        config = {
            'max_time_minutes': data.get('maxTime', 180),
            'min_success_states': data.get('minSuccessStates', 50),
            'num_envs': data.get('numEnvs', 4096),
            'seed': data.get('seed', 42),
            'new_run': data.get('newRun', False),
            'skip_complete': data.get('skipComplete', True),
            'headless': data.get('headless', True),
            'video': data.get('video', False),
            'video_interval': data.get('videoInterval', 2000),
            'video_length': data.get('videoLength', 200),
            # NEW – early-exit tuning
            'min_training_steps_primitive': data.get('minTrainingStepsPrimitive', 30000),
            'min_training_steps_composite': data.get('minTrainingStepsComposite', 50000),
            'enforce_min_steps': data.get('enforceMinSteps', True),
            'checkpoint_interval': data.get('checkpointInterval', 1000)
        }
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Ensure we don't overwrite an existing queue created earlier (e.g., by the API call)
        if session_id not in training_progress_queues:
            training_progress_queues[session_id] = queue.Queue()
            print(f"[DEBUG] Created training progress queue for session {session_id}")
        else:
            print(f"[DEBUG] Reusing existing training progress queue for session {session_id} (size={training_progress_queues[session_id].qsize()})")
        
        # Store session info
        active_training_sessions[session_key] = {
            'session_id': session_id,
            'task_name': task_name,
            'robot': robot,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'current_skill': ''
        }
        
        # Send initial progress
        send_training_progress(session_id, 'starting', 'Initializing training orchestrator...', {
            'totalSkills': 0,
            'currentSkillIndex': 0,
            'currentSkill': '',
            'skillStatus': 'initializing'
        })
        
        # Start training in background thread
        thread = Thread(target=start_training_async, args=(session_id, task_name, robot, config))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'sessionId': session_id,
            'message': 'Training started successfully'
        })
        
    except Exception as e:
        print(f"Error starting training: {e}")
        return jsonify({'error': 'Failed to start training'}), 500

@app.route('/api/tasks/<task_name>/training/cancel', methods=['POST'])
def cancel_training(task_name):
    """Cancel active training for a task"""
    try:
        robot = request.args.get('robot', 'G1')
        session_key = f"{robot}_{task_name}"
        
        if session_key not in active_training_sessions:
            return jsonify({'error': 'No active training session found'}), 404
        
        session_info = active_training_sessions[session_key]
        session_id = session_info['session_id']
        
        # Cancel the training process if it exists
        if session_id in active_processes:
            process = active_processes[session_id]
            print(f"[DEBUG] Cancelling training process for session {session_id}")
            try:
                # Kill process group first
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                # Fallback to individual process termination
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            except Exception as e:
                print(f"[DEBUG] Error terminating process: {e}")
            
            if session_id in active_processes:
                del active_processes[session_id]
        
        # Clean up any remaining Isaac Sim processes
        # cleanup_isaac_sim_processes()
        
        # Send cancellation notification
        send_training_completion(session_id, False, 'Training cancelled by user')
        
        # Clean up session
        if session_key in active_training_sessions:
            del active_training_sessions[session_key]
        
        return jsonify({'success': True, 'message': 'Training cancelled successfully'})
        
    except Exception as e:
        print(f"Error cancelling training: {e}")
        return jsonify({'error': 'Failed to cancel training'}), 500

@app.route('/api/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger cleanup of all Isaac Sim processes"""
    try:
        print("[DEBUG] Manual cleanup requested")
        cleanup_all()
        return jsonify({'success': True, 'message': 'Cleanup completed successfully'})
    except Exception as e:
        print(f"Error during manual cleanup: {e}")
        return jsonify({'error': 'Failed to perform cleanup'}), 500

def do_restart():
    """Helper function to run the restart in a background thread."""
    time.sleep(1) # Give the server a moment to return the response
    print("[INFO] Executing server restart...")
    # Clean up before restarting
    cleanup_all()
    
    # Restart the process
    python = sys.executable
    os.execv(python, [python] + sys.argv)

@app.route('/api/server/restart', methods=['POST'])
def restart_server():
    """Restart the Flask server by spawning a background thread."""
    print("[INFO] Received request to restart server...")
    try:
        thread = Thread(target=do_restart)
        thread.daemon = True
        thread.start()
        return jsonify({'message': 'Server restart initiated'}), 202
    except Exception as e:
        print(f"Error initiating server restart: {e}")
        return jsonify({'error': 'Failed to initiate server restart'}), 500

@app.route('/api/tasks/<task_name>/training/progress/<session_id>')
def training_progress_stream(task_name, session_id):
    """Stream training progress via Server-Sent Events"""
    def event_stream():
        # Store connection
        training_progress_connections[session_id] = True
        print(f"[DEBUG] Training SSE connection established for session {session_id}")
        
        # Create queue if it doesn't exist (like task generation does)
        if session_id not in training_progress_queues:
            training_progress_queues[session_id] = queue.Queue()
            print(f"[DEBUG] Created training progress queue for SSE connection {session_id}")
        
        # Send initial connection confirmation
        yield f"data: {json.dumps({'type': 'connected', 'sessionId': session_id})}\n\n"
        
        try:
            while session_id in training_progress_connections:
                try:
                    # Wait for a message with timeout (queue should always exist now)
                    data = training_progress_queues[session_id].get(timeout=30)
                    print(f"[DEBUG] Retrieved training message from queue for {session_id}: {data}")
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # Check if this is a completion message
                    if data.get('type') == 'complete':
                        print(f"[DEBUG] Training completion message sent, ending SSE for {session_id}")
                        break
                        
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    print(f"[DEBUG] Sending training heartbeat for {session_id}")
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    
        except GeneratorExit:
            print(f"[DEBUG] Client disconnected from training SSE for {session_id}")
            pass
        finally:
            # Clean up
            print(f"[DEBUG] Cleaning up training SSE connection for {session_id}")
            if session_id in training_progress_connections:
                del training_progress_connections[session_id]
            if session_id in training_progress_queues:
                del training_progress_queues[session_id]
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'})

def start_training_async(session_id, task_name, robot, config):
    """Execute training asynchronously and stream real-time output"""
    try:
        print(f'[DEBUG] Starting training async for session {session_id}')
        
        # Ensure we don't overwrite an existing queue created earlier (e.g., by the API call)
        if session_id not in training_progress_queues:
            training_progress_queues[session_id] = queue.Queue()
            print(f"[DEBUG] Created training progress queue for session {session_id}")
        else:
            print(f"[DEBUG] Reusing existing training progress queue for session {session_id} (size={training_progress_queues[session_id].qsize()})")
        
        # Get task and skills paths
        task_path = get_task_path(robot, task_name)
        skills_base_path = get_skills_path(robot, task_name)
        
        if not path_exists(task_path) or not path_exists(skills_base_path):
            send_training_completion(session_id, False, 'Task not found', 'Invalid task or skills path')
            return
        
        # Initialize the orchestrator to get training order and skill info
        sys.path.insert(0, GENHRL_PATH)
        from genhrl.training import TrainingOrchestrator
        from genhrl.training.orchestrator import TrainingConfig
        
        print("=== TRAINING INITIALIZATION ===", flush=True)
        send_training_terminal_output(session_id, "=== TRAINING INITIALIZATION ===")
        send_training_terminal_output(session_id, f"Task: {task_name}")
        send_training_terminal_output(session_id, f"Robot: {robot}")
        send_training_terminal_output(session_id, f"IsaacLab Path: {ISAACLAB_PATH}")
        
        # Create orchestrator (but we'll execute commands directly)
        orchestrator = TrainingOrchestrator(
            isaaclab_path=ISAACLAB_PATH,
            task_name=task_name,
            robot=robot
        )
        
        # Get training order
        training_order = orchestrator.get_training_order()
        
        # If user requested a fresh run, clean previous training artefacts
        if config.get('new_run', False):
            try:
                # Clean GPU memory first to prevent conflicts
                _cleanup_gpu_processes(session_id)
                
                _clean_all_training_data(robot, task_name, training_order, session_id)
                send_training_terminal_output(session_id, "✅ Previous training data cleaned (success states, policies, logs)")
            except Exception as e:
                send_training_terminal_output(session_id, f"⚠️ Error cleaning previous data: {e}")
        
        primitive_skills = [s for s in training_order if orchestrator.is_skill_primitive(s)]
        composite_skills = [s for s in training_order if not orchestrator.is_skill_primitive(s)]
        
        send_training_terminal_output(session_id, "=== STARTING TRAINING SEQUENCE ===")
        send_training_terminal_output(session_id, f"Full Training Order for {task_name}:")
        for i, skill in enumerate(training_order):
            skill_type = "[P]" if orchestrator.is_skill_primitive(skill) else "[C]"
            send_training_terminal_output(session_id, f"{i+1}. {skill} {skill_type}")
        
        # Clean previous training data if requested
        if config.get('new_run', False):
            send_training_terminal_output(session_id, "Cleaning previous training data...")
        
        send_training_terminal_output(session_id, "Starting sequential training...")
        send_training_progress(session_id, 'planning', 'Planning training sequence...', {
            'totalSkills': len(training_order),
            'primitiveCount': len(primitive_skills),
            'compositeCount': len(composite_skills)
        })
        
        # Execute training for each skill
        primitive_idx = 0
        composite_idx = 0
        
        for i, skill_name in enumerate(training_order):
            is_primitive = orchestrator.is_skill_primitive(skill_name)
            
            # Skip if already complete and user wants to skip
            if config.get('skip_complete', True):
                if check_success_states(task_name, robot, skill_name, config.get('min_success_states', 50)):
                    send_training_terminal_output(session_id, f"✅ Skipping skill '{skill_name}' - already has sufficient success states.")
                    time.sleep(1)
                    
                    # Finalize to copy policies, etc.
                    try:
                        _finalize_skill(task_name, robot, skill_name, session_id)
                        send_training_terminal_output(session_id, f"✅ Finalized pre-completed skill {skill_name}")
                    except Exception as e:
                        send_training_terminal_output(session_id, f"❌ ERROR during finalization for pre-completed skill {skill_name}: {e}")
                        send_training_completion(session_id, False, 'Training failed during skill finalization', str(e))
                        return
                        
                    continue # Move to next skill

            # If not skipping, clean its previous success states unless a full 'new_run' was requested
            # (which would have already cleaned it). This prevents stale data from causing a premature exit.
            if not config.get('new_run', False):
                _clean_skill_success_states(robot, task_name, skill_name, session_id)

            if is_primitive:
                skill_type_msg = f"PRIMITIVE ({primitive_idx+1}/{len(primitive_skills)})"
                primitive_idx += 1
            else:
                skill_type_msg = f"COMPOSITE ({composite_idx+1}/{len(composite_skills)})"
                composite_idx += 1
            
            send_training_terminal_output(session_id, f"--- Processing {skill_type_msg}: {skill_name} ---")
            send_training_progress(session_id, 'training', f'Training {skill_name}', {
                'currentSkill': skill_name,
                'skillIndex': i + 1,
                'currentSkillIndex': i + 1,
                'totalSkills': len(training_order),
                'skillType': 'primitive' if is_primitive else 'composite',
                'progressPercent': int((i / len(training_order)) * 100)
            })
            
            # Check if skill should be skipped
            min_states = config.get('min_success_states', 50)
            if config.get('skip_complete', True) and orchestrator.has_sufficient_success_states(skill_name, min_states):
                send_training_terminal_output(session_id, f"Skipping {skill_name} - already has sufficient success states")
                continue
            
            # Get success state count
            skill_dir = os.path.join(GENHRL_PATH, "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based", f"{robot}_generated", "skills", task_name, "skills", skill_name)
            success_states_dir = os.path.join(skill_dir, "success_states")
            current_success_count = 0
            if os.path.exists(success_states_dir):
                success_files = [f for f in os.listdir(success_states_dir) if f.endswith('.pt')]
                current_success_count = len(success_files)
            
            send_training_terminal_output(session_id, f"Found {current_success_count} success state files for {skill_name}")
            
            # Build training command using the same logic as orchestrator
            # Format gym task name to include task name prefix (matching orchestrator)
            task_parts = task_name.split('_')
            skill_parts = skill_name.split('_')
            formatted_task = ''.join(part.capitalize() for part in task_parts)
            formatted_skill = ''.join(part.capitalize() for part in skill_parts)
            gym_task_name_suffix = f"{formatted_task}{formatted_skill}"
            skill_level = orchestrator.get_skill_level(skill_name)
            
            if is_primitive:  # Level 0 - Primitive skills
                base_command = (
                    f"./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py "
                    f"--task Isaac-RobotFlat{gym_task_name_suffix}-v0 "
                    f"--num_envs {config['num_envs']} --seed {config['seed']} "
                    f"--checkpoint_interval {config.get('checkpoint_interval', 1000)}"
                )
            elif skill_level == 1:  # Level 1 - First level composite skills
                base_command = (
                    f"./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_l1.py "
                    f"--task Isaac-RobotComposite{gym_task_name_suffix}-v0 "
                    f"--skill_name {skill_name} "
                    f"--num_envs {config['num_envs']} --seed {config['seed']} "
                    f"--checkpoint_interval {config.get('checkpoint_interval', 1000)}"
                )
            elif skill_level == 2:  # Level 2 - Second level composite skills
                base_command = (
                    f"./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_l2.py "
                    f"--task Isaac-RobotComposite{gym_task_name_suffix}-v0 "
                    f"--skill_name {skill_name} "
                    f"--num_envs {config['num_envs']} --seed {config['seed']} "
                    f"--checkpoint_interval {config.get('checkpoint_interval', 1000)}"
                )
            else:  # Level 3+ - Higher level composite skills (fallback to L2)
                send_training_terminal_output(session_id, f"Warning: Skill {skill_name} is at level {skill_level}, using train_l2.py as fallback")
                base_command = (
                    f"./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_l2.py "
                    f"--task Isaac-RobotComposite{gym_task_name_suffix}-v0 "
                    f"--skill_name {skill_name} "
                    f"--num_envs {config['num_envs']} --seed {config['seed']} "
                    f"--checkpoint_interval {config.get('checkpoint_interval', 1000)}"
                )
            
            if config.get('video', True):
                base_command += f" --video --video_interval {config.get('video_interval', 2000)} --video_length {config.get('video_length', 200)}"
            
            if config.get('headless', True):
                base_command += " --headless"
            
            # Environment setup with conda activation and malloc fix
            object_config_path = os.path.join(skills_base_path, 'object_config.json')
            conda_activation = (
                "eval \"$(conda shell.bash hook)\" && "
                "conda activate env_isaaclab && "
            )
            malloc_fix = (
                "unset PYTORCH_CUDA_ALLOC_CONF && "
                "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 && "
            )
            full_command = (
                f"{conda_activation}"
                f"{malloc_fix}"
                f"export CARB_LOG_LEVEL=FATAL && "
                f"export OBJECT_CONFIG_PATH='{object_config_path}' && "
                f"export GENHRL_TASK_NAME='{task_name}' && "
                f"export GENHRL_ROBOT='{robot}' && "
                f"echo OBJECT_CONFIG_PATH is now set to $OBJECT_CONFIG_PATH && "
                f"echo GENHRL_TASK_NAME is now set to $GENHRL_TASK_NAME && "
                f"{base_command}"
            )
            
            send_training_terminal_output(session_id, f"Running command: {full_command}")
            
            # Execute training command directly with proper output capture
            success = execute_skill_training(session_id, task_name, robot, skill_name, full_command, config, is_primitive)
            
            if not success:
                send_training_completion(session_id, False, f'Training failed for skill: {skill_name}', 
                                       f'Skill {skill_name} did not complete successfully')
                return
            
            # Handle post-training cleanup (policy copying and success state transfer)
            send_training_terminal_output(session_id, f"Handling completion for {skill_name}...")
            try:
                _finalize_skill(task_name, robot, skill_name, session_id)
                send_training_terminal_output(session_id, f"✅ Completed cleanup for {skill_name}")
            except Exception as e:
                send_training_terminal_output(session_id, f"❌ CRITICAL ERROR during cleanup for {skill_name}: {e}")
                send_training_terminal_output(session_id, f"❌ Training cannot continue without proper cleanup - stopping training sequence")
                send_training_completion(session_id, False, f'Training failed during cleanup for skill: {skill_name}', 
                                       f'Cleanup failed for {skill_name}: {e}')
                return
                
            send_training_terminal_output(session_id, f"--- Completed {skill_name} ---")
            
            # Brief pause between skills
            time.sleep(2)
        
        send_training_terminal_output(session_id, "Training sequence completed!")
        send_training_completion(session_id, True, 'All skills trained successfully!')
                
    except Exception as e:
        print(f'[ERROR] Error in training async: {e}')
        import traceback
        traceback.print_exc()
        send_training_completion(session_id, False, 'Failed to start training process', str(e))
    finally:
        # Clean up
        if session_id in active_processes:
            del active_processes[session_id]
        
        # Clean up session
        session_key = f"{robot}_{task_name}"
        if session_key in active_training_sessions:
            del active_training_sessions[session_key]

def execute_skill_training(session_id, task_name, robot, skill_name, command, config, is_primitive):
    """Execute training for a single skill with proper output streaming and non-blocking reads."""
    import re
    import signal
    import time
    import select
    import fcntl
    
    process = None
    try:
        print(f"[DEBUG] Starting training process for {skill_name}")
        
        process = subprocess.Popen(
            command,
            shell=True,
            executable='/bin/bash',  # Use bash instead of sh for conda activation
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=ISAACLAB_PATH,
            preexec_fn=os.setsid
        )
        
        active_processes[session_id] = process
        
        # Set stdout to be non-blocking
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        
        start_time = time.time()
        max_time_seconds = config.get('max_time_minutes', 180) * 60
        min_success_states = config.get('min_success_states', 50)
        current_step, total_steps = 0, 0
        current_reward, episode_length, success_rate = None, None, None

        while True:
            if process.poll() is not None:
                print(f"[DEBUG] Process for {skill_name} exited with code {process.returncode}.")
                break 

            ready_to_read, _, _ = select.select([process.stdout], [], [], 0.2)
            
            if ready_to_read:
                output = process.stdout.read()
                if output:
                    for line in output.splitlines():
                        if not line:
                            continue
                        
                        send_training_terminal_output(session_id, line)
                        
                        # Try multiple step patterns to catch different formats
                        step_match = (
                            re.search(r'(\d+)/(\d+)', line) or 
                            re.search(r'step\s*(\d+)\s*/\s*(\d+)', line, re.IGNORECASE) or 
                            re.search(r'(\d+)\s*of\s*(\d+)', line) or
                            re.search(r'iteration\s*(\d+)\s*/\s*(\d+)', line, re.IGNORECASE) or
                            re.search(r'epoch\s*(\d+)\s*/\s*(\d+)', line, re.IGNORECASE) or
                            re.search(r'progress:\s*(\d+)\s*/\s*(\d+)', line, re.IGNORECASE) or
                            re.search(r'(\d+)\s*steps?\s*/\s*(\d+)', line, re.IGNORECASE) or
                            re.search(r'(\d+)\s*/\s*(\d+)\s*steps?', line, re.IGNORECASE) or
                            re.search(r'step\s*(\d+)\s*of\s*(\d+)', line, re.IGNORECASE) or
                            re.search(r'(\d+)\s*/\s*(\d+)\s*iterations?', line, re.IGNORECASE)
                        )
                        if step_match:
                            current_step, total_steps = int(step_match.group(1)), int(step_match.group(2))
                            print(f"[DEBUG] Step parsed for {skill_name}: {current_step}/{total_steps} from line: {line.strip()}")
                            
                            # Send progress update to UI with step information
                            estimated_time = estimate_time_remaining(start_time, current_step, total_steps)
                            progress_data = {
                                'currentSkill': skill_name,
                                'currentStep': current_step,
                                'totalSteps': total_steps,
                                'estimatedTimeRemaining': estimated_time,
                                'skillType': 'primitive' if is_primitive else 'composite'
                            }
                            print(f"[DEBUG] Sending progress update for {skill_name}: {progress_data}")
                            send_training_progress(session_id, 'training', f'Training {skill_name} - Step {current_step}/{total_steps}', progress_data)
                            
                            # Early-exit check
                            min_steps_required = (
                                config.get('min_training_steps_primitive', 6000) if is_primitive
                                else config.get('min_training_steps_composite', 10000)
                            )
                            min_steps_ok = (not config.get('enforce_min_steps', True)) or (current_step >= min_steps_required)
                            
                            if min_steps_ok:
                                if check_success_states(task_name, robot, skill_name, min_success_states):
                                    send_training_terminal_output(session_id, f"INFO: Early exit for '{skill_name}' - success states and min steps met.")
                                    return True # This will trigger the finally block to kill the process

                        # Parse reward information if available
                        reward_match = re.search(r'Reward.*?mean=([\d.-]+)', line)
                        if reward_match:
                            current_reward = float(reward_match.group(1))
                            
                        # Parse episode length if available
                        episode_match = re.search(r'Episode.*?length_mean=([\d.-]+)', line)
                        if episode_match:
                            episode_length = float(episode_match.group(1))
                            
                        # Parse success rate if available
                        success_match = re.search(r'success_rate=([\d.-]+)', line)
                        if success_match:
                            success_rate = float(success_match.group(1))
                            
                        # Send updated metrics if we have new data
                        if current_reward is not None or episode_length is not None or success_rate is not None:
                            send_training_progress(session_id, 'training', f'Training {skill_name} - Step {current_step}/{total_steps}', {
                                'currentSkill': skill_name,
                                'currentStep': current_step,
                                'totalSteps': total_steps,
                                'currentReward': current_reward,
                                'episodeLength': episode_length,
                                'successRate': success_rate,
                                'estimatedTimeRemaining': estimate_time_remaining(start_time, current_step, total_steps),
                                'skillType': 'primitive' if is_primitive else 'composite'
                            })
                        
                        # Send periodic progress updates even without step info to keep UI alive
                        if current_step == 0 and total_steps == 0 and time.time() - start_time > 30:  # After 30 seconds
                            # Send a basic progress update to show training is still active
                            send_training_progress(session_id, 'training', f'Training {skill_name} - Active', {
                                'currentSkill': skill_name,
                                'currentStep': 0,
                                'totalSteps': 0,
                                'skillType': 'primitive' if is_primitive else 'composite'
                            })
                            # Reset the timer to avoid spam
                            start_time = time.time()

            # Check for cancellation or timeout
            if session_id not in training_progress_connections:
                send_training_terminal_output(session_id, f"INFO: Training for '{skill_name}' cancelled by user.")
                return False
            
            if (time.time() - start_time) > max_time_seconds:
                send_training_terminal_output(session_id, f"ERROR: Max training time reached for '{skill_name}'.")
                return False # Let finally block handle termination
        
        # After loop: process has finished
        return_code = process.returncode
        if return_code == 0:
            return check_success_states(task_name, robot, skill_name, min_success_states)
        else:
            send_training_terminal_output(session_id, f"ERROR: Training script for '{skill_name}' failed with code {return_code}.")
            return False

    except Exception as e:
        error_msg = f"ERROR: Exception during training execution for '{skill_name}': {e}"
        send_training_terminal_output(session_id, error_msg)
        print(error_msg)
        traceback.print_exc()
        return False
    finally:
        if process and process.poll() is None:

            print(f"[DEBUG] Terminating process group for session {session_id}")
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            process.wait(timeout=5)

        
        if session_id in active_processes:
            del active_processes[session_id]
        
        # Final cleanup of any stray processes, just in case
        cleanup_isaac_sim_processes()
        
        # Additional memory cleanup to prevent corruption between skills
        cleanup_memory_between_skills()

def check_success_states(task_name, robot, skill_name, min_success_states):
    """Return True if the skill has >= min_success_states '*.pt' files in its success_states folder."""
    try:
        # Success-states folder path mirrors the orchestrator
        success_dir = os.path.join(
            ISAACLAB_PATH,
            'source', 'isaaclab_tasks', 'isaaclab_tasks', 'manager_based',
            f'{robot}_generated', 'skills',
            task_name, 'skills', skill_name, 'success_states')

        if not os.path.exists(success_dir):
            return False

        files = [f for f in os.listdir(success_dir) if f.startswith('success_states_') and f.endswith('.pt')]
        return len(files) >= min_success_states
    except Exception as e:
        print(f"[DEBUG] Error checking success states for {skill_name}: {e}")
        return False

def estimate_time_remaining(start_time, current_step, total_steps):
    """Estimate remaining training time"""
    if current_step <= 0 or total_steps <= 0:
        return None
        
    elapsed_time = time.time() - start_time
    progress_ratio = current_step / total_steps
    
    if progress_ratio <= 0:
        return None
        
    estimated_total_time = elapsed_time / progress_ratio
    remaining_time = estimated_total_time - elapsed_time
    
    # Convert to minutes and format
    remaining_minutes = int(remaining_time / 60)
    remaining_seconds = int(remaining_time % 60)
    
    return f"{remaining_minutes}m {remaining_seconds}s"

# === VIDEO AND METRICS ENDPOINTS ===

@app.route('/api/tasks/<task_name>/videos', methods=['GET'])
def get_training_videos(task_name):
    """Get list of training videos for a task."""
    try:
        robot = request.args.get('robot', 'G1')
        current_skill = request.args.get('currentSkill', None)  # New parameter for filtering by current skill
        
        # Find the latest training logs directory
        logs_base = os.path.join(ISAACLAB_PATH, 'logs', 'skrl')
        videos = []
        
        # Look for all skills in this task
        task_path = get_task_path(robot, task_name)
        if not path_exists(task_path):
            return jsonify({'videos': []})
        
        # Determine which skills to get videos for
        skill_names = []
        if current_skill:
            # Only get videos for the current skill when training is active
            skill_names = [current_skill]
        else:
            # Get skill names from skills hierarchy (fallback to all skills)
            skills_hierarchy_path = os.path.join(task_path, 'skills_hierarchy.json')
            if path_exists(skills_hierarchy_path):
                hierarchy_data = read_json(skills_hierarchy_path)
                
                # Extract all skill names from hierarchy
                def extract_skill_names(node):
                    names = []
                    if isinstance(node, dict):
                        if 'name' in node:
                            names.append(node['name'])
                        if 'children' in node:
                            for child in node['children']:
                                names.extend(extract_skill_names(child))
                    elif isinstance(node, list):
                        for item in node:
                            names.extend(extract_skill_names(item))
                    return names
                
                skill_names = extract_skill_names(hierarchy_data)
        
        # Find videos for specified skills (current skill only when training, all skills otherwise)
        for skill_name in skill_names:
            skill_log_dir = os.path.join(logs_base, skill_name.lower())
            if os.path.exists(skill_log_dir):
                # Find most recent experiment
                experiments = [d for d in os.listdir(skill_log_dir) 
                             if os.path.isdir(os.path.join(skill_log_dir, d))]
                if experiments:
                    # Sort by modification time to get most recent
                    experiments.sort(key=lambda x: os.path.getmtime(os.path.join(skill_log_dir, x)), reverse=True)
                    latest_exp = experiments[0]
                    
                    videos_dir = os.path.join(skill_log_dir, latest_exp, 'videos', 'train')
                    if os.path.exists(videos_dir):
                        video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
                        for video_file in video_files:
                            video_path = os.path.join(videos_dir, video_file)
                            video_info = {
                                'skill_name': skill_name,
                                'filename': video_file,
                                'path': video_path,
                                'relative_path': f'/api/tasks/{task_name}/videos/{skill_name}/{video_file}',
                                'created_at': os.path.getmtime(video_path),
                                'size': os.path.getsize(video_path),
                                'experiment': latest_exp
                            }
                            videos.append(video_info)
        
        # Sort videos by creation time (newest first)
        videos.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({'videos': videos})
        
    except Exception as e:
        print(f"Error getting videos: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<task_name>/videos/<skill_name>/<filename>')
def serve_video(task_name, skill_name, filename):
    """Serve a training video file."""
    try:
        robot = request.args.get('robot', 'G1')
        
        # Find the video file
        logs_base = os.path.join(ISAACLAB_PATH, 'logs', 'skrl')
        skill_log_dir = os.path.join(logs_base, skill_name.lower())
        
        if not os.path.exists(skill_log_dir):
            return jsonify({'error': 'Skill log directory not found'}), 404
        
        # Find most recent experiment
        experiments = [d for d in os.listdir(skill_log_dir) 
                     if os.path.isdir(os.path.join(skill_log_dir, d))]
        if not experiments:
            return jsonify({'error': 'No experiments found'}), 404
        
        experiments.sort(key=lambda x: os.path.getmtime(os.path.join(skill_log_dir, x)), reverse=True)
        latest_exp = experiments[0]
        
        video_path = os.path.join(skill_log_dir, latest_exp, 'videos', 'train', filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        return send_file(video_path, as_attachment=False, mimetype='video/mp4')
        
    except Exception as e:
        print(f"Error serving video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<task_name>/metrics', methods=['GET'])
def get_training_metrics(task_name):
    """Get training metrics/logs for a task."""
    try:
        robot = request.args.get('robot', 'G1')
        
        # Find all skills for this task
        task_path = get_task_path(robot, task_name)
        if not path_exists(task_path):
            return jsonify({'metrics': {}})
        
        metrics_data = {}
        
        # Get skill names from skills hierarchy
        skills_hierarchy_path = os.path.join(task_path, 'skills_hierarchy.json')
        if path_exists(skills_hierarchy_path):
            hierarchy_data = read_json(skills_hierarchy_path)
            
            # Extract all skill names from hierarchy
            def extract_skill_names(node):
                names = []
                if isinstance(node, dict):
                    if 'name' in node:
                        names.append(node['name'])
                    if 'children' in node:
                        for child in node['children']:
                            names.extend(extract_skill_names(child))
                elif isinstance(node, list):
                    for item in node:
                        names.extend(extract_skill_names(item))
                return names
            
            skill_names = extract_skill_names(hierarchy_data)
            
            # Get metrics for each skill
            logs_base = os.path.join(ISAACLAB_PATH, 'logs', 'skrl')
            
            for skill_name in skill_names:
                skill_log_dir = os.path.join(logs_base, skill_name.lower())
                if os.path.exists(skill_log_dir):
                    # Find most recent experiment
                    experiments = [d for d in os.listdir(skill_log_dir) 
                                 if os.path.isdir(os.path.join(skill_log_dir, d))]
                    if experiments:
                        experiments.sort(key=lambda x: os.path.getmtime(os.path.join(skill_log_dir, x)), reverse=True)
                        latest_exp = experiments[0]
                        exp_dir = os.path.join(skill_log_dir, latest_exp)
                        
                        # Parse TensorBoard logs
                        tb_logs = parse_tensorboard_logs(exp_dir)
                        
                        # Get basic info
                        metrics_data[skill_name] = {
                            'experiment_name': latest_exp,
                            'log_dir': exp_dir,
                            'created_at': os.path.getmtime(exp_dir),
                            'tensorboard_data': tb_logs,
                            'has_checkpoints': os.path.exists(os.path.join(exp_dir, 'checkpoints')),
                            'has_videos': os.path.exists(os.path.join(exp_dir, 'videos', 'train')),
                            'skill_type': 'primitive' if any(skill_name.lower() in fname.lower() for fname in ['flat', 'primitive']) else 'composite'
                        }
        
        return jsonify({'metrics': metrics_data})
        
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500

def parse_tensorboard_logs(log_dir):
    """Parse TensorBoard log files and extract metrics."""
    try:
        # Try to import tensorboard log parsing
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            has_tensorboard = True
        except ImportError:
            has_tensorboard = False
        
        if not has_tensorboard:
            # Fallback: look for event files and return basic info
            event_files = []
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if 'tfevents' in file:
                        event_files.append(os.path.join(root, file))
            
            return {
                'event_files': event_files,
                'has_data': len(event_files) > 0,
                'message': 'TensorBoard not available for parsing. Install tensorboard to view detailed metrics.'
            }
        
        # Find TensorBoard event files
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if 'tfevents' in file:
                    event_files.append(os.path.join(root, file))
        
        if not event_files:
            return {'has_data': False, 'message': 'No TensorBoard event files found'}
        
        # Parse the most recent event file
        latest_file = max(event_files, key=os.path.getmtime)
        
        ea = EventAccumulator(os.path.dirname(latest_file))
        ea.Reload()
        
        # Extract scalar data
        tags = ea.Tags()['scalars']
        scalar_data = {}
        
        for tag in tags:
            scalar_events = ea.Scalars(tag)
            scalar_data[tag] = [
                {
                    'step': event.step,
                    'wall_time': event.wall_time,
                    'value': event.value
                }
                for event in scalar_events
            ]
        
        return {
            'has_data': True,
            'scalars': scalar_data,
            'tags': tags,
            'event_file': latest_file
        }
        
    except Exception as e:
        return {
            'has_data': False,
            'error': str(e)
        }

@app.route('/api/tasks/<task_name>/metrics/<skill_name>/tensorboard')
def get_skill_tensorboard_data(task_name, skill_name):
    """Get detailed TensorBoard data for a specific skill."""
    try:
        robot = request.args.get('robot', 'G1')
        
        logs_base = os.path.join(ISAACLAB_PATH, 'logs', 'skrl')
        skill_log_dir = os.path.join(logs_base, skill_name.lower())
        
        if not os.path.exists(skill_log_dir):
            return jsonify({'error': 'Skill log directory not found'}), 404
        
        # Find most recent experiment
        experiments = [d for d in os.listdir(skill_log_dir) 
                     if os.path.isdir(os.path.join(skill_log_dir, d))]
        if not experiments:
            return jsonify({'error': 'No experiments found'}), 404
        
        experiments.sort(key=lambda x: os.path.getmtime(os.path.join(skill_log_dir, x)), reverse=True)
        latest_exp = experiments[0]
        exp_dir = os.path.join(skill_log_dir, latest_exp)
        
        # Parse TensorBoard logs with more detail
        tb_data = parse_tensorboard_logs(exp_dir)
        
        return jsonify(tb_data)
        
    except Exception as e:
        print(f"Error getting TensorBoard data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/gpu', methods=['GET'])
def get_gpu_status():
    """Get GPU memory status"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                used, total = line.split(', ')
                gpu_info.append({
                    'id': i,
                    'memory_used_mb': int(used),
                    'memory_total_mb': int(total),
                    'memory_free_mb': int(total) - int(used),
                    'memory_used_percent': round((int(used) / int(total)) * 100, 1)
                })
            return jsonify({'success': True, 'gpus': gpu_info})
        else:
            return jsonify({'success': False, 'error': 'nvidia-smi command failed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/tasks/<task_name>/training/finish', methods=['POST'])
def finish_current_training(task_name):
    """User requests to finish the current skill immediately and advance."""
    try:
        robot = request.args.get('robot', 'G1')
        session_key = f"{robot}_{task_name}"
        if session_key not in active_training_sessions:
            return jsonify({'error': 'No active training session'}), 404
 
        session_id = active_training_sessions[session_key]['session_id']
        early_finish_sessions.add(session_id)
        return jsonify({'success': True, 'message': 'Requested early finish'}), 200
    except Exception as e:
        print(f"Error requesting early finish: {e}")
        return jsonify({'error': 'Failed to request early finish'}), 500

# -------------------------------------------------------------
# Helper to perform the same clean-up the orchestrator would do
# when a skill finishes (copy policy, transfer success states)
# -------------------------------------------------------------

def _finalize_skill(task_name: str, robot: str, skill_name: str, session_id=None):
    """Run TrainingOrchestrator._handle_training_completion on-demand."""
    try:
        debug_msg = f"Starting finalization for skill: {skill_name}"
        print(f"[DEBUG] {debug_msg}")
        if session_id is not None:
            send_training_terminal_output(session_id, debug_msg)
            
        from genhrl.training.orchestrator import TrainingOrchestrator
        orch = TrainingOrchestrator(ISAACLAB_PATH, task_name, robot=robot)
        order = orch.get_training_order()
        
        debug_msg = f"Training order: {order}"
        print(f"[DEBUG] {debug_msg}")
        if session_id is not None:
            send_training_terminal_output(session_id, debug_msg)
        
        next_skill = None
        if skill_name in order:
            idx = order.index(skill_name)
            if idx + 1 < len(order):
                next_skill = order[idx + 1]
                debug_msg = f"Next skill for state transfer: {next_skill}"
                print(f"[DEBUG] {debug_msg}")
                if session_id is not None:
                    send_training_terminal_output(session_id, debug_msg)
            else:
                debug_msg = f"{skill_name} is the final skill in the sequence"
                print(f"[DEBUG] {debug_msg}")
                if session_id is not None:
                    send_training_terminal_output(session_id, debug_msg)
        else:
            debug_msg = f"Warning: {skill_name} not found in training order"
            print(f"[DEBUG] {debug_msg}")
            if session_id is not None:
                send_training_terminal_output(session_id, debug_msg)
            
        debug_msg = f"Handling post-training tasks for {skill_name} -> {next_skill}"
        print(f"[DEBUG] {debug_msg}")
        if session_id is not None:
            send_training_terminal_output(session_id, debug_msg)
            
        # Simple post-training handling: copy latest policy and success states
        _handle_simple_training_completion(task_name, robot, skill_name, next_skill, session_id)
        
        debug_msg = f"Successfully completed finalization for {skill_name}"
        print(f"[DEBUG] {debug_msg}")
        if session_id is not None:
            send_training_terminal_output(session_id, debug_msg)
    except Exception as e:
        error_msg = f"Orchestrator finalization error for {skill_name}: {e}"
        print(f"[DEBUG] {error_msg}")
        if session_id is not None:
            send_training_terminal_output(session_id, error_msg)
        import traceback
        traceback.print_exc()

# =============================================================
# Real-time terminal output forwarding for GENERATION sessions
# =============================================================

def send_generation_terminal_output(session_id, line):
    """Forward raw stdout/stderr lines from the generation subprocess to the
    frontend via the same progress SSE channel (type == 'terminal')."""

    if session_id not in progress_queues:
        # Ensure a queue exists to avoid missing early lines
        progress_queues[session_id] = queue.Queue()

    data = {
        'type': 'terminal',
        'line': line,
        'timestamp': datetime.now().isoformat()
    }

    try:
        progress_queues[session_id].put(data, timeout=0.1)
    except queue.Full:
        # Silently drop if client isn't consuming fast enough to avoid blocking
        pass

def _clean_skill_success_states(robot: str, task_name: str, skill_name: str, session_id: str):
    """Remove previous success states for a single skill to ensure a clean run."""
    try:
        skills_base_path = get_skills_path(robot, task_name)
        success_states_dir = os.path.join(skills_base_path, 'skills', skill_name, 'success_states')
        
        if path_exists(success_states_dir):
            send_training_terminal_output(session_id, f"INFO: Cleaning previous success states for '{skill_name}'...")
            shutil.rmtree(success_states_dir)
            
        # Recreate the directory so the training script doesn't fail
        os.makedirs(success_states_dir, exist_ok=True)
        return True
    except Exception as e:
        error_msg = f"ERROR: Could not clean success states for '{skill_name}': {e}"
        send_training_terminal_output(session_id, error_msg)
        print(error_msg)
        return False

def _clean_all_training_data(robot: str, task_name: str, training_order: list, session_id: str):
    """Clean all previous training data for all skills in the task."""
    try:
        skills_base_path = get_skills_path(robot, task_name)
        
        for skill_name in training_order:
            skill_dir = os.path.join(skills_base_path, 'skills', skill_name)
            
            # Clean success states
            success_states_dir = os.path.join(skill_dir, 'success_states')
            if path_exists(success_states_dir):
                shutil.rmtree(success_states_dir)
                os.makedirs(success_states_dir, exist_ok=True)
            
            # Clean current task start states  
            start_states_dir = os.path.join(skill_dir, 'current_task_start_states')
            if path_exists(start_states_dir):
                shutil.rmtree(start_states_dir)
                os.makedirs(start_states_dir, exist_ok=True)
                
            # Clean policy directory
            policy_dir = os.path.join(skill_dir, 'policy')
            if path_exists(policy_dir):
                shutil.rmtree(policy_dir)
                os.makedirs(policy_dir, exist_ok=True)
        
        send_training_terminal_output(session_id, f"Cleaned training data for {len(training_order)} skills")
        return True
    except Exception as e:
        error_msg = f"ERROR: Could not clean training data: {e}"
        send_training_terminal_output(session_id, error_msg)
        print(error_msg)
        return False

def _cleanup_gpu_processes(session_id=None):
    """Clean up any lingering Isaac Sim processes to free GPU memory."""
    try:
        # Kill any Isaac Sim processes
        cleanup_commands = [
            "pkill -f isaac-sim.sh || true",
            "pkill -f IsaacSim || true", 
            "pkill -f python.*train || true",
            "pkill -f omni.isaac || true"
        ]
        
        for cmd in cleanup_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                if session_id and result.stdout:
                    send_training_terminal_output(session_id, f"GPU cleanup: {result.stdout.strip()}")
            except subprocess.TimeoutExpired:
                if session_id:
                    send_training_terminal_output(session_id, f"GPU cleanup timeout for: {cmd}")
            except Exception as e:
                if session_id:
                    send_training_terminal_output(session_id, f"GPU cleanup error for {cmd}: {e}")
        
        # Clear CUDA cache if possible
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if session_id:
                    send_training_terminal_output(session_id, "✅ CUDA cache cleared")
        except ImportError:
            pass  # torch not available in this environment
        except Exception as e:
            if session_id:
                send_training_terminal_output(session_id, f"CUDA cache clear failed: {e}")
                
        # Wait a moment for processes to fully terminate
        time.sleep(2)
        
        if session_id:
            send_training_terminal_output(session_id, "✅ GPU cleanup completed")
        
        return True
        
    except Exception as e:
        error_msg = f"ERROR: GPU cleanup failed: {e}"
        if session_id:
            send_training_terminal_output(session_id, error_msg)
        print(error_msg)
        return False

def _handle_simple_training_completion(task_name, robot, skill_name, next_skill=None, session_id=None):
    """Handle basic post-training tasks: copy policy and prepare for next skill."""
    try:
        # Find the latest policy from Isaac Lab logs
        logs_base = os.path.join(ISAACLAB_PATH, 'logs', 'skrl')
        skill_log_dir = os.path.join(logs_base, skill_name.lower())
        
        skills_base_path = get_skills_path(robot, task_name)
        skill_dir = os.path.join(skills_base_path, 'skills', skill_name)
        policy_dir = os.path.join(skill_dir, 'policy')
        
        # Ensure policy directory exists
        os.makedirs(policy_dir, exist_ok=True)
        
        if os.path.exists(skill_log_dir):
            # Find most recent experiment
            experiments = [d for d in os.listdir(skill_log_dir) 
                         if os.path.isdir(os.path.join(skill_log_dir, d))]
            if experiments:
                experiments.sort(key=lambda x: os.path.getmtime(os.path.join(skill_log_dir, x)), reverse=True)
                latest_exp = experiments[0]
                
                # Copy latest checkpoint as policy
                checkpoints_dir = os.path.join(skill_log_dir, latest_exp, 'checkpoints')
                if os.path.exists(checkpoints_dir):
                    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
                    if checkpoint_files:
                        # Get the latest checkpoint (highest number)
                        checkpoint_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'), reverse=True)
                        latest_checkpoint = checkpoint_files[0]
                        
                        src_path = os.path.join(checkpoints_dir, latest_checkpoint)
                        dst_path = os.path.join(policy_dir, 'agent.pt')
                        
                        shutil.copy2(src_path, dst_path)
                        if session_id:
                            send_training_terminal_output(session_id, f"✅ Copied policy for {skill_name}: {latest_checkpoint} -> agent.pt")
        
        # If there's a next skill, copy success states to its start states
        if next_skill:
            next_skill_dir = os.path.join(skills_base_path, 'skills', next_skill)
            current_success_dir = os.path.join(skill_dir, 'success_states')
            next_start_states_dir = os.path.join(next_skill_dir, 'current_task_start_states')
            
            if os.path.exists(current_success_dir):
                os.makedirs(next_start_states_dir, exist_ok=True)
                
                # Copy success states as start states for next skill
                success_files = [f for f in os.listdir(current_success_dir) if f.endswith('.pt')]
                copied_count = 0
                for i, success_file in enumerate(success_files[:100]):  # Limit to avoid too many files
                    src_path = os.path.join(current_success_dir, success_file)
                    dst_path = os.path.join(next_start_states_dir, f'start_state_{i:06d}.pt')
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                
                if session_id and copied_count > 0:
                    send_training_terminal_output(session_id, f"✅ Transferred {copied_count} success states from {skill_name} to {next_skill}")
        
        return True
        
    except Exception as e:
        error_msg = f"ERROR: Post-training handling failed for {skill_name}: {e}"
        if session_id:
            send_training_terminal_output(session_id, error_msg)
        print(error_msg)
        return False

if __name__ == '__main__':
    print(f"GenHRL UI Flask Server starting...")
    print(f"IsaacLab path: {ISAACLAB_PATH}")
    print(f"GenHRL path: {GENHRL_PATH}")
    print(f"Cleanup functions registered - signal handlers active")
    print(f"Server will automatically clean up Isaac Sim processes on shutdown")
    print(f"Use POST /api/cleanup for manual cleanup if needed")
    
    # Perform initial cleanup in case there are leftover processes
    print(f"Performing initial cleanup of any leftover Isaac Sim processes...")
    cleanup_isaac_sim_processes()
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)