
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import *
import torch
from pathlib import Path
from .TaskRewardsCfg import *
import datetime

# Global dictionary to store success states in memory
success_states_dict = {}

def save_success_state(env, env_id, skill_name):
    """Save the environment state when success is achieved."""
    # Set camera view to environment 0
    if hasattr(env, 'window') and hasattr(env.window, 'camera_controller'):
        env.window.camera_controller.set_view_env_index(0)
    
    # Get the state from the scene for all environments
    state_dict = env.scene.get_state(is_relative=True)
    
    # Extract state for the specific environment
    state = {
        "articulation": {
            name: {
                "joint_position": data["joint_position"][env_id].clone().cpu(),
                "joint_velocity": data["joint_velocity"][env_id].clone().cpu(),
                "root_pose": data["root_pose"][env_id].clone().cpu(),  # Contains both position and quaternion
                "root_velocity": data["root_velocity"][env_id].clone().cpu()
            }
            for name, data in state_dict["articulation"].items()
        },
        "rigid_object": {
            name: {
                "root_pose": data["root_pose"][env_id].clone().cpu(),  # Contains both position and quaternion
                "root_velocity": data["root_velocity"][env_id].clone().cpu()
            }
            for name, data in state_dict["rigid_object"].items()
        }
    }
    
    # Add new state to in-memory dictionary
    if skill_name not in success_states_dict:
        success_states_dict[skill_name] = []
    success_states_dict[skill_name].append(state)
    
    # Save to disk when we have 1000 states
    if len(success_states_dict[skill_name]) >= 1000:
        save_states_to_disk(skill_name)
        # Reset the dictionary for this skill after saving
        success_states_dict[skill_name] = []

def save_states_to_disk(skill_name):
    """Save success states to disk for a specific skill."""
    success_states_dir = Path(__file__).parent / "success_states"
    success_states_dir.mkdir(exist_ok=True)
    
    # Create a filename with timestamp to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    states_file = success_states_dir / f"success_states_{skill_name}_{timestamp}.pt"
    
    # Save the states for this skill
    torch.save(success_states_dict[skill_name], states_file)

def load_success_state(skill_name):
    """Load a random success state for a given skill."""
    success_states_dir = Path(__file__).parent / "success_states"
    
    # Get all state files for this skill
    state_files = list(success_states_dir.glob(f"success_states_{skill_name}_*.pt"))
    if not state_files:
        return None
    
    # Randomly select a file and load states from it
    selected_file = random.choice(state_files)
    states = torch.load(selected_file)
    
    # Return a random state from the loaded states
    if states:
        return random.choice(states)
    return None

def load_all_states():
    """Load all success states from disk at the start of training."""
    global success_states_dict
    
    success_states_dir = Path(__file__).parent / "success_states"
    if not success_states_dir.exists():
        success_states_dict = {}
        return
    
    # Initialize empty dictionary
    success_states_dict = {}
    
    # Load all state files
    for state_file in success_states_dir.glob("success_states_*.pt"):
        # Extract skill name from filename
        skill_name = state_file.stem.split('_')[2]  # Format: success_states_skillname_timestamp
        states = torch.load(state_file)
        
        # Add states to dictionary
        if skill_name not in success_states_dict:
            success_states_dict[skill_name] = []
        success_states_dict[skill_name].extend(states)

def check_success_duration(env, success, skill_name, duration = 1.0):
    """Check if success condition has been maintained for sufficient duration."""
    if not hasattr(env, f'{skill_name}_success_start_time'):
        setattr(env, f'{skill_name}_success_start_time', torch.zeros(env.num_envs, device=env.device))
    
    # Update success timers
    dt = 0.005  # 5ms is a common physics timestep
    current_time = env.common_step_counter * dt
    success_start_time = getattr(env, f'{skill_name}_success_start_time')
    
    # Reset timer for environments where success is no longer detected
    success_start_time = torch.where(
        success,
        success_start_time,
        current_time
    )
    setattr(env, f'{skill_name}_success_start_time', success_start_time)
    
    # Calculate duration for which success has been maintained
    success_duration = current_time - success_start_time
    
    # Check which environments have maintained success for required duration
    sustained_success = success_duration > duration
    
    # Reset timers for environments that have achieved sustained success
    if sustained_success.any():
        new_success_start_time = success_start_time.clone()
        new_success_start_time[sustained_success] = current_time
        setattr(env, f'{skill_name}_success_start_time', new_success_start_time)
    
    return sustained_success  # Return which environments have maintained success for required duration
        
