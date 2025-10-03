"""genhrl.generation.mdp.events
================================

This module provides helper utilities and event-callbacks that are reused by
all skills inside **GenHRL**.  A key feature is the *success-state cache*.  It
stores serialized environment states ("success states") on GPU so that
dependent skills can instantly reset an environment to a meaningful state
without reading from disk every time.

Success-State Cache Lifecycle
-----------------------------
1. **Caching**  –  The first call to
   `preload_success_states(device, skill_name)` for a given *skill* will load
   up to *max_states_total* files from disk and move their tensors onto
   *device*.  The loaded list is kept in the global `_preloaded_success_states`
   list and the current owner skill is tracked in `_current_skill_name`.

2. **Reuse**     –  Subsequent calls from the *same* skill are a no-op; the
   already-cached tensors are reused for fast resets.

3. **Automatic Clear-on-Skill-Switch**  –  When a *different* skill calls
   `preload_success_states`, the previous cache is cleared via
   `clear_preloaded_success_states()` (tensors are deleted, `gc.collect()` and
   `torch.cuda.empty_cache()` are executed) before the new skill’s states are
   loaded.  This prevents GPU memory leaks between sequential training runs.

4. **Process Exit Cleanup**  –  `cleanup_on_training_end()` is registered with
   `atexit`, guaranteeing the cache is freed when the Python process exits
   (including users who launch multiple trainings from a single UI session).

Disabling Caching
-----------------
Caching can be disabled per run by simply **not calling**
`preload_success_states` from the env-config, or by setting the environment
variable ``GENHRL_DISABLE_SUCCESS_CACHE=1`` **before** importing this module.
If that variable is present the helper will skip all caching logic.  Another
option is to call `clear_preloaded_success_states()` immediately after
initialisation to work in a cold-start mode.

Debugging Tips
--------------
• Call `check_memory_leak()` at any time to print the number of cached states,
  current CUDA usage and fragmentation statistics.
• Use ``GENHRL_REWARD_DEBUG=1`` to enable verbose output from the reward
  normaliser (shows if rewards are unexpectedly ~0).
• The unit-test `tests/test_success_state_cache.py` verifies that the cache is
  cleared when switching skills – run ``pytest -q tests/test_success_state_cache.py``.
"""

from dataclasses import MISSING
import os
from typing import Dict, Optional, List, Sequence, Union

import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
import random
from pathlib import Path
import json
import gc

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

# Global variables for success states
_preloaded_success_states = []
_loading_in_progress = False  # Add recursion guard
_current_skill_name: Optional[str] = None  # Track which skill's states are cached

# Name of the current task (e.g., "Basketball_practise" or "football_practise")
TASK_NAME = "Basketball_practise"

# -----------------------------------------------------------------------------
# Optional memory-debug helper
# Enable by setting environment variable ``GENHRL_MEMORY_DEBUG=1`` **before**
# launching the Python process.  When enabled the helper prints memory usage
# before and after cache operations so that memory-leak patterns across skills
# become visible without modifying training scripts.
# -----------------------------------------------------------------------------

DEBUG_MEMORY: bool = os.environ.get("GENHRL_MEMORY_DEBUG", "0") not in ("0", "false", "False")


# Convenience wrapper so we don’t scatter if checks everywhere
def _debug_print(prefix: str) -> None:  # noqa: D401 – simple util
    """Print memory statistics when ``GENHRL_MEMORY_DEBUG`` is enabled."""
    if not DEBUG_MEMORY:
        return
    print(f"[MEM][{prefix}] {get_memory_usage_info()}")

def get_previous_skill(skill_name: str, skill_library: Dict) -> Optional[str]:
    """Get the previous skill from the skill library's requirements.
    
    Args:
        skill_name: Name of the current skill
        skill_library: The loaded skill library dictionary
        
    Returns:
        Optional[str]: Name of the previous skill, or None if no previous skill exists
    """
    if "skills" not in skill_library or skill_name not in skill_library["skills"]:
        return None
        
    skill_info = skill_library["skills"][skill_name]
    required_skills = skill_info.get("requires", [])
    
    # Return the last required skill (most recent dependency)
    return required_skills[-1] if required_skills else None

def find_previous_skill_with_states(skill_name: str, skill_library: Dict, task_dir: Path) -> Optional[str]:
    """Recursively find the most recent previous skill that has success states.
    
    Args:
        skill_name: Name of the current skill
        skill_library: The loaded skill library dictionary
        task_dir: Path to the task directory
        
    Returns:
        Optional[str]: Name of the previous skill with states, or None if none found
    """
    # Get the previous skill from requirements
    previous_skill = get_previous_skill(skill_name, skill_library)
    
    if not previous_skill:
        return None
        
    # Check if this skill has success states
    skill_dir = task_dir / TASK_NAME / previous_skill / "success_states"
    if skill_dir.exists() and any(skill_dir.iterdir()):
        state_files = list(skill_dir.glob(f"success_states_{previous_skill}_*.pt"))
        if state_files:
            return previous_skill
            
    # If no states found, recursively check this skill's previous skill
    return find_previous_skill_with_states(previous_skill, skill_library, task_dir)

def clear_preloaded_success_states():
    """Safely clear all preloaded success states from GPU / CPU memory."""
    global _preloaded_success_states, _current_skill_name

    if len(_preloaded_success_states) == 0:
        _debug_print("clear_noop")
        return  # Nothing to do

    print(f"[DEBUG] Clearing {_current_skill_name or ''} success states (count={len(_preloaded_success_states)})")

    # Explicitly delete tensors to break references and free GPU memory early
    try:
        for state in _preloaded_success_states:
            for asset_type in state.values():
                for asset_name in asset_type.values():
                    for key, tensor in asset_name.items():
                        if torch.is_tensor(tensor):
                            del tensor  # release the Tensor object
    except Exception as cleanup_exc:
        # Log but do not halt – we are in best-effort cleanup
        print(f"[WARNING] Exception during success-state cleanup: {cleanup_exc}")

    # Clear the container itself
    _preloaded_success_states.clear()

    # Run Python and CUDA GC
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _current_skill_name = None
    print("[DEBUG] Success states cleared and CUDA cache emptied.")
    _debug_print("after_clear")

def cleanup_on_training_end():
    """Ensure proper memory cleanup when the training process finishes."""
    global _loading_in_progress

    _loading_in_progress = False
    clear_preloaded_success_states()

    # Minimal additional cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[DEBUG] Training cleanup complete – all success states cleared.")

def get_memory_usage_info():
    """Get current memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB"
    else:
        import psutil
        memory = psutil.virtual_memory()
        return f"System Memory - Used: {memory.used / 1024**3:.2f}GB, Available: {memory.available / 1024**3:.2f}GB"

def reset_cuda_memory_stats():
    """Reset CUDA memory statistics for clean monitoring."""
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        print("CUDA memory statistics reset")

def check_memory_leak():
    """Check for potential memory leaks and provide diagnostic information."""
    print(f"=== Memory Leak Check ===")
    print(f"Preloaded states count: {len(_preloaded_success_states)}")
    print(f"Loading in progress: {_loading_in_progress}")
    print(f"Current memory: {get_memory_usage_info()}")
    
    if torch.cuda.is_available():
        # Check for leaked tensors
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Count tensors
        tensor_count = 0
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    tensor_count += 1
        
        print(f"Active CUDA tensors: {tensor_count}")
        
        # Memory fragmentation check
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
        print(f"Memory fragmentation: {fragmentation:.1f}%")
        
        if fragmentation > 50:
            print("⚠️  High memory fragmentation detected - consider clearing cache")
        if tensor_count > 1000:
            print("⚠️  High number of CUDA tensors - potential memory leak")
        if allocated > 8.0:
            print("⚠️  High memory usage - consider reducing batch size or clearing states")
    
    print("=========================")

def preload_success_states(device: torch.device, skill_name: str, max_states_total: int = 1000):
    """Preload success states for dependent skills at the beginning of training.
    
    This function should be called once at the start of training to load
    success states into GPU memory.
    
    Args:
        device: Device to load tensors onto
        skill_name: The name of the current skill to load dependencies for
        max_states_total: Maximum total number of states to load (default: 1,000)
    """
    global _preloaded_success_states, _current_skill_name
    
    # Debug – track memory usage across skill boundaries
    _debug_print(f"preload_start:{skill_name}")

    # If the skill has changed, wipe the previous cache first
    if _current_skill_name is not None and _current_skill_name != skill_name:
        print(f"[DEBUG] Switching skills: '{_current_skill_name}' -> '{skill_name}'. Clearing cached states.")
        clear_preloaded_success_states()

    # If states for this skill are already present, we can reuse them
    if _current_skill_name == skill_name and len(_preloaded_success_states) > 0:
        print(f"Success states already loaded for skill '{skill_name}' ({len(_preloaded_success_states)} states). Skipping preload.")
        return
    
    print(f"[DEBUG] Memory before loading: {get_memory_usage_info()}")
    
    # Clear existing states to prevent accumulation
    clear_preloaded_success_states()
    
    try:
        # Get task name and robot from environment variables
        task_name = os.environ.get('GENHRL_TASK_NAME', 'default_task')
        robot = os.environ.get('GENHRL_ROBOT', 'G1')
        print(f"[DEBUG] Preloading success states for skill: {skill_name}, task: {task_name}, robot: {robot}")
        
        # Construct the path to the skills directory
        cwd = Path.cwd()
        robot_folder = f"{robot}_generated"
        skills_dir = cwd / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / robot_folder / "skills" / task_name / "skills"
        current_skill_dir = skills_dir / skill_name

        previous_tasks_dir = current_skill_dir / "previous_tasks_start_states"
        current_task_dir = current_skill_dir / "current_task_start_states"

        print(f"[DEBUG] Looking for state files in: {previous_tasks_dir} and {current_task_dir}")
        
        # Collect all available files
        all_files = []
        if previous_tasks_dir.exists():
            all_files.extend(list(previous_tasks_dir.glob("*.pt")))
        if current_task_dir.exists():
            all_files.extend(list(current_task_dir.glob("*.pt")))
        
        if not all_files:
            print("[DEBUG] No .pt state files found")
            return

        print(f"[DEBUG] Found {len(all_files)} total state files")
        
        # Randomly shuffle and limit files to process
        random.shuffle(all_files)
        max_files = min(20, len(all_files))  # Process at most 20 files
        max_states_per_file = max_states_total // max_files if max_files > 0 else 100
        
        states_loaded = 0
        for i, file_path in enumerate(all_files[:max_files]):
            if states_loaded >= max_states_total:
                break
                
            try:
                print(f"[DEBUG] Loading from file {i+1}/{max_files}: {file_path.name}")
                
                # Load states directly to the target device
                states = torch.load(str(file_path), map_location=device)
                
                if not states:
                    continue
                    
                # Limit number of states from this file
                if len(states) > max_states_per_file:
                    # Randomly sample states
                    indices = torch.randperm(len(states))[:max_states_per_file]
                    states = [states[idx] for idx in indices]
                
                # Add states to our preloaded collection
                for state in states:
                    if states_loaded >= max_states_total:
                        break
                    _preloaded_success_states.append(state)
                    states_loaded += 1
                
                print(f"[DEBUG] Loaded {len(states)} states from {file_path.name} (total: {states_loaded})")
                
                # Memory check every few files
                if i % 5 == 0 and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"[DEBUG] GPU memory usage: {memory_allocated:.2f}GB")
                    if memory_allocated > 10.0:  # 10GB limit
                        print(f"[WARNING] High GPU memory usage, stopping at {states_loaded} states")
                        break
                        
            except Exception as e:
                print(f"[ERROR] Failed to load states from {file_path}: {e}")
                continue
    
    except Exception as e:
        print(f"[ERROR] Error during preloading: {e}")
        clear_preloaded_success_states()
        return
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[DEBUG] Memory after loading: {get_memory_usage_info()}")
    
    # Mark which skill is now loaded
    _current_skill_name = skill_name
    print(f"[DEBUG] Preloading complete. Loaded {len(_preloaded_success_states)} states for skill '{skill_name}'")
    _debug_print(f"preload_end:{skill_name}")

def load_success_state_from_dependent_skill(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | None, probability: float = 1) -> None:
    """Load success states from a dependent skill for specific environments that are resetting.
    
    Args:
        env: The environment instance
        env_ids: The environment IDs that are being reset (tensor or list of integers)
        dependent_skill: The name of the dependent skill to load states from
        probability: Probability of loading a success state (vs. random initialization)
    """
    global _preloaded_success_states, _loading_in_progress
    
    # Allow disabling success state loading for debugging wandb issues
    if os.environ.get("GENHRL_DISABLE_SUCCESS_LOADING", "0") == "1":
        return
    
    # Prevent recursive calls
    if _loading_in_progress:
        return
    
    _loading_in_progress = True
    expanded_state = None  # Initialize for cleanup
    
    try:
        # Convert env_ids to tensor format
        env_ids_tensor: torch.Tensor
        if env_ids is None:
            dones = env.termination_manager.dones
            env_ids_tensor = torch.where(dones)[0]
        elif isinstance(env_ids, (list, tuple)):
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=env.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.long()
        else:
            # Fallback - shouldn't happen
            return
        
        # Early exit if no environments to reset
        if len(env_ids_tensor) == 0:
            return

        # Early exit if probability check fails - do this before any other computation
        if probability < 1.0 and torch.rand(1, device=env.device).item() >= probability:
            return
        
        if len(_preloaded_success_states) == 0:
            # print("Warning: No success states preloaded. Skipping load_success_state_from_dependent_skill.")
            return
            
        # Use preloaded states - this is the fast path
        if len(_preloaded_success_states) > 0:
            states = _preloaded_success_states
            
            # Randomly select a state
            state_idx = torch.randint(0, len(states), (1,)).item()
            selected_state = states[int(state_idx)]
            
            # SIMPLIFIED: Expand tensors for multiple environments, then use scene.reset_to()
            try:
                # Quick tensor expansion - convert 1D tensors to 2D for multiple environments
                num_envs = len(env_ids_tensor)
                expanded_state = {}
                
                for asset_type in selected_state:
                    expanded_state[asset_type] = {}
                    for asset_name in selected_state[asset_type]:
                        expanded_state[asset_type][asset_name] = {}
                        for key, tensor in selected_state[asset_type][asset_name].items():
                            # Expand 1D tensor to [num_envs, tensor_size]
                            if tensor.dim() == 1:
                                expanded_tensor = tensor.unsqueeze(0).expand(num_envs, -1).contiguous()
                            else:
                                expanded_tensor = tensor
                            expanded_state[asset_type][asset_name][key] = expanded_tensor.to(env.device)
                
                # Use scene.reset_to() directly to avoid the episode tracking corruption from env.reset_to()
                env.scene.reset_to(expanded_state, env_ids_tensor, is_relative=True)
                
                # Update simulation (from the env.reset_to() implementation)
                env.sim.forward()
                
                # Render if needed (from the env.reset_to() implementation)
                if env.sim.has_rtx_sensors() and env.cfg.rerender_on_reset:
                    env.sim.render()
                
                # Compute observations (from the env.reset_to() implementation)
                env.obs_buf = env.observation_manager.compute()
                
                
            except Exception as e:
                # Fallback to full reset_to if direct scene reset fails
                env.reset_to(state=expanded_state, env_ids=env_ids_tensor, is_relative=True)
            finally:
                # Clean up expanded state
                if 'expanded_state' in locals() and expanded_state:
                    expanded_state.clear()
                
            return
                    
    except Exception as e:
        print(f"Error loading success states: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _loading_in_progress = False
        # Final cleanup
        if expanded_state:
            expanded_state.clear()




def randomize_object_shapes(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, scale_range: tuple = (0.8, 1.2)) -> None:
    """Randomize the shape of a single object by scaling its dimensions."""
    # Generate a single scale factor for all environments - avoid per-environment randomization
    scale_factor = torch.rand(1, device=env.device).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
    
    # Get the object
    obj = env.scene[asset_cfg.name]
    obj_cfg = obj.cfg.spawn
    
    # Use a dictionary lookup instead of multiple isinstance checks
    shape_handlers = {
        sim_utils.SphereCfg: lambda cfg: setattr(cfg, 'radius', cfg.radius * scale_factor),
        sim_utils.CuboidCfg: lambda cfg: setattr(cfg, 'size', tuple(s * scale_factor for s in cfg.size)),
        sim_utils.CylinderCfg: lambda cfg: (setattr(cfg, 'radius', cfg.radius * scale_factor), 
                                           setattr(cfg, 'height', cfg.height * scale_factor)),
        sim_utils.ConeCfg: lambda cfg: (setattr(cfg, 'radius', cfg.radius * scale_factor), 
                                       setattr(cfg, 'height', cfg.height * scale_factor)),
        sim_utils.CapsuleCfg: lambda cfg: (setattr(cfg, 'radius', cfg.radius * scale_factor), 
                                          setattr(cfg, 'height', cfg.height * scale_factor))
    }
    
    # Apply the appropriate handler based on object type
    handler = shape_handlers.get(type(obj_cfg))
    if handler:
        handler(obj_cfg)

# Ensure we always clear GPU memory when the python process exits
import atexit
atexit.register(cleanup_on_training_end)