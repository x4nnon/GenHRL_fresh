import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Space
import torch
import numpy as np
import os
import json
import importlib
import importlib.util
import sys
import types
import random
import pickle # Import standard pickle
from typing import Any, Dict, List, Tuple, Union, Type, Optional, Sequence

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# Need yaml to load agent config
import yaml 

from skrl.models.torch import Model
from skrl import logger

from skrl.utils.runner.torch import Runner

# Add wandb for tracking policy selections
import wandb

# Do not import env types here if they cause issues, use string hints if necessary
# from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
# from isaaclab_rl.skrl import SkrlVecEnvWrapper # Avoid potential import cycle


class SkillBuffer:
    """
    Optimized skill buffer for temporary storage of transitions during skill execution.
    Commits to success replay buffer only on success, discards on failure.
    """
    
    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        # Pre-allocate buffers for proper circular buffer behavior
        self.obs_buffer: List[torch.Tensor] = [None] * max_capacity
        self.act_buffer: List[torch.Tensor] = [None] * max_capacity
        self.rewards_buffer: List[float] = [0.0] * max_capacity
        self.timesteps_buffer: List[int] = [0] * max_capacity
        self.is_active = False
        self._head = 0  # Circular buffer head pointer
        self._size = 0  # Current buffer size
        
    def start(self) -> None:
        """Start recording a new skill execution."""
        self.clear()
        self.is_active = True
        
    def append(self, obs: torch.Tensor, action: torch.Tensor, reward: float, timestep: int) -> None:
        """Append a transition to the buffer with optimized memory management."""
        if not self.is_active:
            return
            
        # Use circular buffer for efficient capacity management
        if self._size < self.max_capacity:
            # Buffer not full - append normally
            self.obs_buffer[self._size] = obs.detach()
            self.act_buffer[self._size] = action.detach()
            self.rewards_buffer[self._size] = reward
            self.timesteps_buffer[self._size] = timestep
            self._size += 1
        else:
            # Buffer full - overwrite oldest entry
            idx = self._head % self.max_capacity
            self.obs_buffer[idx] = obs.detach()
            self.act_buffer[idx] = action.detach()
            self.rewards_buffer[idx] = reward
            self.timesteps_buffer[idx] = timestep
            self._head += 1
    
    def commit_to_replay(self, success_replay_buffer: Dict[str, List], success_signal: float = 1.0) -> None:
        """Commit all buffered transitions to the success replay buffer."""
        if not self.is_active or self._size == 0:
            return
        
        # Add only the valid transitions (avoid pre-allocated None entries)
        for i in range(self._size):
            idx = (self._head + i) % self.max_capacity if self._size == self.max_capacity else i
            obs = self.obs_buffer[idx]
            act = self.act_buffer[idx]
            if obs is None or act is None:
                continue
            success_replay_buffer["obs"].append(obs)
            success_replay_buffer["act"].append(act)
            success_replay_buffer["sig"].append(success_signal)
            
        # Enforce replay buffer capacity
        replay_capacity = 2048  # Default capacity
        while len(success_replay_buffer["obs"]) > replay_capacity:
            success_replay_buffer["obs"].pop(0)
            success_replay_buffer["act"].pop(0)
            success_replay_buffer["sig"].pop(0)
    
    def discard(self) -> None:
        """Discard all buffered transitions."""
        self.clear()
    
    def clear(self) -> None:
        """Clear the buffer."""
        # Reset pre-allocated buffers
        for i in range(self.max_capacity):
            self.obs_buffer[i] = None
            self.act_buffer[i] = None
            self.rewards_buffer[i] = 0.0
            self.timesteps_buffer[i] = 0
        self.is_active = False
        self._head = 0
        self._size = 0
    
    def get_length(self) -> int:
        """Get the number of transitions in the buffer."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return self._size == 0


class HierarchicalVecActionWrapper:
    """
    A custom wrapper (not a gym.Wrapper) for SKRL vectorized environments
    that interprets discrete actions from a higher-level policy (L1)
    to select and execute lower-level policies (L0 primitives) for each parallel environment.

    Expects to wrap an object like SkrlVecEnvWrapper which has num_envs, device,
    observation_space, step, reset, close methods.
    """
    # Define observation and action space properties for skrl compatibility
    _observation_space: Space
    _action_space: Discrete # L1 action space

    def __init__(self,
                 # env: SkrlVecEnvWrapper, # Use string hint or Any if direct import is problematic
                 env: Any, # The SKRL Vec Env Wrapper instance
                 sub_policy_checkpoint_paths: List[str],
                 sub_policy_registered_names: List[str],
                 sub_skill_folder_names: Optional[List[str]] = None,
                 skills_root_path: Optional[str] = None,
                 l1_skill_folder_name: Optional[str] = None,
                 steps_per_l0_policy: int = 10,
                 l1_action_frequency: int = 10,
                 debug_mode: bool = False,
                 use_random_policies: bool = False,
                 base_env: Optional[Any] = None,
                 early_terminate_on_success: bool = True,
                 adapt_l0: bool = False,
                 l0_adapt_lr: float = 1e-5,
                 l0_adapt_std: float = 0.2,
                 l0_adapt_every_n_updates: int = 1,
                 l0_adapt_signal: str = "reward",
                 disable_success_state_saving: bool = True,
                 # Skill buffer parameters
                 use_skill_buffers: bool = True,
                 skill_buffer_capacity: int = 10000,
                 # Adaptation/bootstrap controls and signals
                 min_success_samples_for_imitation: int = 1024,
                 use_reward_prior_when_empty: bool = True,
                 success_signal: float = 1.0,
                 failure_signal: float = 0.0,
                 episode_success_bonus: float = 0.0,
                 # Task-level success detection (optional)
                 task_success_dir: Optional[str] = None,
                 # device: Optional[Union[str, torch.device]] = None # Get device from wrapped env
                 ):
        """
        Initializes the vectorized hierarchical wrapper.

        :param env: The vectorized base environment to wrap (e.g., SkrlVecEnvWrapper).
        :param sub_policy_checkpoint_paths: List of paths to L0 agent checkpoints.
        :param sub_policy_registered_names: List of task names that the L0 agents were trained on.
        :param steps_per_l0_policy: Max steps an L0 policy runs before needing a new L1 action.
        :param l1_action_frequency: Number of steps between L1 actions. Often equals steps_per_l0_policy.
        :param debug_mode: Enable debug logging and info prints.
        :param use_random_policies: If True, create random policies when checkpoint paths are empty/missing.
        """

        logger.info("Initializing HierarchicalVecActionWrapper...")

        self.skrl_vec_env = env # Store the wrapped SKRL environment
        self.sub_policy_registered_names = sub_policy_registered_names
        self.sub_skill_folder_names = sub_skill_folder_names or []
        self.skills_root_path = skills_root_path
        self.l1_skill_folder_name = l1_skill_folder_name
        self.debug_mode = debug_mode
        self.use_random_policies = use_random_policies
        self.base_env = base_env
        self.early_terminate_on_success = early_terminate_on_success
        self.adapt_l0 = adapt_l0
        self.l0_adapt_lr = float(l0_adapt_lr)
        self.l0_adapt_std = float(l0_adapt_std)
        self.l0_adapt_every_n_updates = int(max(1, l0_adapt_every_n_updates))
        self._l0_update_counter = 0
        self.l0_adapt_signal = l0_adapt_signal
        self.disable_success_state_saving = disable_success_state_saving
        # Signals and gating
        self.min_success_samples_for_imitation = int(max(0, min_success_samples_for_imitation))
        self.use_reward_prior_when_empty = bool(use_reward_prior_when_empty)
        self._success_signal = float(success_signal)
        self._failure_signal = float(failure_signal)
        self._episode_success_bonus = float(episode_success_bonus)
        # Task success config
        self.task_success_dir = task_success_dir
        self._task_success_fn: Optional[Any] = None
        # --- Get properties from wrapped environment ---
        if not hasattr(self.skrl_vec_env, "num_envs"):
             raise TypeError("The wrapped environment must have a 'num_envs' attribute.")
        self._num_envs = self.skrl_vec_env.num_envs

        if not hasattr(self.skrl_vec_env, "device"):
            # Try to infer device if not directly available
            if hasattr(self.skrl_vec_env, "_device"):
                 self._device = self.skrl_vec_env._device # Common convention in skrl wrappers
            else:
                 # Fallback or raise error
                 self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 logger.warning(f"Could not automatically detect device from wrapped env. Using default: {self._device}")
        else:
             self._device = self.skrl_vec_env.device
        logger.info(f"Wrapper using device: {self.device}")
        # --- End getting properties ---


        if not sub_policy_checkpoint_paths:
            raise ValueError("`sub_policy_checkpoint_paths` cannot be empty.")

        self.sub_policy_paths = sub_policy_checkpoint_paths
        self.num_sub_policies = len(self.sub_policy_paths)
        self._steps_per_l0_policy = steps_per_l0_policy
        self._l1_action_frequency = l1_action_frequency # How often to expect L1 action input
        self._debug_mode = debug_mode

        logger.info(f"Number of L0 sub-policies: {self.num_sub_policies}")
        logger.info(f"Steps per L0 policy execution: {self._steps_per_l0_policy}")
        logger.info(f"L1 action frequency: {self._l1_action_frequency}")
        logger.info(f"Debug mode: {self._debug_mode}")


        # --- Define Action/Observation Spaces for L1 ---
        # L1 Action Space: Discrete selection of L0 policies for each env
        # The wrapper *receives* this batched action from the L1 agent.
        self._action_space = Discrete(self.num_sub_policies) # L1 outputs index per env
        logger.info(f"Wrapper L1 action space: {self.action_space}")

        # Observation Space: Same as the base vectorized environment's observation space
        if not hasattr(self.skrl_vec_env, "observation_space"):
             raise TypeError("The wrapped environment must have an 'observation_space' attribute.")
        self._observation_space = self.skrl_vec_env.observation_space
        if isinstance(self._observation_space, Space):
            logger.info(f"Wrapper L1 observation space: {self.observation_space}")
        else:
             # Handle cases where it might be dict/tuple spaces
             logger.info(f"Wrapper L1 observation space (structure): {self.observation_space}")

        self.sub_policies: Dict[int, torch.nn.Module] = {}
        
        # Load L0 policies as direct neural networks (much more memory efficient)
        logger.info("Loading L0 policies as direct neural networks...")
        
        for i in range(self.num_sub_policies):
            resume_path = self.sub_policy_paths[i]
            
            try:
                # Check if we should use random policy (empty path or missing file)
                if self.use_random_policies and (not resume_path or not os.path.exists(resume_path)):
                    logger.info(f"Creating random L0 policy {i+1}/{self.num_sub_policies} (testing mode)")
                    policy_net = self._create_random_policy_network(i)
                    self.sub_policies[i] = policy_net
                    logger.info(f"Created random L0 policy {i+1}/{self.num_sub_policies}")
                else:
                    # Load the policy directly as a neural network
                    policy_net = self._load_policy_network(resume_path, i)
                    self.sub_policies[i] = policy_net
                    logger.info(f"Loaded L0 policy {i+1}/{self.num_sub_policies}: {resume_path}")
            except Exception as e:
                if self.use_random_policies:
                    logger.warning(f"Failed to load L0 policy {i}, falling back to random policy: {e}")
                    policy_net = self._create_random_policy_network(i)
                    self.sub_policies[i] = policy_net
                    logger.info(f"Created fallback random L0 policy {i+1}/{self.num_sub_policies}")
                else:
                    logger.error(f"Failed to load L0 policy {i}: {e}")
                    raise
                    
        logger.info(f"Successfully loaded all {len(self.sub_policies)} L0 policies as neural networks")

        # Setup per-policy optimizer if adaptation is enabled
        self._l0_optimizers: Dict[int, torch.optim.Optimizer] = {}
        if self.adapt_l0:
            for idx, net in self.sub_policies.items():
                # Use a tiny LR; do not add weight decay or momentum
                self._l0_optimizers[idx] = torch.optim.Adam(net.parameters(), lr=self.l0_adapt_lr)
            logger.info(f"L0 adaptation enabled with lr={self.l0_adapt_lr}, std={self.l0_adapt_std}")
            logger.info(f"L0 adaptation will run every {self.l0_adapt_every_n_updates} PPO updates")
            if self._debug_mode:
                print(f"[DEBUG][adapt] Initialized {len(self._l0_optimizers)} L0 optimizers")

        # Replay buffers for (obs, action, signal) pairs per policy
        self._l0_replay: Dict[int, Dict[str, list]] = {i: {"obs": [], "act": [], "sig": []} for i in range(self.num_sub_policies)}
        self._l0_replay_capacity: int = 2048
        self._l0_replay_batch: int = 64
        self._prestep_obs: Optional[torch.Tensor] = None
        self._prestep_actions: Optional[torch.Tensor] = None
        
        # Skill buffer parameters - only enable if adaptation is enabled
        self.use_skill_buffers = use_skill_buffers and self.adapt_l0
        self.skill_buffer_capacity = skill_buffer_capacity
        self.skill_buffer_frequency = 1  # Process every N steps to reduce overhead
        
        # Skill buffers for recording full trajectories during execution
        # Structure: {skill_idx: {env_idx: SkillBuffer}}
        self._l0_skill_buffers: Dict[int, Dict[int, SkillBuffer]] = {}
        for i in range(self.num_sub_policies):
            self._l0_skill_buffers[i] = {}
            for env_idx in range(self._num_envs):
                self._l0_skill_buffers[i][env_idx] = SkillBuffer(max_capacity=self.skill_buffer_capacity)
        
        if self.use_skill_buffers:
            logger.info(f"Skill buffers enabled with capacity {self.skill_buffer_capacity} per skill per environment")
        else:
            logger.info("Skill buffers disabled - using single transition recording")
        # Tracking per-skill finishes
        self._skill_finish_success = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self.device)
        self._skill_finish_failure = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self.device)

        # --- Load success termination functions for each sub-skill if requested ---
        self._success_functions: Dict[int, Any] = {}
        self._success_fn_loaded_indices: Dict[int, str] = {}
        if self.early_terminate_on_success:
            self._load_success_functions()
            # Cache a reference to a suitable env for success functions
            self._success_env_ref = self._resolve_success_env()
        # Attempt to infer task-level success directory if not provided
        self._infer_task_success_dir()
        # Load task-level success function if provided / inferred
        self._load_task_success_function()
        # Load current L1 skill success function if name provided
        self._load_l1_skill_success_function()

        # --- Internal State (per environment) ---
        # Index of the currently active L0 policy for each env
        self._active_l0_policy_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Counter for how many steps the current L0 policy has run for each env
        self._current_l0_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Counter for overall steps since last L1 action (PER ENVIRONMENT)
        self._steps_since_l1_action = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Cache of per-env per-skill success within an episode to avoid repeating completed skills
        self._skill_completed_mask = torch.zeros((self.num_envs, self.num_sub_policies), dtype=torch.bool, device=self.device)

        

        # Need to know the action shape of L0 policies for batching

        self._l0_action_space = self._get_l0_action_space()
        if not isinstance(self._l0_action_space, Box):
            # This implementation currently assumes Box action space for L0 for easy batching
             raise NotImplementedError("HierarchicalVecActionWrapper currently only supports Box action spaces for L0 policies.")
        self._l0_action_shape = self._l0_action_space.shape
        logger.info(f"Detected L0 action space: {self._l0_action_space}")

        # --- Debug tracking for policy selections after resets ---
        self._post_reset_policy_selections = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self.device)
        self._total_post_reset_decisions = 0
        self._environments_reset_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_count = 0  # Global step counter for wandb logging
        
        # --- Skill selection tracking for wandb ---
        self._skill_selection_counts = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self.device)
        self._total_selections = 0
        self._wandb_log_frequency = 100  # Log every 100 steps
        # Keep human-readable names if available for nicer wandb keys
        self._skill_names_for_logging = self.sub_policy_registered_names if isinstance(self.sub_policy_registered_names, list) and len(self.sub_policy_registered_names) == self.num_sub_policies else [f"skill_{i}" for i in range(self.num_sub_policies)]
        # Track per-skill successful terminations
        self._skill_success_counts = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self.device)
        self._total_successes = 0
        # Track cached-success (immediate terminations due to prior success)
        self._skill_success_cached_counts = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self.device)
        self._total_successes_cached = 0

        logger.info("Policy selection tracking initialized")
        logger.info(f"Wandb skill tracking enabled - will log every {self._wandb_log_frequency} steps")

        # --- Track per-step remapping of completed skill selections ---
        self._last_remap_from = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._last_remap_to = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    # --- Expose properties required by SKRL Runner ---
    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    @property
    def action_space(self) -> Discrete:
        return self._action_space # L1 action space

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def state_space(self) -> Space: # Added state_space property
        # Assuming the wrapped env has state_space, same as observation_space usually for single agent RL
        # or specific state space for MARL.
        if hasattr(self.skrl_vec_env, "state_space"):
            return self.skrl_vec_env.state_space
        else:
            # Fallback or raise error if wrapped env doesn't have it
            logger.warning("Wrapped env does not have 'state_space' attribute. Returning 'observation_space' as fallback.")
            return self.observation_space # Common fallback

    @property
    def num_agents(self) -> int: # Added num_agents property
        # From the perspective of the L1 agent interacting with this wrapper,
        # it's a single-agent setup (one action per environment selecting L0 policy).
        return 1
    # --- End properties ---


    def _get_l0_action_space(self) -> Space:
        """Helper to get the action space from the first loaded L0 policy network."""
        if not self.sub_policies:
            # If no policies loaded yet, get from environment action space
            env_action_space = self.skrl_vec_env.action_space
            logger.info("Getting L0 action space from environment (no policies loaded yet)")
            return env_action_space
        
        # Get action space from the first policy network output shape
        first_policy = self.sub_policies[0]
        
        # Infer action space from network architecture
        # Get the output layer to determine action dimensions
        if hasattr(first_policy, 'children'):
            layers = list(first_policy.children())
        elif hasattr(first_policy, 'modules'):
            layers = list(first_policy.modules())
        else:
            layers = [first_policy]
        
        # Find the last linear layer
        output_layer = None
        for layer in reversed(layers):
            if isinstance(layer, torch.nn.Linear):
                output_layer = layer
                break
        
        if output_layer is None:
            raise RuntimeError("Could not find output layer in policy network")
        
        # Create Box action space based on output dimension
        action_dim = output_layer.out_features
        logger.info(f"Inferred L0 action space dimension: {action_dim}")
        
        # Assume continuous actions with standard bounds (can be adjusted if needed)
        return Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    
    
    def _disable_wandb_in_config(self, config: Dict[str, Any]) -> None:
        """Recursively disable wandb logging in configuration dictionary."""
        # Look for wandb settings at the top level
        if "experiment" in config:
            if "wandb" in config["experiment"]:
                logger.info("Disabling wandb logging in L0 policy config")
                config["experiment"]["wandb"] = False
        
        # Look for wandb settings in the logger section
        if "logger" in config:
            if "wandb" in config["logger"]:
                logger.info("Disabling wandb in logger config")
                config["logger"]["wandb"] = False
        
        # Check for other common wandb config locations
        if "tracking" in config and "wandb" in config["tracking"]:
            logger.info("Disabling wandb in tracking config")
            config["tracking"]["wandb"] = False
        
        # Recursively check nested dictionaries
        for key, value in config.items():
            if isinstance(value, dict):
                self._disable_wandb_in_config(value)

    def _load_policy_network(self, checkpoint_path: str, policy_idx: int) -> torch.nn.Module:
        """
        Load a policy checkpoint as a direct neural network.
        This is much more memory efficient than loading full SKRL agents.
        Handles SKRL checkpoint format which includes GaussianMixin policies.
        
        Args:
            checkpoint_path: Path to the policy checkpoint file
            policy_idx: Index of the policy being loaded
            
        Returns:
            Neural network module ready for inference
        """
        try:
            # Set CUDA linear algebra backend for policy loading robustness
            try:
                torch.backends.cuda.preferred_linalg_library("cusolver")
            except Exception:
                try:
                    torch.backends.cuda.preferred_linalg_library("cublas")
                except Exception:
                    pass  # Use default
            
            # First, check for simplified network weights export (faster loading)
            from pathlib import Path
            checkpoint_dir = Path(checkpoint_path).parent
            simplified_path = checkpoint_dir / "network_weights.pt"
            training_params_path = checkpoint_dir / "training_params.json"
            
            # Load training parameters if available
            training_params = None
            if training_params_path.exists():
                try:
                    import json
                    with open(training_params_path, 'r') as f:
                        training_params = json.load(f)
                    logger.info(f"Loaded training parameters for policy {policy_idx}")
                except Exception as e:
                    logger.warning(f"Failed to load training parameters: {e}")
            
            if simplified_path.exists():
                logger.info(f"Loading simplified network weights from: {simplified_path}")
                try:
                    try:
                        network_weights = torch.load(simplified_path, map_location=self.device, weights_only=True)  # PyTorch >=2.4
                    except TypeError:
                        network_weights = torch.load(simplified_path, map_location=self.device)
                    policy_network = self._create_policy_network_from_state_dict(
                        network_weights, policy_idx, training_params
                    )
                    policy_network.to(self.device)
                    policy_network.eval()
                    logger.info(f"Successfully loaded simplified policy network {policy_idx}")
                    return policy_network
                except Exception as e:
                    logger.warning(f"Failed to load simplified format, falling back to SKRL checkpoint: {e}")
            
            # Fallback to full SKRL checkpoint
            logger.info(f"Loading SKRL checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Expected dict checkpoint format, got {type(checkpoint)}")
            
            # SKRL checkpoint format analysis
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Extract policy network state dict from SKRL format
            policy_state_dict = None
            
            # Try different SKRL checkpoint formats
            if 'policy' in checkpoint and isinstance(checkpoint['policy'], dict):
                # Standard SKRL format
                policy_state_dict = checkpoint['policy']
                logger.info("Found 'policy' key in checkpoint")
            elif 'models' in checkpoint and 'policy' in checkpoint['models']:
                # Nested models format  
                policy_state_dict = checkpoint['models']['policy']
                logger.info("Found nested 'models.policy' key in checkpoint")
            elif 'state_dict' in checkpoint:
                # Generic state dict format
                policy_state_dict = checkpoint['state_dict']
                logger.info("Found 'state_dict' key in checkpoint")
            else:
                # Try to find policy-related keys
                policy_keys = [k for k in checkpoint.keys() if 'policy' in k.lower()]
                if policy_keys:
                    policy_state_dict = checkpoint[policy_keys[0]]
                    logger.info(f"Found policy key: {policy_keys[0]}")
                else:
                    raise ValueError(f"Could not find policy weights in checkpoint. Available keys: {list(checkpoint.keys())}")
            
            # Filter to get only the network weights (exclude log_std, value layers, and other parameters)
            network_state_dict = {}
            for key, value in policy_state_dict.items():
                # Include main network weights (net_container.X or net.X format) AND policy_layer for action output
                # Exclude value_layer and SKRL-specific parameters, but KEEP policy_layer for L0 policies
                if (('net_container.' in key or 'net.' in key) and 
                    not ('value_layer' in key or 'log_std' in key)):
                    network_state_dict[key] = value
                elif 'policy_layer' in key and not 'log_std' in key:
                    # Include policy_layer for final action output
                    network_state_dict[key] = value
                elif (key.startswith('0.') or key.startswith('1.') or key.startswith('2.')) and \
                     not ('value_layer' in key):
                    # Direct layer numbering
                    network_state_dict[key] = value
                elif (key in ['weight', 'bias'] or key.endswith('.weight') or key.endswith('.bias')) and \
                     not ('value_layer' in key or 'log_std' in key):
                    # Direct layer weights (but not value layers)
                    network_state_dict[key] = value
                    
            if not network_state_dict:
                # If no filtered weights found, use all policy weights
                logger.warning("No filtered network weights found, using all policy weights")
                network_state_dict = policy_state_dict
                
            logger.info(f"Extracted network state dict with keys: {list(network_state_dict.keys())}")
            
            # Create the policy network and copy weights by value (avoids key-name mismatches)
            policy_network = self._create_policy_network_from_state_dict(
                network_state_dict, policy_idx, training_params
            )
                
            policy_network.to(self.device)
            policy_network.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded SKRL policy network {policy_idx}")
            return policy_network
            
        except Exception as e:
            logger.error(f"Failed to load SKRL policy network from {checkpoint_path}: {e}")
            logger.error(f"Error details: {str(e)}")
            raise

    def _create_policy_network_from_state_dict(self, state_dict: Dict[str, torch.Tensor], policy_idx: int, training_params: Optional[Dict] = None) -> torch.nn.Module:
        """
        Create a neural network architecture based on the state dict and training parameters,
        and load weights into it by value (not by key names).
        """
        import torch.nn as nn

        # Decide whether to construct from saved architecture or infer from state dict
        network_from_arch = None
        if training_params and "model_architecture" in training_params:
            arch_info = training_params["model_architecture"]
            logger.info(f"Using saved architecture info for policy {policy_idx}: {len(arch_info.get('layers', []))} layers")
            if arch_info.get("layers"):
                network_from_arch = self._create_network_from_architecture_info(arch_info, state_dict, policy_idx)

        # Parse the provided state dict into ordered layer weights/biases
        logger.info(f"Parsing state dict for policy {policy_idx}")

        layer_weights: Dict[int, torch.Tensor] = {}
        layer_biases: Dict[int, torch.Tensor] = {}
        policy_layer_weight = None
        policy_layer_bias = None

        for key, tensor in state_dict.items():
            if key == "policy_layer.weight":
                policy_layer_weight = tensor
                continue
            elif key == "policy_layer.bias":
                policy_layer_bias = tensor
                continue
            elif "net_container." in key and ".weight" in key:
                parts = key.split(".")
                if len(parts) >= 3 and parts[0] == "net_container":
                    try:
                        layer_num = int(parts[1])
                        layer_weights[layer_num] = tensor
                    except ValueError:
                        pass
            elif "net_container." in key and ".bias" in key:
                parts = key.split(".")
                if len(parts) >= 3 and parts[0] == "net_container":
                    try:
                        layer_num = int(parts[1])
                        layer_biases[layer_num] = tensor
                    except ValueError:
                        pass
            elif "net." in key and ".weight" in key:
                parts = key.split(".")
                if len(parts) >= 3 and parts[0] == "net":
                    try:
                        layer_num = int(parts[1])
                        layer_weights[layer_num] = tensor
                    except ValueError:
                        pass
            elif "net." in key and ".bias" in key:
                parts = key.split(".")
                if len(parts) >= 3 and parts[0] == "net":
                    try:
                        layer_num = int(parts[1])
                        layer_biases[layer_num] = tensor
                    except ValueError:
                        pass
            elif key.endswith(".weight"):
                parts = key.split(".")
                if len(parts) >= 2:
                    try:
                        layer_num = int(parts[0])
                        layer_weights[layer_num] = tensor
                    except ValueError:
                        pass
            elif key.endswith(".bias"):
                parts = key.split(".")
                if len(parts) >= 2:
                    try:
                        layer_num = int(parts[0])
                        layer_biases[layer_num] = tensor
                    except ValueError:
                        pass

        if policy_layer_weight is not None:
            if layer_weights:
                max_layer_num = max(layer_weights.keys())
                final_layer_num = max_layer_num + 2
            else:
                final_layer_num = 0
            layer_weights[final_layer_num] = policy_layer_weight
            if policy_layer_bias is not None:
                layer_biases[final_layer_num] = policy_layer_bias

        if not layer_weights:
            raise ValueError(f"No valid layer weights found in state dict keys: {list(state_dict.keys())}")

        sorted_layers = sorted(layer_weights.keys())
        logger.info(f"Policy {policy_idx} found layers: {sorted_layers}")

        # Determine activation for building a fallback network if needed
        activation_type = "elu"
        if training_params and "model_architecture" in training_params:
            activation_type = training_params["model_architecture"].get("activation", "elu")
        elif training_params and "agent_config" in training_params:
            agent_cfg = training_params["agent_config"]
            if "models" in agent_cfg and "policy" in agent_cfg["models"]:
                network_cfg = agent_cfg["models"]["policy"].get("network", [])
                for layer_cfg in network_cfg:
                    if "activations" in layer_cfg:
                        activation_type = layer_cfg["activations"]
                        break

        if activation_type.lower() == "elu":
            activation_fn = nn.ELU()
        elif activation_type.lower() == "relu":
            activation_fn = nn.ReLU()
        elif activation_type.lower() == "tanh":
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.ELU()

        # Build network if we didn't have one from arch; otherwise reuse it
        if network_from_arch is None:
            layers_seq = []
            for i, layer_num in enumerate(sorted_layers):
                weight = layer_weights[layer_num]
                input_size = weight.shape[1]
                output_size = weight.shape[0]
                linear = nn.Linear(input_size, output_size)
                layers_seq.append(linear)
                if i < len(sorted_layers) - 1:
                    layers_seq.append(activation_fn.__class__())
            network = nn.Sequential(*layers_seq)
        else:
            network = network_from_arch

        # Copy weights into the network sequentially
        with torch.no_grad():
            linear_layer_idx = 0
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    if linear_layer_idx >= len(sorted_layers):
                        break
                    layer_num = sorted_layers[linear_layer_idx]
                    module.weight.copy_(layer_weights[layer_num])
                    if layer_num in layer_biases:
                        module.bias.copy_(layer_biases[layer_num])
                    linear_layer_idx += 1

        logger.info(f"Created policy network with {len([m for m in network.modules() if isinstance(m, nn.Linear)])} linear layers for policy {policy_idx}")
        return network

    def _create_network_from_architecture_info(self, arch_info: Dict, state_dict: Dict[str, torch.Tensor], policy_idx: int) -> torch.nn.Module:
        """Create network using saved architecture information for maximum accuracy."""
        import torch.nn as nn
        
        layers_info = arch_info.get("layers", [])
        activation_type = arch_info.get("activation", "elu")
        
        # Create activation function
        if activation_type.lower() == "elu":
            activation_fn = nn.ELU()
        elif activation_type.lower() == "relu":
            activation_fn = nn.ReLU()
        elif activation_type.lower() == "tanh":
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.ELU()
        
        layers = []
        for i, layer_info in enumerate(layers_info):
            input_size = layer_info["input_size"]
            output_size = layer_info["output_size"]
            
            # Create linear layer
            linear = nn.Linear(input_size, output_size)
            layers.append(linear)
            
            # Add activation for all layers except the last one
            if i < len(layers_info) - 1:
                layers.append(activation_fn.__class__())
        
        network = nn.Sequential(*layers)
        
        # Load weights using the standard approach
        # This will be handled by the calling function
        logger.info(f"Created network from saved architecture: {len(layers_info)} layers, {activation_type} activation")
        
        return network

    def _create_random_policy_network(self, policy_idx: int) -> torch.nn.Module:
        """
        Create a random policy network when no checkpoint is available.
        This is used for testing builds when L0 policies haven't been trained yet.
        
        Args:
            policy_idx: Index of the policy being created
            
        Returns:
            Neural network module with random weights for inference
        """
        import torch.nn as nn
        
        # Get action space size directly from the wrapped environment instead of loaded policies
        # This avoids the circular dependency issue
        env_action_space = self.skrl_vec_env.action_space
        if hasattr(env_action_space, 'shape'):
            action_dim = env_action_space.shape[0]
        elif hasattr(env_action_space, 'n'):
            action_dim = env_action_space.n
        else:
            # Fallback to common robot action dimension
            action_dim = 12  # Typical for humanoid robots
            logger.warning(f"Could not determine action dimension from environment, using fallback: {action_dim}")
        
        # Get observation space size
        obs_space = self.observation_space
        shape = getattr(obs_space, 'shape', None)
        if hasattr(obs_space, 'shape') and shape is not None and len(shape) > 0:
            obs_dim = int(shape[0])
        elif hasattr(obs_space, 'n'):
            obs_dim = int(getattr(obs_space, 'n'))
        else:
            obs_dim = 64  # Fallback to reasonable size
        
        # Create a simple 3-layer network similar to typical SKRL policies
        # Standard RL network: obs -> 256 -> 256 -> action
        hidden_dim = 256
        
        layers = [
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        ]
        
        network = nn.Sequential(*layers)
        
        # Initialize with small random weights (better than default)
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        network.to(self.device)
        network.eval()
        
        logger.info(f"Created random policy network {policy_idx}: {obs_dim} -> {hidden_dim} -> {hidden_dim} -> {action_dim}")
        
        return network

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, reset_master=True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Resets the base environment and the wrapper's internal state."""
        logger.debug("HierarchicalVecActionWrapper reset called.")
        self._environments_reset_this_step.fill_(True)
        self._active_l0_policy_idx.fill_(-1)  # Use -1 to indicate no policy selected yet
        self._current_l0_step.zero_()
        self._steps_since_l1_action.zero_()
        self._skill_completed_mask.zero_()
        if reset_master:
            obs, info = self.skrl_vec_env.reset()
        else: 
            obs, info = self.skrl_vec_env._observations, self.skrl_vec_env._info
        self._last_obs = obs  # Store the initial observation
        return obs, info

    def step(self, action: Optional[Union[np.ndarray, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps the environment using the currently active L0 policies.
        Accepts a new L1 action periodically.

        :param action: Batched actions from the L1 policy (shape: [num_envs,]).
                       This is expected every `l1_action_frequency` steps.
                       Can be None on intermediate steps (though Runner usually provides actions).
        :return: Batched observation, reward, terminated, truncated, info tuple.
        """
        obs = self._last_obs  # Use the last stored observation
        # Default cached-success mask for this step (updated after L1 selection)
        completed_by_cache = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            # DEBUG: Start of step
            if self._debug_mode:
                print("[DEBUG][step] Called with action type:", type(action), "action shape:", getattr(action, 'shape', None))
                print("[DEBUG][step] obs shape:", getattr(obs, 'shape', None))
                print("[DEBUG][step] _steps_since_l1_action:", self._steps_since_l1_action)
                print("[DEBUG][step] _active_l0_policy_idx:", self._active_l0_policy_idx)
                print("[DEBUG][step] _current_l0_step:", self._current_l0_step)

            needs_l1_action = (self._steps_since_l1_action % self._l1_action_frequency == 0)
            if self._debug_mode:
                print("[DEBUG][step] needs_l1_action mask:", needs_l1_action)

            if needs_l1_action.any():
                if self._debug_mode:
                    print("[DEBUG][step] L1 action required for envs:", torch.where(needs_l1_action)[0].tolist())
                if action is None:
                    if self._debug_mode:
                        print("[DEBUG][step] ERROR: L1 action expected but received None.")
                    logger.error(f"L1 action expected for {needs_l1_action.sum().item()} environments, but received None.")
                    raise ValueError(f"L1 action expected but received None.")
                # Convert action to tensor on the correct device with explicit dtype
                if isinstance(action, np.ndarray):
                    l1_actions = torch.tensor(action, dtype=torch.long, device=self.device)
                elif isinstance(action, torch.Tensor):
                    l1_actions = action.to(device=self.device, dtype=torch.long)
                else:
                    if self._debug_mode:
                        print("[DEBUG][step] ERROR: Unsupported L1 action type:", type(action))
                    raise TypeError(f"Unsupported L1 action type: {type(action)}")
                if self._debug_mode:
                    print("[DEBUG][step] l1_actions tensor:", l1_actions)
                    print("[DEBUG][step] l1_actions shape:", l1_actions.shape)
                if l1_actions.shape != (self.num_envs,):
                    if l1_actions.shape == (self.num_envs, 1):
                        l1_actions = l1_actions.squeeze(-1)
                        if self._debug_mode:
                            print("[DEBUG][step] Squeezed l1_actions to shape:", l1_actions.shape)
                    else:
                        if self._debug_mode:
                            print("[DEBUG][step] ERROR: Unexpected l1_actions shape:", l1_actions.shape)
                        raise ValueError(f"Expected L1 action shape ({self.num_envs},) or ({self.num_envs}, 1), but got {l1_actions.shape}")
                min_val = torch.min(l1_actions)
                max_val = torch.max(l1_actions)
                if self._debug_mode:
                    print(f"[DEBUG][step] l1_actions min: {min_val}, max: {max_val}, num_sub_policies: {self.num_sub_policies}")
                if min_val < 0 or max_val >= self.num_sub_policies:
                    invalid_indices = torch.where((l1_actions < 0) | (l1_actions >= self.num_sub_policies))[0]
                    if self._debug_mode:
                        print("[DEBUG][step] ERROR: Invalid L1 action indices detected!", invalid_indices.cpu().tolist())
                    logger.error(f"Invalid L1 action indices detected! Min: {min_val}, Max: {max_val}, Num Policies: {self.num_sub_policies}")
                    logger.error(f"Problematic env indices: {invalid_indices.cpu().tolist()}")
                    logger.error(f"Full L1 actions tensor: {l1_actions.cpu().tolist()}")
                    raise ValueError("L1 policy produced invalid action indices (out of bounds).")
                # Remap selections that correspond to already-completed skills in this episode
                try:
                    l1_actions = self._apply_remap_completed_skills(l1_actions, needs_l1_action)
                except Exception as e:
                    if self._debug_mode:
                        print(f"[DEBUG][step] WARNING: remap failed: {e}")
                self._active_l0_policy_idx[needs_l1_action] = l1_actions[needs_l1_action]
                self._current_l0_step[needs_l1_action] = 0 # Reset count for envs receiving new L1 action
                
                # NEW: Start skill buffers for environments receiving new L1 actions
                if self.use_skill_buffers:
                    for skill_idx in torch.unique(l1_actions[needs_l1_action]):
                        skill_mask = (l1_actions == skill_idx) & needs_l1_action
                        for env_idx in torch.where(skill_mask)[0].tolist():
                            self._l0_skill_buffers[skill_idx.item()][env_idx].start()
                        if self._debug_mode:
                            print(f"[SKILL_BUFFER] Started L0 skill {skill_idx.item()} for {skill_mask.sum().item()} environments")
                
                if self._debug_mode:
                    print("[DEBUG][step] Updated _active_l0_policy_idx:", self._active_l0_policy_idx)
                    print("[DEBUG][step] Updated _current_l0_step:", self._current_l0_step)

                # Compute cached-success mask for the newly selected skills (per env)
                try:
                    valid = self._active_l0_policy_idx >= 0
                    if valid.any():
                        env_ids = torch.arange(self.num_envs, device=self.device)[valid]
                        skill_ids = self._active_l0_policy_idx[valid]
                        completed_by_cache[valid] = self._skill_completed_mask[env_ids, skill_ids]
                except Exception:
                    completed_by_cache = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                if self._debug_mode:
                    print("[DEBUG][step] completed_by_cache after selection:", completed_by_cache)

                # Update selection counts for wandb logging
                try:
                    # Use remapped selections for logging
                    selected_actions = l1_actions[needs_l1_action]
                    # Ensure 1D long tensor
                    selected_actions = selected_actions.view(-1).to(dtype=torch.long)
                    counts = torch.bincount(selected_actions, minlength=self.num_sub_policies)
                    # Align device
                    if counts.device != self._skill_selection_counts.device:
                        counts = counts.to(self._skill_selection_counts.device)
                    self._skill_selection_counts += counts
                    self._total_selections += int(needs_l1_action.sum().item())
                except Exception as e:
                    if self._debug_mode:
                        print(f"[DEBUG][step] WARNING: Failed to update selection counts: {e}")

            elif action is not None:
                if self._debug_mode:
                    print("[DEBUG][step] Action provided but not needed this step. Ignoring.")
                pass

            if not hasattr(self, '_l0_action_shape') or not self._l0_action_shape:
                if self._debug_mode:
                    print("[DEBUG][step] ERROR: Invalid _l0_action_shape attribute")
                logger.error("Invalid _l0_action_shape attribute")
                raise ValueError(f"Invalid _l0_action_shape: {getattr(self, '_l0_action_shape', None)}")
            if self._debug_mode:
                print("[DEBUG][step] _l0_action_shape:", self._l0_action_shape)
            try:
                l0_actions_batch = torch.zeros((self.num_envs, *self._l0_action_shape), 
                                              dtype=torch.float32, 
                                              device="cpu")
                l0_actions_batch = l0_actions_batch.to(self.device)
            except Exception as e:
                if self._debug_mode:
                    print("[DEBUG][step] ERROR: Error creating l0_actions_batch tensor:", e)
                logger.error(f"Error creating l0_actions_batch tensor: {e}")
                logger.error(f"num_envs: {self.num_envs}, l0_action_shape: {self._l0_action_shape}")
                raise
            if self._debug_mode:
                print("[DEBUG][step] l0_actions_batch shape:", l0_actions_batch.shape)
            current_obs_for_l0 = obs
            # Save pre-step observations for potential replay storage
            try:
                self._prestep_obs = current_obs_for_l0.detach().to("cpu") if torch.is_tensor(current_obs_for_l0) else None
            except Exception:
                self._prestep_obs = None
            no_policy_mask = (self._active_l0_policy_idx == -1)
            if self._debug_mode:
                print("[DEBUG][step] no_policy_mask:", no_policy_mask)
            if no_policy_mask.any():
                if self._debug_mode:
                    print(f"[DEBUG][step] Using zero actions for {no_policy_mask.sum().item()} environments with no policy selected")
                l0_actions_batch[no_policy_mask] = 0.0
            for k in range(self.num_sub_policies):
                try:
                    active_policy_idx = self._active_l0_policy_idx.detach().clone()
                    active_policy_idx = active_policy_idx.to(dtype=torch.long, device=self.device)
                    mask_k = (active_policy_idx == k)
                    # Skip inference for envs where this skill already succeeded in this episode
                    if completed_by_cache.any():
                        mask_k = mask_k & (~completed_by_cache)
                    if self._debug_mode:
                        print(f"[DEBUG][step] Processing L0 policy {k}, mask_k sum: {mask_k.sum().item()}")
                    if mask_k.any():
                        policy_net_k = self.sub_policies[k]
                        obs_k = current_obs_for_l0[mask_k]
                        if self._debug_mode:
                            print(f"[DEBUG][step] obs_k shape for L0 policy {k}:", obs_k.shape)
                        if obs_k.shape[0] == 0:
                            if self._debug_mode:
                                print(f"[DEBUG][step] Skipping L0 policy {k} due to empty obs_k")
                            continue
                        if obs_k.device != self.device:
                            if self._debug_mode:
                                print(f"[DEBUG][step] Moving obs_k to device {self.device} for L0 policy {k}")
                            obs_k = obs_k.to(self.device)
                        
                        # Direct neural network inference (much more efficient)
                        with torch.set_grad_enabled(False):
                            actions_k = policy_net_k(obs_k)
                            
                        if self._debug_mode:
                            print(f"[DEBUG][step] actions_k shape for L0 policy {k}:", actions_k.shape)
                        l0_actions_batch[mask_k] = actions_k

                        # Step-time adaptation removed; we will adapt after PPO updates
                        
                except Exception as e:
                    if self._debug_mode:
                        print(f"[DEBUG][step] ERROR: Error processing L0 policy {k}:", e)
                    logger.error(f"Error processing L0 policy {k}: {e}")
                    logger.error(f"active_policy_idx shape: {self._active_l0_policy_idx.shape}, dtype: {self._active_l0_policy_idx.dtype}")
                    logger.error(f"active_policy_idx values: {self._active_l0_policy_idx}")
                    raise
            if self._debug_mode:
                print("[DEBUG][step] Final l0_actions_batch:", l0_actions_batch)
                print("[DEBUG][step] l0_actions_batch shape", l0_actions_batch.shape)
            next_obs_batch_orig, rewards_batch, terminated_batch, truncated_batch, infos = self.skrl_vec_env.step(l0_actions_batch)
            if self._debug_mode:
                print("[DEBUG][step] skrl_vec_env.step returned:")
                print("  next_obs_batch_orig shape:", getattr(next_obs_batch_orig, 'shape', None))
                print("  rewards_batch shape:", getattr(rewards_batch, 'shape', None))
                print("  terminated_batch shape:", getattr(terminated_batch, 'shape', None))
                print("  truncated_batch shape:", getattr(truncated_batch, 'shape', None))
                print("  infos type:", type(infos))
                # On first few steps, try evaluating one known success function to sanity check
                if self._step_count < 5 and self._success_functions:
                    try:
                        env_for_success = self._success_env_ref or self._resolve_success_env()
                        # Choose the first loaded function by index for stable mapping
                        first_idx = sorted(self._success_functions.keys())[0]
                        test_fn = self._success_functions[first_idx]
                        test_out = test_fn(env_for_success) if env_for_success is not None else None
                        print("  [DEBUG][success] probe output type:", type(test_out), "shape:", getattr(test_out, 'shape', None), "skill:", self._success_fn_loaded_indices.get(first_idx, 'unknown'))
                    except Exception as e:
                        print("  [DEBUG][success] probe call failed:", e)
            next_obs_batch = next_obs_batch_orig
            rewards_batch = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
            terminated_batch = torch.as_tensor(terminated_batch, dtype=torch.bool, device=self.device)
            truncated_batch = torch.as_tensor(truncated_batch, dtype=torch.bool, device=self.device)
            # Save pre-step actions for potential replay storage
            try:
                self._prestep_actions = l0_actions_batch.detach().to("cpu")
            except Exception:
                self._prestep_actions = None
            
            # OPTIMIZED: Append transitions to skill buffers with batching (every N steps)
            if (self.use_skill_buffers and self._prestep_obs is not None and 
                self._prestep_actions is not None and 
                self._step_count % self.skill_buffer_frequency == 0):
                # Batch process active skills only
                active_skills = torch.unique(self._active_l0_policy_idx[self._active_l0_policy_idx >= 0])
                for skill_idx in active_skills.tolist():
                    skill_mask = (self._active_l0_policy_idx == skill_idx)
                    if skill_mask.any():
                        env_indices = torch.where(skill_mask)[0]
                        # Convert indices to CPU for indexing CPU tensors
                        env_indices_cpu = env_indices.cpu()
                        # Batch detach operations
                        obs_batch = self._prestep_obs[env_indices_cpu].detach()
                        act_batch = self._prestep_actions[env_indices_cpu].detach()
                        rewards_batch_cpu = rewards_batch[env_indices_cpu].cpu().numpy()
                        
                        # Append to each environment's buffer
                        for i, env_idx in enumerate(env_indices_cpu.tolist()):
                            self._l0_skill_buffers[skill_idx][env_idx].append(
                                obs_batch[i], 
                                act_batch[i],
                                float(rewards_batch_cpu[i]),  # Convert to float once
                                self._step_count
                            )
            
            self._steps_since_l1_action += 1
            self._current_l0_step += 1
            l0_limit_reached = (self._current_l0_step >= self._steps_per_l0_policy)
            env_done = terminated_batch | truncated_batch
            # --- Early termination on success ---
            # Start with cached successes from earlier selections (if computed)
            success_mask = completed_by_cache.clone()
            if self.early_terminate_on_success and self._success_functions:
                try:
                    env_for_success = self._success_env_ref or self._resolve_success_env()
                    if env_for_success is None:
                        raise RuntimeError("No suitable env for success functions (no 'scene' found)")
                    active_policy_idx = self._active_l0_policy_idx.detach().clone().to(dtype=torch.long, device=self.device)
                    unique_indices = torch.unique(active_policy_idx)
                    for idx in unique_indices.tolist():
                        if idx < 0:
                            continue
                        fn = self._success_functions.get(int(idx))
                        if fn is None:
                            continue
                        # Evaluate success for all envs, then mask by those using this policy
                        success_all = fn(env_for_success)
                        if not isinstance(success_all, torch.Tensor):
                            if self._debug_mode:
                                print(f"[DEBUG][success] function for skill idx {idx} returned non-tensor: {type(success_all)}")
                            continue
                        success_all = success_all.to(device=self.device)
                        # Ensure boolean [N] shape
                        if success_all.dtype != torch.bool:
                            success_all = success_all > 0.5
                        if success_all.dim() == 2 and success_all.size(-1) == 1:
                            success_all = success_all.squeeze(-1)
                        if success_all.dim() == 1 and success_all.shape[0] != self.num_envs:
                            if self._debug_mode:
                                print(f"[DEBUG][success] shape mismatch: got {success_all.shape}, expected ({self.num_envs},)")
                            # Try to broadcast or trim if possible
                            try:
                                success_all = success_all.view(-1)[: self.num_envs]
                            except Exception:
                                pass
                        policy_mask = (active_policy_idx == int(idx))
                        if policy_mask.any():
                            success_mask = success_mask | (success_all & policy_mask)
                except Exception as e:
                    if self._debug_mode:
                        print(f"[DEBUG][step] WARNING: success function evaluation failed: {e}")
            # Determine episode-level success for bonus signals
            episode_success_list = self._extract_overall_success_mask(infos, default=False)
            if (episode_success_list is None or not any(episode_success_list)) and (self._task_success_fn is not None):
                try:
                    env_for_success = self._success_env_ref or self._resolve_success_env()
                    if env_for_success is not None:
                        task_succ = self._task_success_fn(env_for_success)
                        if self._debug_mode or self._step_count < 5:
                            try:
                                print("[L1][task_success probe] type:", type(task_succ), "shape:", getattr(task_succ, 'shape', None))
                            except Exception:
                                pass
                        # Normalize to list[bool]
                        if isinstance(task_succ, torch.Tensor):
                            if task_succ.dtype != torch.bool:
                                task_succ = task_succ > 0.5
                            if task_succ.dim() > 1 and task_succ.size(-1) == 1:
                                task_succ = task_succ.squeeze(-1)
                            task_succ = task_succ.detach().to('cpu').view(-1)
                            episode_success_list = [bool(x.item()) for x in task_succ[: self.num_envs]]
                        elif isinstance(task_succ, (list, tuple)):
                            episode_success_list = [bool(x) for x in list(task_succ)[: self.num_envs]]
                        elif isinstance(task_succ, bool):
                            episode_success_list = [task_succ] * self.num_envs
                        else:
                            episode_success_list = [False] * self.num_envs
                    else:
                        episode_success_list = [False] * self.num_envs
                except Exception:
                    episode_success_list = [False] * self.num_envs
            if episode_success_list is None:
                episode_success_list = [False] * self.num_envs
            # Update per-env per-skill completion cache for new successes (not just cached hits)
            try:
                if success_mask.any():
                    valid = self._active_l0_policy_idx >= 0
                    # Only those envs with actual success and valid selected skill
                    upd = success_mask & valid
                    if upd.any():
                        env_ids = torch.arange(self.num_envs, device=self.device)[upd]
                        skill_ids = self._active_l0_policy_idx[upd]
                        self._skill_completed_mask[env_ids, skill_ids] = True
                        if self._debug_mode:
                            # Print first few envs that succeeded
                            succ_envs = env_ids[:8].detach().cpu().tolist()
                            print(f"[DEBUG][success] success envs: {succ_envs}, skills: {skill_ids[:8].detach().cpu().tolist()}")
                        # NEW: Commit successful skill buffers to replay buffer
                        if self.use_skill_buffers:
                            try:
                                env_ids_cpu = env_ids.detach().cpu()
                                skill_ids_cpu = skill_ids.detach().cpu()
                                for e_idx, s_idx in zip(env_ids_cpu.tolist(), skill_ids_cpu.tolist()):
                                    if s_idx < 0 or s_idx >= self.num_sub_policies:
                                        continue
                                    # Include per-skill success signal and optional episode bonus
                                    sig_val = self._success_signal + (self._episode_success_bonus if bool(episode_success_list[e_idx]) else 0.0)
                                    # Commit the entire skill buffer to replay with computed signal
                                    self._l0_skill_buffers[s_idx][e_idx].commit_to_replay(self._l0_replay[s_idx], success_signal=sig_val)
                                    if self._debug_mode:
                                        buffer_length = self._l0_skill_buffers[s_idx][e_idx].get_length()
                                        print(f"[SKILL_BUFFER] Committed L0 skill {s_idx} success for env {e_idx} with {buffer_length} transitions")
                            except Exception as e:
                                if self._debug_mode:
                                    print(f"[SKILL_BUFFER] Failed to commit success: {e}")
                        else:
                            # LEGACY: Store single successful transition
                            try:
                                if self._prestep_obs is not None and self._prestep_actions is not None:
                                    env_ids_cpu = env_ids.detach().cpu()
                                    skill_ids_cpu = skill_ids.detach().cpu()
                                    for e_idx, s_idx in zip(env_ids_cpu.tolist(), skill_ids_cpu.tolist()):
                                        if s_idx < 0 or s_idx >= self.num_sub_policies:
                                            continue
                                        obs_item = self._prestep_obs[e_idx].clone()
                                        act_item = self._prestep_actions[e_idx].clone()
                                        sig_val = self._success_signal + (self._episode_success_bonus if bool(episode_success_list[e_idx]) else 0.0)
                                        buf = self._l0_replay[s_idx]
                                        buf["obs"].append(obs_item)
                                        buf["act"].append(act_item)
                                        buf["sig"].append(sig_val)
                                        # Enforce capacity
                                        if len(buf["obs"]) > self._l0_replay_capacity:
                                            buf["obs"].pop(0)
                                            buf["act"].pop(0)
                                            buf["sig"].pop(0)
                            except Exception as e:
                                if self._debug_mode:
                                    print(f"[DEBUG][replay] Failed to store success samples: {e}")
                        
                        # Increment per-skill success counters
                        with torch.no_grad():
                            binc = torch.bincount(skill_ids, minlength=self.num_sub_policies)
                            self._skill_finish_success += binc.to(self._skill_finish_success.device)
            except Exception:
                pass
            # Combine all decision/termination triggers
            env_done_mask = env_done.squeeze(-1) if env_done.dim() > 1 else env_done
            needs_reset_mask = (l0_limit_reached | env_done_mask | success_mask).to(self.device)
            env_done_mask = env_done.squeeze(-1) if env_done.dim() > 1 else env_done
            if self._debug_mode:
                print("[DEBUG][step] l0_limit_reached:", l0_limit_reached)
                print("[DEBUG][step] env_done:", env_done)
                print("[DEBUG][step] success_mask:", success_mask)
                print("[DEBUG][step] needs_reset_mask:", needs_reset_mask)
            # Update per-skill success counts (count only once per success event)
            try:
                if success_mask.any():
                    with torch.no_grad():
                        valid_mask = success_mask & (self._active_l0_policy_idx >= 0)
                        if valid_mask.any():
                            idxs = self._active_l0_policy_idx[valid_mask].to(dtype=torch.long)
                            counts = torch.bincount(idxs, minlength=self.num_sub_policies)
                            if counts.device != self._skill_success_counts.device:
                                counts = counts.to(self._skill_success_counts.device)
                            self._skill_success_counts += counts
                            self._total_successes += int(valid_mask.sum().item())
                # Track cached-only successes (immediate termination due to memory)
                if completed_by_cache.any():
                    with torch.no_grad():
                        valid_cached = completed_by_cache & (self._active_l0_policy_idx >= 0)
                        if valid_cached.any():
                            idxs_c = self._active_l0_policy_idx[valid_cached].to(dtype=torch.long)
                            counts_c = torch.bincount(idxs_c, minlength=self.num_sub_policies)
                            if counts_c.device != self._skill_success_cached_counts.device:
                                counts_c = counts_c.to(self._skill_success_cached_counts.device)
                            self._skill_success_cached_counts += counts_c
                            self._total_successes_cached += int(valid_cached.sum().item())
            except Exception:
                # Tracking should not break stepping
                pass
            if needs_reset_mask.any():
                if self._debug_mode:
                    print(f"[DEBUG][step] Resetting L0 step counters for {needs_reset_mask.sum().item()} envs")
                
                # NEW: On failures/timeouts, commit with small negative signal, then discard buffers
                if self.use_skill_buffers:
                    failure_mask = needs_reset_mask & ~success_mask  # Reset but not success
                    if failure_mask.any():
                        for env_idx in torch.where(failure_mask)[0].tolist():
                            skill_idx = self._active_l0_policy_idx[env_idx].item()
                            if skill_idx >= 0:
                                try:
                                    # Include failure base signal and optional episode bonus if any
                                    sig_val = self._failure_signal + (self._episode_success_bonus if bool(episode_success_list[env_idx]) else 0.0)
                                    self._l0_skill_buffers[skill_idx][env_idx].commit_to_replay(self._l0_replay[skill_idx], success_signal=sig_val)
                                except Exception:
                                    pass
                                self._l0_skill_buffers[skill_idx][env_idx].discard()
                                if self._debug_mode:
                                    print(f"[SKILL_BUFFER] Committed failure for L0 skill {skill_idx} env {env_idx} with signal {sig_val} and discarded buffer")
                
                self._current_l0_step[needs_reset_mask] = 0
                # Trigger fresh L1 selection next step for any env that reached limit, done, or success
                self._steps_since_l1_action[needs_reset_mask] = 0
            # Do not store negatives for finishes without success; only successes contribute to adaptation
            # Decision boundary if scheduled OR success OR episode done OR max steps reached
            is_decision_step = (needs_l1_action | success_mask | env_done_mask | l0_limit_reached)
            if self._debug_mode:
                print("[DEBUG][step] is_decision_step:", is_decision_step)
            if isinstance(infos, list):
                final_infos = infos  # type: ignore[assignment]
                for env_idx, env_info in enumerate(final_infos):
                    if env_info is None:
                        env_info = {}
                        final_infos[env_idx] = env_info
                    env_info["wrapper_l0_policy_idx"] = int(self._active_l0_policy_idx[env_idx].item())
                    env_info["wrapper_l0_step"] = int(self._current_l0_step[env_idx].item())
                    env_info["is_decision_step"] = bool(is_decision_step[env_idx].item())
                    env_info["decision_mask"] = is_decision_step[env_idx]
                    # Annotate if a selection was remapped due to prior completion
                    try:
                        from_idx = int(self._last_remap_from[env_idx].item())
                        to_idx = int(self._last_remap_to[env_idx].item())
                        if from_idx >= 0:
                            env_info["wrapper_l0_remapped_from"] = from_idx
                            if to_idx >= 0:
                                env_info["wrapper_l0_remapped_to"] = to_idx
                            env_info["wrapper_l0_remap_reason"] = "completed_in_episode"
                    except Exception:
                        pass
                    # Explicit flag: preceding skill ended due to env termination at this decision (exclude truncation)
                    term_mask = terminated_batch.squeeze(-1) if terminated_batch.dim() > 1 else terminated_batch
                    env_info["decision_terminated"] = bool(term_mask[env_idx].item())
                    # Annotate success and reason
                    env_success = bool(success_mask[env_idx].item())
                    env_cached = bool(completed_by_cache[env_idx].item())
                    env_info["skill_success"] = env_success
                    env_info["skill_success_cached"] = env_cached
                    if env_success:
                        env_info["decision_reason"] = "cached_success" if env_cached else "success"
                    elif bool(env_done_mask[env_idx].item()):
                        env_info["decision_reason"] = "env_done"
                    elif bool(l0_limit_reached[env_idx].item()):
                        env_info["decision_reason"] = "max_steps"
                    elif bool(needs_l1_action[env_idx].item()):
                        env_info["decision_reason"] = "schedule"
                    else:
                        env_info["decision_reason"] = "none"
            else:
                final_infos = {}
                if isinstance(infos, dict):
                    final_infos.update(infos)  # Pass through everything from wrapped env
                final_infos["wrapper_l0_policy_idx"] = self._active_l0_policy_idx.cpu().numpy()
                final_infos["wrapper_l0_step"] = self._current_l0_step.cpu().numpy()
                final_infos["is_decision_step"] = is_decision_step.cpu().numpy()
                final_infos["decision_mask"] = is_decision_step
                # decision_terminated uses 'terminated' only
                term_mask = terminated_batch.squeeze(-1) if terminated_batch.dim() > 1 else terminated_batch
                final_infos["decision_terminated"] = term_mask.cpu().numpy()
                final_infos["skill_success"] = success_mask.cpu().numpy()
                final_infos["skill_success_cached"] = completed_by_cache.cpu().numpy()
                # Provide remap arrays (values -1 mean no remap)
                try:
                    final_infos["wrapper_l0_remapped_from"] = self._last_remap_from.detach().cpu().numpy()
                    final_infos["wrapper_l0_remapped_to"] = self._last_remap_to.detach().cpu().numpy()
                except Exception:
                    pass
                # Provide a coarse reason array
                # 0: none, 1: success, 2: env_done, 3: max_steps, 4: schedule
                reason = torch.zeros_like(success_mask, dtype=torch.int32)
                # 5: cached_success (takes priority over plain success)
                cached = completed_by_cache
                reason = torch.where(cached, torch.tensor(5, dtype=torch.int32, device=self.device), reason)
                reason = torch.where(~cached & success_mask, torch.tensor(1, dtype=torch.int32, device=self.device), reason)
                reason = torch.where(env_done_mask, torch.tensor(2, dtype=torch.int32, device=self.device), reason)
                reason = torch.where(l0_limit_reached, torch.tensor(3, dtype=torch.int32, device=self.device), reason)
                reason = torch.where(needs_l1_action, torch.tensor(4, dtype=torch.int32, device=self.device), reason)
                final_infos["decision_reason_code"] = reason.cpu().numpy()
            reward_to_agent = rewards_batch
            if hasattr(self, '_cumulative_rewards'):
                self._cumulative_rewards += rewards_batch
            else:
                self._cumulative_rewards = rewards_batch.clone()
            if env_done_mask.any():
                if hasattr(self, '_cumulative_rewards'):
                    self._cumulative_rewards[env_done_mask] = 0.0
                if env_done_mask.any() and self._debug_mode:
                    completed_count = env_done_mask.sum().item()
                    print(f"[DEBUG][step] Episode completion: {completed_count} environments completed episodes")
                    logger.info(f"Episode completion: {completed_count} environments completed episodes")
                self._environments_reset_this_step[env_done_mask] = True
                # Clear cached successes for completed environments
                try:
                    self._skill_completed_mask[env_done_mask] = False
                except Exception:
                    pass
            self._step_count += 1
            if self._debug_mode:
                print("[DEBUG][step] End of step. Returning observations, rewards, dones, infos.")
            if self._debug_mode:
                if isinstance(final_infos, list):
                    print("[DEBUG][step] decision_mask in final_infos:", [info.get("decision_mask", None) for info in final_infos])
                elif isinstance(final_infos, dict):
                    print("[DEBUG][step] decision_mask in final_infos:", final_infos.get("decision_mask", None))

            # Periodic wandb logging of selection percentages under "choices"
            try:
                if self._step_count % self._wandb_log_frequency == 0 and (self._total_selections > 0 or self._total_successes > 0):
                    log_dict = {}
                    # Move to CPU to avoid GPU<->CPU overhead in wandb
                    sel_counts_cpu = self._skill_selection_counts.detach().cpu().tolist()
                    total_sel = float(self._total_selections)
                    for i, count in enumerate(sel_counts_cpu):
                        name = self._skill_names_for_logging[i] if i < len(self._skill_names_for_logging) else f"skill_{i}"
                        percentage = (float(count) / total_sel) * 100.0 if total_sel > 0 else 0.0
                        log_dict[f"choices/{name}_count"] = count
                        log_dict[f"choices/{name}_percentage"] = percentage
                    log_dict["choices/total_decisions"] = int(total_sel)

                    # Success logging: counts and success rate per selection
                    succ_counts_cpu = self._skill_success_counts.detach().cpu().tolist()
                    for i, succ in enumerate(succ_counts_cpu):
                        name = self._skill_names_for_logging[i] if i < len(self._skill_names_for_logging) else f"skill_{i}"
                        sel = float(sel_counts_cpu[i]) if i < len(sel_counts_cpu) else 0.0
                        rate = (float(succ) / sel) * 100.0 if sel > 0 else 0.0
                        log_dict[f"success/{name}_count"] = int(succ)
                        log_dict[f"success/{name}_rate_pct"] = rate
                    log_dict["success/total_successes"] = int(self._total_successes)
                    overall_rate = (float(self._total_successes) / total_sel) * 100.0 if total_sel > 0 else 0.0
                    log_dict["success/overall_rate_pct"] = overall_rate

                    # Cached success logging
                    succ_cached_cpu = self._skill_success_cached_counts.detach().cpu().tolist()
                    for i, succ_c in enumerate(succ_cached_cpu):
                        name = self._skill_names_for_logging[i] if i < len(self._skill_names_for_logging) else f"skill_{i}"
                        sel = float(sel_counts_cpu[i]) if i < len(sel_counts_cpu) else 0.0
                        rate_c = (float(succ_c) / sel) * 100.0 if sel > 0 else 0.0
                        log_dict[f"success_cached/{name}_count"] = int(succ_c)
                        log_dict[f"success_cached/{name}_rate_pct"] = rate_c
                    log_dict["success_cached/total_successes"] = int(self._total_successes_cached)
                    overall_rate_c = (float(self._total_successes_cached) / total_sel) * 100.0 if total_sel > 0 else 0.0
                    log_dict["success_cached/overall_rate_pct"] = overall_rate_c
                    wandb.log(log_dict, step=self._step_count)
            except Exception:
                # Swallow wandb errors silently to avoid breaking training
                pass
            self._last_obs = next_obs_batch  # Update the stored observation
            return next_obs_batch, rewards_batch, terminated_batch, truncated_batch, final_infos  # type: ignore[return-value]
        except Exception as e:
            if self._debug_mode:
                print(f"[DEBUG][step] ERROR: Exception in step: {e}")
            logger.error(f"Error in HierarchicalVecActionWrapper.step: {e}", exc_info=True)
            logger.error(f"Current state - steps_since_l1_action: {self._steps_since_l1_action}")
            logger.error(f"l1_action_frequency: {self._l1_action_frequency}")
            logger.error(f"active_l0_policy_idx: {self._active_l0_policy_idx}")
            logger.error(f"current_l0_step: {self._current_l0_step}")
            logger.error(f"l0_action_shape: {self._l0_action_shape}")
            logger.error(f"num_envs: {self.num_envs}")
            logger.error(f"device: {self.device}")
            raise

    

    def close(self) -> None:
        """Closes the underlying SKRL vec environment."""
        logger.info("Closing HierarchicalVecActionWrapper and underlying environment.")
        if hasattr(self.skrl_vec_env, "close"):
            return self.skrl_vec_env.close()
        else:
            logger.warning("Wrapped environment does not have a 'close' method.")

    

    def render(self, *args, **kwargs):
        """No-op render method for compatibility with Gym/SKRL. Forwards to underlying env if available."""
        if hasattr(self.skrl_vec_env, "render"):
            return self.skrl_vec_env.render(*args, **kwargs)
        return None

    # --- Modular helpers to enable reuse from higher-level wrappers (no env stepping) ---
    def compute_l0_actions(self, observations: torch.Tensor, active_l0_indices: torch.Tensor) -> torch.Tensor:
        """Compute continuous L0 actions for a batch of observations given selected L0 indices.

        This method does not step the environment and does not modify internal counters.

        Args:
            observations: Tensor [B, obs_dim] on any device
            active_l0_indices: Long tensor [B] with indices into self.sub_policies

        Returns:
            actions: Tensor [B, action_dim] on self.device
        """
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, dtype=torch.float32)
        B = observations.shape[0]
        if not isinstance(active_l0_indices, torch.Tensor):
            active_l0_indices = torch.as_tensor(active_l0_indices, dtype=torch.long)
        active_l0_indices = active_l0_indices.to(dtype=torch.long, device=self.device)
        # Prepare output
        actions = torch.zeros((B, *self._l0_action_shape), dtype=torch.float32, device=self.device)
        # Ensure observations on device
        obs_dev = observations.to(self.device)
        # Group by policy index
        unique = torch.unique(active_l0_indices)
        for idx in unique.tolist():
            if idx < 0 or idx >= self.num_sub_policies:
                continue
            mask = (active_l0_indices == int(idx))
            if not mask.any():
                continue
            net = self.sub_policies.get(int(idx))
            if net is None:
                actions[mask] = 0.0
                continue
            obs_k = obs_dev[mask]
            with torch.no_grad():
                acts = net(obs_k)
            actions[mask] = acts
        return actions

    def evaluate_success_mask(self, env_for_success: Any, active_l0_indices: torch.Tensor) -> torch.Tensor:
        """Evaluate success functions for a batch, returning a boolean success mask [B].

        Args:
            env_for_success: Base env with 'scene' attribute
            active_l0_indices: Long tensor [B] of selected L0 indices

        Returns:
            success_mask: Bool tensor [B]
        """
        B = int(active_l0_indices.shape[0])
        if not self.early_terminate_on_success or not self._success_functions:
            return torch.zeros(B, dtype=torch.bool, device=self.device)
        try:
            success_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
            active_l0_indices = active_l0_indices.to(dtype=torch.long, device=self.device)
            unique = torch.unique(active_l0_indices)
            for idx in unique.tolist():
                if idx < 0:
                    continue
                fn = self._success_functions.get(int(idx))
                if fn is None:
                    continue
                succ_all = fn(env_for_success)
                if not isinstance(succ_all, torch.Tensor):
                    continue
                if succ_all.dtype != torch.bool:
                    succ_all = succ_all > 0.5
                if succ_all.dim() > 1 and succ_all.size(-1) == 1:
                    succ_all = succ_all.squeeze(-1)
                if succ_all.device != self.device:
                    succ_all = succ_all.to(self.device)
                # Align length
                if succ_all.numel() < B:
                    succ_all = torch.nn.functional.pad(succ_all, (0, B - succ_all.numel()), value=False)
                mask = (active_l0_indices == int(idx))
                if mask.any():
                    success_mask = success_mask | (succ_all[:B].to(dtype=torch.bool) & mask)
            return success_mask
        except Exception:
            return torch.zeros(B, dtype=torch.bool, device=self.device)

    def forward_prepare(self, observations: torch.Tensor, env_mask: torch.Tensor, l1_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Prepare an L1 step without stepping the environment.

        Updates selection state for masked envs, computes L0 actions for masked envs, and records
        pre-step observations/actions for replay/adaptation. Does not change global counters that depend on env.step.

        Args:
            observations: Current observations [N, obs_dim]
            env_mask: Bool mask [N] indicating envs controlled by this L1 wrapper
            l1_actions: Optional long tensor [N] with L0 indices; only values where env_mask & needs_l1_action are True are used

        Returns:
            l0_actions_batch: Tensor [N, action_dim] with actions filled for masked envs and zeros elsewhere
        """
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(observations, dtype=torch.float32)
        if not torch.is_tensor(env_mask):
            env_mask = torch.as_tensor(env_mask, dtype=torch.bool)
        env_mask = env_mask.to(device=self.device)
        obs = observations.to(self.device)
        # Determine which envs under this L1 need a new L1 action now
        needs_l1_action = (self._steps_since_l1_action % self._l1_action_frequency == 0) & env_mask
        self._last_needs_l1_action = needs_l1_action.clone()
        # Process L1 actions if provided
        if l1_actions is not None:
            if not torch.is_tensor(l1_actions):
                l1_actions = torch.as_tensor(l1_actions, dtype=torch.long)
            l1_actions = l1_actions.to(device=self.device, dtype=torch.long)
            # Bounds check
            if (l1_actions[needs_l1_action] < 0).any() or (l1_actions[needs_l1_action] >= self.num_sub_policies).any():
                invalid = torch.where((l1_actions < 0) | (l1_actions >= self.num_sub_policies))[0]
                raise ValueError(f"Invalid L1 action indices: {invalid.detach().cpu().tolist()}")
            # Remap completed selections for this subset of envs
            if needs_l1_action.any():
                try:
                    l1_actions = self._apply_remap_completed_skills(l1_actions, needs_l1_action)
                except Exception:
                    pass
            # Apply
            if needs_l1_action.any():
                self._active_l0_policy_idx[needs_l1_action] = l1_actions[needs_l1_action]
                self._current_l0_step[needs_l1_action] = 0
        # Build L0 actions
        if not hasattr(self, '_l0_action_shape') or not self._l0_action_shape:
            raise ValueError(f"Invalid _l0_action_shape: {getattr(self, '_l0_action_shape', None)}")
        l0_actions_batch = torch.zeros((self.num_envs, *self._l0_action_shape), dtype=torch.float32, device=self.device)
        no_policy_mask = (self._active_l0_policy_idx == -1) & env_mask
        if no_policy_mask.any():
            l0_actions_batch[no_policy_mask] = 0.0
        # Save pre-step obs (only if needed for adaptation or debugging)
        if self.adapt_l0 or self._debug_mode:
            try:
                self._prestep_obs = obs.detach().to("cpu")
            except Exception:
                self._prestep_obs = None
        # Compute per-policy actions on masked envs (only over active indices in the mask)
        with torch.no_grad():
            active_idx = torch.unique(self._active_l0_policy_idx[env_mask])
            for k in active_idx.tolist():
                if k < 0:
                    continue
                mask_k = (self._active_l0_policy_idx == k) & env_mask
                if not mask_k.any():
                    continue
                policy_net_k = self.sub_policies.get(int(k))
                if policy_net_k is None:
                    l0_actions_batch[mask_k] = 0.0
                    continue
                obs_k = obs[mask_k]
                if obs_k.shape[0] == 0:
                    continue
                actions_k = policy_net_k(obs_k)
                l0_actions_batch[mask_k] = actions_k
        # Save pre-step actions (only if needed for adaptation or debugging)
        if self.adapt_l0 or self._debug_mode:
            try:
                self._prestep_actions = l0_actions_batch.detach().to("cpu")
            except Exception:
                self._prestep_actions = None
        return l0_actions_batch

    def forward_finalize(self,
                          next_obs_batch: torch.Tensor,
                          rewards_batch: torch.Tensor,
                          terminated_batch: torch.Tensor,
                          truncated_batch: torch.Tensor,
                          infos_from_env: Any,
                          env_mask: torch.Tensor) -> Dict[int, Dict[str, Any]]:
        """Finalize L1 step after env.step. Updates counters, evaluates success, and produces per-env info.

        Returns a dict mapping env index -> info updates for those envs in env_mask.
        """
        if not torch.is_tensor(env_mask):
            env_mask = torch.as_tensor(env_mask, dtype=torch.bool)
        env_mask = env_mask.to(device=self.device)
        # Normalize shapes
        rewards_batch = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        terminated_batch = torch.as_tensor(terminated_batch, dtype=torch.bool, device=self.device)
        truncated_batch = torch.as_tensor(truncated_batch, dtype=torch.bool, device=self.device)
        env_done = terminated_batch | truncated_batch
        # Increment counters for masked envs
        self._steps_since_l1_action[env_mask] += 1
        self._current_l0_step[env_mask] += 1
        l0_limit_reached = (self._current_l0_step >= self._steps_per_l0_policy)
        # Success evaluation
        success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.early_terminate_on_success and self._success_functions:
            try:
                env_for_success = self._resolve_success_env()
                if env_for_success is not None:
                    active_policy_idx = self._active_l0_policy_idx.detach().clone().to(dtype=torch.long, device=self.device)
                    unique_indices = torch.unique(active_policy_idx)
                    for idx in unique_indices.tolist():
                        if idx < 0:
                            continue
                        fn = self._success_functions.get(int(idx))
                        if fn is None:
                            continue
                        success_all = fn(env_for_success)
                        if isinstance(success_all, torch.Tensor):
                            if success_all.dtype != torch.bool:
                                success_all = success_all > 0.5
                            if success_all.dim() > 1 and success_all.size(-1) == 1:
                                success_all = success_all.squeeze(-1)
                            success_all = success_all.to(device=self.device)
                        else:
                            continue
                        policy_mask = (active_policy_idx == int(idx)) & env_mask
                        if policy_mask.any():
                            success_mask = success_mask | (success_all & policy_mask)
            except Exception:
                pass
        # Reset cadence on triggers
        needs_l1_action = getattr(self, '_last_needs_l1_action', torch.zeros_like(env_mask))
        is_decision_step = (needs_l1_action | success_mask | env_done | l0_limit_reached) & env_mask
        if is_decision_step.any():
            self._current_l0_step[is_decision_step] = 0
            self._steps_since_l1_action[is_decision_step] = 0
        # Build info dicts
        out: Dict[int, Dict[str, Any]] = {}
        for i in torch.where(env_mask)[0].tolist():
            info = {}
            info["wrapper_l0_policy_idx"] = int(self._active_l0_policy_idx[i].item())
            info["wrapper_l0_step"] = int(self._current_l0_step[i].item())
            info["is_decision_step"] = bool(is_decision_step[i].item())
            info["decision_mask"] = bool(is_decision_step[i].item())
            info["skill_success"] = bool(success_mask[i].item())
            # Reason
            if bool(success_mask[i].item()):
                info["decision_reason"] = "success"
            elif bool(env_done[i].item()):
                info["decision_reason"] = "env_done"
            elif bool(l0_limit_reached[i].item()):
                info["decision_reason"] = "max_steps"
            elif bool(needs_l1_action[i].item()):
                info["decision_reason"] = "schedule"
            else:
                info["decision_reason"] = "none"
            out[i] = info
        return out

    def compute_needs_l1_action_mask(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of envs (subset by env_mask) that require a new L1 selection now."""
        if not torch.is_tensor(env_mask):
            env_mask = torch.as_tensor(env_mask, dtype=torch.bool)
        env_mask = env_mask.to(device=self.device)
        return (self._steps_since_l1_action % self._l1_action_frequency == 0) & env_mask

    def _apply_remap_completed_skills(self, l1_actions: torch.Tensor, process_mask: torch.Tensor) -> torch.Tensor:
        """Remap L1 selections that point to already-completed skills within the current episode.

        For envs where the selected skill is marked completed in `self._skill_completed_mask`,
        choose the next available skill index (cyclically) that is not completed. If all skills
        are completed for an environment, the original selection is kept. Per-env remaps are
        recorded in `self._last_remap_from` and `self._last_remap_to` for info annotation.

        Args:
            l1_actions: Long tensor [N] of proposed selections on self.device
            process_mask: Bool tensor [N] indicating which envs to consider (e.g., needs_l1_action)

        Returns:
            Remapped actions as a new tensor [N] on self.device
        """
        if not torch.is_tensor(l1_actions):
            l1_actions = torch.as_tensor(l1_actions, dtype=torch.long, device=self.device)
        actions = l1_actions.to(device=self.device, dtype=torch.long).clone()
        if not torch.is_tensor(process_mask):
            process_mask = torch.as_tensor(process_mask, dtype=torch.bool, device=self.device)
        else:
            process_mask = process_mask.to(device=self.device)

        # Reset last remap tracking
        try:
            self._last_remap_from.fill_(-1)
            self._last_remap_to.fill_(-1)
        except Exception:
            pass

        # Determine which envs need remap (selected skill already completed)
        env_ids = torch.arange(self.num_envs, device=self.device)
        clamped = torch.clamp(actions, 0, self.num_sub_policies - 1)
        try:
            selected_completed = self._skill_completed_mask[env_ids, clamped]
        except Exception:
            # Fallback if indexing fails for any reason
            selected_completed = torch.zeros_like(process_mask, dtype=torch.bool, device=self.device)
        need_remap = (selected_completed & process_mask)
        if not need_remap.any():
            return actions

        for env_idx in torch.where(need_remap)[0].tolist():
            try:
                sel = int(actions[env_idx].item())
                # Skip invalid selections defensively
                if sel < 0 or sel >= int(self.num_sub_policies):
                    continue
                row = self._skill_completed_mask[env_idx]
                # Find next not-completed skill cyclically
                new_sel = -1
                for offset in range(1, int(self.num_sub_policies) + 1):
                    cand = (sel + offset) % int(self.num_sub_policies)
                    completed = bool(row[cand].item())
                    if not completed:
                        new_sel = cand
                        break
                if new_sel >= 0:
                    actions[env_idx] = int(new_sel)
                    try:
                        self._last_remap_from[env_idx] = int(sel)
                        self._last_remap_to[env_idx] = int(new_sel)
                    except Exception:
                        pass
                else:
                    # All skills completed for this env; keep original selection
                    try:
                        self._last_remap_from[env_idx] = int(sel)
                        self._last_remap_to[env_idx] = -1
                    except Exception:
                        pass
            except Exception:
                continue
        return actions
    def _extract_overall_success_mask(self, infos: Any, default: bool = False) -> Optional[list]:
        """Best-effort extraction of overall task success per-env from infos.

        Looks for common keys: 'episode_success', 'task_success', or inside a dict list.
        Returns a Python list of bools (length=num_envs) or None if unavailable.
        """
        try:
            if isinstance(infos, list) and len(infos) == self.num_envs:
                out = []
                for d in infos:
                    if isinstance(d, dict):
                        val = d.get("episode_success", d.get("task_success", default))
                        if isinstance(val, torch.Tensor):
                            val = bool(val.item())
                        out.append(bool(val))
                    else:
                        out.append(default)
                return out
            if isinstance(infos, dict):
                val = infos.get("episode_success", infos.get("task_success", default))
                if isinstance(val, torch.Tensor):
                    val = bool(val.item())
                return [bool(val)] * self.num_envs
        except Exception:
            return None
        return None
    # --- Internal helpers ---
    def perform_l0_adaptation(self) -> None:
        """Run a tiny adaptation step for each L0 network using a self-supervised smoothness prior.

        This is intentionally conservative and does not persist to disk.
        """
        if not self.adapt_l0:
            return
        try:
            if self._debug_mode:
                print("[DEBUG][adapt] perform_l0_adaptation called")
            # Use the last observation as a minibatch; for stability, detach targets
            if not hasattr(self, '_last_obs') or self._last_obs is None:
                if self._debug_mode:
                    print("[DEBUG][adapt] No _last_obs available; skipping")
                return
            obs = self._last_obs
            if not torch.is_tensor(obs):
                if self._debug_mode:
                    print(f"[DEBUG][adapt] _last_obs not a tensor (type={type(obs)}); skipping")
                return
            for k, net in self.sub_policies.items():
                opt = self._l0_optimizers.get(k)
                if opt is None:
                    continue
                try:
                    # Sample from replay; check size for imitation gating
                    buf = self._l0_replay.get(k, {"obs": [], "act": [], "sig": []})
                    replay_size = len(buf["obs"]) if isinstance(buf, dict) and "obs" in buf else 0
                    has_min_success = replay_size >= self.min_success_samples_for_imitation
                    use_imitation = has_min_success and (self.l0_adapt_signal in ("success", "both"))

                    if replay_size > 0:
                        batch_size = min(self._l0_replay_batch, replay_size)
                        idxs = random.sample(range(replay_size), batch_size)
                        obs_batch = torch.stack([buf["obs"][i] for i in idxs]).to(self.device)
                        act_batch = torch.stack([buf["act"][i] for i in idxs]).to(self.device)
                        sig_batch = torch.tensor([buf["sig"][i] for i in idxs], dtype=torch.float32, device=self.device).view(-1, 1)
                    else:
                        # No replay yet: avoid zero-loss bootstrap
                        obs_batch = obs
                        with torch.no_grad():
                            # Optional: add tiny noise teacher to avoid exact self-imitation if ever used
                            act_batch = net(obs_batch).detach()
                        sig_batch = torch.ones((obs_batch.shape[0], 1), device=self.device)

                    net.train()
                    preds = net(obs_batch)

                    # Imitation term only if we have minimum successes
                    if use_imitation:
                        loss_im = ((preds - act_batch) ** 2 * sig_batch).mean()
                    else:
                        loss_im = None

                    # Reward prior term
                    if self.l0_adapt_signal in ("reward", "both") and (self.use_reward_prior_when_empty or replay_size > 0):
                        with torch.no_grad():
                            rew = torch.tensor(0.0, device=self.device)
                            try:
                                rew = torch.clamp(self._cumulative_rewards.mean(), min=-1.0, max=1.0) if hasattr(self, '_cumulative_rewards') else torch.tensor(0.0, device=self.device)
                            except Exception:
                                pass
                        loss_pg = (preds.pow(2).mean()) * (-rew)
                    else:
                        loss_pg = None

                    # If neither term applies (e.g., no replay and reward prior disabled), skip update
                    if loss_im is None and loss_pg is None:
                        if self._debug_mode:
                            print(f"[DEBUG][adapt] Skipping update for policy {k}: no valid loss terms (replay={replay_size}, has_min_success={has_min_success})")
                        net.eval()
                        continue

                    if loss_im is not None and loss_pg is not None:
                        loss = 0.5 * loss_im + 0.5 * loss_pg
                    elif loss_im is not None:
                        loss = loss_im
                    else:
                        loss = loss_pg

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
                    opt.step()
                    net.eval()
                    if self._debug_mode:
                        first_w = None
                        for p in net.parameters():
                            first_w = p.detach().view(-1)[0].item()
                            break
                        print(f"[DEBUG][adapt] Policy {k} adapted; loss={loss.item():.6f}, mode={self.l0_adapt_signal}, replay={replay_size}, imitate={use_imitation}, w0={first_w}")
                except Exception as e:
                    if self._debug_mode:
                        print(f"[DEBUG][adapt] perform_l0_adaptation for policy {k} failed: {e}")
        except Exception as e:
            if self._debug_mode:
                print(f"[DEBUG][adapt] perform_l0_adaptation failed: {e}")
    def _load_success_functions(self) -> None:
        """Load per-skill success functions from SuccessTerminationCfg.py files.

        It tries to resolve a callable named "*_success" or falls back to
        module.SuccessTerminationCfg.success.func if available.
        """
        if not self.skills_root_path or not self.sub_skill_folder_names:
            logger.warning("Early termination requested but skills_root_path or sub_skill_folder_names not provided. Skipping success loading.")
            return
        if len(self.sub_skill_folder_names) != self.num_sub_policies:
            logger.warning("Mismatch between number of sub skills and sub policies. Success loading may be incorrect.")

        for idx, skill_folder in enumerate(self.sub_skill_folder_names):
            try:
                success_dir = os.path.join(self.skills_root_path, skill_folder)
                success_file = os.path.join(success_dir, "SuccessTerminationCfg.py")
                if not os.path.exists(success_file):
                    logger.warning(f"SuccessTerminationCfg.py not found for sub-skill '{skill_folder}' at {success_file}")
                    continue

                mod = None
                # Preferred: import as package module so relative imports (from .base_success ...) work
                try:
                    # Build module path starting from 'isaaclab_tasks' package
                    normalized = os.path.normpath(success_dir)
                    parts = normalized.split(os.sep)
                    if "isaaclab_tasks" in parts:
                        idx_pkg = parts.index("isaaclab_tasks")
                        module_path = ".".join(parts[idx_pkg:] + ["SuccessTerminationCfg"])  # isaaclab_tasks....SuccessTerminationCfg
                        mod = importlib.import_module(module_path)
                except Exception as e:
                    mod = None

                # Fallback: construct a synthetic package so relative imports work
                if mod is None:
                    try:
                        # Ensure a root package exists
                        root_pkg_name = "genhrl_dynamic_success"
                        if root_pkg_name not in sys.modules:
                            root_pkg = types.ModuleType(root_pkg_name)
                            root_pkg.__path__ = []  # type: ignore[attr-defined]
                            sys.modules[root_pkg_name] = root_pkg

                        # Build a safe subpackage name from the folder
                        safe_folder = "".join(ch if ch.isalnum() or ch == '_' else '_' for ch in skill_folder)
                        pkg_name = f"{root_pkg_name}.{safe_folder}"
                        if pkg_name not in sys.modules:
                            pkg_mod = types.ModuleType(pkg_name)
                            pkg_mod.__path__ = [success_dir]  # type: ignore[attr-defined]
                            sys.modules[pkg_name] = pkg_mod

                        # Now load SuccessTerminationCfg as a submodule of that package
                        submodule_name = f"{pkg_name}.SuccessTerminationCfg"
                        spec = importlib.util.spec_from_file_location(submodule_name, success_file)
                        if spec is None or spec.loader is None:
                            logger.warning(f"Failed to load spec for success module: {success_file}")
                            continue
                        mod = importlib.util.module_from_spec(spec)
                        # Set proper package so relative imports (from .base_success ...) resolve to pkg_name
                        mod.__package__ = pkg_name  # type: ignore[attr-defined]
                        sys.modules[submodule_name] = mod
                        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning(f"Failed to import success module for '{skill_folder}': {e}")
                        continue

                # Optionally patch out success-state saving when not training this L0 skill
                if self.disable_success_state_saving and hasattr(mod, "save_success_state"):
                    try:
                        def _no_op_save_success_state(*args, **kwargs):
                            return None
                        mod.save_success_state = _no_op_save_success_state  # type: ignore[attr-defined]
                        if self._debug_mode:
                            print(f"[DEBUG][success] Disabled save_success_state for skill module: {skill_folder}")
                    except Exception:
                        pass

                # Try to find a function named '*_success'
                success_fn = None
                for attr_name in dir(mod):
                    if attr_name.endswith("_success") and callable(getattr(mod, attr_name)):
                        success_fn = getattr(mod, attr_name)
                        break
                if success_fn is None and hasattr(mod, "SuccessTerminationCfg"):
                    try:
                        cfg = getattr(mod, "SuccessTerminationCfg")
                        # Attempt to extract underlying function
                        if hasattr(cfg, "success"):
                            term = getattr(cfg, "success")
                            # Common pattern: DoneTerm(func=...)
                            if hasattr(term, "func") and callable(term.func):
                                success_fn = term.func
                    except Exception:
                        success_fn = None

                if success_fn is not None:
                    self._success_functions[idx] = success_fn
                    self._success_fn_loaded_indices[idx] = skill_folder
                    logger.info(f"Loaded success function for sub-skill '{skill_folder}' (policy {idx})")
                else:
                    logger.warning(f"Could not locate success function in {success_file}")
            except Exception as e:
                logger.warning(f"Failed to load success function for sub-skill '{skill_folder}': {e}")

    def _resolve_success_env(self) -> Optional[Any]:
        """Resolve the underlying Isaac Lab env to pass to success functions.

        Preference:
        - Explicit base_env if provided and has a 'scene'
        - Try common wrapper attributes on skrl_vec_env to find an object with 'scene'
        """
        try:
            if self.base_env is not None and hasattr(self.base_env, "scene"):
                return self.base_env
        except Exception:
            pass
        # Try direct unwrapped
        try:
            cand = getattr(self.skrl_vec_env, "unwrapped", None)
            if cand is not None and hasattr(cand, "scene"):
                return cand
        except Exception:
            pass
        # Try common inner attributes
        for attr in ("env", "_env", "vecenv", "_vecenv"):
            try:
                obj = getattr(self.skrl_vec_env, attr, None)
                if obj is None:
                    continue
                inner = getattr(obj, "unwrapped", obj)
                if hasattr(inner, "scene"):
                    return inner
            except Exception:
                continue
        return None


    def _load_task_success_function(self) -> None:
        """Load task-level success function if a directory is provided.

        Looks for a module file named 'TaskSuccessTerminationCfg.py' inside task_success_dir,
        and tries to resolve a callable named '*_success' or module.SuccessTerminationCfg.success.func
        similar to sub-skill success loading.
        """
        self._task_success_fn = None
        try:
            if not self.task_success_dir:
                return
            success_file = os.path.join(self.task_success_dir, "TaskSuccessTerminationCfg.py")
            if not os.path.exists(success_file):
                # Fallback: allow generic SuccessTerminationCfg.py at task root
                alt_file = os.path.join(self.task_success_dir, "SuccessTerminationCfg.py")
                if os.path.exists(alt_file):
                    success_file = alt_file
                else:
                    logger.warning(f"Task-level success file not found in: {self.task_success_dir} (looked for TaskSuccessTerminationCfg.py / SuccessTerminationCfg.py)")
                    return
            # Load as a unique module
            pkg_name = "genhrl_task_success"
            submodule_name = f"{pkg_name}.TaskSuccessTerminationCfg"
            spec = importlib.util.spec_from_file_location(submodule_name, success_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Failed to create import spec for task success file: {success_file}")
                return
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg_name  # type: ignore[attr-defined]
            sys.modules[submodule_name] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            # Try to find a function named '*_success'
            success_fn = None
            for attr_name in dir(mod):
                if attr_name.endswith("_success") and callable(getattr(mod, attr_name)):
                    success_fn = getattr(mod, attr_name)
                    break
            if success_fn is None and hasattr(mod, "SuccessTerminationCfg"):
                try:
                    cfg = getattr(mod, "SuccessTerminationCfg")
                    if hasattr(cfg, "success"):
                        term = getattr(cfg, "success")
                        if hasattr(term, "func") and callable(term.func):
                            success_fn = term.func
                except Exception:
                    success_fn = None
            self._task_success_fn = success_fn
            if self._task_success_fn is not None:
                logger.info("Loaded task-level success function")
                try:
                    print(f"[L1] Loaded task success from: {success_file}")
                    print(f"[L1] Task success callable: {getattr(self._task_success_fn, '__name__', str(self._task_success_fn))}")
                except Exception:
                    pass
            else:
                try:
                    exported = [n for n in dir(mod) if n.endswith('_success')]
                except Exception:
                    exported = []
                logger.warning(f"No task-level success callable found in {success_file}. Exported *_success: {exported}")
        except Exception as e:
            logger.warning(f"Failed to load task-level success function: {e}")

    def _load_l1_skill_success_function(self) -> None:
        """Load success function for the current L1 training skill.

        It looks for skills_root_path/<l1_skill_folder_name>/SuccessTerminationCfg.py
        and resolves a '*_success' callable or SuccessTerminationCfg.success.func.
        """
        self._l1_success_fn = None
        try:
            if not self.skills_root_path or not self.l1_skill_folder_name:
                return
            # Try common layouts:
            # A) <skills_root>/<L1>/SuccessTerminationCfg.py
            # B) <skills_root>/<L1>/skills/<L1>/SuccessTerminationCfg.py (nested pattern)
            success_dirs = [
                os.path.join(self.skills_root_path, self.l1_skill_folder_name),
                os.path.join(self.skills_root_path, self.l1_skill_folder_name, "skills", self.l1_skill_folder_name),
            ]
            success_file = None
            for cand_dir in success_dirs:
                cand_file = os.path.join(cand_dir, "SuccessTerminationCfg.py")
                if os.path.exists(cand_file):
                    success_file = cand_file
                    success_dir = cand_dir
                    break
            if not success_file:
                logger.warning(
                    f"L1-skill success file not found in tried locations: "
                    f"{os.path.join(self.skills_root_path, self.l1_skill_folder_name, 'SuccessTerminationCfg.py')} | "
                    f"{os.path.join(self.skills_root_path, self.l1_skill_folder_name, 'skills', self.l1_skill_folder_name, 'SuccessTerminationCfg.py')}"
                )
                return
            # Prepare synthetic package hierarchy to support relative imports in the module
            pkg_root = "genhrl_l1_success"
            # Sanitize folder name for a valid package identifier
            safe_folder = "".join(ch if (ch.isalnum() or ch == '_') else '_' for ch in self.l1_skill_folder_name)
            pkg_name = f"{pkg_root}.{safe_folder}"
            # Ensure root package exists
            if pkg_root not in sys.modules:
                root_pkg = types.ModuleType(pkg_root)
                root_pkg.__path__ = []  # type: ignore[attr-defined]
                sys.modules[pkg_root] = root_pkg
            # Ensure skill package exists with __path__ so relative imports work
            if pkg_name not in sys.modules:
                pkg_mod = types.ModuleType(pkg_name)
                pkg_mod.__path__ = [success_dir]  # type: ignore[attr-defined]
                sys.modules[pkg_name] = pkg_mod

            submodule_name = f"{pkg_name}.SuccessTerminationCfg"
            spec = importlib.util.spec_from_file_location(submodule_name, success_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Failed to create import spec for L1-skill success file: {success_file}")
                return
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg_name  # type: ignore[attr-defined]
            sys.modules[submodule_name] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            # Resolve callable
            success_fn = None
            for attr_name in dir(mod):
                if attr_name.endswith("_success") and callable(getattr(mod, attr_name)):
                    success_fn = getattr(mod, attr_name)
                    break
            if success_fn is None and hasattr(mod, "SuccessTerminationCfg"):
                try:
                    cfg = getattr(mod, "SuccessTerminationCfg")
                    if hasattr(cfg, "success"):
                        term = getattr(cfg, "success")
                        if hasattr(term, "func") and callable(term.func):
                            success_fn = term.func
                except Exception:
                    success_fn = None
            self._l1_success_fn = success_fn
            if self._l1_success_fn is not None:
                try:
                    print(f"[L1] Loaded L1-skill success from: {success_file}")
                    print(f"[L1] L1-skill success callable: {getattr(self._l1_success_fn, '__name__', str(self._l1_success_fn))}")
                except Exception:
                    pass
            else:
                try:
                    exported = [n for n in dir(mod) if n.endswith('_success')]
                except Exception:
                    exported = []
                logger.warning(f"No L1-skill success callable found in {success_file}. Exported *_success: {exported}")
        except Exception as e:
            logger.warning(f"Failed to load L1-skill success function: {e}")

    def _infer_task_success_dir(self) -> None:
        """Infer task success directory from skills_root_path if possible.

        Expected layout (as used in train scripts):
        .../manager_based/<ROBOT_FOLDER>/skills/<TASK_NAME>/skills  -> skills_root_path
        .../manager_based/<ROBOT_FOLDER>/tasks/<TASK_NAME>         -> task dir we need
        """
        try:
            if self.task_success_dir:
                return
            if not self.skills_root_path:
                return
            norm = os.path.normpath(self.skills_root_path)
            # Expect trailing .../skills/<TASK_NAME>/skills
            parts = norm.split(os.sep)
            if len(parts) < 4:
                return
            # Find the segment 'skills' that precedes <TASK_NAME>
            try:
                idx_skills = len(parts) - 1 - parts[::-1].index("skills")  # last occurrence
            except ValueError:
                return
            # parts[idx_skills] == 'skills', parts[idx_skills+1] == <TASK_NAME>, parts[idx_skills+2] == 'skills'
            if idx_skills + 2 >= len(parts):
                return
            task_name = parts[idx_skills + 1]
            # Rebuild path to manager_based root: .../manager_based/<ROBOT_FOLDER>
            # This is two levels above the 'skills' that encloses TASK_NAME: .../skills/<TASK_NAME>/skills
            manager_based_root = os.path.join(*parts[:idx_skills - 1]) if idx_skills - 1 >= 0 else None
            if not manager_based_root:
                return
            # Construct tasks/<TASK_NAME>
            candidate = os.path.join(manager_based_root, "tasks", task_name)
            if os.path.isdir(candidate):
                self.task_success_dir = candidate
                logger.info(f"Inferred task_success_dir: {self.task_success_dir}")
                try:
                    print(f"[L1] Inferred task success directory: {self.task_success_dir}")
                except Exception:
                    pass
        except Exception:
            return

