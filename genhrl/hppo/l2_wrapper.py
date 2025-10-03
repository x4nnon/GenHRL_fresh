"""
L2 Wrapper for Hierarchical PPO - Clean Implementation

This wrapper coordinates L1 policies, which in turn coordinate L0 policies.
Architecture: L2 Agent → L1 Policy → L0 Policy → Environment Actions

No wrapper nesting - direct coordination with proper index alignment.
"""

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
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

from skrl import logger
import wandb


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


class L2Wrapper:
    """
    Clean L2 wrapper that directly coordinates L1→L0 policies.
    
    Flow: L2 selects L1 → L1 selects L0 → L0 outputs env actions
    """
    
    def __init__(self,
                 env: Any,  # SKRL Vec Env Wrapper
                 sub_policy_checkpoint_paths: List[str],  # L1 policy paths
                 sub_policy_registered_names: List[str],  # L1 skill names
                 sub_skill_folder_names: List[str],  # L1 skill folders
                 skills_root_path: str,
                 steps_per_l1_policy: int = 200,
                 l2_action_frequency: int = 200,
        base_env: Optional[Any] = None,
        early_terminate_on_success: bool = True,
                 use_random_policies: bool = False,
                 debug_mode: bool = False,
        disable_success_state_saving: bool = True,
                 test_in_order_only: bool = False,
                 # L0 and L1 adaptation parameters
                 adapt_l0: bool = False,
                 adapt_l1: bool = False,
                 l0_adapt_lr: float = 1e-6,
                 l1_adapt_lr: float = 1e-6,
                 l0_adapt_std: float = 0.2,
                 l1_adapt_std: float = 0.2,
                 l0_adapt_every_n_updates: int = 1,
                 l1_adapt_every_n_updates: int = 1,
                 l0_adapt_signal: str = "success",
                 l1_adapt_signal: str = "success",
                 # Skill buffer parameters
                 use_skill_buffers: bool = True,
                 skill_buffer_capacity: int = 10000):
        """
        Initialize L2 wrapper with clean L1→L0 coordination.
        
        Args:
            env: SKRL vectorized environment
            sub_policy_checkpoint_paths: Paths to L1 policy checkpoints
            sub_policy_registered_names: L1 skill names
            sub_skill_folder_names: L1 skill folder names
            skills_root_path: Root path to skills directory
            steps_per_l1_policy: How long each L1 policy runs
            l2_action_frequency: How often L2 makes decisions
            early_terminate_on_success: Enable early termination on success
            use_random_policies: Use random policies for testing
            debug_mode: Enable debug logging
            test_in_order_only: Test mode - select L1 skills in order
            adapt_l0: Enable L0 policy adaptation
            adapt_l1: Enable L1 policy adaptation
            l0_adapt_lr: Learning rate for L0 adaptation
            l1_adapt_lr: Learning rate for L1 adaptation
            l0_adapt_std: Action std for L0 adaptation
            l1_adapt_std: Action std for L1 adaptation
            l0_adapt_every_n_updates: Apply L0 adaptation every N updates
            l1_adapt_every_n_updates: Apply L1 adaptation every N updates
            l0_adapt_signal: Signal for L0 adaptation (success/reward/both)
            l1_adapt_signal: Signal for L1 adaptation (success/reward/both)
        """
        
        logger.info("Initializing L2Wrapper with clean L1→L0 coordination...")
        
        # Store wrapped environment and basic config
        self.skrl_vec_env = env
        self.skills_root_path = skills_root_path
        self.early_terminate_on_success = early_terminate_on_success
        self.use_random_policies = use_random_policies
        self.debug_mode = debug_mode
        self.disable_success_state_saving = disable_success_state_saving
        self.test_in_order_only = test_in_order_only
        
        # Adaptation parameters
        self.adapt_l0 = adapt_l0
        self.adapt_l1 = adapt_l1
        self.l0_adapt_lr = float(l0_adapt_lr)
        self.l1_adapt_lr = float(l1_adapt_lr)
        self.l0_adapt_std = float(l0_adapt_std)
        self.l1_adapt_std = float(l1_adapt_std)
        self.l0_adapt_every_n_updates = int(max(1, l0_adapt_every_n_updates))
        self.l1_adapt_every_n_updates = int(max(1, l1_adapt_every_n_updates))
        self.l0_adapt_signal = l0_adapt_signal
        self.l1_adapt_signal = l1_adapt_signal
        self._l0_update_counter = 0
        self._l1_update_counter = 0
        
        # Environment properties
        self._num_envs = self.skrl_vec_env.num_envs
        self._device = getattr(self.skrl_vec_env, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # L1 policy configuration
        self.sub_policy_paths = sub_policy_checkpoint_paths
        self.sub_policy_registered_names = sub_policy_registered_names
        self.sub_skill_folder_names = sub_skill_folder_names
        self.num_sub_policies = len(sub_policy_checkpoint_paths)  # Number of L1 policies
        
        # Timing configuration
        self._steps_per_l1_policy = steps_per_l1_policy
        self._l2_action_frequency = l2_action_frequency
        
        # Action and observation spaces
        self._observation_space = self.skrl_vec_env.observation_space
        self._action_space = Discrete(self.num_sub_policies)  # L2 selects L1 policy
        self._l1_action_space = self.skrl_vec_env.action_space  # Environment actions
        self._l1_action_shape = self._l1_action_space.shape
        
        logger.info(f"L2 wrapper config:")
        logger.info(f"  - {self.num_sub_policies} L1 policies")
        logger.info(f"  - {self._num_envs} environments")
        logger.info(f"  - L1 duration: {self._steps_per_l1_policy} steps")
        logger.info(f"  - L2 frequency: {self._l2_action_frequency} steps")
        logger.info(f"  - L2 action space: {self._action_space}")
        logger.info(f"  - Environment action space: {self._l1_action_space}")
        
        # === POLICY LOADING ===
        
        # L1 policies: {l1_idx: l1_policy_network}
        self.l1_policies: Dict[int, torch.nn.Module] = {}
        
        # L0 policies: {l1_idx: {l0_idx: l0_policy_network}}
        self.l0_policies: Dict[int, Dict[int, torch.nn.Module]] = {}
        
        # L0 index mapping: {l1_idx: {l0_skill_name: l0_idx}}
        self.l0_index_mapping: Dict[int, Dict[str, int]] = {}
        
        # Load all policies
        self._load_all_policies()
        
        # === ADAPTATION SETUP ===
        
        # L0 optimizers: {l1_idx: {l0_idx: optimizer}}
        self._l0_optimizers: Dict[int, Dict[int, torch.optim.Optimizer]] = {}
        
        # L1 optimizers: {l1_idx: optimizer}
        self._l1_optimizers: Dict[int, torch.optim.Optimizer] = {}
        
        if self.adapt_l0:
            for l1_idx in self.l0_policies:
                self._l0_optimizers[l1_idx] = {}
                for l0_idx, net in self.l0_policies[l1_idx].items():
                    self._l0_optimizers[l1_idx][l0_idx] = torch.optim.Adam(net.parameters(), lr=self.l0_adapt_lr)
            logger.info(f"L0 adaptation enabled with lr={self.l0_adapt_lr}, std={self.l0_adapt_std}")
            logger.info(f"L0 adaptation will run every {self.l0_adapt_every_n_updates} PPO updates")
        
        if self.adapt_l1:
            for l1_idx, net in self.l1_policies.items():
                self._l1_optimizers[l1_idx] = torch.optim.Adam(net.parameters(), lr=self.l1_adapt_lr)
            logger.info(f"L1 adaptation enabled with lr={self.l1_adapt_lr}, std={self.l1_adapt_std}")
            logger.info(f"L1 adaptation will run every {self.l1_adapt_every_n_updates} PPO updates")
        
        # Replay buffers for (obs, action, signal) pairs per policy
        # L0 replay: {l1_idx: {l0_idx: {"obs": [], "act": [], "sig": []}}}
        self._l0_replay: Dict[int, Dict[int, Dict[str, list]]] = {}
        for l1_idx in self.l0_policies:
            self._l0_replay[l1_idx] = {}
            for l0_idx in self.l0_policies[l1_idx]:
                self._l0_replay[l1_idx][l0_idx] = {"obs": [], "act": [], "sig": []}
        
        # L1 replay: {l1_idx: {"obs": [], "act": [], "sig": []}}
        self._l1_replay: Dict[int, Dict[str, list]] = {}
        for l1_idx in self.l1_policies:
            self._l1_replay[l1_idx] = {"obs": [], "act": [], "sig": []}
        
        self._l0_replay_capacity: int = 2048
        self._l1_replay_capacity: int = 2048
        self._l0_replay_batch: int = 64
        self._l1_replay_batch: int = 64
        
        # Skill buffer parameters - only enable if adaptation is enabled
        self.use_skill_buffers = use_skill_buffers and (self.adapt_l0 or self.adapt_l1)
        self.skill_buffer_capacity = skill_buffer_capacity
        self.skill_buffer_frequency = 1  # Process every N steps to reduce overhead
        
        # Skill buffers for recording full trajectories during execution
        # L0 skill buffers: {l1_idx: {l0_idx: {env_idx: SkillBuffer}}}
        self._l0_skill_buffers: Dict[int, Dict[int, Dict[int, SkillBuffer]]] = {}
        for l1_idx in self.l0_policies:
            self._l0_skill_buffers[l1_idx] = {}
            for l0_idx in self.l0_policies[l1_idx]:
                self._l0_skill_buffers[l1_idx][l0_idx] = {}
                for env_idx in range(self._num_envs):
                    self._l0_skill_buffers[l1_idx][l0_idx][env_idx] = SkillBuffer(max_capacity=self.skill_buffer_capacity)
        
        # L1 skill buffers: {l1_idx: {env_idx: SkillBuffer}}
        self._l1_skill_buffers: Dict[int, Dict[int, SkillBuffer]] = {}
        for l1_idx in self.l1_policies:
            self._l1_skill_buffers[l1_idx] = {}
            for env_idx in range(self._num_envs):
                self._l1_skill_buffers[l1_idx][env_idx] = SkillBuffer(max_capacity=self.skill_buffer_capacity)
        
        if self.use_skill_buffers:
            logger.info(f"Skill buffers enabled with capacity {self.skill_buffer_capacity} per skill per environment")
        else:
            logger.info("Skill buffers disabled - using single transition recording")
        
        # Pre-step storage for adaptation
        self._prestep_obs: Optional[torch.Tensor] = None
        self._prestep_l0_actions: Optional[torch.Tensor] = None
        self._prestep_l1_actions: Optional[torch.Tensor] = None
        
        # Success tracking per skill (for adaptation)
        self._l0_skill_finish_success = {}
        self._l1_skill_finish_success = {}
        for l1_idx in self.l0_policies:
            self._l0_skill_finish_success[l1_idx] = {}
            for l0_idx in self.l0_policies[l1_idx]:
                self._l0_skill_finish_success[l1_idx][l0_idx] = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        for l1_idx in self.l1_policies:
            self._l1_skill_finish_success[l1_idx] = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        
        # === SUCCESS FUNCTION LOADING ===
        
        # L1 success functions: {l1_idx: success_function}
        self.l1_success_functions: Dict[int, Any] = {}
        
        # L0 success functions: {l1_idx: {l0_idx: success_function}}
        self.l0_success_functions: Dict[int, Dict[int, Any]] = {}
        
        if self.early_terminate_on_success:
            self._load_all_success_functions()
        
        # === STATE TRACKING (per environment) ===
        
        # Which L1 policy is active per env
        self._active_l1_policy_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        
        # Which L0 policy is active per env (within the active L1)
        self._active_l0_policy_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        
        # Timing counters
        self._steps_since_l2_action = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._steps_since_l1_action = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._current_l1_step = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._current_l0_step = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        
        # Success tracking per skill (for caching within episode)
        self._l1_completed_mask = torch.zeros((self._num_envs, self.num_sub_policies), dtype=torch.bool, device=self._device)
        self._l0_completed_mask = torch.zeros((self._num_envs, self.num_sub_policies, self._max_l0_policies_per_l1()), dtype=torch.bool, device=self._device)
        
        # === WANDB TRACKING ===
        
        # L1 choice tracking
        self._l1_selection_counts = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self._device)
        self._l1_success_counts = torch.zeros(self.num_sub_policies, dtype=torch.long, device=self._device)
        self._total_l1_selections = 0
        self._total_l1_successes = 0
        
        # L0 choice tracking (flattened across all L1s)
        total_l0_policies = sum(len(l0_dict) for l0_dict in self.l0_policies.values())
        self._l0_selection_counts = torch.zeros(total_l0_policies, dtype=torch.long, device=self._device)
        self._l0_success_counts = torch.zeros(total_l0_policies, dtype=torch.long, device=self._device)
        self._total_l0_selections = 0
        self._total_l0_successes = 0
        
        # Build skill names for logging
        self._l1_skill_names = self.sub_skill_folder_names
        self._l0_skill_names = self._build_l0_skill_names()
        
        self._wandb_log_frequency = 100
        self._step_count = 0
        
        # Test mode state
        if self.test_in_order_only:
            self._test_l1_order_counter = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
            logger.info("🧪 TEST MODE: L1 skills will be selected in order")
        
        logger.info("✅ L2Wrapper initialization complete")

    # === PROPERTIES FOR SKRL COMPATIBILITY ===
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def observation_space(self) -> Space:
        return self._observation_space
    
    @property
    def action_space(self) -> Discrete:
        return self._action_space
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def state_space(self) -> Space:
        if hasattr(self.skrl_vec_env, "state_space"):
            return self.skrl_vec_env.state_space
        return self.observation_space
    
    @property
    def num_agents(self) -> int:
        return 1

    def _max_l0_policies_per_l1(self) -> int:
        """Get the maximum number of L0 policies for any L1."""
        if not self.l0_policies:
            return 10  # Default before loading
        return max(len(l0_dict) for l0_dict in self.l0_policies.values()) if self.l0_policies else 10

    def _build_l0_skill_names(self) -> List[str]:
        """Build flattened list of L0 skill names for logging."""
        names = []
        for l1_idx in sorted(self.l0_policies.keys()):
            l0_dict = self.l0_policies[l1_idx]
            for l0_idx in sorted(l0_dict.keys()):
                # Get skill name from index mapping
                l1_mapping = self.l0_index_mapping.get(l1_idx, {})
                skill_name = None
                for name, idx in l1_mapping.items():
                    if idx == l0_idx:
                        skill_name = name
                        break
                if skill_name is None:
                    skill_name = f"l1_{l1_idx}_l0_{l0_idx}"
                names.append(f"{self._l1_skill_names[l1_idx]}.{skill_name}")
        return names

    # === POLICY LOADING ===
    
    def _load_all_policies(self):
        """Load all L1 and L0 policies with proper index alignment."""
        logger.info("Loading all L1 and L0 policies...")
        
        # Load skill library
        skill_library = self._load_skill_library()
        
        for l1_idx in range(self.num_sub_policies):
            l1_skill_name = self.sub_skill_folder_names[l1_idx]
            l1_checkpoint_path = self.sub_policy_paths[l1_idx]
            
            logger.info(f"Loading L1 skill {l1_idx}: '{l1_skill_name}'")
            
            # Load L1 policy
            self._load_l1_policy(l1_idx, l1_checkpoint_path)
            
            # Load L0 policies for this L1
            self._load_l0_policies_for_l1(l1_idx, l1_skill_name, skill_library)
        
        logger.info(f"✅ Loaded {len(self.l1_policies)} L1 policies")
        total_l0 = sum(len(l0_dict) for l0_dict in self.l0_policies.values())
        logger.info(f"✅ Loaded {total_l0} L0 policies across all L1s")

    def _load_skill_library(self) -> Dict[str, Any]:
        """Load skill library JSON."""
        skills_root = Path(self.skills_root_path)
        skill_library_path = skills_root.parent / "skill_library.json"
        
        if not skill_library_path.exists():
            logger.warning(f"Skill library not found: {skill_library_path}")
            return {}
        
        with open(skill_library_path, 'r') as f:
            return json.load(f)

    def _load_l1_policy(self, l1_idx: int, checkpoint_path: str):
        """Load a single L1 policy."""
        try:
            if self.use_random_policies and (not checkpoint_path or not os.path.exists(checkpoint_path)):
                logger.info(f"Creating random L1 policy {l1_idx} (testing mode)")
                policy_net = self._create_random_policy_network()
            else:
                logger.info(f"Loading L1 policy from: {checkpoint_path}")
                policy_net = self._load_policy_network(checkpoint_path)
            
            self.l1_policies[l1_idx] = policy_net
            logger.info(f"✅ Loaded L1 policy {l1_idx}: {self.sub_skill_folder_names[l1_idx]}")
            
        except Exception as e:
            if self.use_random_policies:
                logger.warning(f"Failed to load L1 policy {l1_idx}, using random: {e}")
                self.l1_policies[l1_idx] = self._create_random_policy_network()
            else:
                logger.error(f"Failed to load L1 policy {l1_idx}: {e}")
                raise

    def _load_l0_policies_for_l1(self, l1_idx: int, l1_skill_name: str, skill_library: Dict[str, Any]):
        """Load all L0 policies for a specific L1 skill."""
        try:
            # Get sub-skills for this L1 from skill library
            if l1_skill_name not in skill_library.get('skills', {}):
                logger.warning(f"L1 skill '{l1_skill_name}' not found in skill library")
                return
            
            l1_skill_info = skill_library['skills'][l1_skill_name]
            sub_skills = l1_skill_info.get('sub_skills', [])
            
            if not sub_skills:
                logger.warning(f"No sub-skills found for L1 skill '{l1_skill_name}'")
                return
            
            # Initialize L0 storage for this L1
            self.l0_policies[l1_idx] = {}
            self.l0_index_mapping[l1_idx] = {}
            
            skills_root = Path(self.skills_root_path)
            
            # Load each L0 sub-skill
            for l0_local_idx, l0_skill_name in enumerate(sub_skills):
                try:
                    l0_skill_dir = skills_root / l0_skill_name
                    policy_file = l0_skill_dir / "policy" / "agent.pt"
                    
                    if policy_file.exists():
                        logger.info(f"  Loading L0[{l0_local_idx}]: {l0_skill_name}")
                        l0_policy = self._load_policy_network(str(policy_file))
                        self.l0_policies[l1_idx][l0_local_idx] = l0_policy
                        self.l0_index_mapping[l1_idx][l0_skill_name] = l0_local_idx
                        logger.info(f"  ✅ Loaded L0 policy: {l0_skill_name}")
                    else:
                        logger.warning(f"  ❌ L0 policy not found: {policy_file}")
                        if self.use_random_policies:
                            l0_policy = self._create_random_policy_network()
                            self.l0_policies[l1_idx][l0_local_idx] = l0_policy
                            self.l0_index_mapping[l1_idx][l0_skill_name] = l0_local_idx
                
                except Exception as e:
                    logger.error(f"  Error loading L0 skill {l0_skill_name}: {e}")
                    if self.use_random_policies:
                        l0_policy = self._create_random_policy_network()
                        self.l0_policies[l1_idx][l0_local_idx] = l0_policy
                        self.l0_index_mapping[l1_idx][l0_skill_name] = l0_local_idx
            
            num_l0_loaded = len(self.l0_policies[l1_idx])
            logger.info(f"  ✅ Loaded {num_l0_loaded} L0 policies for L1 '{l1_skill_name}'")
            
        except Exception as e:
            logger.error(f"Error loading L0 policies for L1 {l1_skill_name}: {e}")

    def _load_policy_network(self, checkpoint_path: str) -> torch.nn.Module:
        """Load policy network from checkpoint (same method as L1 wrapper)."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Expected dict checkpoint, got {type(checkpoint)}")
            
            # Extract policy network state dict from SKRL format
            policy_state_dict = None
            
            if 'policy' in checkpoint:
                policy_state_dict = checkpoint['policy']
            elif 'models' in checkpoint and 'policy' in checkpoint['models']:
                policy_state_dict = checkpoint['models']['policy']
            elif 'state_dict' in checkpoint:
                policy_state_dict = checkpoint['state_dict']
            else:
                policy_keys = [k for k in checkpoint.keys() if 'policy' in k.lower()]
                if policy_keys:
                    policy_state_dict = checkpoint[policy_keys[0]]
                else:
                    raise ValueError(f"Could not find policy in checkpoint keys: {list(checkpoint.keys())}")
            
            # Filter network weights (exclude value_layer, log_std)
            network_state_dict = {}
            for key, value in policy_state_dict.items():
                if (('net_container.' in key or 'net.' in key) and 
                    not ('value_layer' in key or 'log_std' in key)):
                    network_state_dict[key] = value
                elif 'policy_layer' in key and not 'log_std' in key:
                    network_state_dict[key] = value
                elif (key.startswith('0.') or key.startswith('1.') or key.startswith('2.')) and \
                     not ('value_layer' in key):
                    network_state_dict[key] = value
                elif (key in ['weight', 'bias'] or key.endswith('.weight') or key.endswith('.bias')) and \
                     not ('value_layer' in key or 'log_std' in key):
                    network_state_dict[key] = value
            
            if not network_state_dict:
                network_state_dict = policy_state_dict
            
            # Create network from state dict
            policy_network = self._create_policy_network_from_state_dict(network_state_dict)
            policy_network.to(self._device)
            policy_network.eval()
            
            return policy_network
            
        except Exception as e:
            logger.error(f"Failed to load policy from {checkpoint_path}: {e}")
            raise

    def _create_policy_network_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """Create neural network from state dict."""
        import torch.nn as nn
        
        # Parse state dict to extract layer weights/biases
        layer_weights: Dict[int, torch.Tensor] = {}
        layer_biases: Dict[int, torch.Tensor] = {}
        
        for key, tensor in state_dict.items():
            if "net_container." in key and ".weight" in key:
                layer_num = int(key.split(".")[1])
                layer_weights[layer_num] = tensor
            elif "net_container." in key and ".bias" in key:
                layer_num = int(key.split(".")[1])
                layer_biases[layer_num] = tensor
            elif "policy_layer.weight" in key:
                # Find max layer number and add policy layer
                max_layer = max(layer_weights.keys()) if layer_weights else -1
                layer_weights[max_layer + 2] = tensor
            elif "policy_layer.bias" in key:
                max_layer = max(layer_weights.keys()) if layer_weights else -1
                layer_biases[max_layer + 2] = tensor
        
        if not layer_weights:
            raise ValueError("No layer weights found in state dict")
        
        # Build network
        sorted_layers = sorted(layer_weights.keys())
        layers = []
        
        for i, layer_num in enumerate(sorted_layers):
            weight = layer_weights[layer_num]
            input_size = weight.shape[1]
            output_size = weight.shape[0]
            
            linear = nn.Linear(input_size, output_size)
            layers.append(linear)
            
            # Add activation for all layers except the last
            if i < len(sorted_layers) - 1:
                layers.append(nn.ELU())
        
        network = nn.Sequential(*layers)
        
        # Load weights
        with torch.no_grad():
            linear_idx = 0
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    layer_num = sorted_layers[linear_idx]
                    module.weight.copy_(layer_weights[layer_num])
                    if layer_num in layer_biases:
                        module.bias.copy_(layer_biases[layer_num])
                    linear_idx += 1
        
        return network

    def _create_random_policy_network(self) -> torch.nn.Module:
        """Create random policy network for testing."""
        import torch.nn as nn
        
        # Get dimensions
        env_action_space = self.skrl_vec_env.action_space
        action_dim = env_action_space.shape[0] if hasattr(env_action_space, 'shape') else 12
        
        obs_space = self.observation_space
        obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') and obs_space.shape else 64
        
        # Create network
        hidden_dim = 256
        layers = [
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        ]
        
        network = nn.Sequential(*layers)
        
        # Initialize
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        network.to(self._device)
        network.eval()
        
        return network

    # === SUCCESS FUNCTION LOADING ===
    
    def _load_all_success_functions(self):
        """Load L1 and L0 success functions."""
        logger.info("Loading success functions...")
        
        # Load L1 success functions
        for l1_idx, l1_skill_name in enumerate(self.sub_skill_folder_names):
            self._load_l1_success_function(l1_idx, l1_skill_name)
        
        # Load L0 success functions
        for l1_idx in self.l0_policies:
            self.l0_success_functions[l1_idx] = {}
            l1_mapping = self.l0_index_mapping.get(l1_idx, {})
            for l0_skill_name, l0_idx in l1_mapping.items():
                self._load_l0_success_function(l1_idx, l0_idx, l0_skill_name)

    def _load_l1_success_function(self, l1_idx: int, l1_skill_name: str):
        """Load success function for L1 skill."""
        try:
            success_dir = os.path.join(self.skills_root_path, l1_skill_name)
            success_file = os.path.join(success_dir, "SuccessTerminationCfg.py")
            
            if not os.path.exists(success_file):
                logger.warning(f"L1 success file not found: {success_file}")
                return
            
            success_fn = self._extract_success_function_from_file(success_file, l1_skill_name)
            if success_fn:
                self.l1_success_functions[l1_idx] = success_fn
                logger.info(f"✅ Loaded L1 success function: {l1_skill_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load L1 success function for {l1_skill_name}: {e}")

    def _load_l0_success_function(self, l1_idx: int, l0_idx: int, l0_skill_name: str):
        """Load success function for L0 skill."""
        try:
            success_dir = os.path.join(self.skills_root_path, l0_skill_name)
            success_file = os.path.join(success_dir, "SuccessTerminationCfg.py")
            
            if not os.path.exists(success_file):
                logger.warning(f"L0 success file not found: {success_file}")
                return
            
            success_fn = self._extract_success_function_from_file(success_file, l0_skill_name)
            if success_fn:
                self.l0_success_functions[l1_idx][l0_idx] = success_fn
                logger.info(f"✅ Loaded L0 success function: {l0_skill_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load L0 success function for {l0_skill_name}: {e}")

    def _extract_success_function_from_file(self, success_file: str, skill_name: str) -> Optional[Any]:
        """Extract success function from SuccessTerminationCfg.py file."""
        try:
            # Create synthetic package for relative imports
            root_pkg_name = "genhrl_dynamic_success"
            if root_pkg_name not in sys.modules:
                root_pkg = types.ModuleType(root_pkg_name)
                root_pkg.__path__ = []
                sys.modules[root_pkg_name] = root_pkg
            
            # Create safe subpackage name
            safe_skill = "".join(ch if ch.isalnum() or ch == '_' else '_' for ch in skill_name)
            pkg_name = f"{root_pkg_name}.{safe_skill}"
            
            if pkg_name not in sys.modules:
                pkg_mod = types.ModuleType(pkg_name)
                pkg_mod.__path__ = [os.path.dirname(success_file)]
                sys.modules[pkg_name] = pkg_mod
            
            # Load module
            submodule_name = f"{pkg_name}.SuccessTerminationCfg"
            spec = importlib.util.spec_from_file_location(submodule_name, success_file)
            if spec is None or spec.loader is None:
                return None
            
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg_name
            sys.modules[submodule_name] = mod
            spec.loader.exec_module(mod)
            
            # Disable success state saving if requested
            if self.disable_success_state_saving and hasattr(mod, "save_success_state"):
                def _no_op_save_success_state(*args, **kwargs):
                    return None
                mod.save_success_state = _no_op_save_success_state
            
            # Find success function
            success_fn = None
            for attr_name in dir(mod):
                if attr_name.endswith("_success") and callable(getattr(mod, attr_name)):
                    success_fn = getattr(mod, attr_name)
                    break
            
            if success_fn is None and hasattr(mod, "SuccessTerminationCfg"):
                cfg = getattr(mod, "SuccessTerminationCfg")
                if hasattr(cfg, "success") and hasattr(cfg.success, "func"):
                    success_fn = cfg.success.func
            
            return success_fn
            
        except Exception as e:
            logger.warning(f"Failed to extract success function from {success_file}: {e}")
            return None

    # === STEP IMPLEMENTATION ===
    
    def step(self, action: Optional[Union[np.ndarray, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Step the environment with hierarchical L2→L1→L0 coordination.
        
        Args:
            action: L2 action (discrete L1 selection) when needed, None otherwise
            
        Returns:
            next_obs, rewards, terminated, truncated, infos
        """
        
        obs = getattr(self, '_last_obs', None)
        if obs is None:
            # Get current observations from environment if not stored
            obs = getattr(self.skrl_vec_env, '_observations', None)
            if obs is None:
                raise RuntimeError("No observations available - call reset() first")
        
        try:
            # === SUCCESS EVALUATION (BEFORE DECISION POINTS) ===
            
            # Evaluate success using current observations (before environment step)
            l0_success_mask, l1_success_mask = self._evaluate_success(obs)
            
            # Track successes for wandb logging
            self._track_l1_successes(l1_success_mask)
            self._track_l0_successes(l0_success_mask)
            
            # === STORE SUCCESS SAMPLES IN REPLAY BUFFERS ===
            
            # NEW: Commit successful skill buffers to replay buffer
            if self.use_skill_buffers:
                # Commit L0 success samples
                if self.adapt_l0 and l0_success_mask.any():
                    try:
                        valid_mask = l0_success_mask & (self._active_l1_policy_idx >= 0) & (self._active_l0_policy_idx >= 0)
                        if valid_mask.any():
                            env_ids = torch.where(valid_mask)[0]
                            l1_ids = self._active_l1_policy_idx[valid_mask]
                            l0_ids = self._active_l0_policy_idx[valid_mask]
                            
                            for i, (env_idx, l1_idx, l0_idx) in enumerate(zip(env_ids.tolist(), l1_ids.tolist(), l0_ids.tolist())):
                                if l1_idx in self._l0_skill_buffers and l0_idx in self._l0_skill_buffers[l1_idx]:
                                    # Commit the entire L0 skill buffer to replay
                                    self._l0_skill_buffers[l1_idx][l0_idx][env_idx].commit_to_replay(self._l0_replay[l1_idx][l0_idx])
                                    if self.debug_mode:
                                        buffer_length = self._l0_skill_buffers[l1_idx][l0_idx][env_idx].get_length()
                                        logger.debug(f"[SKILL_BUFFER] Committed L0[{l1_idx}][{l0_idx}] success for env {env_idx} with {buffer_length} transitions")
                    except Exception as e:
                        if self.debug_mode:
                            logger.warning(f"[SKILL_BUFFER] Failed to commit L0 success: {e}")
                
                # Commit L1 success samples
                if self.adapt_l1 and l1_success_mask.any():
                    try:
                        valid_mask = l1_success_mask & (self._active_l1_policy_idx >= 0)
                        if valid_mask.any():
                            env_ids = torch.where(valid_mask)[0]
                            l1_ids = self._active_l1_policy_idx[valid_mask]
                            
                            for i, (env_idx, l1_idx) in enumerate(zip(env_ids.tolist(), l1_ids.tolist())):
                                if l1_idx in self._l1_skill_buffers:
                                    # Commit the entire L1 skill buffer to replay
                                    self._l1_skill_buffers[l1_idx][env_idx].commit_to_replay(self._l1_replay[l1_idx])
                                    if self.debug_mode:
                                        buffer_length = self._l1_skill_buffers[l1_idx][env_idx].get_length()
                                        logger.debug(f"[SKILL_BUFFER] Committed L1[{l1_idx}] success for env {env_idx} with {buffer_length} transitions")
                    except Exception as e:
                        if self.debug_mode:
                            logger.warning(f"[SKILL_BUFFER] Failed to commit L1 success: {e}")
            else:
                # LEGACY: Store single successful transitions
                # Store L0 success samples
                if self.adapt_l0 and l0_success_mask.any():
                    try:
                        if self._prestep_obs is not None and self._prestep_l0_actions is not None:
                            valid_mask = l0_success_mask & (self._active_l1_policy_idx >= 0) & (self._active_l0_policy_idx >= 0)
                            if valid_mask.any():
                                env_ids = torch.where(valid_mask)[0]
                                l1_ids = self._active_l1_policy_idx[valid_mask]
                                l0_ids = self._active_l0_policy_idx[valid_mask]
                                
                                for i, (env_idx, l1_idx, l0_idx) in enumerate(zip(env_ids.tolist(), l1_ids.tolist(), l0_ids.tolist())):
                                    if l1_idx in self._l0_replay and l0_idx in self._l0_replay[l1_idx]:
                                        obs_item = self._prestep_obs[env_idx].clone()
                                        act_item = self._prestep_l0_actions[env_idx].clone()
                                        sig_val = 1.0  # Success signal
                                        
                                        buf = self._l0_replay[l1_idx][l0_idx]
                                        buf["obs"].append(obs_item)
                                        buf["act"].append(act_item)
                                        buf["sig"].append(sig_val)
                                        
                                        # Enforce capacity
                                        if len(buf["obs"]) > self._l0_replay_capacity:
                                            buf["obs"].pop(0)
                                            buf["act"].pop(0)
                                            buf["sig"].pop(0)
                    except Exception as e:
                        if self.debug_mode:
                            logger.warning(f"Failed to store L0 success samples: {e}")
                
                # Store L1 success samples
                if self.adapt_l1 and l1_success_mask.any():
                    try:
                        if self._prestep_obs is not None and self._prestep_l1_actions is not None:
                            valid_mask = l1_success_mask & (self._active_l1_policy_idx >= 0)
                            if valid_mask.any():
                                env_ids = torch.where(valid_mask)[0]
                                l1_ids = self._active_l1_policy_idx[valid_mask]
                                
                                for i, (env_idx, l1_idx) in enumerate(zip(env_ids.tolist(), l1_ids.tolist())):
                                    if l1_idx in self._l1_replay:
                                        obs_item = self._prestep_obs[env_idx].clone()
                                        act_item = self._prestep_l1_actions[i].clone() if i < len(self._prestep_l1_actions) else torch.tensor(0)
                                        sig_val = 1.0  # Success signal
                                        
                                        buf = self._l1_replay[l1_idx]
                                        buf["obs"].append(obs_item)
                                        buf["act"].append(act_item)
                                        buf["sig"].append(sig_val)
                                        
                                        # Enforce capacity
                                        if len(buf["obs"]) > self._l1_replay_capacity:
                                            buf["obs"].pop(0)
                                            buf["act"].pop(0)
                                            buf["sig"].pop(0)
                    except Exception as e:
                        if self.debug_mode:
                            logger.warning(f"Failed to store L1 success samples: {e}")
            
            # === DETERMINE DECISION POINTS ===
            
            # L2 decision points: every l2_action_frequency steps OR L1 success/timeout
            l1_timeout = (self._current_l1_step >= self._steps_per_l1_policy)
            needs_l2_decision = (self._steps_since_l2_action % self._l2_action_frequency == 0) | l1_success_mask | l1_timeout
            
            # L1 decision points: every steps_per_l1_policy/4 steps OR L0 success/timeout  
            l1_decision_frequency = max(1, self._steps_per_l1_policy // 4)  # L1 decides 4x per L1 execution
            l0_timeout = (self._current_l0_step >= (self._steps_per_l1_policy // 4))
            needs_l1_decision = (self._steps_since_l1_action % l1_decision_frequency == 0) | l0_success_mask | l0_timeout
            
            if self.debug_mode:
                logger.debug(f"Step {self._step_count}: L2 decisions: {needs_l2_decision.sum()}, L1 decisions: {needs_l1_decision.sum()}")
                if l1_success_mask.any():
                    logger.debug(f"L1 success triggered for envs: {torch.where(l1_success_mask)[0].tolist()}")
                if l0_success_mask.any():
                    logger.debug(f"L0 success triggered for envs: {torch.where(l0_success_mask)[0].tolist()}")
            
            # === L2 DECISION (SELECT L1 POLICY) ===
            
            if needs_l2_decision.any():
                if action is None:
                    raise ValueError("L2 action expected but received None")
                
                # Convert action to tensor
                if isinstance(action, np.ndarray):
                    l2_actions = torch.tensor(action, dtype=torch.long, device=self._device)
                elif isinstance(action, torch.Tensor):
                    l2_actions = action.to(device=self._device, dtype=torch.long)
                else:
                    raise TypeError(f"Unsupported action type: {type(action)}")
                
                # Handle action shape
                if l2_actions.shape == (self._num_envs, 1):
                    l2_actions = l2_actions.squeeze(-1)
                elif l2_actions.shape != (self._num_envs,):
                    raise ValueError(f"Expected action shape ({self._num_envs},), got {l2_actions.shape}")
                
                # Bounds check
                if (l2_actions < 0).any() or (l2_actions >= self.num_sub_policies).any():
                    invalid = torch.where((l2_actions < 0) | (l2_actions >= self.num_sub_policies))[0]
                    raise ValueError(f"Invalid L2 actions: {l2_actions[invalid].tolist()}")
                
                # Test mode: override with ordered selection
                if self.test_in_order_only:
                    for env_idx in torch.where(needs_l2_decision)[0]:
                        l2_actions[env_idx] = self._test_l1_order_counter[env_idx] % self.num_sub_policies
                        self._test_l1_order_counter[env_idx] += 1
                
                # Update active L1 policies
                self._active_l1_policy_idx[needs_l2_decision] = l2_actions[needs_l2_decision]
                self._current_l1_step[needs_l2_decision] = 0
                self._steps_since_l1_action[needs_l2_decision] = 0  # Reset L1 timer
                
                # NEW: Start L1 skill buffers for environments receiving new L2 actions
                if self.use_skill_buffers:
                    for l1_idx in torch.unique(l2_actions[needs_l2_decision]):
                        l1_mask = (l2_actions == l1_idx) & needs_l2_decision
                        for env_idx in torch.where(l1_mask)[0].tolist():
                            self._l1_skill_buffers[l1_idx.item()][env_idx].start()
                        if self.debug_mode:
                            logger.debug(f"[SKILL_BUFFER] Started L1 skill {l1_idx.item()} for {l1_mask.sum().item()} environments")
                
                # Track L1 selections
                self._track_l1_selections(l2_actions[needs_l2_decision])
                
                if self.debug_mode:
                    logger.debug(f"L2 decision: updated {needs_l2_decision.sum()} envs, L1 indices: {l2_actions[needs_l2_decision].tolist()}")
            
            # === L1 DECISION (SELECT L0 POLICY) ===
            
            if needs_l1_decision.any():
                # Get observations for environments needing L1 decisions
                obs_for_l1 = obs[needs_l1_decision]
                active_l1_indices = self._active_l1_policy_idx[needs_l1_decision]
                
                # Generate L1 actions (L0 selections) for each active L1 policy
                new_l0_indices = torch.zeros_like(active_l1_indices)
                
                for l1_idx in range(self.num_sub_policies):
                    l1_mask = (active_l1_indices == l1_idx)
                    if l1_mask.any():
                        l1_policy = self.l1_policies[l1_idx]
                        obs_subset = obs_for_l1[l1_mask]
                        
                        with torch.no_grad():
                            l1_output = l1_policy(obs_subset)
                            
                            # Convert L1 output to discrete L0 indices
                            num_l0_for_this_l1 = len(self.l0_policies.get(l1_idx, {}))
                            if num_l0_for_this_l1 > 0:
                                if l1_output.dim() > 1 and l1_output.shape[-1] == num_l0_for_this_l1:
                                    # Discrete output (logits)
                                    l0_indices = torch.argmax(l1_output, dim=-1)
                                else:
                                    # Continuous output - convert to discrete
                                    l1_continuous = l1_output.view(-1)
                                    l0_indices = torch.clamp(
                                        torch.round(l1_continuous).to(dtype=torch.long),
                                        min=0, max=num_l0_for_this_l1 - 1
                                    )
                                
                                new_l0_indices[l1_mask] = l0_indices
                
                # Update active L0 policies
                self._active_l0_policy_idx[needs_l1_decision] = new_l0_indices
                self._current_l0_step[needs_l1_decision] = 0
                
                # NEW: Start L0 skill buffers for environments receiving new L1 actions
                if self.use_skill_buffers:
                    for env_idx in torch.where(needs_l1_decision)[0].tolist():
                        l1_idx = active_l1_indices[torch.where(needs_l1_decision)[0] == env_idx][0].item()
                        l0_idx = new_l0_indices[torch.where(needs_l1_decision)[0] == env_idx][0].item()
                        if l1_idx in self._l0_skill_buffers and l0_idx in self._l0_skill_buffers[l1_idx]:
                            self._l0_skill_buffers[l1_idx][l0_idx][env_idx].start()
                    if self.debug_mode:
                        logger.debug(f"[SKILL_BUFFER] Started L0 skill buffers for {needs_l1_decision.sum().item()} environments")
                
                # Track L0 selections
                self._track_l0_selections(active_l1_indices, new_l0_indices)
                
                if self.debug_mode:
                    logger.debug(f"L1 decision: updated {needs_l1_decision.sum()} envs, L0 indices: {new_l0_indices.tolist()}")
            
            # === L0 ACTION GENERATION (ENVIRONMENT ACTIONS) ===
            
            env_actions = torch.zeros((self._num_envs, *self._l1_action_shape), dtype=torch.float32, device=self._device)
            
            # Store pre-step observations for adaptation
            if self.adapt_l0 or self.adapt_l1 or self.debug_mode:
                try:
                    self._prestep_obs = obs.detach().to("cpu") if torch.is_tensor(obs) else None
                except Exception:
                    self._prestep_obs = None
            
            for l1_idx in range(self.num_sub_policies):
                l1_mask = (self._active_l1_policy_idx == l1_idx)
                if l1_mask.any():
                    l0_policies_for_l1 = self.l0_policies.get(l1_idx, {})
                    
                    for l0_idx in l0_policies_for_l1:
                        l0_mask = l1_mask & (self._active_l0_policy_idx == l0_idx)
                        if l0_mask.any():
                            l0_policy = l0_policies_for_l1[l0_idx]
                            obs_subset = obs[l0_mask]
                            
                            with torch.no_grad():
                                actions_subset = l0_policy(obs_subset)
                            
                            env_actions[l0_mask] = actions_subset
            
            # === ENVIRONMENT STEP ===
            
            next_obs, rewards, terminated, truncated, infos = self.skrl_vec_env.step(env_actions)
            
            # Convert to tensors
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self._device)
            terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self._device)
            truncated = torch.as_tensor(truncated, dtype=torch.bool, device=self._device)
            
            # Store pre-step actions for adaptation
            if self.adapt_l0 or self.adapt_l1 or self.debug_mode:
                try:
                    self._prestep_l0_actions = env_actions.detach().to("cpu")
                    # Store L1 actions (L0 selections) for L1 adaptation
                    if needs_l1_decision.any():
                        # Store a full-sized vector aligned with env indices
                        decision_envs_cpu = torch.where(needs_l1_decision)[0].detach().to("cpu")
                        full_l1_actions = torch.full((self._num_envs,), -1, dtype=torch.long)
                        full_l1_actions[decision_envs_cpu] = new_l0_indices.detach().to("cpu")
                        self._prestep_l1_actions = full_l1_actions
                    else:
                        self._prestep_l1_actions = None
                except Exception:
                    self._prestep_l0_actions = None
                    self._prestep_l1_actions = None
            
            # OPTIMIZED: Append transitions to skill buffers with batching (every N steps)
            if (self.use_skill_buffers and self._prestep_obs is not None and 
                self._prestep_l0_actions is not None and 
                self._step_count % self.skill_buffer_frequency == 0):
                # Batch process active L1-L0 combinations only
                active_l1_l0_pairs = []
                for env_idx in range(self._num_envs):
                    l1_idx = self._active_l1_policy_idx[env_idx].item()
                    l0_idx = self._active_l0_policy_idx[env_idx].item()
                    if l1_idx >= 0 and l0_idx >= 0:
                        active_l1_l0_pairs.append((env_idx, l1_idx, l0_idx))
                
                # Group by L1-L0 pairs for batch processing
                l1_l0_groups = {}
                for env_idx, l1_idx, l0_idx in active_l1_l0_pairs:
                    key = (l1_idx, l0_idx)
                    if key not in l1_l0_groups:
                        l1_l0_groups[key] = []
                    l1_l0_groups[key].append(env_idx)
                
                # Process each L1-L0 group in batches
                for (l1_idx, l0_idx), env_indices in l1_l0_groups.items():
                    # Convert indices to CPU for indexing CPU tensors
                    env_indices_cpu = torch.tensor(env_indices, device='cpu')
                    
                    # Batch detach operations
                    obs_batch = self._prestep_obs[env_indices_cpu].detach()
                    act_batch = self._prestep_l0_actions[env_indices_cpu].detach()
                    rewards_batch_cpu = rewards[env_indices_cpu].cpu().numpy()
                    
                    # Append to L0 skill buffers
                    if l1_idx in self._l0_skill_buffers and l0_idx in self._l0_skill_buffers[l1_idx]:
                        for i, env_idx in enumerate(env_indices):
                            self._l0_skill_buffers[l1_idx][l0_idx][env_idx].append(
                                obs_batch[i], 
                                act_batch[i],
                                float(rewards_batch_cpu[i]),
                                self._step_count
                            )
                    
                    # Append to L1 skill buffers ONLY on L1 decision steps
                    if l1_idx in self._l1_skill_buffers and self._prestep_l1_actions is not None:
                        for i, env_idx in enumerate(env_indices):
                            # Restrict to envs that required an L1 decision this step
                            if bool(needs_l1_decision[env_idx].item()):
                                l1_action_value = self._prestep_l1_actions[env_idx]
                                self._l1_skill_buffers[l1_idx][env_idx].append(
                                    obs_batch[i],
                                    l1_action_value,
                                    float(rewards_batch_cpu[i]),
                                    self._step_count
                                )
            
            # === UPDATE COUNTERS ===
            
            self._steps_since_l2_action += 1
            self._steps_since_l1_action += 1
            self._current_l1_step += 1
            self._current_l0_step += 1
            self._step_count += 1
            
            # === RESET COUNTERS ON SUCCESS/TIMEOUT ===
            
            env_done = terminated | truncated
            
            # L0 success or timeout triggers L1 decision
            l1_reset_trigger = l0_success_mask | l0_timeout
            if l1_reset_trigger.any():
                # NEW: Discard L0 skill buffers for failed executions (not successes)
                if self.use_skill_buffers:
                    l0_failure_mask = l1_reset_trigger & ~l0_success_mask  # Reset but not success
                    if l0_failure_mask.any():
                        for env_idx in torch.where(l0_failure_mask)[0].tolist():
                            l1_idx = self._active_l1_policy_idx[env_idx].item()
                            l0_idx = self._active_l0_policy_idx[env_idx].item()
                            if l1_idx >= 0 and l0_idx >= 0 and l1_idx in self._l0_skill_buffers and l0_idx in self._l0_skill_buffers[l1_idx]:
                                self._l0_skill_buffers[l1_idx][l0_idx][env_idx].discard()
                                if self.debug_mode:
                                    logger.debug(f"[SKILL_BUFFER] Discarded L0[{l1_idx}][{l0_idx}] failure for env {env_idx}")
                
                self._steps_since_l1_action[l1_reset_trigger] = 0
                self._current_l0_step[l1_reset_trigger] = 0
            
            # L1 success or timeout triggers L2 decision  
            l2_reset_trigger = l1_success_mask | l1_timeout
            if l2_reset_trigger.any():
                # NEW: Discard L1 skill buffers for failed executions (not successes)
                if self.use_skill_buffers:
                    l1_failure_mask = l2_reset_trigger & ~l1_success_mask  # Reset but not success
                    if l1_failure_mask.any():
                        for env_idx in torch.where(l1_failure_mask)[0].tolist():
                            l1_idx = self._active_l1_policy_idx[env_idx].item()
                            if l1_idx >= 0 and l1_idx in self._l1_skill_buffers:
                                self._l1_skill_buffers[l1_idx][env_idx].discard()
                                if self.debug_mode:
                                    logger.debug(f"[SKILL_BUFFER] Discarded L1[{l1_idx}] failure for env {env_idx}")
                
                self._steps_since_l2_action[l2_reset_trigger] = 0
                self._current_l1_step[l2_reset_trigger] = 0
                self._steps_since_l1_action[l2_reset_trigger] = 0
            
            # Environment done resets everything
            if env_done.any():
                # print(f"Environments done: {env_done}")
                # print(f"shape: {env_done.shape}")
                self._steps_since_l2_action[env_done.squeeze(-1)] = 0
                self._steps_since_l1_action[env_done.squeeze(-1)] = 0
                self._current_l1_step[env_done.squeeze(-1)] = 0
                self._current_l0_step[env_done.squeeze(-1)] = 0
                # Clear success caches for completed episodes
                self._l1_completed_mask[env_done.squeeze(-1)] = False
                self._l0_completed_mask[env_done.squeeze(-1)] = False
            
            # === BUILD INFO DICT ===
            
            # Determine decision points for this step
            is_l2_decision_step = needs_l2_decision | l2_reset_trigger | env_done
            is_l1_decision_step = needs_l1_decision | l1_reset_trigger | env_done
            
            final_infos = self._build_info_dict(
                infos, is_l2_decision_step, is_l1_decision_step, 
                l0_success_mask, l1_success_mask, env_done
            )
            
            # === WANDB LOGGING ===
            
            if self._step_count % self._wandb_log_frequency == 0:
                self._log_to_wandb()
            
            # Store observations for next step
            self._last_obs = next_obs

            return next_obs, rewards, terminated, truncated, final_infos
            
        except Exception as e:
            logger.error(f"Error in L2Wrapper.step: {e}", exc_info=True)
            raise

    def _evaluate_success(self, env_state) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate L0 and L1 success functions."""
        l0_success = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        l1_success = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        
        if not self.early_terminate_on_success:
            return l0_success, l1_success
        
        try:
            # Get environment for success evaluation
            env_for_success = self._resolve_success_env()
            if env_for_success is None:
                return l0_success, l1_success
            
            # Evaluate L0 success
            for l1_idx in range(self.num_sub_policies):
                l1_mask = (self._active_l1_policy_idx == l1_idx)
                if l1_mask.any():
                    l0_success_fns = self.l0_success_functions.get(l1_idx, {})
                    
                    for l0_idx in l0_success_fns:
                        l0_mask = l1_mask & (self._active_l0_policy_idx == l0_idx)
                        if l0_mask.any():
                            success_fn = l0_success_fns[l0_idx]
                            success_result = success_fn(env_for_success)
                            
                            if isinstance(success_result, torch.Tensor):
                                success_result = success_result.to(device=self._device, dtype=torch.bool)
                                if success_result.dim() > 1:
                                    success_result = success_result.squeeze(-1)
                                l0_success[l0_mask] = success_result[l0_mask]
            
            # Evaluate L1 success
            for l1_idx in range(self.num_sub_policies):
                l1_mask = (self._active_l1_policy_idx == l1_idx)
                if l1_mask.any() and l1_idx in self.l1_success_functions:
                    success_fn = self.l1_success_functions[l1_idx]
                    success_result = success_fn(env_for_success)
                    
                    if isinstance(success_result, torch.Tensor):
                        success_result = success_result.to(device=self._device, dtype=torch.bool)
                        if success_result.dim() > 1:
                            success_result = success_result.squeeze(-1)
                        l1_success[l1_mask] = success_result[l1_mask]
            
        except Exception as e:
            if self.debug_mode:
                logger.warning(f"Success evaluation failed: {e}")
        
        return l0_success, l1_success

    def _resolve_success_env(self) -> Optional[Any]:
        """Get environment for success function evaluation."""
        # Try base_env first
        base_env = getattr(self, 'base_env', None)
        if base_env is not None and hasattr(base_env, "scene"):
            return base_env
        
        # Try unwrapped environment
        if hasattr(self.skrl_vec_env, "unwrapped") and hasattr(self.skrl_vec_env.unwrapped, "scene"):
            return self.skrl_vec_env.unwrapped
        
        # Try common attributes
        for attr in ("env", "_env", "vecenv", "_vecenv"):
            obj = getattr(self.skrl_vec_env, attr, None)
            if obj is not None:
                inner = getattr(obj, "unwrapped", obj)
                if hasattr(inner, "scene"):
                    return inner
        
        return None

    def _track_l1_selections(self, selected_l1_indices: torch.Tensor):
        """Track L1 policy selections for wandb."""
        try:
            counts = torch.bincount(selected_l1_indices, minlength=self.num_sub_policies)
            self._l1_selection_counts += counts.to(self._l1_selection_counts.device)
            self._total_l1_selections += int(selected_l1_indices.numel())
        except Exception:
            pass

    def _track_l0_selections(self, l1_indices: torch.Tensor, l0_indices: torch.Tensor):
        """Track L0 policy selections for wandb."""
        try:
            # Convert L1+L0 indices to global L0 indices for tracking
            global_l0_indices = []
            global_offset = 0
            
            for l1_idx in range(self.num_sub_policies):
                l1_mask = (l1_indices == l1_idx)
                if l1_mask.any():
                    local_l0_indices = l0_indices[l1_mask]
                    global_indices = local_l0_indices + global_offset
                    global_l0_indices.append(global_indices)
                
                global_offset += len(self.l0_policies.get(l1_idx, {}))
            
            if global_l0_indices:
                all_global_indices = torch.cat(global_l0_indices)
                counts = torch.bincount(all_global_indices, minlength=len(self._l0_skill_names))
                self._l0_selection_counts += counts.to(self._l0_selection_counts.device)
                self._total_l0_selections += int(all_global_indices.numel())
                
        except Exception:
            pass

    def _track_l1_successes(self, l1_success_mask: torch.Tensor):
        """Track L1 skill successes for wandb."""
        try:
            if l1_success_mask.any():
                valid_mask = l1_success_mask & (self._active_l1_policy_idx >= 0)
                if valid_mask.any():
                    l1_indices = self._active_l1_policy_idx[valid_mask]
                    counts = torch.bincount(l1_indices, minlength=self.num_sub_policies)
                    self._l1_success_counts += counts.to(self._l1_success_counts.device)
                    self._total_l1_successes += int(valid_mask.sum().item())
        except Exception:
            pass

    def _track_l0_successes(self, l0_success_mask: torch.Tensor):
        """Track L0 skill successes for wandb."""
        try:
            if l0_success_mask.any():
                valid_mask = l0_success_mask & (self._active_l1_policy_idx >= 0) & (self._active_l0_policy_idx >= 0)
                if valid_mask.any():
                    l1_indices = self._active_l1_policy_idx[valid_mask]
                    l0_indices = self._active_l0_policy_idx[valid_mask]
                    
                    # Convert to global L0 indices
                    global_l0_indices = []
                    global_offset = 0
                    
                    for l1_idx in range(self.num_sub_policies):
                        l1_mask = (l1_indices == l1_idx)
                        if l1_mask.any():
                            local_l0_indices = l0_indices[l1_mask]
                            global_indices = local_l0_indices + global_offset
                            global_l0_indices.append(global_indices)
                        
                        global_offset += len(self.l0_policies.get(l1_idx, {}))
                    
                    if global_l0_indices:
                        all_global_indices = torch.cat(global_l0_indices)
                        counts = torch.bincount(all_global_indices, minlength=len(self._l0_skill_names))
                        self._l0_success_counts += counts.to(self._l0_success_counts.device)
                        self._total_l0_successes += int(all_global_indices.numel())
        except Exception:
            pass

    def _build_info_dict(self, infos, is_l2_decision_step, is_l1_decision_step, 
                        l0_success, l1_success, env_done) -> Dict[str, Any]:
        """Build info dictionary for SKRL."""
        
        if isinstance(infos, list):
            final_infos = infos.copy()
            for env_idx in range(len(final_infos)):
                if final_infos[env_idx] is None:
                    final_infos[env_idx] = {}
                
                info = final_infos[env_idx]
                info["active_l1_policy_idx"] = int(self._active_l1_policy_idx[env_idx].item())
                info["active_l0_policy_idx"] = int(self._active_l0_policy_idx[env_idx].item())
                info["is_l2_decision_step"] = bool(is_l2_decision_step[env_idx].item())
                info["decision_mask"] = bool(is_l2_decision_step[env_idx].item())  # For SKRL compatibility
                info["is_l1_decision_step"] = bool(is_l1_decision_step[env_idx].item())
                # Explicit flag: preceding skill (L1 period) ended due to env termination
                info["decision_terminated"] = bool(env_done[env_idx].item())
                info["l0_skill_success"] = bool(l0_success[env_idx].item())
                info["l1_skill_success"] = bool(l1_success[env_idx].item())
                info["skill_success"] = bool(l1_success[env_idx].item())  # For SKRL compatibility (L1 success)
        else:
            final_infos = infos.copy() if isinstance(infos, dict) else {}
            final_infos["active_l1_policy_idx"] = self._active_l1_policy_idx.cpu().numpy()
            final_infos["active_l0_policy_idx"] = self._active_l0_policy_idx.cpu().numpy()
            final_infos["is_l2_decision_step"] = is_l2_decision_step.cpu().numpy()
            final_infos["decision_mask"] = is_l2_decision_step.cpu().numpy()
            final_infos["is_l1_decision_step"] = is_l1_decision_step.cpu().numpy()
            final_infos["decision_terminated"] = env_done.cpu().numpy()
            final_infos["l0_skill_success"] = l0_success.cpu().numpy()
            final_infos["l1_skill_success"] = l1_success.cpu().numpy()
            final_infos["skill_success"] = l1_success.cpu().numpy()
        
        return final_infos

    def _log_to_wandb(self):
        """Log selection and success statistics to wandb."""
        try:
            if self._total_l1_selections == 0:
                return
            
            log_dict = {}
            
            # === L1 CHOICES AND SUCCESSES ===
            l1_counts = self._l1_selection_counts.detach().cpu().tolist()
            l1_success_counts = self._l1_success_counts.detach().cpu().tolist()
            
            for i, count in enumerate(l1_counts):
                name = self._l1_skill_names[i] if i < len(self._l1_skill_names) else f"l1_{i}"
                
                # Choice statistics
                choice_percentage = (count / self._total_l1_selections) * 100.0 if self._total_l1_selections > 0 else 0.0
                log_dict[f"l1_choices/{name}_count"] = count
                log_dict[f"l1_choices/{name}_percentage"] = choice_percentage
                
                # Success statistics
                success_count = l1_success_counts[i] if i < len(l1_success_counts) else 0
                success_rate = (success_count / count) * 100.0 if count > 0 else 0.0
                log_dict[f"l1_success/{name}_count"] = success_count
                log_dict[f"l1_success/{name}_rate_pct"] = success_rate
            
            log_dict["l1_choices/total_decisions"] = self._total_l1_selections
            log_dict["l1_success/total_successes"] = self._total_l1_successes
            overall_l1_success_rate = (self._total_l1_successes / self._total_l1_selections) * 100.0 if self._total_l1_selections > 0 else 0.0
            log_dict["l1_success/overall_rate_pct"] = overall_l1_success_rate
            
            # === L0 CHOICES AND SUCCESSES ===
            if self._total_l0_selections > 0:
                l0_counts = self._l0_selection_counts.detach().cpu().tolist()
                l0_success_counts = self._l0_success_counts.detach().cpu().tolist()
                
                for i, count in enumerate(l0_counts):
                    if i < len(self._l0_skill_names):
                        name = self._l0_skill_names[i]
                        
                        # Choice statistics
                        choice_percentage = (count / self._total_l0_selections) * 100.0
                        log_dict[f"l0_choices/{name}_count"] = count
                        log_dict[f"l0_choices/{name}_percentage"] = choice_percentage
                        
                        # Success statistics
                        success_count = l0_success_counts[i] if i < len(l0_success_counts) else 0
                        success_rate = (success_count / count) * 100.0 if count > 0 else 0.0
                        log_dict[f"l0_success/{name}_count"] = success_count
                        log_dict[f"l0_success/{name}_rate_pct"] = success_rate
                
                log_dict["l0_choices/total_decisions"] = self._total_l0_selections
                log_dict["l0_success/total_successes"] = self._total_l0_successes
                overall_l0_success_rate = (self._total_l0_successes / self._total_l0_selections) * 100.0 if self._total_l0_selections > 0 else 0.0
                log_dict["l0_success/overall_rate_pct"] = overall_l0_success_rate
            
            wandb.log(log_dict, step=self._step_count)
            
        except Exception:
            # Don't break training on wandb errors
            pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, reset_master=True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset environment and wrapper state."""
        logger.debug("L2Wrapper reset called")
        
        # Reset state tracking
        self._active_l1_policy_idx.fill_(-1)
        self._active_l0_policy_idx.fill_(-1)
        self._steps_since_l2_action.zero_()
        self._steps_since_l1_action.zero_()
        self._current_l1_step.zero_()
        self._current_l0_step.zero_()
        
        # Reset success caches
        self._l1_completed_mask.zero_()
        self._l0_completed_mask.zero_()
        
        # Reset test mode counters
        if self.test_in_order_only:
            self._test_l1_order_counter.zero_()
        
        # Reset environment
        if reset_master:
            obs, info = self.skrl_vec_env.reset()
        else:
            obs = getattr(self.skrl_vec_env, '_observations', None)
            info = getattr(self.skrl_vec_env, '_info', {})
            if obs is None:
                obs, info = self.skrl_vec_env.reset()
        
        self._last_obs = obs
        return obs, info

    def close(self) -> None:
        """Close the environment."""
        logger.info("Closing L2Wrapper")
        if hasattr(self.skrl_vec_env, "close"):
            self.skrl_vec_env.close()

    def render(self, *args, **kwargs):
        """Render the environment."""
        if hasattr(self.skrl_vec_env, "render"):
            return self.skrl_vec_env.render(*args, **kwargs)
        return None
    
    # === ADAPTATION METHODS ===
    
    def perform_l0_adaptation(self) -> None:
        """Run adaptation step for L0 policies using success signals."""
        if not self.adapt_l0:
            return
        
        try:
            if self.debug_mode:
                logger.debug("perform_l0_adaptation called")
            
            # Use the last observation as a minibatch if no replay data
            if not hasattr(self, '_last_obs') or self._last_obs is None:
                if self.debug_mode:
                    logger.debug("No _last_obs available; skipping L0 adaptation")
                return
            
            obs = self._last_obs
            if not torch.is_tensor(obs):
                if self.debug_mode:
                    logger.debug(f"_last_obs not a tensor (type={type(obs)}); skipping L0 adaptation")
                return
            
            for l1_idx in self.l0_policies:
                for l0_idx, net in self.l0_policies[l1_idx].items():
                    opt = self._l0_optimizers.get(l1_idx, {}).get(l0_idx)
                    if opt is None:
                        continue
                    
                    try:
                        # Sample from replay; fall back to last obs if empty
                        buf = self._l0_replay.get(l1_idx, {}).get(l0_idx, {"obs": [], "act": [], "sig": []})
                        if len(buf["obs"]) > 0:
                            batch_size = min(self._l0_replay_batch, len(buf["obs"]))
                            idxs = random.sample(range(len(buf["obs"])), batch_size)
                            obs_batch = torch.stack([buf["obs"][i] for i in idxs]).to(self._device)
                            act_batch = torch.stack([buf["act"][i] for i in idxs]).to(self._device)
                            sig_batch = torch.tensor([buf["sig"][i] for i in idxs], dtype=torch.float32, device=self._device).view(-1, 1)
                        else:
                            obs_batch = obs
                            with torch.no_grad():
                                act_batch = net(obs_batch).detach()
                            sig_batch = torch.ones((obs_batch.shape[0], 1), device=self._device)
                        
                        net.train()
                        preds = net(obs_batch)
                        
                        # Success-weighted imitation learning
                        if self.l0_adapt_signal in ("success", "both"):
                            loss_im = ((preds - act_batch) ** 2 * sig_batch).mean()
                        else:
                            loss_im = None
                        
                        # Reward-based policy gradient (if enabled)
                        if self.l0_adapt_signal in ("reward", "both"):
                            # Simple reward-weighted log-prob surrogate
                            with torch.no_grad():
                                rew = torch.tensor(0.0, device=self._device)  # Could use recent reward
                            loss_pg = (preds.pow(2).mean()) * (-rew)
                        else:
                            loss_pg = None
                        
                        # Combine losses
                        if loss_im is not None and loss_pg is not None:
                            loss = 0.5 * loss_im + 0.5 * loss_pg
                        elif loss_im is not None:
                            loss = loss_im
                        elif loss_pg is not None:
                            loss = loss_pg
                        else:
                            loss = torch.mean((preds - act_batch) ** 2)
                        
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
                        opt.step()
                        net.eval()
                        
                        if self.debug_mode:
                            logger.debug(f"L0[{l1_idx}][{l0_idx}] adapted; loss={loss.item():.6f}, batch={len(buf['obs']) if len(buf['obs'])>0 else 'bootstrap'}")
                    
                    except Exception as e:
                        if self.debug_mode:
                            logger.warning(f"L0 adaptation failed for [{l1_idx}][{l0_idx}]: {e}")
        
        except Exception as e:
            if self.debug_mode:
                logger.warning(f"perform_l0_adaptation failed: {e}")
    
    def perform_l1_adaptation(self) -> None:
        """Run adaptation step for L1 policies using success signals."""
        if not self.adapt_l1:
            return
        
        try:
            if self.debug_mode:
                logger.debug("perform_l1_adaptation called")
            
            # Use the last observation as a minibatch if no replay data
            if not hasattr(self, '_last_obs') or self._last_obs is None:
                if self.debug_mode:
                    logger.debug("No _last_obs available; skipping L1 adaptation")
                return
            
            obs = self._last_obs
            if not torch.is_tensor(obs):
                if self.debug_mode:
                    logger.debug(f"_last_obs not a tensor (type={type(obs)}); skipping L1 adaptation")
                return
            
            for l1_idx, net in self.l1_policies.items():
                opt = self._l1_optimizers.get(l1_idx)
                if opt is None:
                    continue
                
                try:
                    # Sample from replay; fall back to last obs if empty
                    buf = self._l1_replay.get(l1_idx, {"obs": [], "act": [], "sig": []})
                    if len(buf["obs"]) > 0:
                        batch_size = min(self._l1_replay_batch, len(buf["obs"]))
                        idxs = random.sample(range(len(buf["obs"])), batch_size)
                        obs_batch = torch.stack([buf["obs"][i] for i in idxs]).to(self._device)
                        act_batch = torch.stack([buf["act"][i] for i in idxs]).to(self._device)
                        sig_batch = torch.tensor([buf["sig"][i] for i in idxs], dtype=torch.float32, device=self._device).view(-1, 1)
                    else:
                        obs_batch = obs
                        with torch.no_grad():
                            # Generate L0 selection actions
                            l1_output = net(obs_batch)
                            num_l0_for_this_l1 = len(self.l0_policies.get(l1_idx, {}))
                            if num_l0_for_this_l1 > 0:
                                if l1_output.dim() > 1 and l1_output.shape[-1] == num_l0_for_this_l1:
                                    act_batch = torch.argmax(l1_output, dim=-1).float()
                                else:
                                    act_batch = l1_output.view(-1)
                        sig_batch = torch.ones((obs_batch.shape[0], 1), device=self._device)
                    
                    net.train()
                    preds = net(obs_batch)
                    
                    # Success-weighted imitation learning
                    if self.l1_adapt_signal in ("success", "both"):
                        loss_im = ((preds - act_batch) ** 2 * sig_batch).mean()
                    else:
                        loss_im = None
                    
                    # Reward-based policy gradient (if enabled)
                    if self.l1_adapt_signal in ("reward", "both"):
                        with torch.no_grad():
                            rew = torch.tensor(0.0, device=self._device)  # Could use recent reward
                        loss_pg = (preds.pow(2).mean()) * (-rew)
                    else:
                        loss_pg = None
                    
                    # Combine losses
                    if loss_im is not None and loss_pg is not None:
                        loss = 0.5 * loss_im + 0.5 * loss_pg
                    elif loss_im is not None:
                        loss = loss_im
                    elif loss_pg is not None:
                        loss = loss_pg
                    else:
                        loss = torch.mean((preds - act_batch) ** 2)
                    
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
                    opt.step()
                    net.eval()
                    
                    if self.debug_mode:
                        logger.debug(f"L1[{l1_idx}] adapted; loss={loss.item():.6f}, batch={len(buf['obs']) if len(buf['obs'])>0 else 'bootstrap'}")
                
                except Exception as e:
                    if self.debug_mode:
                        logger.warning(f"L1 adaptation failed for [{l1_idx}]: {e}")
        
        except Exception as e:
            if self.debug_mode:
                logger.warning(f"perform_l1_adaptation failed: {e}")