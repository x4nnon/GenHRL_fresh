import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Space
import torch
import numpy as np
import os
import json
from typing import Any, Dict, List, Tuple, Union, Optional

from skrl import logger


class HierarchicalVecActionWrapperL3:
    """
    Level-3 hierarchical wrapper for SKRL vectorized environments.

    L3 selects among L2 policies; selected L2 policies select among L1 policies;
    selected L1 policies select among L0 (primitive) policies that output continuous actions.

    This wrapper mirrors the design of HierarchicalVecActionWrapperL2 but adds
    one more decision level on top. It performs direct neural network inference
    for all policy levels for efficiency (no SKRL agent objects).
    """

    _observation_space: Space
    _action_space: Discrete  # L3 action space (selects L2 policy)

    def __init__(
        self,
        env: Any,
        l2_policy_checkpoint_paths: List[str],
        l2_policy_registered_names: List[str],
        l2_skill_names: List[str],
        # L1 policies referenced by the set of L2 skills
        l1_policy_checkpoint_paths: List[str],
        l1_policy_registered_names: List[str],
        # L0 primitives referenced by the set of L1 skills
        l0_policy_checkpoint_paths: List[str],
        l0_policy_registered_names: List[str],
        # Mappings and paths
        skill_library_path: str,
        skills_path: str,
        # Temporal abstraction frequencies
        steps_per_l2_policy: int = 300,
        l3_action_frequency: int = 300,
        steps_per_l1_policy: int = 100,
        l2_action_frequency: int = 100,
        steps_per_l0_policy: int = 50,
        l1_action_frequency: int = 50,
        use_random_policies: bool = False,
    ) -> None:
        logger.info("Initializing HierarchicalVecActionWrapperL3...")

        self.skrl_vec_env = env
        self.use_random_policies = use_random_policies
        self.skills_path = skills_path
        self.skill_library_path = skill_library_path

        # Base env properties
        if not hasattr(self.skrl_vec_env, "num_envs"):
            raise TypeError("The wrapped environment must have a 'num_envs' attribute.")
        self._num_envs = self.skrl_vec_env.num_envs

        if not hasattr(self.skrl_vec_env, "device"):
            self._device = getattr(self.skrl_vec_env, "_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self._device = self.skrl_vec_env.device
        logger.info(f"Wrapper L3 using device: {self.device}")

        # Load skill library JSON
        try:
            with open(self.skill_library_path, 'r') as f:
                self.skill_library: Dict[str, Any] = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load skill library from {self.skill_library_path}: {e}")
            raise

        # Store provided policy path lists
        self.l2_policy_paths = l2_policy_checkpoint_paths
        self.l2_registered_names = l2_policy_registered_names
        self.l2_skill_names = l2_skill_names

        self.l1_policy_paths = l1_policy_checkpoint_paths
        self.l1_registered_names = l1_policy_registered_names

        self.l0_policy_paths = l0_policy_checkpoint_paths
        self.l0_registered_names = l0_policy_registered_names

        # Frequencies
        self._steps_per_l2_policy = steps_per_l2_policy
        self._l3_action_frequency = l3_action_frequency
        self._steps_per_l1_policy = steps_per_l1_policy
        self._l2_action_frequency = l2_action_frequency
        self._steps_per_l0_policy = steps_per_l0_policy
        self._l1_action_frequency = l1_action_frequency

        # Action / observation spaces
        self._action_space = Discrete(len(self.l2_policy_paths))
        if not hasattr(self.skrl_vec_env, "observation_space"):
            raise TypeError("The wrapped environment must have an 'observation_space' attribute.")
        self._observation_space = self.skrl_vec_env.observation_space

        # Policy stores
        self.l2_policies: Dict[int, torch.nn.Module] = {}
        self.l1_policies: Dict[int, torch.nn.Module] = {}
        self.l0_policies: Dict[int, torch.nn.Module] = {}

        # Action spaces inferred from networks
        self.l1_action_spaces: Dict[int, Space] = {}
        self.l0_action_spaces: Dict[int, Space] = {}

        # Mappings: L2 index -> list of global L1 indices; L1 index -> list of global L0 indices
        self.l2_to_l1_mapping: Dict[int, List[int]] = {}
        self.l1_to_l0_mapping: Dict[int, List[int]] = {}

        # Internal state (per environment)
        self._active_l2_policy_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._current_l2_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_l3_action = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._active_l1_policy_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._current_l1_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_l2_action = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._active_l0_policy_global_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._current_l0_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_l1_action = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Infer base action shape
        if not hasattr(self.skrl_vec_env, 'action_space') or not hasattr(self.skrl_vec_env.action_space, 'shape'):
            raise RuntimeError("Cannot determine base environment action space shape")
        self._base_action_shape = self.skrl_vec_env.action_space.shape

        # Load all policy networks and build mappings
        self._load_all_policies_and_build_mappings()

    # Properties required by SKRL runner
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
        """Expose state_space for SKRL Runner compatibility (fallback to observation_space)."""
        if hasattr(self.skrl_vec_env, "state_space"):
            return self.skrl_vec_env.state_space  # type: ignore[attr-defined]
        logger.warning("Wrapped env does not have 'state_space'. Using observation_space as fallback.")
        return self.observation_space

    @property
    def num_agents(self) -> int:
        """Single-agent from Runner perspective (one action per env)."""
        return 1

    # --- Loading and mapping helpers ---
    def _load_all_policies_and_build_mappings(self) -> None:
        """Load L2/L1/L0 policy networks and construct L2->L1 and L1->L0 mappings from the skill library."""
        logger.info("Loading L0/L1/L2 policies for L3 wrapper...")

        # Load L0 policies
        for i, resume_path in enumerate(self.l0_policy_paths):
            self.l0_policies[i] = self._load_or_create_policy(resume_path, policy_level="L0")
            self.l0_action_spaces[i] = self._infer_action_space_from_network(self.l0_policies[i], policy_level="L0")

        # Load L1 policies and map L1 -> L0 via skill library
        for i, resume_path in enumerate(self.l1_policy_paths):
            self.l1_policies[i] = self._load_or_create_policy(resume_path, policy_level="L1")
            self.l1_action_spaces[i] = self._infer_action_space_from_network(self.l1_policies[i], policy_level="L1")
            # Build mapping from this L1 skill to its primitive L0 dependencies if available later
            # The actual mapping will be injected by train script; keep empty default here
            if i not in self.l1_to_l0_mapping:
                self.l1_to_l0_mapping[i] = []

        # Load L2 policies and map L2 -> L1 via skill library
        for i, resume_path in enumerate(self.l2_policy_paths):
            self.l2_policies[i] = self._load_or_create_policy(resume_path, policy_level="L2")
            if i not in self.l2_to_l1_mapping:
                self.l2_to_l1_mapping[i] = []

        logger.info("Finished loading hierarchical policies for L3 wrapper")

    def register_mappings(self, l2_to_l1: Dict[int, List[int]], l1_to_l0: Dict[int, List[int]]) -> None:
        """Register mappings after construction if not provided at init time."""
        self.l2_to_l1_mapping.update(l2_to_l1)
        self.l1_to_l0_mapping.update(l1_to_l0)

    def _load_or_create_policy(self, resume_path: str, policy_level: str) -> torch.nn.Module:
        """Load a torch.nn.Module policy from checkpoint or create a random fallback."""
        import torch.nn as nn

        if self.use_random_policies or not resume_path or not os.path.exists(resume_path):
            # Create simple random network based on observation/action spaces
            obs_space = self.observation_space
            obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else 64
            if policy_level == "L0":
                action_dim = self._base_action_shape[0]
            else:
                # Discrete selector; reasonable default
                action_dim = 4
            net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(),
                nn.Linear(256, action_dim)
            )
            net.to(self.device)
            net.eval()
            logger.info(f"Created random {policy_level} policy (fallback)")
            return net

        # Load checkpoint dict and try known keys
        checkpoint = torch.load(resume_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format for {policy_level}: {type(checkpoint)}")
        state_dict: Dict[str, torch.Tensor]
        if 'policy' in checkpoint and isinstance(checkpoint['policy'], dict):
            state_dict = checkpoint['policy']
        elif 'models' in checkpoint and isinstance(checkpoint['models'], dict) and 'policy' in checkpoint['models'] and isinstance(checkpoint['models']['policy'], dict):
            state_dict = checkpoint['models']['policy']
        elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            state_dict = checkpoint['state_dict']
        else:
            raise ValueError(f"Could not find policy weights in {policy_level} checkpoint")

        # Infer simple architecture from state_dict
        import torch.nn as nn
        # Find last linear layer dim by scanning keys
        out_dim = None
        for key, tensor in state_dict.items():
            if key.endswith('.weight') and tensor.dim() == 2:
                out_dim = tensor.shape[0]
        if out_dim is None:
            out_dim = self._base_action_shape[0] if policy_level == "L0" else 4

        obs_space = self.observation_space
        obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else 64
        net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, out_dim)
        )
        net.to(self.device)
        load_result = net.load_state_dict(state_dict, strict=False)
        if getattr(load_result, 'missing_keys', []):
            logger.warning(f"{policy_level} policy missing keys: {getattr(load_result, 'missing_keys', [])}")
        net.eval()
        logger.info(f"Loaded {policy_level} policy from checkpoint")
        return net

    def _infer_action_space_from_network(self, policy_network: torch.nn.Module, policy_level: str) -> Space:
        # Identify last linear layer
        last_linear = None
        for module in policy_network.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        if last_linear is None:
            # Fallbacks
            return Box(low=-1.0, high=1.0, shape=self._base_action_shape, dtype=np.float32) if policy_level == "L0" else Discrete(4)
        out_dim = last_linear.out_features
        if policy_level == "L0":
            return Box(low=-1.0, high=1.0, shape=(out_dim,), dtype=np.float32)
        else:
            return Discrete(out_dim)

    # --- SKRL Runner API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, reset_master: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._active_l2_policy_idx.zero_()
        self._current_l2_step.zero_()
        self._steps_since_l3_action.zero_()

        self._active_l1_policy_idx.zero_()
        self._current_l1_step.zero_()
        self._steps_since_l2_action.zero_()

        self._active_l0_policy_global_idx.fill_(-1)
        self._current_l0_step.zero_()
        self._steps_since_l1_action.zero_()

        if reset_master:
            obs, info = self.skrl_vec_env.reset()
        else:
            obs, info = self.skrl_vec_env._observations, self.skrl_vec_env._info
        self._last_obs = obs
        return obs, info

    def step(self, action: Optional[Union[np.ndarray, torch.Tensor]], obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Steps the environment using L3 -> L2 -> L1 -> L0 hierarchy.
        Accepts the current observation batch from the Runner (like L2 wrapper).
        """

        # Determine if new L3 decisions are needed
        needs_l3_action = (self._steps_since_l3_action % self._l3_action_frequency == 0)

        # Prepare L3 action tensor
        if needs_l3_action.any():
            if action is None:
                raise ValueError("L3 action expected but received None.")
            if isinstance(action, np.ndarray):
                l3_actions = torch.tensor(action, dtype=torch.long, device=self.device)
            elif isinstance(action, torch.Tensor):
                l3_actions = action.to(device=self.device, dtype=torch.long)
            else:
                raise TypeError(f"Unsupported L3 action type: {type(action)}")
            if l3_actions.shape != (self.num_envs,):
                if l3_actions.shape == (self.num_envs, 1):
                    l3_actions = l3_actions.squeeze(-1)
                else:
                    raise ValueError(f"Expected L3 action shape ({self.num_envs},) or ({self.num_envs}, 1), got {l3_actions.shape}")
            # Bounds check
            if (l3_actions.min() < 0) or (l3_actions.max() >= len(self.l2_policies)):
                raise ValueError("L3 action indices out of bounds for L2 policies")
            self._active_l2_policy_idx[needs_l3_action] = l3_actions[needs_l3_action]
            self._current_l2_step[needs_l3_action] = 0

        # Determine if new L2 decisions are needed (within currently selected L2 policy)
        needs_new_l2 = (self._steps_since_l2_action % self._l2_action_frequency == 0)

        # For envs needing L2 decisions, run corresponding L2 policy to pick an L1 index (local, then map to global)
        if needs_new_l2.any():
            env_indices = torch.where(needs_new_l2)[0]
            for l2_idx in torch.unique(self._active_l2_policy_idx[env_indices]):
                l2_idx_int = int(l2_idx.item())
                env_mask = needs_new_l2 & (self._active_l2_policy_idx == l2_idx_int)
                if not env_mask.any():
                    continue
                policy_net = self.l2_policies[l2_idx_int]
                batch_obs = obs[env_mask]
                with torch.no_grad():
                    l2_out = policy_net(batch_obs)
                    if l2_out.dim() > 1 and l2_out.shape[-1] > 1:
                        local_l1_actions = torch.argmax(l2_out, dim=-1)
                    else:
                        local_l1_actions = l2_out.squeeze(-1).long()
                # Map local L1 index to global L1 index using l2_to_l1_mapping
                mapped_global_l1 = []
                l2_mapping = self.l2_to_l1_mapping.get(l2_idx_int, [])
                for i in range(local_l1_actions.shape[0]):
                    sel = int(local_l1_actions[i].item())
                    if not l2_mapping:
                        mapped_global_l1.append(0)
                    else:
                        mapped_global_l1.append(l2_mapping[min(sel, len(l2_mapping) - 1)])
                mapped_global_l1_tensor = torch.tensor(mapped_global_l1, device=self.device, dtype=torch.long)
                self._active_l1_policy_idx[env_mask] = mapped_global_l1_tensor
                self._current_l1_step[env_mask] = 0

        # Determine if new L1 decisions are needed (select L0 policy)
        needs_new_l1 = (self._steps_since_l1_action % self._l1_action_frequency == 0)
        if needs_new_l1.any():
            env_indices = torch.where(needs_new_l1)[0]
            for global_l1_idx in torch.unique(self._active_l1_policy_idx[env_indices]):
                l1_idx_int = int(global_l1_idx.item())
                env_mask = needs_new_l1 & (self._active_l1_policy_idx == l1_idx_int)
                if not env_mask.any():
                    continue
                # Run the L1 policy to produce a local L0 selection
                l1_policy = self.l1_policies.get(l1_idx_int)
                if l1_policy is None:
                    # Fallback: map to first available L0
                    l0_mapping = self.l1_to_l0_mapping.get(l1_idx_int, [])
                    mapped_global_l0 = 0 if not l0_mapping else l0_mapping[0]
                    self._active_l0_policy_global_idx[env_mask] = int(mapped_global_l0)
                    self._current_l0_step[env_mask] = 0
                    continue
                batch_obs = obs[env_mask]
                with torch.no_grad():
                    l1_out = l1_policy(batch_obs)
                    if l1_out.dim() > 1 and l1_out.shape[-1] > 1:
                        local_l0_actions = torch.argmax(l1_out, dim=-1)
                    else:
                        local_l0_actions = l1_out.squeeze(-1).long()
                # Map local L0 index to global L0 index using l1_to_l0_mapping
                l0_mapping = self.l1_to_l0_mapping.get(l1_idx_int, [])
                mapped_global_l0 = []
                for i in range(local_l0_actions.shape[0]):
                    sel = int(local_l0_actions[i].item())
                    if not l0_mapping:
                        mapped_global_l0.append(0)
                    else:
                        mapped_global_l0.append(l0_mapping[min(sel, len(l0_mapping) - 1)])
                mapped_global_l0_tensor = torch.tensor(mapped_global_l0, device=self.device, dtype=torch.long)
                self._active_l0_policy_global_idx[env_mask] = mapped_global_l0_tensor
                self._current_l0_step[env_mask] = 0

        # Build primitive action batch using selected L0 policies
        primitive_actions_batch = torch.zeros((self.num_envs, *self._base_action_shape), dtype=torch.float32, device=self.device)

        # Group envs by selected L0 policy
        unique_l0 = torch.unique(self._active_l0_policy_global_idx)
        for global_l0_idx in unique_l0:
            idx_int = int(global_l0_idx.item())
            if idx_int < 0:
                # No policy selected yet → zero actions
                continue
            env_mask = (self._active_l0_policy_global_idx == idx_int)
            if not env_mask.any():
                continue
            l0_policy = self.l0_policies.get(idx_int)
            if l0_policy is None:
                continue
            batch_obs = obs[env_mask]
            with torch.no_grad():
                actions = l0_policy(batch_obs)
            primitive_actions_batch[env_mask] = actions

        # Step base environment
        next_obs, rewards_batch, terminated_batch, truncated_batch, infos = self.skrl_vec_env.step(primitive_actions_batch)

        # Update counters
        self._steps_since_l3_action += 1
        self._current_l2_step += 1
        self._steps_since_l2_action += 1
        self._current_l1_step += 1
        self._steps_since_l1_action += 1
        self._current_l0_step += 1

        # Reset counters for done envs
        env_done = torch.as_tensor(terminated_batch, dtype=torch.bool, device=self.device) | torch.as_tensor(truncated_batch, dtype=torch.bool, device=self.device)
        env_done_mask = env_done.squeeze(-1) if env_done.dim() > 1 else env_done
        if env_done_mask.any():
            self._current_l2_step[env_done_mask] = 0
            self._steps_since_l3_action[env_done_mask] = 0
            self._current_l1_step[env_done_mask] = 0
            self._steps_since_l2_action[env_done_mask] = 0
            self._current_l0_step[env_done_mask] = 0
            self._steps_since_l1_action[env_done_mask] = 0

        # Build info dict
        final_infos: Dict[str, Any] = {}
        if isinstance(infos, dict):
            final_infos.update(infos)
        final_infos["wrapper_l3_active_l2_idx"] = self._active_l2_policy_idx.cpu().numpy()
        final_infos["wrapper_l2_active_l1_idx"] = self._active_l1_policy_idx.cpu().numpy()
        final_infos["wrapper_l1_active_l0_idx"] = self._active_l0_policy_global_idx.cpu().numpy()

        self._last_obs = next_obs
        # Cast rewards and dones to tensors on device
        rewards_batch = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        terminated_batch = torch.as_tensor(terminated_batch, dtype=torch.bool, device=self.device)
        truncated_batch = torch.as_tensor(truncated_batch, dtype=torch.bool, device=self.device)
        return next_obs, rewards_batch, terminated_batch, truncated_batch, final_infos

    def close(self) -> None:
        if hasattr(self.skrl_vec_env, "close"):
            return self.skrl_vec_env.close()
        return None


