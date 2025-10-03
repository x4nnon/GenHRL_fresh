from typing import Any, Dict, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

import dill as pickle
import os


# fmt: off
# [start-config-dict-torch]
HPPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": True,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance
    
    "debug_mode": False,            # enable debug printing for hierarchical learning

    # Ablation toggles (defaults preserve baseline behavior)
    "use_hierarchical_gae": True,                  # if False, skip hierarchical GAE computation
    "normalize_advantages_at_decisions": True,     # if False, skip advantage normalization on decision points
    "pass_decision_mask": True,                    # if False, pass zero mask (behaves like flat PPO)
    "zero_bootstrap_on_decision": False,           # if True, do not bootstrap across decision boundaries

    # Numerical stability parameters
    "value_scale": 1.0,                           # scale factor for value function outputs
    "max_value_magnitude": 1000.0,               # maximum absolute value for value estimates

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },

    "sub_agents": {
        "paths": []
    }
}
# [end-config-dict-torch]
# fmt: on


class HPPO(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        sub_agent_paths: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Proximal Policy Optimization (HPPO)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        :param sub_agent_paths: Optional mapping from sub-agent names to file paths
                                where their instances (saved with save_instance) are stored.
        :type sub_agent_paths: dict[str, str], optional

        :raises KeyError: If the models dictionary is missing a required key
        :raises ValueError: If loading a sub-agent fails.
        """
        _cfg = copy.deepcopy(HPPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        print(f"HPPO: cfg: {_cfg}")

        # Determine device early for loading sub-agents
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._device_type = self.device.type

        # Initialize sub_agents dictionary *before* calling super().__init__
        # It will store the loaded agent instances.
        self.sub_agents: Mapping[str, 'HPPO'] = {}

        # Load sub-agents from paths *before* initializing parent components
        # that might depend on them or need to be moved to the device.
        if sub_agent_paths:
            logger.info(f"Loading {len(sub_agent_paths)} sub-agent(s)...")
            for name, path in sub_agent_paths.items():
                try:
                    logger.info(f"Loading sub-agent '{name}' from {path} onto device {self.device}")
                    # Load the instance and move it to the correct device
                    loaded_agent = HPPO.load_instance(path, device=self.device)
                    self.sub_agents[name] = loaded_agent
                    logger.info(f"Sub-agent '{name}' loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load sub-agent '{name}' from path {path}: {e}")
                    # Decide on error handling: raise, skip, or log and continue?
                    # Raising prevents partial initialization.
                    raise ValueError(f"Failed to load sub-agent '{name}' from path {path}") from e
            logger.info("All specified sub-agents loaded.")


        # Now call the parent __init__ with the determined device
        # Pass the already determined device to the parent constructor
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device, # Pass the resolved device
            cfg=_cfg,
        )

        # Optional: reference to the managed environment for external coordination
        self._managed_env = None
        
        # Store hierarchy level for conditional decision detection
        self._hierarchy_level = None
        
        # CRITICAL FIX: Configure hierarchical memory parameters after initialization
        if hasattr(self, 'memory'):
            memory_instance = self.memory[0] if isinstance(self.memory, tuple) else self.memory
            if hasattr(memory_instance, 'hierarchy_level'):
                # Store hierarchy level for conditional decision detection
                self._hierarchy_level = getattr(memory_instance, 'hierarchy_level', 1)
                if hasattr(memory_instance, 'gae_lambda'):
                    memory_instance.gae_lambda = self.cfg.get("lambda", 0.95)  # type: ignore
                if hasattr(memory_instance, 'skill_reward_discount'):
                    memory_instance.skill_reward_discount = self.cfg.get("skill_reward_discount", 0.99)  # type: ignore
                if hasattr(memory_instance, 'discount_factor'):
                    memory_instance.discount_factor = self.cfg.get("discount_factor", 0.99)  # type: ignore
                if hasattr(memory_instance, 'zero_bootstrap_on_decision'):
                    memory_instance.zero_bootstrap_on_decision = self.cfg.get("zero_bootstrap_on_decision", False)  # type: ignore
                logger.info(f"Configured hierarchical memory with GAE lambda: {getattr(memory_instance, 'gae_lambda', 'N/A')}, "
                           f"skill reward discount: {getattr(memory_instance, 'skill_reward_discount', 'N/A')}, "
                           f"main discount: {getattr(memory_instance, 'discount_factor', 'N/A')}, "
                           f"zero bootstrap on decision: {getattr(memory_instance, 'zero_bootstrap_on_decision', 'N/A')}, "
                           f"hierarchy level: {getattr(memory_instance, 'hierarchy_level', 'N/A')}")
            else:
                logger.info("Using non-hierarchical memory (RandomMemory)")
        else:
            logger.warning("No memory instance found for configuration")

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]
        
        # Add value scaling for numerical stability
        self._value_scale = self.cfg.get("value_scale", 1.0)
        self._max_value_magnitude = self.cfg.get("max_value_magnitude", 1000.0)
        
        # Debug mode for hierarchical learning
        self._debug_mode = self.cfg["debug_mode"]

        # Add debugging state tracking focused on memory issues
        self._debug_step_count = 0
        self._debug_env0_history = []  # Detailed history for environment 0
        self._debug_memory_calls = []  # Track all memory method calls
        self._debug_storage_attempts = 0  # Count storage attempts
        self._debug_successful_storage = 0  # Count successful storage
        
        # Diagnostics: track log_prob fallback usage
        self._diag_zero_logprob_fallbacks = 0
        self._diag_logprob_samples = 0
        
        # Diagnostics: reward propagation
        self._diag_reward_total = 0
        self._diag_reward_nonzero = 0
        self._diag_reward_max = float('-inf')
        self._diag_reward_large = 0
        self._diag_reward_large_threshold = self.cfg.get("success_reward_threshold", 500.0)

        # set up automatic mixed precision - device is already known
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
        else:
            # ... (scaler fallback logic) ...
             if self._mixed_precision and not self.device.type == 'cuda':
                 logger.warning("Mixed precision requested but device is not CUDA. Disabling scaler.")
                 self.scaler = torch.cuda.amp.GradScaler(enabled=False)
             elif self._mixed_precision:
                 self.scaler = torch.cuda.amp.GradScaler(enabled=True)
             else:
                 self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
            else:
                 self.scheduler = None

            self.checkpoint_modules["optimizer"] = self.optimizer
            if self.scheduler:
                 self.checkpoint_modules["scheduler"] = self.scheduler

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
            if hasattr(self._state_preprocessor, 'to'):
                 self._state_preprocessor.to(self.device)
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
            if hasattr(self._value_preprocessor, 'to'):
                 self._value_preprocessor.to(self.device)
        else:
            self._value_preprocessor = self._empty_preprocessor

        # Ensure all components initialized *after* loading sub-agents are correctly on self.device.
        # The existing logic (moving models, preprocessors) should handle this.

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        # self.set_mode("eval")

        # Debug memory type and capabilities
        if self._debug_mode:
            print(f"\n🔍 MEMORY DEBUG INITIALIZATION:")
            print(f"  Memory type: {type(self.memory)}")
            print(f"  Memory class name: {self.memory.__class__.__name__}")
            print(f"  Has hierarchy_level: {hasattr(self.memory, 'hierarchy_level')}")
            print(f"  Has _filled_size: {hasattr(self.memory, '_filled_size')}")
            print(f"  Has create_tensors: {hasattr(self.memory, 'create_tensors')}")
            print(f"  Has sample_decision_steps_only: {hasattr(self.memory, 'sample_decision_steps_only')}")
            if hasattr(self.memory, 'hierarchy_level'):
                print(f"  Hierarchy level: {getattr(self.memory, 'hierarchy_level', 'N/A')}")

        # create tensors in memory
        if self.memory is not None:
            # Check if this is a DecisionPointMemory that has its own tensor creation method
            if hasattr(self.memory, 'create_tensors'):
                # Use DecisionPointMemory's method that includes decision_mask
                print(f"🔧 Using DecisionPointMemory.create_tensors()")
                self.memory.create_tensors(self.observation_space, self.action_space)  # type: ignore
            else:
                # Standard memory creation for regular memories
                print(f"🔧 Using standard Memory.create_tensor()")
                # Handle the case where memory might be a tuple
                memory_instance = self.memory[0] if isinstance(self.memory, tuple) else self.memory
                if self.observation_space is not None and self.action_space is not None:
                    memory_instance.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
                    memory_instance.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
                    memory_instance.create_tensor(name="rewards", size=1, dtype=torch.float32)
                    memory_instance.create_tensor(name="terminated", size=1, dtype=torch.bool)
                    memory_instance.create_tensor(name="truncated", size=1, dtype=torch.bool)
                    memory_instance.create_tensor(name="log_prob", size=1, dtype=torch.float32)
                    memory_instance.create_tensor(name="values", size=1, dtype=torch.float32)
                    memory_instance.create_tensor(name="returns", size=1, dtype=torch.float32)
                    memory_instance.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            base_tensors = ["states", "actions", "log_prob", "values", "returns", "advantages"]
            # Add decision_mask if it exists (used for filtering at sampling time)
            if hasattr(self.memory, 'hierarchy_level'):
                self._tensors_names = base_tensors + ["decision_mask"]
            else:
                self._tensors_names = base_tensors

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
            self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory"""
        # DEBUG: Print flow information every 50 steps
        # if timestep % 50 == 0:
        #     print(f"\n🔍 RECORD_TRANSITION DEBUG (Step {timestep}):")
        #     print(f"  States shape: {states.shape}")
        #     print(f"  Actions shape: {actions.shape}")
        #     print(f"  Rewards shape: {rewards.shape}")
        #     print(f"  Infos type: {type(infos)}")
        #     print(f"  Hierarchy level: {self._hierarchy_level}")
        #     print(f"  Decision flag name: {self._get_decision_flag_name()}")
            
        #     # Print infos content
        #     if isinstance(infos, list) and len(infos) > 0:
        #         env0_info = infos[0] if isinstance(infos[0], dict) else {}
        #         print(f"  Env0 info keys: {list(env0_info.keys())}")
        #         print(f"  Env0 decision_mask: {env0_info.get('decision_mask', 'NOT_FOUND')}")
        #         print(f"  Env0 {self._get_decision_flag_name()}: {env0_info.get(self._get_decision_flag_name(), 'NOT_FOUND')}")
        #     elif isinstance(infos, dict):
        #         print(f"  Dict info keys: {list(infos.keys())}")
        #         print(f"  Dict decision_mask: {infos.get('decision_mask', 'NOT_FOUND')}")
        #         print(f"  Dict {self._get_decision_flag_name()}: {infos.get(self._get_decision_flag_name(), 'NOT_FOUND')}")
        
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)
            
            # Reward diagnostics
            try:
                self._diag_reward_total += rewards.numel()
                self._diag_reward_nonzero += int((rewards != 0).sum().item())
                step_max = float(rewards.max().item())
                if step_max > self._diag_reward_max:
                    self._diag_reward_max = step_max
                self._diag_reward_large += int((rewards >= self._diag_reward_large_threshold).sum().item())
            except Exception:
                pass

            # compute values for ALL states - let DecisionPointMemory filter later during sampling
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)
                
                # Clip values to prevent extreme magnitudes
                values = torch.clamp(values, -self._max_value_magnitude, self._max_value_magnitude)
                
                # Scale values if configured
                if self._value_scale != 1.0:
                    values = values * self._value_scale

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            decision_info = self._analyze_decision_detection(infos, timestep)
            
            # DEBUG: Print decision detection results every 50 steps
            # if timestep % 50 == 0:
            #     print(f"  Decision detection result: {decision_info}")
            
            # Track environment 0 in detail
            self._debug_track_env0_step(timestep, states, actions, rewards, infos, decision_info)

            # --- Store samples in memory (SIMPLIFIED) ---
            # Always store all transitions - let DecisionPointMemory handle filtering
            self._debug_storage_attempts += 1
            
            try:
                # Handle tuple memory case
                memory_instance = self.memory[0] if isinstance(self.memory, tuple) else self.memory
                
                # Ensure log_prob is not None
                log_prob_to_store = self._current_log_prob
                # Count samples processed for diagnostics
                self._diag_logprob_samples += states.shape[0]
                if log_prob_to_store is None:
                    if self._debug_mode:
                        print("❌ WARNING: log_prob is None, using zero tensor")
                    self._diag_zero_logprob_fallbacks += states.shape[0]
                    log_prob_to_store = torch.zeros(states.shape[0], 1, device=self.device)
                
                # Build decision_mask explicitly from infos to robustly support list/dict formats
                decision_mask_value = decision_info["decision_mask_value"]
                if isinstance(decision_mask_value, np.ndarray):
                    # Convert numpy array to torch tensor and ensure correct shape
                    if decision_mask_value.ndim == 2:
                        # 2D array: take the first row or flatten
                        if decision_mask_value.shape[0] == 1:
                            # Single row: [env0, env1, env2, ...]
                            decision_mask_tensor = torch.from_numpy(decision_mask_value[0]).to(device=self.device, dtype=torch.bool).unsqueeze(-1)
                        else:
                            # Multiple rows: take first row
                            decision_mask_tensor = torch.from_numpy(decision_mask_value[0]).to(device=self.device, dtype=torch.bool).unsqueeze(-1)
                    else:
                        # 1D array: [env0, env1, env2, ...]
                        decision_mask_tensor = torch.from_numpy(decision_mask_value).to(device=self.device, dtype=torch.bool).unsqueeze(-1)
                
                else:
                    decision_mask_tensor = decision_mask_value
                
                # else:
                #     print(f"decision_mask_value: {decision_mask_value}")
                #     raise ValueError("decision_mask error")
                
                # DEBUG: Print decision mask building
                # if timestep % 50 == 0:
                #     print(f"  Building decision mask...")
                #     print(f"    Decision_mask_tensor shape: {decision_mask_tensor.shape}")
                #     print(f"    Decision_mask_tensor values: {decision_mask_tensor.flatten().tolist()}")
                #     print(f"    Decision_mask sum: {decision_mask_tensor.sum().item()}")
                
                # Ablation A3: optionally disable decision mask (behaves like flat sampling)
                if not self.cfg.get("pass_decision_mask", True):
                    decision_mask_tensor = torch.zeros((states.shape[0], 1), dtype=torch.bool, device=self.device)

                # Store transition in memory - always pass full-sized tensors and explicit decision_mask
                # Store transition in memory - always pass full-sized tensors and explicit decision_mask
                # if timestep % 50 == 0:
                #     print(f"  About to store in memory:")
                #     print(f"    States shape: {states.shape}")
                #     print(f"    Decision_mask shape: {decision_mask_tensor.shape}")
                #     print(f"    Decision_mask sum: {decision_mask_tensor.sum().item()}")
                #     print(f"    Memory type: {type(memory_instance)}")
                #     print(f"    Memory has add_samples: {hasattr(memory_instance, 'add_samples')}")
                
                memory_instance.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=log_prob_to_store,
                    values=values,
                    infos=infos,
                    decision_mask=decision_mask_tensor,
                )
                
                self._debug_successful_storage += 1
                
                # if timestep % 50 == 0:
                #     print(f"  ✅ Successfully stored in memory")
                #     if hasattr(memory_instance, '_filled_size'):
                #         print(f"    Memory filled_size: {getattr(memory_instance, '_filled_size', 'N/A')}")
                
                # Check memory state after storage (simplified)
                if self._debug_mode and timestep % 100 == 0:  # Log every 100 steps to reduce noise
                    if hasattr(memory_instance, '_filled_size'):
                        current_size = getattr(memory_instance, '_filled_size', 0)
                        print(f"💾 Memory: {current_size} samples stored")
                    # Log decision point info from infos
                    if isinstance(infos, dict):
                        decision_count = 0
                        if "decision_mask" in infos:
                            decision_flag = infos["decision_mask"]
                            if hasattr(decision_flag, 'sum'):
                                decision_count = decision_flag.sum().item()
                        print(f"🎯 Decision points this step: {decision_count}")
                
            except Exception as e:
                print(f"❌ ERROR storing in memory: {e}")
                    
            # Also store in secondary memories
            for memory in self.secondary_memories:
                try:
                    memory.add_samples(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        terminated=terminated,
                        truncated=truncated,
                        log_prob=log_prob_to_store,
                        values=values,
                        infos=infos,
                    )
                except Exception as e:
                    print(f"❌ ERROR storing in secondary memory: {e}")

            # Periodic debug summary
            if timestep % 100 == 0:
                pass
                # self._debug_print_storage_summary(timestep)

    def _get_decision_flag_name(self) -> str:
        """Get the correct decision flag name based on hierarchy level"""
        if self._hierarchy_level == 2:
            return "is_l2_decision_step"
        elif self._hierarchy_level == 1:
            return "is_decision_step"
        else:
            # Default to L1 for backward compatibility
            return "is_decision_step"

    def _analyze_decision_detection(self, infos: Any, timestep: int) -> dict:
        """Analyze how decision points are being detected"""
        decision_info = {
            "is_decision": False,
            "decision_mask_value": None,
            "source": "none_detected"
        }
        
        # Check for decision_mask first
        if "decision_mask" in infos:
            decision_mask = infos["decision_mask"]
            decision_info["decision_mask_value"] = decision_mask
            
            if isinstance(decision_mask, torch.Tensor):
                if len(decision_mask) > 0:
                    decision_info["is_decision"] = decision_mask[0].item()  # Focus on env 0
                    decision_info["source"] = "decision_mask_tensor"
                else:
                    decision_info["source"] = "decision_mask_empty_tensor"
            elif isinstance(decision_mask, bool):
                decision_info["is_decision"] = decision_mask
                decision_info["source"] = "decision_mask_bool"
            elif isinstance(decision_mask, np.ndarray):
                # Handle numpy arrays (common from L2 wrapper)
                if decision_mask.size > 0:
                    decision_info["is_decision"] = bool(decision_mask.flat[0])  # Focus on first element
                    decision_info["source"] = "decision_mask_numpy"
                else:
                    decision_info["source"] = "decision_mask_empty_numpy"
            elif isinstance(decision_mask, (list, tuple)):
                # Handle lists/tuples
                if len(decision_mask) > 0:
                    decision_info["is_decision"] = bool(decision_mask[0])
                    decision_info["source"] = "decision_mask_list"
                else:
                    decision_info["source"] = "decision_mask_empty_list"
            else:
                decision_info["source"] = f"decision_mask_unknown_type_{type(decision_mask)}"
        
        # Check for hierarchy-specific decision flag as fallback
        else:
            decision_flag_name = self._get_decision_flag_name()
            if decision_flag_name in infos:
                is_decision = infos[decision_flag_name]
                if isinstance(is_decision, torch.Tensor):
                    if len(is_decision) > 0:
                        decision_info["is_decision"] = is_decision[0].item()
                        decision_info["source"] = f"{decision_flag_name}_tensor"
                elif isinstance(is_decision, bool):
                    decision_info["is_decision"] = is_decision
                    decision_info["source"] = f"{decision_flag_name}_bool"
                else:
                    decision_info["source"] = f"{decision_flag_name}_unknown_type_{type(is_decision)}"
            else:
                decision_info["source"] = f"no_{decision_flag_name}_found"
        
        # Debug log for environment 0 decision detection issues
        if self._debug_mode and timestep % 50 == 0:
            env_0_info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else infos
            # print(f"\n🔍 DECISION DETECTION DEBUG (Step {timestep}):")
            # print(f"  Decision detected: {decision_info['is_decision']}")
            # print(f"  Detection source: {decision_info['source']}")
            # print(f"  Available info keys: {list(env_0_info.keys()) if isinstance(env_0_info, dict) else 'not dict'}")
            # # Fix linter errors: check if infos is a dict before using .get
            # if isinstance(infos, dict):
            #     print(f"  decision_mask type: {type(infos.get('decision_mask', None))}")
            #     print(f"  decision_mask value: {infos.get('decision_mask', None)}")
        
        return decision_info

    def _debug_track_env0_step(self, timestep: int, states: torch.Tensor, actions: torch.Tensor, 
                              rewards: torch.Tensor, infos: Any, decision_info: dict) -> None:
        """Track detailed information for environment 0"""
        if not self._debug_mode:
            return
            
        # Extract data for env 0
        env_0_state = states[0] if len(states.shape) > 1 else states
        env_0_action = actions[0] if len(actions.shape) > 1 else actions  
        env_0_reward = rewards[0] if len(rewards.shape) > 0 else rewards
        
        # Extract info for env 0
        env_0_info = {}
        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            env_0_info = infos[0] if isinstance(infos[0], dict) else {}
        elif isinstance(infos, dict):
            env_0_info = infos
            
        # ENHANCED: Resolve skill information from hierarchical wrapper data
        current_skill = "unknown"
        skill_step = -1
        skill_length = -1
        
        # Try to resolve skill names from hierarchical wrapper information
        if "wrapper_l2_policy_idx" in env_0_info:
            # L2 wrapper is active
            l2_policy_idx = env_0_info.get("wrapper_l2_policy_idx", [-1])
            l2_idx = l2_policy_idx[0] if isinstance(l2_policy_idx, (list, np.ndarray)) else l2_policy_idx
            l1_step = env_0_info.get("wrapper_l1_step", [-1])
            l1_step_val = l1_step[0] if isinstance(l1_step, (list, np.ndarray)) else l1_step
            
            # Try to get actual L1 skill name from wrapper
            if "wrapper_l1_skill_names" in env_0_info:
                l1_skill_names = env_0_info.get("wrapper_l1_skill_names", ["unknown"])
                current_skill = l1_skill_names[0] if len(l1_skill_names) > 0 else f"L1_skill_{l2_idx}"
            else:
                current_skill = f"L1_skill_{l2_idx}" if l2_idx >= 0 else "unknown_L1"
            
            skill_step = l1_step_val
            skill_length = env_0_info.get("steps_per_l1_policy", 300)  # Default L1 skill length
            
            # If L0 information is available, show that instead for more detail
            if "wrapper_l0_policy_global_idx" in env_0_info:
                l0_policy_idx = env_0_info.get("wrapper_l0_policy_global_idx", [-1])
                l0_idx = l0_policy_idx[0] if isinstance(l0_policy_idx, (list, np.ndarray)) else l0_policy_idx
                l0_step = env_0_info.get("wrapper_l0_step", [-1])
                l0_step_val = l0_step[0] if isinstance(l0_step, (list, np.ndarray)) else l0_step
                
                # Show both L1 skill and current L0 policy
                l0_skill_name = f"L0_skill_{l0_idx}" if l0_idx >= 0 else "unknown_L0"
                current_skill = f"{current_skill}→{l0_skill_name}"  # Show hierarchy: L1→L0
                skill_step = l0_step_val
                skill_length = env_0_info.get("steps_per_l0_policy", 100)  # Default L0 skill length
        
        elif "wrapper_l1_policy_idx" in env_0_info:
            # L1 wrapper is active (legacy key path)
            l1_policy_idx = env_0_info.get("wrapper_l1_policy_idx", [-1])
            l1_idx = l1_policy_idx[0] if isinstance(l1_policy_idx, (list, np.ndarray)) else l1_policy_idx
            l0_step = env_0_info.get("wrapper_l0_step", [-1])
            l0_step_val = l0_step[0] if isinstance(l0_step, (list, np.ndarray)) else l0_step
            
            current_skill = f"L0_skill_{l1_idx}" if isinstance(l1_idx, (int, np.integer)) and l1_idx >= 0 else "unknown_L0"
            skill_step = l0_step_val
            skill_length = env_0_info.get("steps_per_l0_policy", 100)  # Default L0 skill length
        elif "wrapper_l0_policy_idx" in env_0_info:
            # L1 wrapper (current implementation) exposes the selected L0 index
            l0_policy_idx = env_0_info.get("wrapper_l0_policy_idx", [-1])
            l0_idx = l0_policy_idx[0] if isinstance(l0_policy_idx, (list, np.ndarray)) else l0_policy_idx
            l0_step = env_0_info.get("wrapper_l0_step", [-1])
            l0_step_val = l0_step[0] if isinstance(l0_step, (list, np.ndarray)) else l0_step
            
            # If names are ever provided, prefer them; otherwise fallback to index-based label
            if "wrapper_l0_skill_names" in env_0_info:
                l0_skill_names = env_0_info.get("wrapper_l0_skill_names", ["unknown"])  # type: ignore[assignment]
                current_skill = l0_skill_names[0] if len(l0_skill_names) > 0 else f"L0_skill_{l0_idx}"
            else:
                current_skill = f"L0_skill_{l0_idx}" if isinstance(l0_idx, (int, np.integer)) and l0_idx >= 0 else "unknown_L0"
            skill_step = l0_step_val
            skill_length = env_0_info.get("steps_per_l0_policy", 100)
        
        else:
            # Try fallback to direct skill info if provided
            current_skill = env_0_info.get("current_skill", "unknown")
            skill_step = env_0_info.get("skill_step", -1)
            skill_length = env_0_info.get("skill_length", -1)

        # Create comprehensive debug entry
        debug_entry = {
            "timestep": timestep,
            "is_decision": decision_info.get("is_decision", False),
            "decision_mask_value": decision_info.get("decision_mask_value", None),
            "decision_source": decision_info.get("source", "unknown"),
            "current_skill": current_skill,
            "skill_step": skill_step,
            "skill_length": skill_length,
            "action": env_0_action.cpu().numpy() if torch.is_tensor(env_0_action) else env_0_action,
            "reward": env_0_reward.item() if torch.is_tensor(env_0_reward) else env_0_reward,
            "state_norm": torch.norm(env_0_state).item() if torch.is_tensor(env_0_state) else 0.0,
            "infos_keys": list(env_0_info.keys()) if env_0_info else [],
            # Add hierarchical debug info
            "l2_policy_idx": env_0_info.get("wrapper_l2_policy_idx", "N/A"),
            "l1_step": env_0_info.get("wrapper_l1_step", "N/A"),
            "l0_policy_idx": env_0_info.get("wrapper_l0_policy_global_idx", "N/A"),
            "l0_step": env_0_info.get("wrapper_l0_step", "N/A"),
        }
        
        self._debug_env0_history.append(debug_entry)
        
        # Only log for environment 0 and less frequently  
        if timestep % 100 == 0:  # Every 100 steps
            pass
            # print(f"\n🎯 ENV_0 (Step {timestep}): Skill='{debug_entry['current_skill']}' ({debug_entry['skill_step']}/{debug_entry['skill_length']}), Decision={debug_entry['is_decision']}, Reward={debug_entry['reward']:.3f}")

    def _debug_track_memory_operation(self, operation: str, success: bool, details: Optional[dict] = None) -> None:
        """Track memory operations for debugging"""
        if not self._debug_mode:
            return
            
        entry = {
            "operation": operation,
            "success": success,
            "timestep": self._debug_step_count,
            "details": details or {}
        }
        
        self._debug_memory_calls.append(entry)
        
        # Log failures immediately
        if not success:
            print(f"❌ MEMORY OPERATION FAILED: {operation} at step {self._debug_step_count}")
            if details:
                print(f"   Details: {details}")

    def _debug_print_storage_summary(self, timestep: int) -> None:
        """Print summary of memory storage attempts"""
        if not self._debug_mode:
            return
            
        # Only print every 200 steps to reduce noise
        if timestep % 200 != 0:
            return
            
        print(f"\n📊 ENV_0 SUMMARY (Step {timestep}):")
        print(f"  Storage: {self._debug_successful_storage}/{self._debug_storage_attempts} successful")
        
        # Check current memory state
        if hasattr(self.memory, '_filled_size'):
            filled_size = getattr(self.memory, '_filled_size', 0)
            print(f"  Memory size: {filled_size}")
        
        # Analyze recent decision detection for env 0 only
        if len(self._debug_env0_history) >= 100:
            recent_history = self._debug_env0_history[-100:]
            decision_count = sum(1 for entry in recent_history if entry["is_decision"])
            total_reward = sum(entry["reward"] for entry in recent_history)
            print(f"  Recent decisions (env 0): {decision_count}/100")
            print(f"  Recent avg reward: {total_reward/100:.3f}")

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int, **kwargs) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :param kwargs: Additional keyword arguments (e.g., observations, reward, terminated, truncated, info, next_observations)
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            # self.set_mode("eval")
            # MEMORY CLEANUP: Clear memory after each rollout to prevent accumulation
            self._clear_memory_after_rollout()

        # Only pass expected args to super
        super().post_interaction(timestep, timesteps)

    # Allow external components (runner) to provide the environment wrapper instance
    def set_managed_env(self, env: Any) -> None:
        self._managed_env = env
        if getattr(self, '_debug_mode', False):
            print("[DEBUG][adapt] Managed env set on agent")

    def _update(self, timestep: int, timesteps: int) -> None:
        """Update the policy and value networks using hierarchical PPO following standard SKRL pattern"""
        
        # Check if we have any data to update with
        memory_instance = self.memory[0] if isinstance(self.memory, tuple) else self.memory
        filled_size = getattr(memory_instance, '_filled_size', 0)
        
        # DEBUG: Print memory state before update
        # print(f"\n🔄 UPDATE DEBUG (Timestep {timestep}):")
        # print(f"  Memory type: {type(memory_instance)}")
        # print(f"  Memory filled_size: {filled_size}")
        # print(f"  Memory has get_tensor_by_name: {hasattr(memory_instance, 'get_tensor_by_name')}")
        
        # Fix linter errors for memory access and .item() usage
        # Replace memory_instance['decision_mask'] with get_tensor_by_name('decision_mask')
        # Only call .item() on tensors
        total_decisions = 0
        try:
            if hasattr(memory_instance, 'get_tensor_by_name'):
                decision_mask = memory_instance.get_tensor_by_name("decision_mask")
                # print(f"  Decision_mask tensor: {decision_mask}")
                # if decision_mask is not None:
                #     print(f"  Decision_mask shape: {decision_mask.shape}")
                #     print(f"  Decision_mask dtype: {decision_mask.dtype}")
                #     print(f"  Decision_mask sum: {decision_mask.sum().item()}")
                total_decisions = decision_mask.sum().item() if decision_mask is not None and hasattr(decision_mask, 'sum') else 0
                # For DecisionPointMemory, the REAL check is whether we have decision points
                actual_data_available = total_decisions > 0
            else:
                # For regular memory, use filled_size
                actual_data_available = filled_size > 0
        except Exception as e:
            print(f"  ❌ Error checking decision_mask: {e}")
            # Fallback to filled_size check
            actual_data_available = filled_size > 0
        
        if not actual_data_available:
            print(f"❌ No usable data in memory for update (decision_points: {total_decisions}, filled_size: {filled_size})")
            return
        
        print(f"🔄 Starting HPPO update with {total_decisions} decision points (filled_size: {filled_size})")
        
        # STEP 1: COMPUTE HIERARCHICAL GAE (following standard SKRL PPO pattern)
        # This is where standard PPO computes bootstrap values and GAE
        if self.cfg.get("use_hierarchical_gae", True) and hasattr(memory_instance, 'compute_hierarchical_gae'):
            print("Computing hierarchical GAE...")
            
            # Get required tensors from memory for GAE computation
            rewards = memory_instance.get_tensor_by_name("rewards")
            values = memory_instance.get_tensor_by_name("values")
            terminated = memory_instance.get_tensor_by_name("terminated")
            truncated = memory_instance.get_tensor_by_name("truncated")
            decision_mask = memory_instance.get_tensor_by_name("decision_mask")
            
            # Create dones mask
            dones = terminated | truncated
            
            # Call the GAE method with correct parameters - it will handle the computation internally
            # Allow compute_hierarchical_gae to return optional third value (decision_terminated_mask)
            gae_out = memory_instance.compute_hierarchical_gae(  # type: ignore
                rewards, values, dones, decision_mask, filled_size
            )
            if isinstance(gae_out, tuple) and len(gae_out) >= 2:
                hierarchical_returns, hierarchical_advantages = gae_out[0], gae_out[1]
            else:
                hierarchical_returns, hierarchical_advantages = gae_out

            # Normalize advantages at decision points every update for stability
            if self.cfg.get("normalize_advantages_at_decisions", True):
                try:
                    adv = hierarchical_advantages
                    dm = decision_mask
                    # Squeeze trailing singleton dims to work in [T, N]
                    if adv.dim() == 3 and adv.size(-1) == 1:
                        adv2d = adv.squeeze(-1)
                    else:
                        adv2d = adv
                    if dm is not None and isinstance(dm, torch.Tensor):
                        dm2d = dm.squeeze(-1) if (dm.dim() == 3 and dm.size(-1) == 1) else dm
                        dm_bool = dm2d.to(dtype=torch.bool)
                        # Flatten to 1D mask over T*N
                        mask_flat = dm_bool.flatten()
                        adv_flat = adv2d.flatten()
                        sel = adv_flat[mask_flat]
                        if sel.numel() > 0:
                            sel_mean = sel.mean()
                            sel_std = sel.std()
                            # Use more robust normalization with minimum std threshold
                            min_std = 0.1  # Prevent division by extremely small std
                            if sel_std > min_std:
                                normalized_adv = (sel - sel_mean) / torch.clamp(sel_std, min=min_std)
                                # Clip normalized advantages to prevent extreme values
                                normalized_adv = torch.clamp(normalized_adv, -10.0, 10.0)
                                adv_flat[mask_flat] = normalized_adv
                            else:
                                adv_flat[mask_flat] = 0.0
                            adv2d = adv_flat.view_as(adv2d)
                            hierarchical_advantages = adv2d.unsqueeze(-1) if (adv.dim() == 3 and adv.size(-1) == 1) else adv2d
                except Exception as e:
                    print(f"⚠️ Advantage normalization skipped due to error: {e}")

            # Update the stored returns and advantages in memory
            memory_instance.set_tensor_by_name("returns", hierarchical_returns)
            memory_instance.set_tensor_by_name("advantages", hierarchical_advantages)
            
            print("✅ Hierarchical GAE computation complete")

            # Debug: verify termination segments carry negative returns
            try:
                dm = memory_instance.get_tensor_by_name("decision_mask")
                dt = memory_instance.get_tensor_by_name("decision_terminated")
                rets = hierarchical_returns
                if dm is not None and dt is not None:
                    # Normalize shapes to 2D [T, N]
                    if dm.dim() == 3 and dm.size(-1) == 1:
                        dm2d = dm.squeeze(-1)
                    else:
                        dm2d = dm
                    if dt.dim() == 3 and dt.size(-1) == 1:
                        dt2d = dt.squeeze(-1)
                    else:
                        dt2d = dt
                    if rets.dim() == 3 and rets.size(-1) == 1:
                        rets2d = rets.squeeze(-1)
                    else:
                        rets2d = rets
                    mask_terminated_decisions = (dm2d.bool() & dt2d.bool())
                    if mask_terminated_decisions.any():
                        terminated_returns = rets2d[mask_terminated_decisions]
                        mean_ret = float(terminated_returns.mean().item()) if terminated_returns.numel() > 0 else 0.0
                        count = int(mask_terminated_decisions.sum().item())
                        print(f"[HPPO][debug] Terminated decision points: {count}, mean return: {mean_ret:.4f}")
            except Exception:
                pass
        else:
            # Fallback: compute standard (flat) GAE over all timesteps and then sample decision steps
            try:
                print("⚠️ Hierarchical GAE disabled; computing standard GAE over timesteps")

                rewards = memory_instance.get_tensor_by_name("rewards")
                values = memory_instance.get_tensor_by_name("values")
                terminated = memory_instance.get_tensor_by_name("terminated")
                truncated = memory_instance.get_tensor_by_name("truncated")
                decision_mask = memory_instance.get_tensor_by_name("decision_mask")

                # Shapes -> [T, N]
                squeeze_trailing = rewards.dim() == 3 and rewards.size(-1) == 1
                rew2d = rewards.squeeze(-1) if squeeze_trailing else rewards
                val2d = values.squeeze(-1) if (values.dim() == 3 and values.size(-1) == 1) else values
                ter2d = terminated.squeeze(-1) if (terminated.dim() == 3 and terminated.size(-1) == 1) else terminated
                tru2d = truncated.squeeze(-1) if (truncated.dim() == 3 and truncated.size(-1) == 1) else truncated

                dones2d = (ter2d | tru2d).to(dtype=rew2d.dtype)

                T = rew2d.size(0)
                N = rew2d.size(1)
                adv2d = torch.zeros_like(rew2d)
                ret2d = torch.zeros_like(rew2d)

                gamma = float(self.cfg.get("discount_factor", 0.99))
                lam = float(self.cfg.get("lambda", 0.95))

                next_adv = torch.zeros(N, device=rew2d.device, dtype=rew2d.dtype)
                next_value = val2d[-1]
                next_non_terminal = (1.0 - dones2d[-1])

                for t in reversed(range(T)):
                    if t < T - 1:
                        next_value = val2d[t + 1]
                        next_non_terminal = (1.0 - dones2d[t])
                    # delta_t = r_t + gamma * V_{t+1} * (1-done_t) - V_t
                    delta = rew2d[t] + gamma * next_value * next_non_terminal - val2d[t]
                    next_adv = delta + gamma * lam * next_non_terminal * next_adv
                    adv2d[t] = next_adv
                    ret2d[t] = adv2d[t] + val2d[t]

                # Optional: normalize advantages at decision steps for stability
                if self.cfg.get("normalize_advantages_at_decisions", True) and isinstance(decision_mask, torch.Tensor):
                    dm2d = decision_mask.squeeze(-1) if (decision_mask.dim() == 3 and decision_mask.size(-1) == 1) else decision_mask
                    dm_bool = dm2d.to(dtype=torch.bool)
                    mask_flat = dm_bool.flatten()
                    adv_flat = adv2d.flatten()
                    sel = adv_flat[mask_flat]
                    if sel.numel() > 0:
                        sel_mean = sel.mean()
                        sel_std = sel.std()
                        min_std = 0.1
                        if sel_std > min_std:
                            normalized_adv = (sel - sel_mean) / torch.clamp(sel_std, min=min_std)
                            normalized_adv = torch.clamp(normalized_adv, -10.0, 10.0)
                            adv_flat[mask_flat] = normalized_adv
                        else:
                            adv_flat[mask_flat] = 0.0
                        adv2d = adv_flat.view_as(adv2d)

                # Restore original shape [T, N, 1] if needed and write back
                ret_out = ret2d.unsqueeze(-1) if squeeze_trailing else ret2d
                adv_out = adv2d.unsqueeze(-1) if squeeze_trailing else adv2d
                memory_instance.set_tensor_by_name("returns", ret_out)
                memory_instance.set_tensor_by_name("advantages", adv_out)

                # Basic sanity log
                try:
                    dm2 = decision_mask.squeeze(-1) if (isinstance(decision_mask, torch.Tensor) and decision_mask.dim() == 3 and decision_mask.size(-1) == 1) else decision_mask
                    if isinstance(dm2, torch.Tensor) and dm2.any():
                        sel_adv = adv2d[dm2.to(dtype=torch.bool)]
                        if sel_adv.numel() > 0:
                            print(f"Flat GAE computed: decision-step adv mean={float(sel_adv.mean().item()):.4f}, std={float(sel_adv.std().item()):.4f}")
                except Exception:
                    pass
            except Exception as e:
                print(f"❌ Error computing flat GAE fallback: {e}")

        # STEP 2: SAMPLE MINI-BATCHES (following standard SKRL PPO pattern)
        try:
            if hasattr(memory_instance, 'hierarchy_level') and hasattr(memory_instance, 'sample_decision_steps_only'):
                print("Sampling decision steps only...")
                sampled_batches = memory_instance.sample_decision_steps_only(names=self._tensors_names, mini_batches=self._mini_batches)  # type: ignore
                
                if not sampled_batches:
                    print("❌ No decision step batches available")
                    return
                print(f"✅ Generated {len(sampled_batches)} decision-step mini-batches")
                
                # Quick batch info
                if sampled_batches:
                    first_batch = sampled_batches[0]
                    print(f"First batch: {first_batch[0].shape[0]} samples, advantages mean={first_batch[5].mean().item():.4f}")
            else:
                print("Using regular memory sampling...")
                sampled_batches = memory_instance.sample_all(names=tuple(self._tensors_names), mini_batches=self._mini_batches)
                print(f"✅ Generated {len(sampled_batches)} regular mini-batches")
        except Exception as e:
            print(f"❌ ERROR during sampling: {e}")
            return

        # STEP 3: TRAIN ON MINI-BATCHES (following standard SKRL PPO pattern)
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        
        total_batches_processed = 0

        # learning epochs - simplified logging
        for epoch in range(self._learning_epochs):
            kl_divergences = []
            
            # mini-batches loop
            for batch_idx, (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in enumerate(sampled_batches):

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence (keep gradients for early stopping)
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # Batch diagnostics (no gradients): advantages, log-ratios, ratios, returns, predicted values
                    try:
                        with torch.no_grad():
                            adv = sampled_advantages
                            adv_mean = float(adv.mean().item())
                            adv_std = float(adv.std().item())
                            logratio = (next_log_prob - sampled_log_prob)
                            lr_mean = float(logratio.mean().item())
                            lr_max = float(logratio.max().item())
                            ratio_e = torch.exp(logratio)
                            r_mean = float(ratio_e.mean().item())
                            r_max = float(ratio_e.max().item())
                            ret_mean = float(sampled_returns.mean().item())
                            ret_std = float(sampled_returns.std().item())
                            # Predicted value stats in raw reward space
                            pv, _, _ = self.value.act({"states": sampled_states}, role="value")
                            pv = self._value_preprocessor(pv, inverse=True)
                            pv_mean = float(pv.mean().item())
                            pv_std = float(pv.std().item())
                            pv_min = float(pv.min().item())
                            pv_max = float(pv.max().item())
                            # print(f"[diag][epoch {epoch} batch {batch_idx}] adv μ={adv_mean:.3f} σ={adv_std:.3f} | log-ratio μ={lr_mean:.3f} max={lr_max:.3f} | ratio μ={r_mean:.3f} max={r_max:.3f} | returns μ={ret_mean:.3f} σ={ret_std:.3f} | V̂ μ={pv_mean:.3f} σ={pv_std:.3f} min={pv_min:.3f} max={pv_max:.3f}")
                    except Exception:
                        pass

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss with numerical stability safeguards
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    
                    # Invert value preprocessor so predictions are in raw reward space
                    predicted_values = self._value_preprocessor(predicted_values, inverse=True)
                    
                    # Clip predicted values in raw space to prevent extreme values
                    predicted_values = torch.clamp(predicted_values, -1000.0, 1000.0)
                    
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    
                    # Use Huber loss instead of MSE for better numerical stability
                    value_diff = sampled_returns - predicted_values
                    value_loss = self._value_loss_scale * torch.where(
                        torch.abs(value_diff) < 1.0,
                        0.5 * value_diff ** 2,
                        torch.abs(value_diff) - 0.5
                    ).mean()

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += float(entropy_loss)
                
                total_batches_processed += 1

            # update learning rate
            if self._learning_rate_scheduler and hasattr(self, 'scheduler') and self.scheduler is not None:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # Final summary
        avg_policy_loss = cumulative_policy_loss / max(1, total_batches_processed)
        avg_value_loss = cumulative_value_loss / max(1, total_batches_processed)
        print(f"Update complete: Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}")
        
        # Diagnostics: log_prob fallback usage for this rollout
        total_lp = max(1, self._diag_logprob_samples)
        fallback_pct = (self._diag_zero_logprob_fallbacks / total_lp) * 100.0
        print(f"[diag] log_prob fallbacks this rollout: {self._diag_zero_logprob_fallbacks}/{self._diag_logprob_samples} ({fallback_pct:.2f}%)")
        
        # Diagnostics: reward propagation summary
        total_rw = max(1, self._diag_reward_total)
        nonzero_pct = (self._diag_reward_nonzero / total_rw) * 100.0
        print(f"[diag] rewards: nonzero {self._diag_reward_nonzero}/{self._diag_reward_total} ({nonzero_pct:.2f}%), max {self._diag_reward_max:.3f}, >= {self._diag_reward_large_threshold} count {self._diag_reward_large}")
        # Reset per-rollout counters after update
        self._diag_zero_logprob_fallbacks = 0
        self._diag_logprob_samples = 0
        self._diag_reward_total = 0
        self._diag_reward_nonzero = 0
        self._diag_reward_max = float('-inf')
        self._diag_reward_large = 0
        
        # Show env 0 summary occasionally  
        if timestep % 1000 == 0 and len(self._debug_env0_history) > 0:
            recent_decisions = [h for h in self._debug_env0_history[-100:] if h.get('is_decision', False)]
            recent_reward = sum(h.get('reward', 0) for h in self._debug_env0_history[-100:])
            print(f"ENV_0 Summary: {len(recent_decisions)} recent decisions, avg reward: {recent_reward/100:.3f}")

        # record data
        total_expected_batches = max(1, self._learning_epochs * self._mini_batches)
        self.track_data("Loss / Policy loss", cumulative_policy_loss / total_expected_batches)
        self.track_data("Loss / Value loss", cumulative_value_loss / total_expected_batches)
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / total_expected_batches
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler and hasattr(self, 'scheduler') and self.scheduler is not None:
            # Some schedulers may not implement get_last_lr()
            try:
                last_lr = self.scheduler.get_last_lr()[0]
            except Exception:
                try:
                    last_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                except Exception:
                    last_lr = self._learning_rate
            self.track_data("Learning / Learning rate", last_lr)

        # After main policy update, optionally adapt lower-level skills if the environment supports it
        try:
            if hasattr(self, '_managed_env') and self._managed_env is not None:
                env = self._managed_env
                # Unwrap to our wrapper if nested
                target = env
                # Many wrappers nest env in .env/._env/etc.; attempt a shallow unwrap
                for attr in ("env", "_env"):
                    target = getattr(target, attr, target)
                               
                # L2 training case: adapt both L0 and L1
                # L0 adaptation
                if hasattr(target, 'adapt_l0') and getattr(target, 'adapt_l0'):
                    if not hasattr(target, '_l0_update_counter'):
                        target._l0_update_counter = 0
                    target._l0_update_counter += 1
                    every_l0 = getattr(target, 'l0_adapt_every_n_updates', 1) if hasattr(target, 'l0_adapt_every_n_updates') else 1
                    if target._l0_update_counter % max(1, every_l0) == 0 and hasattr(target, 'perform_l0_adaptation'):
                        if getattr(self, '_debug_mode', False):
                            print(f"[DEBUG][adapt] Triggering L0 adaptation at PPO update {target._l0_update_counter}")
                        target.perform_l0_adaptation()
                        print("L0 adaptation done")
                
                # L1 adaptation
                if hasattr(target, 'adapt_l1') and getattr(target, 'adapt_l1'):
                    if not hasattr(target, '_l1_update_counter'):
                        target._l1_update_counter = 0
                    target._l1_update_counter += 1
                    every_l1 = getattr(target, 'l1_adapt_every_n_updates', 1) if hasattr(target, 'l1_adapt_every_n_updates') else 1
                    if target._l1_update_counter % max(1, every_l1) == 0 and hasattr(target, 'perform_l1_adaptation'):
                        if getattr(self, '_debug_mode', False):
                            print(f"[DEBUG][adapt] Triggering L1 adaptation at PPO update {target._l1_update_counter}")
                        target.perform_l1_adaptation()
                        print("L1 adaptation done")
        except Exception as e:
            print(f"❌ ERROR during L0/L1 adaptation: {e}")
            if getattr(self, '_debug_mode', False):
                print(f"[DEBUG][adapt] L0 post-update adaptation failed: {e}")

    def save_instance(self, path: str) -> None:
        """Save the entire agent instance using dill.

        This saves the complete object, including models, optimizers,
        preprocessors, internal counters, configuration, and references
        to other objects (like sub-agents if implemented).

        :param path: Path to the file where the agent instance will be saved.
        :type path: str
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # Move agent to CPU before saving to avoid GPU-specific serialization issues
            original_device = self.device
            self.to("cpu")
            with open(path, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Agent instance pickled to {path}")
            # Move back to original device
            self.to(original_device)
        except Exception as e:
            logger.error(f"Error pickling agent instance: {e}")
            # Attempt to move back to original device even if saving failed
            self.to(original_device)


    @staticmethod
    def load_instance(path: str, device: Optional[Union[str, torch.device]] = None) -> "HPPO":
        """Load an agent instance from a file using dill.

        Loads the complete object saved by `save_instance`.

        :param path: Path to the file from which the agent instance will be loaded.
        :type path: str
        :param device: The device to move the loaded agent to (e.g., "cuda:0" or "cpu").
                       If None, attempts to use CUDA if available, otherwise CPU.
        :type device: str or torch.device, optional

        :return: The loaded HPPO agent instance.
        :rtype: HPPO
        """
        try:
            with open(path, "rb") as f:
                agent = pickle.load(f)
            logger.info(f"Agent instance unpickled from {path}")

            # Determine target device
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)

            # Move the loaded agent to the target device
            agent.to(device)

            return agent
        except FileNotFoundError:
            logger.error(f"Error unpickling agent: File not found at {path}")
            raise
        except Exception as e:
            logger.error(f"Error unpickling agent instance: {e}")
            raise

    def to(self, device: Optional[Union[str, torch.device]]) -> None:
        """Move the agent and its components to the specified device.

        This includes models, optimizer state, memory buffers, and sub-agents.

        :param device: The target device (e.g., "cuda:0", "cpu").
        :type device: str or torch.device
        """
        if device is None:
            logger.warning("Device is None in agent.to(), no action taken.")
            return
        new_device = torch.device(device)

        # Optimization: Check if device is actually changing for the agent itself
        # We still need to potentially move sub-components even if the top-level device matches
        # because sub-agents might have been loaded to a different device initially if not specified
        # or if `to` is called multiple times with different devices.
        needs_move = self.device != new_device

        # Always iterate through sub-agents to ensure they are on the correct device
        if hasattr(self, 'sub_agents') and self.sub_agents:
            logger.info(f"Ensuring {len(self.sub_agents)} sub-agent(s) are on device: {new_device}")
            for name, sub_agent in self.sub_agents.items():
                if hasattr(sub_agent, 'to'):
                    try:
                        # Recursively call 'to' on sub_agent
                        sub_agent.to(new_device)
                    except Exception as e:
                        logger.error(f"Failed to move sub_agent '{name}' to {new_device}: {e}")
                else:
                    logger.warning(f"Sub-agent '{name}' does not have a 'to' method.")

        # Now proceed only if the main agent's device needs changing
        if not needs_move:
             # logger.debug(f"Agent already on device {new_device}. Skipping main component move.")
             # Still need to update self.device if sub-components were moved? No, self.device reflects the *intended* device.
             return

        self.device = new_device
        self._device_type = self.device.type
        # Careful with calling super().to - ensure it exists and does what's expected
        if hasattr(super(), 'to'):
             try:
                 super().to(self.device) # Call parent's to method if needed
             except Exception as e:
                  logger.error(f"Error calling super().to({self.device}): {e}")

        logger.info(f"Moving agent main components to device: {self.device}")

        if self.policy is not None:
            self.policy.to(self.device)
        if self.value is not None and self.value is not self.policy:
            self.value.to(self.device)

        # Move optimizer state
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            state[k] = v.to(self.device)
                        except Exception as e:
                             logger.error(f"Failed to move optimizer tensor {k} to {self.device}: {e}")


        # Move preprocessor states
        if hasattr(self, '_state_preprocessor') and hasattr(self._state_preprocessor, 'to'):
            try:
                self._state_preprocessor.to(self.device)
            except Exception as e:
                logger.error(f"Failed to move state_preprocessor to {self.device}: {e}")

        if hasattr(self, '_value_preprocessor') and hasattr(self._value_preprocessor, 'to'):
            try:
                self._value_preprocessor.to(self.device)
            except Exception as e:
                logger.error(f"Failed to move value_preprocessor to {self.device}: {e}")


        # Re-initialize scaler for the correct device type
        scaler_state = self.scaler.state_dict() if hasattr(self, 'scaler') else None
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
        else:
            if self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        if scaler_state is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as e:
                logger.error(f"Failed to load scaler state after moving device: {e}")


        # Move memory tensors (assuming base class or manual handling covers this)
        # The check 'if hasattr(super(), 'to')' handles this.
        # ... (memory moving logic if needed) ...


        logger.info(f"Agent main components successfully moved to device: {self.device}")

    def save_debug_data(self, filepath: str) -> None:
        """Save comprehensive debug data to a file for analysis"""
        if not self._debug_mode:
            print("Debug mode not enabled - no data to save")
            return
            
        debug_data = {
            "metadata": {
                "total_steps": self._debug_step_count,
                "storage_attempts": self._debug_storage_attempts,
                "successful_storage": self._debug_successful_storage,
                "memory_type": type(self.memory).__name__,
                "has_hierarchy_level": hasattr(self.memory, 'hierarchy_level'),
                "hierarchy_level": getattr(self.memory, 'hierarchy_level', None),
                "memory_methods": [method for method in dir(self.memory) if not method.startswith('_')]
            },
            "env0_history": self._debug_env0_history,
            "memory_calls": self._debug_memory_calls,
            "analysis": {
                "total_decisions_detected": len([h for h in self._debug_env0_history if h.get('is_decision', False)]),
                "decision_rate": len([h for h in self._debug_env0_history if h.get('is_decision', False)]) / max(1, len(self._debug_env0_history)),
                "unique_skills": list(set(h.get('current_skill', 'unknown') for h in self._debug_env0_history)),
                "avg_reward": sum(h.get('reward', 0) for h in self._debug_env0_history) / max(1, len(self._debug_env0_history))
            }
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                for entry in debug_data["env0_history"]:
                    if 'action' in entry and hasattr(entry['action'], 'tolist'):
                        entry['action'] = entry['action'].tolist()
                
                json.dump(debug_data, f, indent=2)
            print(f"✅ Debug data saved to {filepath}")
            
            # Print quick summary
            print(f"\n📋 DEBUG DATA SUMMARY:")
            print(f"  Total steps tracked: {debug_data['metadata']['total_steps']}")
            print(f"  Decisions detected: {debug_data['analysis']['total_decisions_detected']}")
            print(f"  Decision rate: {debug_data['analysis']['decision_rate']:.3f}")
            print(f"  Unique skills seen: {len(debug_data['analysis']['unique_skills'])}")
            print(f"  Storage success rate: {(self._debug_successful_storage/max(1,self._debug_storage_attempts))*100:.1f}%")
            
        except Exception as e:
            print(f"❌ Error saving debug data: {e}")

    def get_debug_summary(self) -> dict:
        """Get a quick debug summary without saving to file"""
        if not self._debug_mode:
            return {"error": "Debug mode not enabled"}
            
        return {
            "steps_tracked": len(self._debug_env0_history),
            "decisions_detected": len([h for h in self._debug_env0_history if h.get('is_decision', False)]),
            "storage_attempts": self._debug_storage_attempts,
            "successful_storage": self._debug_successful_storage,
            "memory_type": type(self.memory).__name__,
            "has_hierarchical_features": hasattr(self.memory, 'hierarchy_level'),
            "recent_skills": [h.get('current_skill', 'unknown') for h in self._debug_env0_history[-10:]] if self._debug_env0_history else []
        }

    def _clear_memory_after_rollout(self) -> None:
        """Clear memory after each rollout to prevent CUDA memory accumulation."""
        try:
            # Handle tuple memory case
            memory_instance = self.memory[0] if isinstance(self.memory, tuple) else self.memory
            
            # Clear memory using proper methods
            if hasattr(memory_instance, 'clear'):
                memory_instance.clear()
            elif hasattr(memory_instance, 'reset'):
                memory_instance.reset()
            else:
                # Fallback to direct attribute manipulation if no proper methods exist
                if hasattr(memory_instance, '_filled_size'):
                    memory_instance._filled_size = 0
                if hasattr(memory_instance, '_memory_index'):
                    memory_instance._memory_index = 0
                if hasattr(memory_instance, '_memory_pointer'):
                    memory_instance._memory_pointer = 0
                    
                # Clear tensors if they exist
                if hasattr(memory_instance, '_tensors'):
                    for tensor_name, tensor in memory_instance._tensors.items():
                        if tensor is not None:
                            tensor.zero_()
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self._debug_mode:
                memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                print(f"🧹 Memory cleared after rollout: {collected} objects collected, GPU memory: {memory_mb:.1f}MB")
                
        except Exception as e:
            print(f"❌ Error clearing memory after rollout: {e}")
            # Continue execution - memory clearing failure shouldn't stop training
