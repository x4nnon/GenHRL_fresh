from skrl.memories.torch import RandomMemory
import torch
import numpy as np
from typing import Dict, Any, Union, Optional
import wandb

def safe_wandb_log(data: Dict[str, Any], step: Optional[int] = None):
    """Safely log to wandb if initialized, otherwise store for later or print."""
    try:
        if wandb.run is not None:
            # Convert boolean values to integers to avoid wandb media type confusion
            clean_data = {}
            for key, value in data.items():
                if isinstance(value, bool):
                    clean_data[key] = int(value)
                elif isinstance(value, torch.Tensor):
                    clean_data[key] = value.item() if value.numel() == 1 else value.tolist()
                elif isinstance(value, (list, tuple)) and len(value) > 20:
                    # Truncate very long lists to avoid wandb issues
                    clean_data[key] = list(value)[:20]
                else:
                    clean_data[key] = value
            wandb.log(clean_data, step=step)
        else:
            # wandb not initialized yet, could store for later or just skip
            pass
    except Exception as e:
        # If any wandb error occurs, just skip logging
        pass

class DecisionPointMemory(RandomMemory):
    """
    A memory buffer that only stores experiences at decision points for hierarchical RL.
    This is a wrapper around SKRL's RandomMemory that filters samples that are not
    decision points based on info flags.
    
    Works with any level of hierarchy by checking for the appropriate decision flags:
    - is_decision_step: For L1 decisions
    - is_l2_decision_step: For L2 decisions
    """
    
    def __init__(self, num_envs: int, device: Union[str, torch.device], hierarchy_level: int = 1,
                 debug_mode: bool = True, discount_factor: float = 0.99, skill_reward_discount: float = 1.0,
                   use_average_reward: bool = False, temporal_abstraction_method: str = "sum",
                    max_skill_duration: int = 300, scale_rewards: bool = True, gae_lambda: float = 0.95,
                    drop_remapped_decisions: bool = True, **kwargs):
        """
        Initialize the memory with the same parameters as RandomMemory.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to use for tensor operations
            hierarchy_level: Level in the hierarchy (1 for L1, 2 for L2, etc.)
            debug_mode: Enable debug printing for hierarchical learning (default: False)
            discount_factor: Discount factor for L1 policy learning between decisions (default: 0.99)
            skill_reward_discount: Discount factor for reward accumulation during skill execution (default: 1.0 - no discounting)
            use_average_reward: If True, use average reward per step instead of accumulated reward (default: False)
            temporal_abstraction_method: Method for L2 skill-level rewards ("final", "sum", "average") (default: "sum")
            max_skill_duration: Maximum skill duration for scaling (should be passed from training config) (default: 300)
            **kwargs: Additional arguments to pass to RandomMemory
        """
        super().__init__(num_envs=num_envs, device=device, **kwargs)
        self.num_envs = num_envs
        self.device = device
        self.hierarchy_level = hierarchy_level
        self.debug_mode = debug_mode
        self.discount_factor = discount_factor  # For L1 policy learning between decisions
        self.skill_reward_discount = skill_reward_discount  # For reward accumulation during skill execution
        self.use_average_reward = use_average_reward  # Use average instead of accumulation
        self.temporal_abstraction_method = temporal_abstraction_method  # Method for L2 skill-level rewards
        self.max_skill_duration = max_skill_duration  # Maximum skill duration for intelligent scaling
        self.scale_rewards = scale_rewards  # Whether to scale rewards by skill duration
        self.gae_lambda = gae_lambda  # GAE lambda parameter for advantage estimation
        # Control bootstrapping/credit across decision boundaries
        self.zero_bootstrap_on_decision = False
        # Drop decisions where L1 selection was remapped by the wrapper due to prior completion
        self.drop_remapped_decisions = drop_remapped_decisions
        
        # Determine which flag to check based on hierarchy level
        if hierarchy_level == 1:
            self.decision_flag = "is_decision_step"
        elif hierarchy_level == 2:
            self.decision_flag = "is_l2_decision_step"
        else:
            self.decision_flag = f"is_l{hierarchy_level}_decision_step"
            
        if self.debug_mode:
            safe_wandb_log({
                "debug/memory_init/hierarchy_level": hierarchy_level,
                "debug/memory_init/device": str(device),
                "debug/memory_init/l1_policy_discount": discount_factor,
                "debug/memory_init/skill_reward_discount": skill_reward_discount,
                "debug/memory_init/use_average_reward": use_average_reward,
                "debug/memory_init/temporal_abstraction_method": temporal_abstraction_method,
            })
        
    def add_samples(self, **tensors):
        """Add samples to memory, handling the decision_mask properly"""
        # Handle infos and extract decision_mask
        infos = tensors.pop("infos", None)
        decision_mask = None
        decision_terminated = None
        decision_remapped = None
        
        # For L2 training, we need to extract the correct decision flag
        if infos is not None and isinstance(infos, dict):
            if self.hierarchy_level == 2:
                # For L2, look for L2 decision steps (when L2 selects L1 skills)
                if "is_l2_decision_step" in infos:
                    decision_mask = infos.get("is_l2_decision_step")
                elif "decision_mask" in infos:
                    # Fallback to decision_mask if properly set by L2 wrapper
                    decision_mask = infos.get("decision_mask")
                else:
                    raise ValueError("L2 DecisionPointMemory requires 'is_l2_decision_step' flag in infos!")
            elif self.hierarchy_level == 1:
                # For L1, look for L1 decision steps (when L1 selects L0 skills)
                if "is_decision_step" in infos:
                    decision_mask = infos.get("is_decision_step")
                elif "decision_mask" in infos:
                    decision_mask = infos.get("decision_mask")
                else:
                    raise ValueError("L1 DecisionPointMemory requires 'is_decision_step' flag in infos!")
            # Optional: explicit flag indicating the preceding skill ended by env termination
            if "decision_terminated" in infos:
                decision_terminated = infos.get("decision_terminated")
            # Optional: mask of decisions where wrapper remapped the selected action
            # Expect arrays shaped [num_envs] where values >=0 indicate a remap occurred
            if "wrapper_l0_remapped_from" in infos:
                remap_from = infos.get("wrapper_l0_remapped_from")
                try:
                    if not isinstance(remap_from, torch.Tensor):
                        remap_from = torch.tensor(remap_from, dtype=torch.long, device=self.device)
                    else:
                        remap_from = remap_from.to(dtype=torch.long, device=self.device)
                    decision_remapped = (remap_from >= 0)
                except Exception:
                    decision_remapped = None
            
        if "decision_mask" in tensors:
            decision_mask = tensors["decision_mask"]
            tensors.pop("decision_mask")
        if "decision_terminated" in tensors:
            decision_terminated = tensors["decision_terminated"]
            tensors.pop("decision_terminated")
        if "decision_remapped" in tensors:
            decision_remapped = tensors["decision_remapped"]
            tensors.pop("decision_remapped")
            
        # Ensure decision_mask has correct shape and type
        if decision_mask is not None:
            if not isinstance(decision_mask, torch.Tensor):
                decision_mask = torch.tensor(decision_mask, dtype=torch.bool, device=self.device)
            else:
                decision_mask = decision_mask.to(dtype=torch.bool, device=self.device)
            
            # Ensure proper shape [num_envs] -> [num_envs, 1] to match memory format
            if decision_mask.dim() == 1:
                decision_mask = decision_mask.unsqueeze(-1)
            elif decision_mask.dim() == 0:
                # Single environment case
                decision_mask = decision_mask.unsqueeze(0).unsqueeze(-1)

            # Normalize and optionally apply remap filtering
            if decision_remapped is not None:
                if not isinstance(decision_remapped, torch.Tensor):
                    decision_remapped = torch.tensor(decision_remapped, dtype=torch.bool, device=self.device)
                else:
                    # remapped source indices provided -> convert to bool mask
                    if decision_remapped.dtype != torch.bool:
                        decision_remapped = (decision_remapped >= 1) | (decision_remapped >= 0)
                    decision_remapped = decision_remapped.to(dtype=torch.bool, device=self.device)
                if decision_remapped.dim() == 1:
                    decision_remapped = decision_remapped.unsqueeze(-1)
                elif decision_remapped.dim() == 0:
                    decision_remapped = decision_remapped.unsqueeze(0).unsqueeze(-1)

                # Drop remapped decisions from training data if requested
                if self.drop_remapped_decisions:
                    decision_mask = decision_mask & (~decision_remapped)
                
            # Add decision_mask back to tensors for storage
            tensors["decision_mask"] = decision_mask
        else:
            raise ValueError(f"DecisionPointMemory (L{self.hierarchy_level}) requires decision_mask to be provided for hierarchical learning!")

        # Normalize optional decision_terminated shape and type if provided
        if decision_terminated is not None:
            if not isinstance(decision_terminated, torch.Tensor):
                decision_terminated = torch.tensor(decision_terminated, dtype=torch.bool, device=self.device)
            else:
                decision_terminated = decision_terminated.to(dtype=torch.bool, device=self.device)
            if decision_terminated.dim() == 1:
                decision_terminated = decision_terminated.unsqueeze(-1)
            elif decision_terminated.dim() == 0:
                decision_terminated = decision_terminated.unsqueeze(0).unsqueeze(-1)
            tensors["decision_terminated"] = decision_terminated

        # Store decision_remapped mask for debugging/analysis if provided
        if decision_remapped is not None:
            if not isinstance(decision_remapped, torch.Tensor):
                decision_remapped = torch.tensor(decision_remapped, dtype=torch.bool, device=self.device)
            else:
                decision_remapped = decision_remapped.to(dtype=torch.bool, device=self.device)
            if decision_remapped.dim() == 1:
                decision_remapped = decision_remapped.unsqueeze(-1)
            elif decision_remapped.dim() == 0:
                decision_remapped = decision_remapped.unsqueeze(0).unsqueeze(-1)
            tensors["decision_remapped"] = decision_remapped

        for name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                # If tensor is 1D [num_envs], reshape to [num_envs, 1]
                if tensor.dim() == 1:
                    tensors[name] = tensor.unsqueeze(-1)
                # If tensor is 0D, reshape to [1, 1] for single env case
                elif tensor.dim() == 0:
                    tensors[name] = tensor.unsqueeze(0).unsqueeze(-1)
        
        # Get counts for logging
        total_count = decision_mask.shape[0]
        decision_count = decision_mask.sum().item()
        remapped_count = int(decision_remapped.sum().item()) if ('decision_remapped' in tensors) else 0
        
        if self.debug_mode:
            hierarchy_desc = "L2 (skill selection)" if self.hierarchy_level == 2 else "L1 (primitive selection)"
            safe_wandb_log({
                "debug/add_samples/total_samples": total_count,
                "debug/add_samples/decision_samples": decision_count,
                "debug/add_samples/decision_ratio": decision_count / max(1, total_count),
                "debug/add_samples/hierarchy_desc": hierarchy_desc,
                "debug/add_samples/decision_remapped_dropped": remapped_count if self.drop_remapped_decisions else 0,
            })
        
        # Call parent's add_samples with all tensors (including decision_mask)
        super().add_samples(**tensors)

    def create_tensors(self, observation_space, action_space):
        """Create memory tensors including the decision_mask tensor"""
        # Create standard tensors first
        self.create_tensor(name="states", size=observation_space, dtype=torch.float32)
        self.create_tensor(name="actions", size=action_space, dtype=torch.float32)
        self.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.create_tensor(name="truncated", size=1, dtype=torch.bool)
        self.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.create_tensor(name="values", size=1, dtype=torch.float32)
        self.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.create_tensor(name="advantages", size=1, dtype=torch.float32)
        
        # Create decision mask tensor with proper size [memory_size, num_envs]
        # The create_tensor method will handle the full shape [memory_size, num_envs, 1]
        self.create_tensor(name="decision_mask", size=1, dtype=torch.bool)
        # Create optional tensor to mark decisions whose preceding skill ended by env termination
        self.create_tensor(name="decision_terminated", size=1, dtype=torch.bool)
        
        if self.debug_mode:
            safe_wandb_log({
                "debug/create_tensors/num_envs": self.num_envs,
            })

    def _accumulate_hierarchical_rewards(self):
        """
        Accumulate rewards using proper hierarchical structure with Hierarchical GAE.
        
        For L2: Two-level accumulation:
        1. First accumulate primitive rewards at L1 level 
        2. Then accumulate L1 skill rewards at L2 level
        
        For L1: Single-level accumulation over L0 execution periods.
        
        Uses proper Hierarchical GAE that computes advantages between decision points.
        """
        if self.debug_mode:
            safe_wandb_log({
                "debug/accumulate_rewards/hierarchy_level": self.hierarchy_level,
            })
        
        # Get required tensors from memory
        rewards = self.get_tensor_by_name("rewards").clone()
        values = self.get_tensor_by_name("values").clone()
        terminated = self.get_tensor_by_name("terminated").clone()
        truncated = self.get_tensor_by_name("truncated").clone()
        decision_mask = self.get_tensor_by_name("decision_mask").clone()
        
        memory_size = rewards.shape[0]
        if self.debug_mode:
            safe_wandb_log({
                "debug/accumulate_rewards/memory_size": memory_size,
                "debug/accumulate_rewards/num_envs": self.num_envs,
                "debug/accumulate_rewards/rewards_shape": list(rewards.shape),
                "debug/accumulate_rewards/values_shape": list(values.shape),
                "debug/accumulate_rewards/decision_mask_shape": list(decision_mask.shape),
            })
        
        # Ensure tensors are 2D [memory_size, num_envs]
        tensor_squeezed = False
        if rewards.dim() == 3:
            rewards = rewards.squeeze(-1)
            tensor_squeezed = True
        if values.dim() == 3:
            values = values.squeeze(-1)
        if terminated.dim() == 3:
            terminated = terminated.squeeze(-1)
        if truncated.dim() == 3:
            truncated = truncated.squeeze(-1)
        if decision_mask.dim() == 3:
            decision_mask = decision_mask.squeeze(-1)
            
        if self.debug_mode and tensor_squeezed:
            safe_wandb_log({
                "debug/accumulate_rewards/squeezed_rewards_shape": list(rewards.shape),
                "debug/accumulate_rewards/squeezed_decision_mask_shape": list(decision_mask.shape),
            })
        
        # Create dones mask
        dones = terminated | truncated
        
        # Use Hierarchical GAE for both L1 and L2
        if self.debug_mode:
            safe_wandb_log({"debug/accumulate_rewards/mode": "hierarchical_gae"})
        # Prefer an explicitly provided decision_terminated mask if present
        provided_decision_terminated = self.get_tensor_by_name("decision_terminated")
        if provided_decision_terminated is not None and provided_decision_terminated.dim() == 3 and provided_decision_terminated.size(-1) == 1:
            provided_decision_terminated = provided_decision_terminated.squeeze(-1)

        hierarchical_returns, hierarchical_advantages, decision_terminated_mask = self.compute_hierarchical_gae(
                rewards, values, dones, decision_mask, memory_size
            )
        
        # Log advantage statistics but DON'T normalize (let HPPO handle it)
        decision_positions = decision_mask.bool()
        decision_count = decision_positions.sum().item()
        
        if decision_positions.any():
            decision_advantages = hierarchical_advantages[decision_positions]
            if decision_advantages.numel() > 0:
                advantage_mean = decision_advantages.mean()
                advantage_std = decision_advantages.std()
                
                if self.debug_mode:
                    safe_wandb_log({
                        "debug/advantage_stats/raw_mean": advantage_mean.item(),
                        "debug/advantage_stats/raw_std": advantage_std.item(),
                        "debug/advantage_stats/raw_min": decision_advantages.min().item(),
                        "debug/advantage_stats/raw_max": decision_advantages.max().item(),
                        "debug/advantage_stats/advantage_count": decision_advantages.numel(),
                    })
                    
                # Only normalize if advantages are extremely large (>100) to prevent instability
                if advantage_std > 100.0:
                    print(f"⚠️  Large advantage std detected: {advantage_std.item():.2f}, applying conservative normalization")
                    normalized_advantages = hierarchical_advantages[decision_positions] / (advantage_std + 1e-8)
                    hierarchical_advantages[decision_positions] = normalized_advantages
                    
                    if self.debug_mode:
                        safe_wandb_log({
                            "debug/advantage_norm/applied_conservative_norm": True,
                            "debug/advantage_norm/after_mean": normalized_advantages.mean().item(),
                            "debug/advantage_norm/after_std": normalized_advantages.std().item(),
                        })
            else:
                if self.debug_mode:
                    safe_wandb_log({
                        "debug/advantage_norm/warning": "no_decision_advantages",
                        "debug/advantage_norm/advantage_count": decision_advantages.numel(),
                    })
        else:
            if self.debug_mode:
                safe_wandb_log({"debug/advantage_norm/error": "no_decision_positions"})
        
        # Reshape back to original dimensions if needed
        original_returns = self.get_tensor_by_name("returns")
        original_advantages = self.get_tensor_by_name("advantages")
        
        if original_returns.dim() == 3:
            hierarchical_returns = hierarchical_returns.unsqueeze(-1)
            if self.debug_mode:
                safe_wandb_log({"debug/reshape/returns_shape": list(hierarchical_returns.shape)})
        if original_advantages.dim() == 3:
            hierarchical_advantages = hierarchical_advantages.unsqueeze(-1)
            if self.debug_mode:
                safe_wandb_log({"debug/reshape/advantages_shape": list(hierarchical_advantages.shape)})
            
        # Update the stored tensors with hierarchical values
        self.set_tensor_by_name("returns", hierarchical_returns)
        self.set_tensor_by_name("advantages", hierarchical_advantages)
        # Store the decision termination mask for debugging / analysis
        try:
            self.set_tensor_by_name("decision_terminated", decision_terminated_mask.to(dtype=torch.bool))
        except Exception:
            pass
        
        if self.debug_mode:
            safe_wandb_log({
                "debug/accumulate_rewards/completed": True,
                "debug/accumulate_rewards/decision_count": decision_count,
                "debug/accumulate_rewards/reward_type": "hierarchical_gae",
            })

    def _accumulate_rewards_standardized(self, rewards, discount_factor, max_steps=None):
        """
        Standardized reward accumulation method used across all hierarchy levels.
        
        Args:
            rewards: Tensor of rewards to accumulate
            discount_factor: Discount factor to apply
            max_steps: Maximum number of steps to accumulate (None for all)
        
        Returns:
            Float: Accumulated discounted reward
        """
        if len(rewards) == 0:
            return 0.0
            
        # Limit accumulation if max_steps is specified
        rewards_to_use = rewards[:max_steps] if max_steps is not None else rewards
        
        accumulated = 0.0
        for i, reward in enumerate(rewards_to_use):
            accumulated += (discount_factor ** i) * reward.item()
            
        return accumulated

    def compute_hierarchical_gae(self, rewards, values, dones, decision_mask, memory_size):
        """
        Compute Hierarchical Generalized Advantage Estimation (GAE).
        
        This implements proper GAE for hierarchical RL where advantages are computed 
        between decision points rather than every timestep.
        
        Standard GAE: δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
        Hierarchical GAE: δ_decision = R_skill + γV(s_next_decision) - V(s_decision)
        
        Args:
            rewards: Tensor of rewards [memory_size, num_envs]
            values: Tensor of value estimates [memory_size, num_envs] 
            dones: Tensor of termination flags [memory_size, num_envs]
            decision_mask: Tensor indicating decision points [memory_size, num_envs]
            memory_size: Size of memory buffer
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (hierarchical_returns, hierarchical_advantages)
        """
        if self.debug_mode:
            safe_wandb_log({"debug/hierarchical_gae/started": True})
        
        # Initialize hierarchical returns and advantages
        hierarchical_returns = torch.zeros_like(rewards)
        hierarchical_advantages = torch.zeros_like(rewards)
        decision_terminated_mask = torch.zeros_like(rewards, dtype=torch.bool)
        
        total_decision_points = 0
        
        # Try to read an explicitly provided decision-termination mask (from tensors, if any)
        explicit_decision_terminated = self.get_tensor_by_name("decision_terminated")
        if explicit_decision_terminated is not None:
            if explicit_decision_terminated.dim() == 3 and explicit_decision_terminated.size(-1) == 1:
                explicit_decision_terminated = explicit_decision_terminated.squeeze(-1)
            explicit_decision_terminated = explicit_decision_terminated.to(dtype=torch.bool, device=self.device)

        # Process each environment separately
        for env_idx in range(self.num_envs):
            show_debug = (env_idx == 0) and self.debug_mode
            
            # Get data for this environment
            env_decision_mask = decision_mask[:, env_idx]
            env_rewards = rewards[:, env_idx]
            env_values = values[:, env_idx]
            env_dones = dones[:, env_idx]
            
            # Find decision step indices for this environment
            decision_indices = torch.where(env_decision_mask)[0]
            
            if len(decision_indices) == 0:
                if show_debug:
                    safe_wandb_log({f"debug/hierarchical_gae_env_{env_idx}/warning": "no_decisions"})
                continue
                
            if show_debug:
                safe_wandb_log({
                    f"debug/hierarchical_gae_env_{env_idx}/decision_count": len(decision_indices),
                    f"debug/hierarchical_gae_env_{env_idx}/decision_indices": decision_indices.tolist(),
                })
            total_decision_points += len(decision_indices)
            
            # Compute GAE in REVERSE order (standard RL practice)
            advantage = 0.0  # Running advantage for GAE
            
            for i in reversed(range(len(decision_indices))):
                decision_idx = decision_indices[i]
                start_idx = int(decision_idx.item())
                
                # Determine end index and next decision value
                if i + 1 < len(decision_indices):
                    end_idx = int(decision_indices[i + 1].item())
                    next_decision_value = env_values[end_idx]
                else:
                    # This is the last decision point - bootstrap from end of trajectory
                    end_idx = memory_size
                    if end_idx > start_idx:
                        final_step_idx = min(end_idx - 1, memory_size - 1)
                        next_decision_value = env_values[final_step_idx] if not env_dones[final_step_idx] else 0.0
                    else:
                        next_decision_value = 0.0
                
                # Accumulate rewards over skill execution period using standardized method
                skill_rewards = env_rewards[start_idx:end_idx]
                cumulative_reward = self._accumulate_rewards_standardized(
                    skill_rewards, self.skill_reward_discount
                )
                
                # ENHANCED Episode Boundary Handling for Hierarchical GAE
                # Check for episode termination during skill execution
                # Detect if the skill ended with termination
                episode_terminated = env_dones[start_idx:end_idx].any() if end_idx > start_idx else False
                # Override using explicit per-decision flag if provided
                if explicit_decision_terminated is not None:
                    try:
                        # If the NEXT decision step is marked terminated, then this segment ended in termination
                        if i + 1 < len(decision_indices):
                            if bool(explicit_decision_terminated[end_idx, env_idx].item()):
                                episode_terminated = True
                                # Treat termination as happening at end_idx, so include that reward
                                termination_step = end_idx
                            else:
                                termination_step = None
                        else:
                            # For last decision in the buffer, fall back to dones window
                            termination_step = None
                    except Exception:
                        termination_step = None
                else:
                    termination_step = None
                if episode_terminated and termination_step is None and end_idx > start_idx:
                    # Find exactly where termination occurred inside the window
                    termination_mask = env_dones[start_idx:end_idx]
                    termination_indices = torch.where(termination_mask)[0]
                    if len(termination_indices) > 0:
                        termination_step = start_idx + termination_indices[0].item()
                
                # Compute TD error for this decision point with proper episode boundary handling
                # δ = R_skill + γ * V(s_next_decision) - V(s_decision)
                if episode_terminated:
                    # If episode terminated during skill execution, don't bootstrap from next decision
                    next_value = 0.0
                    
                    # CRITICAL FIX: Truncate reward accumulation at termination point (include the termination step)
                    if termination_step is not None and termination_step >= start_idx:
                        truncated_rewards = env_rewards[start_idx:termination_step + 1]
                        cumulative_reward = self._accumulate_rewards_standardized(
                            truncated_rewards, self.skill_reward_discount
                        )
                    if show_debug:
                        safe_wandb_log({
                            f"debug/hierarchical_gae_env_{env_idx}/episode_boundary/termination_step": int(termination_step if termination_step is not None else -1),
                            f"debug/hierarchical_gae_env_{env_idx}/episode_boundary/truncated_reward": cumulative_reward,
                        })
                else:
                    # Respect zero_bootstrap_on_decision when not terminated
                    next_value = 0.0 if self.zero_bootstrap_on_decision else next_decision_value
                
                td_error = cumulative_reward + self.discount_factor * next_value - env_values[start_idx]
                
                # Compute GAE advantage with proper episode boundary handling
                # A = δ + γλδ_next + (γλ)²δ_next_next + ...
                # Since we're going in reverse, we have the future advantage already computed
                if episode_terminated:
                    # Don't propagate advantage across episode boundaries
                    advantage = td_error  # Only immediate TD error, no future advantage
                else:
                    # Normal GAE computation with skill-level discounting
                    if self.zero_bootstrap_on_decision:
                        # Do not propagate advantage across decision boundaries
                        advantage = td_error
                    else:
                        advantage = td_error + self.discount_factor * self.gae_lambda * advantage
                
                # Store hierarchical return and advantage for this decision point
                hierarchical_return = advantage + env_values[start_idx]
                hierarchical_returns[start_idx, env_idx] = hierarchical_return
                hierarchical_advantages[start_idx, env_idx] = advantage
                # Mark decision terminated flag at this decision point if needed
                if episode_terminated:
                    decision_terminated_mask[start_idx, env_idx] = True
                
                # ENHANCED: Reset advantage accumulation after episode termination
                # AND after very long skill executions to prevent excessive accumulation
                skill_duration = end_idx - start_idx
                max_reasonable_skill_duration = self.max_skill_duration * 2  # Allow some flexibility
                
                if episode_terminated:
                    advantage = 0.0  # Always reset on episode boundaries
                elif skill_duration > max_reasonable_skill_duration:
                    # Reset advantage for extremely long skills to prevent instability
                    advantage = advantage * 0.5  # Partial reset instead of full reset
                    if show_debug:
                        safe_wandb_log({
                            f"debug/hierarchical_gae_env_{env_idx}/long_skill_partial_reset": True,
                            f"debug/hierarchical_gae_env_{env_idx}/skill_duration": skill_duration,
                            f"debug/hierarchical_gae_env_{env_idx}/max_duration": max_reasonable_skill_duration,
                        })
                
                if show_debug:
                    safe_wandb_log({
                        f"debug/hierarchical_gae_env_{env_idx}/decision_{len(decision_indices)-i}/cumulative_reward": cumulative_reward,
                        f"debug/hierarchical_gae_env_{env_idx}/decision_{len(decision_indices)-i}/td_error": float(td_error),
                        f"debug/hierarchical_gae_env_{env_idx}/decision_{len(decision_indices)-i}/advantage": float(advantage),
                        f"debug/hierarchical_gae_env_{env_idx}/decision_{len(decision_indices)-i}/hierarchical_return": float(hierarchical_return),
                        f"debug/hierarchical_gae_env_{env_idx}/decision_{len(decision_indices)-i}/episode_terminated": int(episode_terminated),
                        f"debug/hierarchical_gae_env_{env_idx}/decision_{len(decision_indices)-i}/skill_duration": end_idx - start_idx,
                    })
        
        if self.debug_mode:
            safe_wandb_log({
                "debug/hierarchical_gae/total_decision_points": total_decision_points,
                "debug/hierarchical_gae/completed": True,
            })
        
        return hierarchical_returns, hierarchical_advantages, decision_terminated_mask

    def _compute_l2_two_level_accumulation(self, rewards, values, dones, decision_mask, memory_size):
        """
        Compute L2 hierarchical returns using proper two-level accumulation:
        1. First accumulate primitive rewards at L1 level 
        2. Then accumulate L1 skill rewards at L2 level
        """
        if self.debug_mode:
            safe_wandb_log({"debug/l2_accumulation/started": True})
        
        # Initialize hierarchical returns and advantages
        hierarchical_returns = torch.zeros_like(rewards)
        hierarchical_advantages = torch.zeros_like(rewards)
        
        total_l2_decisions = 0
        total_l1_skills_executed = 0
        
        # Process each environment separately
        for env_idx in range(self.num_envs):
            show_debug = (env_idx == 0) and self.debug_mode  # Debug only for env 0
            
            # Get the L2 decision mask and data for this environment
            env_l2_decision_mask = decision_mask[:, env_idx]
            env_rewards = rewards[:, env_idx].clone()
            env_values = values[:, env_idx].clone()
            env_dones = dones[:, env_idx]
            
            # Find L2 decision step indices (when L2 selects L1 skills)
            l2_decision_indices = torch.where(env_l2_decision_mask)[0]
            
            if len(l2_decision_indices) == 0:
                if show_debug:
                    safe_wandb_log({
                        f"debug/l2_env_{env_idx}/warning": "no_l2_decisions",
                    })
                continue
                
            if show_debug:
                safe_wandb_log({
                    f"debug/l2_env_{env_idx}/l2_decision_count": len(l2_decision_indices),
                    f"debug/l2_env_{env_idx}/l2_decision_indices": l2_decision_indices.tolist(),
                })
            total_l2_decisions += len(l2_decision_indices)
            
            # LEVEL 1: Accumulate primitive rewards within each L1 skill execution period
            # For this, we need to identify L1 decision points within each L2 skill period
            
            # REVERSE ORDER: Process L2 decision points from last to first
            for i in reversed(range(len(l2_decision_indices))):
                l2_decision_idx = l2_decision_indices[i]
                l2_start_idx = int(l2_decision_idx.item())
                
                # End index is either the next L2 decision step or end of memory
                if i + 1 < len(l2_decision_indices):
                    l2_end_idx = int(l2_decision_indices[i + 1].item())
                    # FIXED: Use the original value function at the next L2 decision point
                    # The hierarchical_returns might still be zero if we haven't processed that step yet
                    next_l2_decision_step = l2_end_idx
                    if hierarchical_returns[next_l2_decision_step, env_idx] != 0.0:
                        # Use the already computed hierarchical return if available
                        l2_bootstrap_value = hierarchical_returns[next_l2_decision_step, env_idx]
                    else:
                        # Fallback to original value function estimate
                        l2_bootstrap_value = env_values[next_l2_decision_step]
                else:
                    # This is the last L2 decision point
                    l2_end_idx = memory_size
                    # Bootstrap with the final value estimate
                    if l2_end_idx > l2_start_idx:
                        final_step_idx = min(l2_end_idx - 1, memory_size - 1)
                        l2_bootstrap_value = env_values[final_step_idx] if not env_dones[final_step_idx] else 0.0
                    else:
                        l2_bootstrap_value = env_values[l2_start_idx] if not env_dones[l2_start_idx] else 0.0
                
                # L1 skill execution period: from l2_start_idx to l2_end_idx
                l1_skill_period_length = l2_end_idx - l2_start_idx
                
                if show_debug:
                    safe_wandb_log({
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/start_idx": l2_start_idx,
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/end_idx": l2_end_idx,
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/skill_duration": l1_skill_period_length,
                    })
                
                # STEP 1: Compute L1-level accumulated rewards over primitive steps
                # Simulate L1 decision points within this L2 period (based on L1 action frequency)
                l1_accumulated_reward = self._compute_l1_level_rewards_within_l2_period(
                    env_rewards, env_dones, l2_start_idx, l2_end_idx, show_debug
                )
                
                total_l1_skills_executed += 1
                
                # STEP 2: Compute L2-level return using L1 accumulated reward
                episode_terminated = env_dones[l2_start_idx:l2_end_idx].any() if l2_end_idx > l2_start_idx else False
                
                # Use the bootstrap value directly without scaling
                # The value function mismatch is a fundamental issue that needs a proper L2 value function
                # For now, use the original value estimates without artificial scaling
                if episode_terminated:
                    # If L1 skill execution led to termination, no bootstrap
                    l2_hierarchical_return = l1_accumulated_reward
                else:
                    # L2 return: L1 skill reward + discounted future L2 value
                    l2_hierarchical_return = l1_accumulated_reward + self.discount_factor * l2_bootstrap_value
                
                # Store the L2 hierarchical return and compute advantage
                hierarchical_returns[l2_start_idx, env_idx] = l2_hierarchical_return
                hierarchical_advantages[l2_start_idx, env_idx] = l2_hierarchical_return - env_values[l2_start_idx]
                
                if show_debug:
                    advantage_value = hierarchical_advantages[l2_start_idx, env_idx].item()
                    advantage_magnitude = abs(advantage_value)
                    
                    # Log basic values
                    safe_wandb_log({
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/l1_accumulated_reward": l1_accumulated_reward,
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/l2_hierarchical_return": l2_hierarchical_return,
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/l2_value": env_values[l2_start_idx].item(),
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/advantage": advantage_value,
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/advantage_magnitude": advantage_magnitude,
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/episode_terminated": int(episode_terminated),
                        f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/bootstrap_value": l2_bootstrap_value,
                    })
                    
                    # Log warnings for potential issues - adjusted thresholds
                    if advantage_magnitude > 50:  # Lowered threshold
                        safe_wandb_log({f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/warning_large_advantage": advantage_magnitude})
                    if l1_accumulated_reward < -50 or l1_accumulated_reward > 50:  # More reasonable bounds
                        safe_wandb_log({f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/warning_extreme_reward": l1_accumulated_reward})
                    if abs(l2_hierarchical_return) > 100:  # Check for extreme returns
                        safe_wandb_log({f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/warning_extreme_return": l2_hierarchical_return})
                    if abs(env_values[l2_start_idx].item()) > 100:  # Check for extreme value function estimates
                        safe_wandb_log({f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/warning_extreme_value": env_values[l2_start_idx].item()})
                    if not episode_terminated and l2_bootstrap_value == 0:
                        safe_wandb_log({f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/warning_zero_bootstrap": 1})
                        
                    # Log return computation breakdown
                    if not episode_terminated:
                        bootstrap_contribution = self.discount_factor * l2_bootstrap_value
                        safe_wandb_log({
                            f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/return_breakdown_reward": l1_accumulated_reward,
                            f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/return_breakdown_discount": self.discount_factor,
                            f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/return_breakdown_bootstrap": l2_bootstrap_value,
                            f"debug/l2_env_{env_idx}/decision_{len(l2_decision_indices)-i}/return_breakdown_total": l2_hierarchical_return,
                        })
        
        if self.debug_mode:
            safe_wandb_log({
                "debug/l2_accumulation/total_l2_decisions": total_l2_decisions,
                "debug/l2_accumulation/total_l1_skills_executed": total_l1_skills_executed,
            })
        
        return hierarchical_returns, hierarchical_advantages

    def _compute_l1_level_rewards_within_l2_period(self, env_rewards, env_dones, l2_start_idx, l2_end_idx, show_debug=False):
        """
        Compute L1-level accumulated rewards within a single L2 period using standardized accumulation.
        This simulates what the L1 policy would have accumulated over its L0 execution periods.
        """
        if l2_start_idx >= l2_end_idx:
            return 0.0
            
        skill_rewards = env_rewards[l2_start_idx:l2_end_idx]
        
        # Use standardized reward accumulation method
        if self.temporal_abstraction_method == "sum":
            # Standard discounted sum using skill_reward_discount
            l1_accumulated_reward = self._accumulate_rewards_standardized(
                skill_rewards, self.skill_reward_discount
            )
            method_desc = "standardized discounted sum over L1 period"
            
        elif self.temporal_abstraction_method == "average":
            # Average over the period (no discounting for average)
            l1_accumulated_reward = skill_rewards.mean().item() if len(skill_rewards) > 0 else 0.0
            method_desc = "average over L1 period"
            
        elif self.temporal_abstraction_method == "final":
            # Final reward only
            l1_accumulated_reward = skill_rewards[-1].item() if len(skill_rewards) > 0 else 0.0
            method_desc = "final reward of L1 period"
            
        elif self.temporal_abstraction_method == "discounted_sum":
            # Same as sum method - use standardized accumulation
            l1_accumulated_reward = self._accumulate_rewards_standardized(
                skill_rewards, self.skill_reward_discount
            )
            method_desc = "standardized discounted sum over L1 period"
            
        else:
            raise ValueError(f"Unknown temporal abstraction method for L1 simulation: {self.temporal_abstraction_method}")
        
        if show_debug:
            skill_duration = l2_end_idx - l2_start_idx
            safe_wandb_log({
                f"debug/l1_within_l2/method": method_desc,
                f"debug/l1_within_l2/l1_accumulated_reward": l1_accumulated_reward,
                f"debug/l1_within_l2/skill_duration": skill_duration,
                f"debug/l1_within_l2/skill_reward_discount": self.skill_reward_discount,
            })
            if len(skill_rewards) > 1:
                raw_sum = skill_rewards.sum().item()
                safe_wandb_log({
                    f"debug/l1_within_l2/primitive_sum": raw_sum,
                    f"debug/l1_within_l2/primitive_avg": skill_rewards.mean().item(),
                    f"debug/l1_within_l2/primitive_steps": len(skill_rewards),
                })
        
        return l1_accumulated_reward

    def _compute_l1_single_level_accumulation(self, rewards, values, dones, decision_mask, memory_size):
        """
        Compute L1 hierarchical returns using single-level accumulation over L0 execution periods.
        This is the original logic for L1 training.
        """
        if self.debug_mode:
            safe_wandb_log({"debug/l1_accumulation/started": True})
        
        # Initialize hierarchical returns and advantages
        hierarchical_returns = torch.zeros_like(rewards)
        hierarchical_advantages = torch.zeros_like(rewards)
        
        total_decision_points = 0
        total_accumulated_reward = 0.0
        single_timestep_skills = 0
        
        # Process each environment separately (original L1 logic)
        for env_idx in range(self.num_envs):
            show_debug = (env_idx == 0) and self.debug_mode
            
            # Get the decision mask and data for this environment
            env_decision_mask = decision_mask[:, env_idx]
            env_rewards = rewards[:, env_idx].clone()
            env_values = values[:, env_idx].clone()
            env_dones = dones[:, env_idx]
            
            # Find decision step indices for this environment
            decision_indices = torch.where(env_decision_mask)[0]
            
            if len(decision_indices) == 0:
                if show_debug:
                    safe_wandb_log({f"debug/l1_env_{env_idx}/warning": "no_decisions"})
                continue
                
            if show_debug:
                safe_wandb_log({
                    f"debug/l1_env_{env_idx}/decision_count": len(decision_indices),
                    f"debug/l1_env_{env_idx}/decision_indices": decision_indices.tolist(),
                })
            total_decision_points += len(decision_indices)
            
            # REVERSE ORDER: Process decision points from last to first
            for i in reversed(range(len(decision_indices))):
                decision_idx = decision_indices[i]
                start_idx = int(decision_idx.item())
                
                # End index is either the next decision step or end of memory
                if i + 1 < len(decision_indices):
                    reward_end_idx = int(decision_indices[i + 1].item())
                    bootstrap_value = hierarchical_returns[reward_end_idx, env_idx]
                else:
                    reward_end_idx = memory_size
                    if reward_end_idx > start_idx:
                        final_step_idx = min(reward_end_idx - 1, memory_size - 1)
                        bootstrap_value = env_values[final_step_idx] if not env_dones[final_step_idx] else 0.0
                    else:
                        bootstrap_value = env_values[start_idx] if not env_dones[start_idx] else 0.0
                
                # Check if this is a single-timestep skill
                skill_duration = reward_end_idx - start_idx
                is_single_timestep = (skill_duration == 1)
                if is_single_timestep:
                    single_timestep_skills += 1
                
                # L1 uses normal reward accumulation over L0 execution periods
                cumulative_reward, effective_horizon, episode_terminated = self._compute_accumulated_reward(
                    env_rewards, env_dones, start_idx, reward_end_idx, show_debug, is_single_timestep
                )
                
                # Compute hierarchical return using standard recursive formula
                if episode_terminated:
                    hierarchical_return = cumulative_reward
                else:
                    hierarchical_return = cumulative_reward + self.discount_factor * bootstrap_value
                
                # Store the hierarchical return and compute advantage
                hierarchical_returns[start_idx, env_idx] = hierarchical_return
                hierarchical_advantages[start_idx, env_idx] = hierarchical_return - env_values[start_idx]
                
                total_accumulated_reward += cumulative_reward
        
        if self.debug_mode:
            safe_wandb_log({
                "debug/l1_accumulation/total_decision_points": total_decision_points,
                "debug/l1_accumulation/single_timestep_skills": single_timestep_skills,
                "debug/l1_accumulation/single_timestep_ratio": single_timestep_skills / max(1, total_decision_points),
                "debug/l1_accumulation/avg_accumulated_reward": total_accumulated_reward / max(1, total_decision_points),
            })
        
        return hierarchical_returns, hierarchical_advantages

    def _compute_skill_level_reward(self, env_rewards, env_dones, start_idx, end_idx, show_debug=False):
        """
        Compute skill-level reward for L2 using temporal abstraction.
        Different methods available:
        - "final": Use the final reward at end of skill execution
        - "sum": Sum all rewards during skill execution (treated as single skill outcome)
        - "average": Average of all rewards during skill execution
        """
        if start_idx >= end_idx:
            return 0.0
            
        skill_rewards = env_rewards[start_idx:end_idx]
        
        if self.temporal_abstraction_method == "final":
            # Method 1: Use the final reward of the skill execution period
            # This represents the outcome of the entire skill
            skill_reward = skill_rewards[-1].item() if len(skill_rewards) > 0 else 0.0
            method_desc = "final"
            
        elif self.temporal_abstraction_method == "sum":
            # Method 2: Sum all rewards but treat as single skill outcome
            skill_reward = skill_rewards.sum().item()
            method_desc = "sum"
            
        elif self.temporal_abstraction_method == "average":
            # Method 3: Average all rewards during skill execution
            skill_reward = skill_rewards.mean().item() if len(skill_rewards) > 0 else 0.0
            method_desc = "average"
            
        else:
            raise ValueError(f"Unknown temporal abstraction method: {self.temporal_abstraction_method}")
        
        if show_debug:
            print(f"      🏆 Skill-level reward ({method_desc}): {skill_reward:.4f}")
            # For debugging, show statistics of the skill execution period
            if len(skill_rewards) > 0:
                acc_reward = skill_rewards.sum().item()
                avg_reward = skill_rewards.mean().item()
                print(f"      📊 (Debug) Skill period stats: sum={acc_reward:.4f}, avg={avg_reward:.4f}, steps={len(skill_rewards)}")
        
        return skill_reward

    def _compute_l2_skill_level_reward(self, env_rewards, env_dones, start_idx, end_idx, show_debug=False):
        """
        Compute skill-level reward for L2 using temporal abstraction over L1 skill execution.
        
        For L2 training, we need to accumulate rewards over the ENTIRE L1 skill execution period.
        This is the period from when L2 selects an L1 skill until L2 selects the next L1 skill.
        
        Unlike the basic temporal abstraction methods (final, sum, average), this properly
        accounts for the hierarchical structure where L2 decisions control L1 skill execution.
        """
        if start_idx >= end_idx:
            return 0.0
            
        skill_rewards = env_rewards[start_idx:end_idx]
        
        # For L2, we want to accumulate rewards over the L1 skill execution period
        # This gives L2 credit for the outcomes of the L1 skills it selects
        if self.temporal_abstraction_method == "sum":
            # Sum all rewards during L1 skill execution - L2 gets credit for total outcome
            l2_skill_reward = skill_rewards.sum().item()
            method_desc = "sum over L1 execution"
            
        elif self.temporal_abstraction_method == "average":
            # Average reward during L1 skill execution - normalized outcome
            l2_skill_reward = skill_rewards.mean().item() if len(skill_rewards) > 0 else 0.0
            method_desc = "average over L1 execution"
            
        elif self.temporal_abstraction_method == "final":
            # Final reward at end of L1 skill execution - outcome-based
            l2_skill_reward = skill_rewards[-1].item() if len(skill_rewards) > 0 else 0.0
            method_desc = "final reward of L1 execution"
            
        elif self.temporal_abstraction_method == "discounted_sum":
            # Discounted sum with skill_reward_discount - standard temporal abstraction
            l2_skill_reward = 0.0
            for i, reward in enumerate(skill_rewards):
                l2_skill_reward += (self.skill_reward_discount ** i) * reward.item()
            method_desc = "discounted sum over L1 execution"
            
        else:
            raise ValueError(f"Unknown temporal abstraction method for L2: {self.temporal_abstraction_method}")
        
        if show_debug:
            print(f"      🏆 L2 skill-level reward ({method_desc}): {l2_skill_reward:.4f}")
            # For debugging, show statistics of the L1 skill execution period
            if len(skill_rewards) > 0:
                sum_reward = skill_rewards.sum().item()
                avg_reward = skill_rewards.mean().item()
                min_reward = skill_rewards.min().item()
                max_reward = skill_rewards.max().item()
                print(f"      📊 L1 skill execution stats: sum={sum_reward:.4f}, avg={avg_reward:.4f}, min={min_reward:.4f}, max={max_reward:.4f}, steps={len(skill_rewards)}")
        
        return l2_skill_reward

    def _compute_accumulated_reward(self, env_rewards, env_dones, start_idx, end_idx, show_debug=False, is_single_timestep=False):
        """
        Compute accumulated reward for L1 using standardized accumulation method.
        """
        if start_idx >= end_idx:
            return 0.0, 0, False
            
        episode_terminated = False
        step_rewards = []
        
        # Extract rewards for this skill execution period
        skill_rewards = []
        for step_idx in range(start_idx, end_idx):
            if step_idx >= len(env_rewards):
                break
                
            step_reward = env_rewards[step_idx]
            skill_rewards.append(step_reward)
            
            if show_debug:
                step_rewards.append(step_reward.item())
            
            # Check for episode termination
            if env_dones[step_idx]:
                episode_terminated = True
                if show_debug:
                    print(f"       Episode terminated at step {step_idx}")
                break
        
        # Use standardized reward accumulation
        cumulative_reward = self._accumulate_rewards_standardized(
            torch.stack(skill_rewards) if skill_rewards else torch.tensor([]), 
            self.skill_reward_discount
        )
        
        effective_horizon = len(skill_rewards)
        
        if show_debug:
            print(f"      📈 Rewards: {step_rewards[:5]}{'...' if len(step_rewards) > 5 else ''} (total: {len(step_rewards)} steps)")
            print(f"      💰 Standardized accumulated reward: {cumulative_reward:.4f}")
        
        return cumulative_reward, effective_horizon, episode_terminated

    def sample_decision_steps_only(self, names, mini_batches=1):
        """
        Sample only decision steps for training by filtering based on stored decision masks.
        
        Note: GAE computation should be done in HPPO._update() before calling this method,
        following the standard SKRL PPO pattern.
        """
        if self.debug_mode:
            safe_wandb_log({
                "debug/sampling/started": True,
                "debug/sampling/requested_names": names,
                "debug/sampling/mini_batches": mini_batches,
                "debug/sampling/memory_filled_size": getattr(self, '_filled_size', -1),
            })
        
        # Check decision mask statistics before processing
        decision_mask_tensor = self.get_tensor_by_name("decision_mask")
        total_stored_steps = decision_mask_tensor.shape[0] * decision_mask_tensor.shape[1]
        total_decision_steps = decision_mask_tensor.sum().item()
        
        if self.debug_mode:
            safe_wandb_log({
                "debug/sampling/total_stored_steps": total_stored_steps,
                "debug/sampling/total_decision_steps": total_decision_steps,
                "debug/sampling/decision_ratio": total_decision_steps / max(1, total_stored_steps),
            })
        
        # GAE computation is now handled in HPPO._update() - no need to recompute here
        
        # Proceed with normal sampling and filtering
        all_samples = super().sample_all(names=names, mini_batches=mini_batches)
        
        if self.debug_mode:
            safe_wandb_log({
                "debug/sampling/initial_batches": len(all_samples),
            })
        
        # Filter each mini-batch to only include decision steps
        filtered_batches = []
        total_decision_samples = 0
        total_original_samples = 0
        
        for batch_idx, batch in enumerate(all_samples):
            # Find the decision_mask in the batch
            if "decision_mask" in names:
                decision_mask_idx = names.index("decision_mask")
                decision_mask = batch[decision_mask_idx]
                
                # Flatten the decision mask for proper indexing
                if decision_mask.dim() > 1:
                    decision_mask = decision_mask.squeeze(-1)  # Remove last dimension
                    if decision_mask.dim() > 1:
                        decision_mask = decision_mask.flatten()  # Flatten completely if needed
                
                batch_size = decision_mask.shape[0]
                decision_count_batch = decision_mask.sum().item()
                total_original_samples += batch_size

                decision_indices = torch.where(decision_mask.bool())[0]
                if len(decision_indices) == 0:
                    # Check if we have ANY decision steps in memory at all
                    all_decision_mask = self.get_tensor_by_name("decision_mask")
                    if all_decision_mask.dim() > 1:
                        all_decision_mask = all_decision_mask.squeeze(-1)
                    total_decisions = all_decision_mask.sum().item()
                    
                    if total_decisions == 0:
                        if self.debug_mode:
                            safe_wandb_log({"debug/sampling/error": "no_decision_steps_in_memory"})
                        raise RuntimeError("No decision steps found in memory! Check decision point marking logic.")
                    else:
                        # Skip this specific batch but warn about it
                        if self.debug_mode:
                            safe_wandb_log({
                                f"debug/sampling/batch_{batch_idx}/warning": "empty_batch",
                                f"debug/sampling/batch_{batch_idx}/total_decisions_in_memory": total_decisions,
                            })
                        continue
                
                # Filter all tensors in this batch using the decision mask
                filtered_batch = []
                
                for i, tensor in enumerate(batch):
                    if i == decision_mask_idx:
                        # Skip the decision mask itself for training
                        continue
                    else:
                        # Filter tensor using decision mask
                        tensor_name = names[i] if i < len(names) else f"tensor_{i}"
                        
                        if tensor.dim() == 1:
                            # 1D tensor case
                            filtered_tensor = tensor[decision_mask.bool()]
                        elif tensor.dim() == 2:
                            # 2D tensor case [batch_size, feature_dim]
                            filtered_tensor = tensor[decision_mask.bool()]
                        else:
                            # Handle higher dimensional tensors by keeping the last dimensions
                            # and only filtering on the first dimension
                            mask_indices = torch.where(decision_mask.bool())[0]
                            filtered_tensor = tensor[mask_indices]
                        
                        filtered_batch.append(filtered_tensor)
                
                if filtered_batch:  # Only add if we have data
                    decision_count = decision_mask.sum().item()
                    total_decision_samples += decision_count
                    filtered_batches.append(tuple(filtered_batch))
            else:
                # No decision mask, return all samples
                if self.debug_mode:
                    safe_wandb_log({f"debug/sampling/batch_{batch_idx}/warning": "no_decision_mask"})
                filtered_batches.append(batch)
                if batch:
                    total_decision_samples += batch[0].shape[0]  # Assume first tensor gives batch size
                    total_original_samples += batch[0].shape[0]
        
        # Final summary
        if self.debug_mode:
            safe_wandb_log({
                "debug/sampling/original_samples": total_original_samples,
                "debug/sampling/filtered_decision_samples": total_decision_samples,
                "debug/sampling/retention_rate": total_decision_samples / max(1, total_original_samples),
                "debug/sampling/final_batches": len(filtered_batches),
                "debug/sampling/completed": True,
            })
        
        if len(filtered_batches) == 0:
            if self.debug_mode:
                safe_wandb_log({"debug/sampling/error": "all_batches_empty"})
            raise RuntimeError("All batches were empty after filtering! Check decision point frequency and memory size.")
        
        return filtered_batches 