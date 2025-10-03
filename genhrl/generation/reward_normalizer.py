import torch
from typing import Dict, Optional
from dataclasses import dataclass, field
import wandb
import math
from torch.utils.tensorboard import SummaryWriter
import os
import sys

def _get_normalization_mode_from_cli() -> str:
    """Automatically detect normalization mode from CLI args or environment.

    Precedence:
    1) --normaliser on CLI
    2) ENV GENHRL_NORMALISER or GENHRL_REWARD_NORMALISER
    3) default "custom"
    """
    # Check if --normaliser argument is provided
    for i, arg in enumerate(sys.argv):
        if arg == "--normaliser" and i + 1 < len(sys.argv):
            normaliser_value = sys.argv[i + 1]
            if normaliser_value == "standard":
                return "standard"
            elif normaliser_value == "None":
                return "None"
            elif normaliser_value == "custom":
                return "custom"
            else:
                print(f"WARNING: Unknown normaliser value '{normaliser_value}', falling back to 'custom'")
                return "custom"
        elif arg.startswith("--normaliser="):
            normaliser_value = arg.split("=", 1)[1]
            if normaliser_value == "standard":
                return "standard"
            elif normaliser_value == "None":
                return "None"
            elif normaliser_value == "custom":
                return "custom"
            else:
                print(f"WARNING: Unknown normaliser value '{normaliser_value}', falling back to 'custom'")
                return "custom"
    # Check environment variables (Hydra may clear sys.argv)
    env_val = os.environ.get("GENHRL_NORMALISER") or os.environ.get("GENHRL_REWARD_NORMALISER")
    if env_val:
        if env_val in ("standard", "None", "custom"):
            return env_val
        else:
            print(f"WARNING: Unknown GENHRL_NORMALISER '{env_val}', falling back to 'custom'")
            return "custom"

    # Default to custom if no --normaliser argument found
    return "custom"

@dataclass
class RewardStats:
    """Statistics for reward normalization with enhanced stability."""
    running_mean: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    running_var: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    running_mean_sq: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))  # For more stable variance
    count: int = 0
    debug_counter: int = 0
    # Stability tracking
    recent_values: torch.Tensor = field(default_factory=lambda: torch.zeros(100))  # Fixed-size circular buffer
    buffer_idx: int = 0
    buffer_full: bool = False
    # Adaptive parameters
    adaptive_momentum: float = 0.999
    stability_factor: float = 1.0
    # Performance cache
    _cached_std: Optional[torch.Tensor] = None
    _cached_effective_std: Optional[torch.Tensor] = None
    _cache_valid: bool = False


class RewardNormalizer:
    """Robust reward normalizer with adaptive behavior and stability guarantees."""
    
    def __init__(self, device: torch.device, debug_frequency: int = 100, writer: Optional[SummaryWriter] = None, normalization_mode: Optional[str] = None):
        self.device = device
        self.stats: Dict[str, RewardStats] = {}
        self.debug_frequency = debug_frequency  # Configurable debug frequency
        self.writer = writer  # Store TensorBoard writer
        
        # Auto-detect normalization mode from CLI if not explicitly provided
        if normalization_mode is None:
            self.normalization_mode = _get_normalization_mode_from_cli()
        else:
            self.normalization_mode = normalization_mode
        
        # Base parameters - more conservative
        self.base_momentum = 0.995  # Lower base momentum
        self.epsilon = 1e-8
        self.min_std_clamp = 0.01  # Higher minimum to prevent division by tiny numbers
        self.max_std_clamp = 10.0  # Prevent exploding gradients
        
        # Stability parameters
        self.max_norm_value = 3.0  # Hard limit on normalized values
        self.convergence_threshold = 0.01  # When to consider converged
        self.disturbance_threshold = 2.0  # When to adapt to disturbances
        self.buffer_size = 100  # Size of recent values buffer
        
        # Adaptive behavior
        self.min_momentum = 0.99  # Fastest adaptation
        self.max_momentum = 0.9999  # Slowest adaptation
        # Standard normalization EMA momentum (used when normalization_mode == "standard")
        self.standard_momentum = 0.99
        
        # Performance optimizations - pre-compute constants
        self.momentum_smooth_factor = 0.9
        self.momentum_update_factor = 0.1
        self.stability_increase_factor = 1.01
        self.stability_decrease_factor = 0.95
        
    def _invalidate_cache(self, stats: RewardStats):
        """Invalidate cached values for a stats object."""
        stats._cache_valid = False
        
    def _get_cached_std(self, stats: RewardStats) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached standard deviation or compute and cache it."""
        if not stats._cache_valid or stats._cached_std is None:
            stats._cached_std = torch.sqrt(stats.running_var + self.epsilon).clamp_(
                self.min_std_clamp, self.max_std_clamp)
            stats._cached_effective_std = stats._cached_std * stats.stability_factor
            stats._cache_valid = True
        # Ensure both are tensors (should never be None at this point)
        assert stats._cached_std is not None and stats._cached_effective_std is not None
        return stats._cached_std, stats._cached_effective_std
        
    def _detect_convergence(self, stats: RewardStats) -> bool:
        """Detect if the reward has converged (low recent variance)."""
        if not stats.buffer_full and stats.buffer_idx < 50:
            return False
            
        # Calculate variance of recent values - use view to avoid copying
        if stats.buffer_full:
            recent_var = stats.recent_values.var()
        else:
            recent_var = stats.recent_values[:stats.buffer_idx].var()
            
        return recent_var.item() < self.convergence_threshold
    
    def _detect_disturbance(self, stats: RewardStats, new_value: float) -> bool:
        """Detect if there's a significant disturbance from recent pattern."""
        if not stats.buffer_full and stats.buffer_idx < 10:
            return False
            
        # Calculate how far the new value is from recent pattern - optimize tensor ops
        if stats.buffer_full:
            recent_slice = stats.recent_values
        else:
            recent_slice = stats.recent_values[:stats.buffer_idx]
            
        recent_mean = recent_slice.mean()
        recent_std = recent_slice.std()
            
        if recent_std.item() < self.epsilon:
            return abs(new_value - recent_mean.item()) > self.disturbance_threshold
        else:
            z_score = abs(new_value - recent_mean.item()) / (recent_std.item() + self.epsilon)
            return z_score > self.disturbance_threshold
    
    def _update_adaptive_momentum(self, stats: RewardStats, batch_mean: float):
        """Adapt momentum based on convergence and disturbance detection."""
        is_converged = self._detect_convergence(stats)
        is_disturbed = self._detect_disturbance(stats, batch_mean)
        
        if is_converged and not is_disturbed:
            # Converged and stable - use high momentum
            target_momentum = self.max_momentum
            stats.stability_factor = min(stats.stability_factor * self.stability_increase_factor, 2.0)
        elif is_disturbed:
            # Disturbance detected - adapt quickly
            target_momentum = self.min_momentum
            stats.stability_factor = max(stats.stability_factor * self.stability_decrease_factor, 0.5)
        else:
            # Normal operation
            target_momentum = self.base_momentum
            stats.stability_factor = 1.0
            
        # Smooth transition to target momentum - use pre-computed factors
        stats.adaptive_momentum = (self.momentum_smooth_factor * stats.adaptive_momentum + 
                                 self.momentum_update_factor * target_momentum)
        
        # Invalidate cache since stability_factor changed
        self._invalidate_cache(stats)
        
    def update_stats(self, name: str, values: torch.Tensor):
        """Update running statistics with enhanced stability."""
        # Input validation - optimize nan/inf checking
        has_invalid = torch.isnan(values).any() or torch.isinf(values).any()
        if has_invalid:
            print(f"WARNING ({name}): Input values contain NaN/Inf, replacing with zeros")
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
        if name not in self.stats:
            self.stats[name] = RewardStats()
            stats = self.stats[name]
            # Move tensors to device in batch
            stats.running_mean = stats.running_mean.to(self.device)
            stats.running_var = stats.running_var.to(self.device)
            stats.running_mean_sq = stats.running_mean_sq.to(self.device)
            stats.recent_values = stats.recent_values.to(self.device)
        else:
            stats = self.stats[name]
            
        # If using standard normalization, bypass adaptive logic and use exact running stats
        if self.normalization_mode == "standard":
            stats.debug_counter += 1
            # Compute batch moments
            batch_mean = values.mean()
            batch_mean_sq = (values * values).mean()
            batch_count = int(values.numel())

            prev_count = int(stats.count)
            new_count = prev_count + batch_count

            if prev_count == 0:
                # Initialize with first batch
                stats.running_mean = batch_mean
                stats.running_mean_sq = batch_mean_sq
                stats.running_var = (batch_mean_sq - batch_mean.square()).clamp_(min=0.0)
                stats.count = new_count
            else:
                # Combine previous and batch using weighted means
                # New E[X] = (n*mu + k*mu_b)/(n+k)
                # New E[X^2] = (n*mu2 + k*mu2_b)/(n+k)
                weighted_mean = (stats.running_mean * prev_count + batch_mean * batch_count) / new_count
                weighted_mean_sq = (stats.running_mean_sq * prev_count + batch_mean_sq * batch_count) / new_count
                stats.running_mean = weighted_mean
                stats.running_mean_sq = weighted_mean_sq
                stats.running_var = (weighted_mean_sq - weighted_mean.square()).clamp_(min=0.0)
                stats.count = new_count
            # Invalidate cache for std
            self._invalidate_cache(stats)
            # Optional logging at debug cadence
            if stats.debug_counter % self.debug_frequency == 0:
                current_std, _ = self._get_cached_std(stats)
                log_dict = {f"raw_rewards/{name}": batch_mean.item(), f"running_stats/mean_{name}": stats.running_mean.item(), f"running_stats/std_{name}": current_std.item()}
                wandb.log(log_dict, step=stats.count)
                if self.writer is not None:
                    self.writer.add_scalar(f"raw_rewards/{name}", batch_mean.item(), stats.count)
                    self.writer.add_scalar(f"running_stats/mean_{name}", stats.running_mean.item(), stats.count)
                    self.writer.add_scalar(f"running_stats/std_{name}", current_std.item(), stats.count)
            return

        stats.count += 1
        stats.debug_counter += 1
        
        # OUTLIER CLIPPING: Clip values to prevent outliers from corrupting statistics
        if stats.count > 50:  # Only after we have some stable statistics
            # Use cached std computation
            current_std, _ = self._get_cached_std(stats)
            current_mean = stats.running_mean
            
            # Clip values to be within 3 standard deviations of current mean
            clip_range = 3.0 * current_std
            # Use in-place clamp for better performance
            clipped_values = values.clamp(current_mean - clip_range, current_mean + clip_range)
            
            # Count how many values were clipped - only for debugging
            if stats.debug_counter % self.debug_frequency == 0:
                num_clipped = (values != clipped_values).sum().item()
                if num_clipped > 0:
                    pass
                    # print(f"OUTLIER CLIPPING ({name}): Clipped {num_clipped}/{len(values)} values "
                    #       f"(mean={current_mean.item():.4f}, std={current_std.item():.4f})")
            
            values = clipped_values
        
        # Calculate batch statistics - combine operations for efficiency
        batch_mean = values.mean()
        batch_var = values.var().clamp_(min=0.0)  # In-place clamp
        batch_mean_sq = (values * values).mean()  # Avoid ** operator
        
        # Update circular buffer of recent values - single operation
        stats.recent_values[stats.buffer_idx] = batch_mean
        stats.buffer_idx = (stats.buffer_idx + 1) % self.buffer_size
        if stats.buffer_idx == 0:
            stats.buffer_full = True
            
        # Update adaptive momentum
        self._update_adaptive_momentum(stats, batch_mean.item())
        
        # Initialize or update statistics
        if stats.count == 1:
            stats.running_mean = batch_mean
            stats.running_var = batch_var
            stats.running_mean_sq = batch_mean_sq
        else:
            momentum = stats.adaptive_momentum
            inv_momentum = 1.0 - momentum  # Pre-compute
            
            # Update mean - use in-place operations where possible
            stats.running_mean.mul_(momentum).add_(batch_mean, alpha=inv_momentum)
            
            # Update variance using more stable method
            stats.running_mean_sq.mul_(momentum).add_(batch_mean_sq, alpha=inv_momentum)
            # Variance = E[X²] - E[X]²
            computed_var = stats.running_mean_sq - stats.running_mean.square()
            stats.running_var = computed_var.clamp_(min=0.0)
            
        # Invalidate cache after updating stats
        self._invalidate_cache(stats)
            
        # Debug logging and wandb tracking (configurable frequency)
        if stats.debug_counter % self.debug_frequency == 0:
            # Additional diagnostic: warn if running mean is ~0 and variance huge -> likely zeroed rewards
            if os.environ.get("GENHRL_REWARD_DEBUG", "0") == "1":
                # This diagnostic does not know the term weight directly, but can infer if values are tiny
                msg_extra = ""
                if abs(stats.running_mean.item()) < 1e-6:
                    msg_extra = " [Running mean ~0 — check term weight or reward calc]"
                print(f"[REWARD DEBUG] Term '{name}': mean={stats.running_mean.item():.4e}, std={current_std.item():.4e}{msg_extra}")

            current_std, _ = self._get_cached_std(stats)
            # print(f"Reward {name}: mean={stats.running_mean.item():.4f}, "
            #       f"std={current_std.item():.4f}, momentum={stats.adaptive_momentum:.3f}, "
            #       f"stability={stats.stability_factor:.2f}")
            
            # Log to wandb - batch the logging
            log_dict = {
                f"raw_rewards/{name}": batch_mean.item(),
                f"running_stats/mean_{name}": stats.running_mean.item(),
                f"running_stats/std_{name}": current_std.item(),
                f"adaptive_params/momentum_{name}": stats.adaptive_momentum,
                f"adaptive_params/stability_factor_{name}": stats.stability_factor,
            }
            wandb.log(log_dict, step=stats.count)

            # Log to TensorBoard if writer is available
            if self.writer is not None:
                self.writer.add_scalar(f"raw_rewards/{name}", batch_mean.item(), stats.count)
                self.writer.add_scalar(f"running_stats/mean_{name}", stats.running_mean.item(), stats.count)
                self.writer.add_scalar(f"running_stats/std_{name}", current_std.item(), stats.count)
                self.writer.add_scalar(f"adaptive_params/momentum_{name}", stats.adaptive_momentum, stats.count)
                self.writer.add_scalar(f"adaptive_params/stability_factor_{name}", stats.stability_factor, stats.count)
    
    def normalize(self, name: str, values: torch.Tensor) -> torch.Tensor:
        """Normalize values based on the normalization mode."""
        # Input validation - optimize checking
        has_invalid = torch.isnan(values).any() or torch.isinf(values).any()
        if has_invalid:
            print(f"WARNING ({name}): Input values to normalize contain NaN/Inf")
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle "None" mode - no normalization
        if self.normalization_mode == "None":
            return values
        
        # Handle "standard" mode - simple mean/std normalization
        if self.normalization_mode == "standard":
            if name not in self.stats:
                # No stats yet - return raw values
                return values
                
            stats = self.stats[name]
            mean = stats.running_mean
            std = torch.sqrt(stats.running_var + self.epsilon)
            
            # Simple normalization: (values - mean) / std
            normalized = (values - mean) / std
            
            # Basic safety checks
            has_invalid_norm = torch.isnan(normalized).any() or torch.isinf(normalized).any()
            if has_invalid_norm:
                print(f"WARNING ({name}): Standard normalization produced invalid values, clamping")
                normalized = torch.nan_to_num(normalized, nan=0.0, posinf=3.0, neginf=-3.0)
            
            return normalized
        
        # Handle "custom" mode - original complex adaptive behavior
        if self.normalization_mode == "custom":
            if name not in self.stats:
                # No stats yet - return clamped raw values
                return values.clamp(-self.max_norm_value, self.max_norm_value)
                
            stats = self.stats[name]
            
            # Early stage - use simple scaling
            if stats.count < 10:
                abs_max = values.abs().max()
                if abs_max < self.epsilon:
                    return values
                scale_factor = min(1.0, self.max_norm_value / (abs_max.item() + self.epsilon))
                return values * scale_factor
            
            # Robust normalization
            mean = stats.running_mean
            var = stats.running_var
            
            # Enhanced stability checks - batch the validation
            mean_invalid = torch.isnan(mean) or torch.isinf(mean)
            var_invalid = torch.isnan(var) or torch.isinf(var) or var < 0
            
            if mean_invalid:
                print(f"WARNING ({name}): Running mean is invalid, resetting")
                stats.running_mean.zero_()
                mean = stats.running_mean
                self._invalidate_cache(stats)
                
            if var_invalid:
                print(f"WARNING ({name}): Running variance is invalid, resetting")
                stats.running_var.fill_(1.0)
                var = stats.running_var
                self._invalidate_cache(stats)
            
            # Use cached standard deviation computation
            _, effective_std = self._get_cached_std(stats)
            
            # Normalize - use in-place operations where safe
            normalized = (values - mean) / effective_std
            
            # Final safety checks and clamping
            has_invalid_norm = torch.isnan(normalized).any() or torch.isinf(normalized).any()
            if has_invalid_norm:
                print(f"WARNING ({name}): Normalization produced invalid values, clamping")
                normalized = torch.nan_to_num(normalized, nan=0.0, 
                                            posinf=self.max_norm_value, neginf=-self.max_norm_value)
            
            # Soft clamping with preserving gradients - in-place
            normalized.clamp_(-self.max_norm_value, self.max_norm_value)
            
            # Log normalized rewards occasionally
            if stats.debug_counter % self.debug_frequency == 0:
                log_dict = {
                    f"normalized_rewards/{name}": normalized.mean().item(),
                    f"normalized_rewards/std_{name}": normalized.std().item(),
                }
                wandb.log(log_dict, step=stats.count)
            
            return normalized
        
        # Fallback for unknown modes - treat as "custom"
        print(f"WARNING: Unknown normalization mode '{self.normalization_mode}', falling back to 'custom'")
        self.normalization_mode = "custom"
        return self.normalize(name, values)
    
    def reset_stats(self, name: str):
        """Reset statistics for a specific reward term."""
        if name in self.stats:
            del self.stats[name]
            
    def get_stats_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all current statistics."""
        summary = {}
        for name, stats in self.stats.items():
            std, _ = self._get_cached_std(stats)
            summary[name] = {
                'mean': stats.running_mean.item(),
                'std': std.item(),
                'count': stats.count,
                'momentum': stats.adaptive_momentum,
                'stability_factor': stats.stability_factor
            }
        return summary

# Global normalizer instance
_NORMALIZER: Optional[RewardNormalizer] = None

def get_normalizer(device: Optional[torch.device] = None, debug_frequency: int = 100, writer: Optional[SummaryWriter] = None, normalization_mode: Optional[str] = None) -> RewardNormalizer:
    """Get or create the global normalizer instance.
    
    Args:
        device: PyTorch device to use
        debug_frequency: How often to log debug information
        writer: TensorBoard writer for logging
        normalization_mode: One of "custom", "standard", or "None". If None, auto-detects from CLI args.
            - "custom": Complex adaptive normalization (default, original behavior)
            - "standard": Simple mean/std normalization
            - "None": No normalization (returns raw values)
    """
    global _NORMALIZER
    if _NORMALIZER is None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _NORMALIZER = RewardNormalizer(device, debug_frequency, writer, normalization_mode)
    return _NORMALIZER


def get_normalizer_from_args(args, device: Optional[torch.device] = None, debug_frequency: int = 100, writer: Optional[SummaryWriter] = None) -> RewardNormalizer:
    """Get or create the global normalizer instance using args.normaliser.
    
    Args:
        args: Arguments object with normaliser attribute
        device: PyTorch device to use
        debug_frequency: How often to log debug information
        writer: TensorBoard writer for logging
        
    Returns:
        RewardNormalizer instance configured based on args.normaliser
    """
    # Determine normalization mode from args
    if not hasattr(args, 'normaliser') or args.normaliser is None or args.normaliser == "custom":
        normalization_mode = "custom"
    elif args.normaliser == "standard":
        normalization_mode = "standard"
    elif args.normaliser == "None":
        normalization_mode = "None"
    else:
        print(f"WARNING: Unknown normaliser value '{args.normaliser}', falling back to 'custom'")
        normalization_mode = "custom"
    
    return get_normalizer(device, debug_frequency, writer, normalization_mode) 