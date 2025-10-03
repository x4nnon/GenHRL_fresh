from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch

 

def build_stairs_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "build_stairs_primary_reward") -> torch.Tensor:
    """
    Single reward: negative XY distance between Object2↔Object1 and Object3↔Object2.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    small_block = env.scene['Object1']
    medium_block = env.scene['Object2']
    large_block = env.scene['Object3']

    # Positions
    small_block_pos = small_block.data.root_pos_w
    medium_block_pos = medium_block.data.root_pos_w
    large_block_pos = large_block.data.root_pos_w

    # XY distances
    dist_obj2_obj1_xy = torch.norm(medium_block_pos[:, :2] - small_block_pos[:, :2], dim=1)
    dist_obj3_obj2_xy = torch.norm(large_block_pos[:, :2] - medium_block_pos[:, :2], dim=1)

    # Reward is negative sum of XY distances
    reward = -(dist_obj2_obj1_xy + dist_obj3_obj2_xy)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

# Removed shaping rewards for simplicity


 


@configclass
class TaskRewardsCfg:
    """
    Configuration for the reward terms used in the Build_Stairs skill.
    """
    # Single reward term: negative XY distances
    build_stairs_primary_reward = RewTerm(
        func=build_stairs_primary_reward,
        weight=0.0,
        params={"normalise": True, "normaliser_name": "build_stairs_primary_reward"},
    )