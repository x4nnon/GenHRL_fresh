from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch

from isaaclab.envs import mdp
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

# Define cube and platform dimensions based on the task description
# "three of these objects should be 0.5m cubed blocks" -> half size is 0.25m
CUBE_HALF_SIZE = 0.25
# "The fourth object should be a platform x=2m y=2m"
PLATFORM_X_SIZE = 2.0
PLATFORM_Y_SIZE = 2.0
# Platform Z height is typically very small, like 0.001m, but we only care about the cube's Z position relative to the platform's Z.
# The platform's Z position is its root, so the top surface is at root_pos_w[:, 2] + PLATFORM_Z_HEIGHT / 2.
# However, the task description says "z=0.001" for the platform, implying its root is at 0.001.
# So, the top surface is effectively at 0.001.
# The cube's center should be at platform_z + cube_half_size.

def main_move_three_objects_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Simple distance-based reward for moving three cubes toward the platform.
    Returns negative absolute distance from platform center for all three blocks combined.
    Closer blocks to platform center = higher reward.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    platform = env.scene['Object4']

    # Access object positions
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w
    platform_pos = platform.data.root_pos_w

    # Calculate distance from each object to platform center
    distance_obj1_x = torch.abs(object1_pos[:, 0] - platform_pos[:, 0])
    distance_obj1_y = torch.abs(object1_pos[:, 1] - platform_pos[:, 1])
    distance_obj1_z = torch.abs(object1_pos[:, 2] - platform_pos[:, 2])
    
    distance_obj2_x = torch.abs(object2_pos[:, 0] - platform_pos[:, 0])
    distance_obj2_y = torch.abs(object2_pos[:, 1] - platform_pos[:, 1])
    distance_obj2_z = torch.abs(object2_pos[:, 2] - platform_pos[:, 2])
    
    distance_obj3_x = torch.abs(object3_pos[:, 0] - platform_pos[:, 0])
    distance_obj3_y = torch.abs(object3_pos[:, 1] - platform_pos[:, 1])
    distance_obj3_z = torch.abs(object3_pos[:, 2] - platform_pos[:, 2])
    
    # Sum distances for all three objects (negative reward - closer is better)
    total_distance = (distance_obj1_x + distance_obj1_y + 
                     distance_obj2_x + distance_obj2_y +
                     distance_obj3_x + distance_obj3_y)
    
    # Return negative distance as reward (closer objects = higher reward)
    total_reward = -total_distance

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, total_reward)
        RewNormalizer.update_stats(normaliser_name, total_reward)
        return scaled_reward
    return total_reward



@configclass
class TaskRewardsCfg:
    """
    Reward terms for the 'move_three_objects_seed42' skill.
    """
    # Primary reward for moving cubes onto the platform
    main_reward = RewTerm(func=main_move_three_objects_reward, weight=1.0,
                          params={"normalise": True, "normaliser_name": "main_reward"})
